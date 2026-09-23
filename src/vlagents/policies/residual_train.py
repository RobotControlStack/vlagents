"""Train the residual edit policy.

    python -m vlagents.policies.residual_train warmstart <lerobot_dir> <out.pt> --replay-dir <duobench replay dir>
    python -m vlagents.policies.residual_train finetune <in.pt> <out.pt> <run_dir> [<run_dir> ...]

warmstart: behaviour cloning from demonstrations. For every window of `chunk` frames the straight-line chunk the
VLM would have written (from the start pose to the reached pose) is the nominal; the label is the demonstrated
deviation from it at each step. Task state (ball, board) is only known for recordings with a simulator state
(DuoBench replay parquet); LeRobot demonstrations get a zero task state with the "valid" flag off.

finetune: dataset aggregation with hindsight corrections. Every executed step recorded by the ResidualEditor is
relabelled with edit + correction of the VLM (from the agent log of the same run) and the policy is trained on the
union of warm-start data and corrections.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import typer
from scipy.spatial.transform import Rotation, Slerp

from vlagents.policies.residual import (
    EDIT_DIM,
    ROBOTS,
    EditLimits,
    ResidualMLP,
    arm_features,
    build_features,
    feature_dim,
    pose_delta,
)

app = typer.Typer(add_completion=False)

BALL_MAZE_TASK_STATE_DIM = 13
BALL_MAZE_TASK_STATE_SCALE = np.array([0.1, 0.1, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 1.0, 1.0, 0.1, 0.2, 0.2], np.float32)
"""fixed normalisation of the task state (m, m/s, rad): data-driven statistics are not available when the warm
start data has no task state"""
GOAL_IN_BOARD = np.array([0.09, -0.09, 0.0033])
BASE_ORIGIN_WORLD_Z = 1.1515  # shared base frame origin above the world origin (FR3 Duo on the Vention table)


def interpolate(start: tuple[np.ndarray, Rotation], end: tuple[np.ndarray, Rotation], n: int) -> list[tuple[np.ndarray, Rotation]]:
    steps = np.linspace(0, 1, n + 1)[1:]
    xyz = start[0] + steps[:, None] * (end[0] - start[0])
    rots = Slerp([0, 1], Rotation.concatenate([start[1], end[1]]))(steps)
    return [(xyz[i], rots[i]) for i in range(n)]


def demo_samples(
    poses: dict[str, list[tuple[np.ndarray, Rotation]]],
    grippers: dict[str, np.ndarray],
    task_states: np.ndarray | None,
    chunk: int,
    stride: int,
    limits: np.ndarray,
    task_state_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Features and labels for every chunk window of one demonstration."""
    length = len(next(iter(poses.values())))
    features, labels = [], []
    for t in range(0, length - chunk - 1, stride):
        goal = {robot: poses[robot][t + chunk] for robot in ROBOTS}
        nominal = {robot: interpolate(poses[robot][t], goal[robot], chunk) for robot in ROBOTS}
        for i in range(chunk):
            arms, label = {}, []
            for robot in ROBOTS:
                current = poses[robot][t + i]
                arms[robot] = arm_features(
                    current, float(grippers[robot][t + i]), nominal[robot][i], float(grippers[robot][t + chunk]), goal[robot]
                )
                label.append(pose_delta(poses[robot][t + i + 1], nominal[robot][i]))
            state = task_states[t + i] if task_states is not None else None
            features.append(build_features(arms, i / (chunk - 1), False, state, task_state_dim))
            labels.append(np.clip(np.concatenate(label), -limits, limits))
    return np.asarray(features, np.float32), np.asarray(labels, np.float32)


def load_lerobot(dataset_dir: Path, robots: tuple[str, ...] = ROBOTS) -> list[dict[str, Any]]:
    """Episodes of a LeRobot v3 dataset as base-frame tool poses per arm (needs RCS forward kinematics)."""
    import pyarrow.parquet as pq
    import rcs
    from rcs.envs.configs import EmptyWorldFR3Duo

    info = json.loads((dataset_dir / "meta" / "info.json").read_text())
    meta = pq.read_table(next((dataset_dir / "meta" / "episodes").rglob("*.parquet"))).to_pandas()
    scene = EmptyWorldFR3Duo()
    cfg = scene.config()
    kinematics = {robot: rcs.common.Pin(*paths) for robot, paths in scene.kinematics_cfg(cfg).items()}
    tcp_offset = cfg.robot_cfgs[robots[0]].tcp_offset
    episodes = []
    for (chunk_index, file_index), rows in meta.groupby(["data/chunk_index", "data/file_index"]):
        data = pq.read_table(dataset_dir / info["data_path"].format(chunk_index=int(chunk_index), file_index=int(file_index))).to_pandas()
        for _, row in rows.iterrows():
            start, length = int(row["dataset_from_index"]), int(row["length"])
            states = np.stack(data.iloc[start : start + length]["observation.state"].to_numpy())
            poses: dict[str, list[tuple[np.ndarray, Rotation]]] = {robot: [] for robot in robots}
            grippers = {}
            for r, robot in enumerate(robots):
                joints = states[:, r * 8 : r * 8 + 7]
                grippers[robot] = states[:, r * 8 + 7]
                for q in joints:
                    pose = cfg.robot_to_shared_base_frame[robot] * kinematics[robot].forward(np.asarray(q, dtype=float), tcp_offset)
                    poses[robot].append((pose.translation(), Rotation.from_quat(pose.rotation_q())))
            episodes.append({"index": int(row["episode_index"]), "poses": poses, "grippers": grippers, "task_states": None})
    return episodes


def ball_maze_task_states(sim_states: np.ndarray, fps: float) -> np.ndarray:
    """Task state vector (see vlagents.envs.duobench.BALL_MAZE_STATE_KEYS) from recorded MuJoCo states whose first
    two joints are the board and the ball free joints (pos xyz + quat wxyz each)."""
    board_pos, board_quat = sim_states[:, 0:3], sim_states[:, 3:7]
    ball_pos = sim_states[:, 7:10]
    rots = Rotation.from_quat(board_quat[:, [1, 2, 3, 0]])
    ball_rel = np.einsum("nij,ni->nj", rots.as_matrix(), ball_pos - board_pos)  # R^T (p_ball - p_board)
    ball_vel = np.gradient(ball_rel, 1.0 / fps, axis=0)
    goal_rel = np.tile(GOAL_IN_BOARD, (len(sim_states), 1))
    rpy = rots.as_euler("xyz")
    return np.stack(
        [
            ball_rel[:, 0],
            ball_rel[:, 1],
            ball_vel[:, 0],
            ball_vel[:, 1],
            goal_rel[:, 0],
            goal_rel[:, 1],
            rpy[:, 0],
            rpy[:, 1],
            np.sin(rpy[:, 2]),
            np.cos(rpy[:, 2]),
            board_pos[:, 2] - BASE_ORIGIN_WORLD_Z,
            board_pos[:, 0],
            board_pos[:, 1],
        ],
        axis=1,
    ).astype(np.float32)


def load_replay(replay_dir: Path, fps: float = 30.0) -> list[dict[str, Any]]:
    """Episodes of a DuoBench/RCS replay recording (parquet with obs, info.sim_state)."""
    import duckdb

    episodes = []
    for uuid in duckdb.sql(f"select distinct uuid from read_parquet('{replay_dir}/*.parquet')").fetchall():
        rows = duckdb.sql(
            "select obs, info.left.sim_state from read_parquet('" + f"{replay_dir}/*.parquet') where uuid = '{uuid[0]}' order by step"
        ).fetchall()
        poses: dict[str, list[tuple[np.ndarray, Rotation]]] = {robot: [] for robot in ROBOTS}
        grippers: dict[str, list[float]] = {robot: [] for robot in ROBOTS}
        sim_states = []
        for obs, sim_state in rows:
            for robot in ROBOTS:
                xyzrpy = np.asarray(obs[robot]["xyzrpy"], dtype=float)
                poses[robot].append((xyzrpy[:3], Rotation.from_euler("xyz", xyzrpy[3:])))
                grippers[robot].append(float(np.squeeze(obs[robot]["gripper"])))
            sim_states.append(sim_state)
        episodes.append(
            {
                "index": uuid[0],
                "poses": poses,
                "grippers": {robot: np.asarray(g) for robot, g in grippers.items()},
                "task_states": ball_maze_task_states(np.asarray(sim_states, dtype=float), fps),
            }
        )
    return episodes


def correction_samples(
    run_dir: Path, limits: np.ndarray, hold_stride: int = 1
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Relabel the residual records of a run with the VLM's hindsight corrections.

    With simulated inference delay most recorded steps are holds (the same nominal repeated while the VLM thinks);
    `hold_stride` keeps only every n-th of them so a chunk's 30 executed steps are not drowned out."""
    records = sorted((run_dir / "residual").glob("episode_*.npz"))
    agent_logs = sorted(p for p in (run_dir / "vlm").iterdir() if p.is_dir()) if (run_dir / "vlm").exists() else []
    features, labels = [], []
    stats = {"chunks": 0, "corrected": 0, "steps": 0}
    for record_path, log_dir in zip(records, agent_logs):
        data = np.load(record_path)
        corrections: dict[int, dict[str, Any]] = {}
        for step_dir in sorted(log_dir.glob("step_*")):
            step = int(step_dir.name.split("_")[1])
            turn = json.loads((step_dir / "turn.json").read_text())
            if turn["commands"].get("correction"):
                corrections[step - 1] = turn["commands"]["correction"]  # step k corrects chunk k-1
        for chunk in np.unique(data["chunk"]):
            mask = data["chunk"] == chunk
            stats["chunks"] += 1
            correction = corrections.get(int(chunk))
            if correction is None:
                continue
            stats["corrected"] += 1
            n = int(np.sum(mask & ~data["holding"]))
            for feats, edit, step, holding in zip(data["features"][mask], data["edit"][mask], data["step"][mask], data["holding"][mask]):
                if holding and step % hold_stride:
                    continue
                phase = 1.0 if holding else step / max(n - 1, 1)
                label = edit.copy()
                for r, robot in enumerate(ROBOTS):
                    entry = correction.get(robot)
                    if entry and entry.get("from", 0.0) <= phase <= entry.get("to", 1.0):
                        delta = np.concatenate(
                            [np.asarray(entry["dxyz"], dtype=float), Rotation.from_euler("xyz", entry["drpy_deg"], degrees=True).as_rotvec()]
                        )
                        label[r * EDIT_DIM : (r + 1) * EDIT_DIM] += delta
                features.append(feats)
                labels.append(np.clip(label, -limits, limits))
                stats["steps"] += 1
    if not features:
        return np.zeros((0, 0), np.float32), np.zeros((0, 0), np.float32), stats
    return np.asarray(features, np.float32), np.asarray(labels, np.float32), stats


def train(
    policy: ResidualMLP,
    features: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    epochs: int,
    lr: float,
    batch_size: int = 256,
    normalize: bool = True,
    weight_decay: float = 1e-4,
    validation: tuple[np.ndarray, np.ndarray] | None = None,
) -> list[float]:
    """Weighted MSE in units of the edit limits. With validation data the best epoch's weights are kept."""
    import copy

    import torch

    x = torch.as_tensor(features)
    y = torch.as_tensor(labels)
    w = torch.as_tensor(weights, dtype=torch.float32)
    if normalize:
        policy.mean = x.mean(0)
        policy.std = x.std(0).clamp_min(1e-3)
        # global flags and the task state keep a fixed scale (they may be constant in the warm-start data)
        n_state = policy.mean.shape[0] - feature_dim(0)
        policy.mean[-n_state - 3 :] = 0.0
        policy.std[-n_state - 3 : -n_state] = 1.0
        policy.std[-n_state:] = torch.as_tensor(BALL_MAZE_TASK_STATE_SCALE[:n_state])
        if not bool((x[:, -n_state - 1 :] != 0).any()):
            # no task state seen: keep those inputs inert until hindsight corrections teach them
            policy.zero_input_columns(list(range(policy.mean.shape[0] - n_state - 1, policy.mean.shape[0])))
            typer.echo("no task state in the training data: task-state inputs start inert")
    opt = torch.optim.AdamW(policy.net.parameters(), lr=lr, weight_decay=weight_decay)
    losses = []
    best, best_state = float("inf"), None
    for epoch in range(epochs):
        perm = torch.randperm(len(x))
        total = 0.0
        for start in range(0, len(x), batch_size):
            idx = perm[start : start + batch_size]
            pred = policy.forward(x[idx])
            loss = (w[idx, None] * (pred - y[idx]) ** 2 / policy.limits**2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        losses.append(total / len(x))
        if validation is not None:
            with torch.no_grad():
                vx, vy = torch.as_tensor(validation[0]), torch.as_tensor(validation[1])
                val = float(((policy.forward(vx) - vy) ** 2 / policy.limits**2).mean())
            if val < best:
                best, best_state = val, copy.deepcopy(policy.net.state_dict())
            if epoch % 5 == 0 or epoch == epochs - 1:
                typer.echo(f"epoch {epoch}: train {losses[-1]:.4f} val {val:.4f}")
    if best_state is not None:
        policy.net.load_state_dict(best_state)
        typer.echo(f"kept the weights with the best validation loss {best:.4f}")
    return losses


def save(policy: ResidualMLP, path: Path, meta: dict[str, Any]):
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": policy.state(), "meta": meta}, path)


@app.command()
def warmstart(
    dataset_dir: Annotated[Path, typer.Argument(help="LeRobot dataset root of the task (joint states only)")],
    output: Annotated[Path, typer.Argument(help="Checkpoint to write (.pt)")],
    replay_dir: Annotated[Path | None, typer.Option(help="DuoBench replay parquet dir with simulator states")] = None,
    episodes: Annotated[int, typer.Option(help="Number of LeRobot episodes to use")] = 50,
    chunk: Annotated[int, typer.Option()] = 30,
    stride: Annotated[int, typer.Option(help="Frames between chunk windows")] = 10,
    epochs: Annotated[int, typer.Option()] = 40,
    lr: Annotated[float, typer.Option()] = 1e-3,
    hidden: Annotated[int, typer.Option()] = 128,
    translation_limit: Annotated[float, typer.Option(help="max edit in m per axis")] = 0.02,
    rotation_limit: Annotated[float, typer.Option(help="max edit in deg per axis")] = 5.0,
    replay_weight: Annotated[float, typer.Option(help="sample weight of replay (task-state) data")] = 3.0,
    val_episodes: Annotated[int, typer.Option(help="last demonstrations held out for validation")] = 10,
    weight_decay: Annotated[float, typer.Option()] = 1e-4,
):
    limits = np.tile(EditLimits(translation_limit, rotation_limit).vector, len(ROBOTS))
    features, labels, weights = [], [], []
    demos = load_lerobot(dataset_dir)[:episodes]
    validation = None
    if val_episodes > 0:
        held_out, demos = demos[-val_episodes:], demos[:-val_episodes]
        vf, vl = zip(*(demo_samples(e["poses"], e["grippers"], None, chunk, stride, limits, BALL_MAZE_TASK_STATE_DIM) for e in held_out))
        validation = (np.concatenate(vf), np.concatenate(vl))
        typer.echo(f"validation: {len(held_out)} demonstrations, {len(validation[0])} samples, zero-edit loss {float(np.mean((validation[1] / limits) ** 2)):.4f}")
    for episode in demos:
        f, l = demo_samples(episode["poses"], episode["grippers"], None, chunk, stride, limits, BALL_MAZE_TASK_STATE_DIM)
        features.append(f), labels.append(l), weights.append(np.ones(len(f)))
    typer.echo(f"{len(demos)} demonstrations -> {sum(len(f) for f in features)} samples (no task state)")
    if replay_dir is not None:
        replays = load_replay(replay_dir)
        n_before = sum(len(f) for f in features)
        for episode in replays:
            f, l = demo_samples(
                episode["poses"], episode["grippers"], episode["task_states"], chunk, max(stride // 3, 1), limits, BALL_MAZE_TASK_STATE_DIM
            )
            features.append(f), labels.append(l), weights.append(np.full(len(f), replay_weight))
        typer.echo(f"{len(replays)} replay episodes -> {sum(len(f) for f in features) - n_before} samples with task state")
    x, y, w = np.concatenate(features), np.concatenate(labels), np.concatenate(weights)
    policy = ResidualMLP(feature_dim(BALL_MAZE_TASK_STATE_DIM), len(ROBOTS) * EDIT_DIM, limits, hidden)
    losses = train(policy, x, y, w, epochs, lr, weight_decay=weight_decay, validation=validation)
    baseline = float(np.mean((y / limits) ** 2))
    typer.echo(f"loss {losses[0]:.4f} -> {losses[-1]:.4f} (zero-edit baseline {baseline:.4f}); label std {y.std(0).round(4).tolist()}")
    meta = {
        "in_dim": feature_dim(BALL_MAZE_TASK_STATE_DIM),
        "out_dim": len(ROBOTS) * EDIT_DIM,
        "hidden": hidden,
        "limits": limits.tolist(),
        "task_state_dim": BALL_MAZE_TASK_STATE_DIM,
        "samples": int(len(x)),
        "losses": losses,
        "stage": "warmstart",
    }
    save(policy, output, meta)
    np.savez_compressed(output.with_suffix(".data.npz"), features=x, labels=y, weights=w)
    typer.echo(f"wrote {output} and {output.with_suffix('.data.npz')}")


@app.command()
def finetune(
    checkpoint: Annotated[Path, typer.Argument(help="Checkpoint to start from")],
    output: Annotated[Path, typer.Argument(help="Checkpoint to write")],
    runs: Annotated[list[Path], typer.Argument(help="Run dirs with residual/ records and vlm/ agent logs")],
    epochs: Annotated[int, typer.Option()] = 60,
    lr: Annotated[float, typer.Option()] = 5e-4,
    prior_weight: Annotated[float, typer.Option(help="sample weight of the warm-start data")] = 0.3,
    correction_weight: Annotated[float, typer.Option(help="sample weight of corrected steps")] = 1.0,
    hold_stride: Annotated[int, typer.Option(help="keep every n-th recorded hold step")] = 10,
):
    import torch

    from vlagents.policies.residual import load_policy

    policy, meta = load_policy(checkpoint)
    limits = np.asarray(meta["limits"], dtype=np.float32)
    features, labels, weights = [], [], []
    prior = checkpoint.with_suffix(".data.npz")
    if prior.exists():
        data = np.load(prior)
        features.append(data["features"]), labels.append(data["labels"]), weights.append(data["weights"] * prior_weight)
        typer.echo(f"prior data: {len(data['features'])} samples")
    total = {"chunks": 0, "corrected": 0, "steps": 0}
    for run in runs:
        f, l, stats = correction_samples(run, limits, hold_stride)
        typer.echo(f"{run}: {stats}")
        for key in total:
            total[key] += stats[key]
        if len(f):
            features.append(f), labels.append(l), weights.append(np.full(len(f), correction_weight))
    if total["steps"] == 0:
        raise typer.BadParameter("no corrected steps found in the given runs")
    x, y, w = np.concatenate(features), np.concatenate(labels), np.concatenate(weights)
    losses = train(policy, x, y, w, epochs, lr, normalize=False)
    typer.echo(f"loss {losses[0]:.4f} -> {losses[-1]:.4f}")
    meta = {**meta, "stage": "finetune", "samples": int(len(x)), "losses": losses, "corrections": total, "runs": [str(r) for r in runs]}
    save(policy, output, meta)
    np.savez_compressed(output.with_suffix(".data.npz"), features=x, labels=y, weights=w / np.where(w > 0, np.maximum(w, 1e-6), 1))
    typer.echo(f"wrote {output}")


@app.command()
def inspect(checkpoint: Annotated[Path, typer.Argument()]):
    import torch

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    meta = dict(ckpt["meta"])
    losses = meta.pop("losses", [])
    typer.echo(json.dumps(meta, indent=2))
    typer.echo(f"losses: {losses[:3]} ... {losses[-3:]}")


if __name__ == "__main__":
    app()
