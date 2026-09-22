"""Export in-context examples for the VLM agent from a LeRobot v3 dataset (e.g. RobotControlStack/duobench).

Takes the first `episodes` episodes, samples every `stride` frames and stores the observation together with the
state `stride` frames later (the target of the next chunk) plus one JPEG keyframe. The agent renders these into
its own state text and command format, so the examples always match the live prompt.

    python -m vlagents.policies.vlm_icl <dataset_dir> <out.json> --episodes 10 --control-mode xyzrpy
"""

import base64
import json
from pathlib import Path
from typing import Annotated

import av
import numpy as np
import pyarrow.parquet as pq
import simplejpeg
import typer
from PIL import Image

app = typer.Typer(add_completion=False)


def fk_tquat(joints: np.ndarray, robot: str, kinematics, robot_to_shared: dict, tcp_offset) -> list[float]:
    pose = robot_to_shared[robot] * kinematics[robot].forward(np.asarray(joints, dtype=float), tcp_offset)
    return np.concatenate([pose.translation(), pose.rotation_q()]).round(5).tolist()


def decode_frames(path: Path, indices: set[int], image_size: int | None) -> dict[int, str]:
    frames = {}
    with av.open(str(path)) as container:
        for i, frame in enumerate(container.decode(video=0)):
            if i > max(indices):
                break
            if i in indices:
                image = frame.to_ndarray(format="rgb24")
                if image_size is not None:
                    image = np.asarray(Image.fromarray(image).resize((image_size, image_size)))
                frames[i] = base64.b64encode(simplejpeg.encode_jpeg(np.ascontiguousarray(image))).decode()
    return frames


@app.command()
def export(
    dataset_dir: Annotated[Path, typer.Argument(help="LeRobot dataset root containing meta/, data/, videos/")],
    output: Annotated[Path, typer.Argument(help="Output json file")],
    episodes: Annotated[int, typer.Option(help="Number of episodes to export (first n)")] = 10,
    stride: Annotated[int, typer.Option(help="Frames between keyframes, equals the agent chunk size")] = 30,
    control_mode: Annotated[str, typer.Option(help="joints or xyzrpy (adds FK tool poses)")] = "xyzrpy",
    camera: Annotated[str, typer.Option(help="Video key used for the keyframe image")] = "observation.images.head",
    image_size: Annotated[int | None, typer.Option(help="Resize keyframes to this square size")] = None,
    robot_keys: Annotated[str, typer.Option(help="Comma separated robot names in state order")] = "left,right",
):
    import rcs
    from rcs.envs.configs import EmptyWorldFR3Duo

    robots = robot_keys.split(",")
    info = json.loads((dataset_dir / "meta" / "info.json").read_text())
    assert episodes <= 10, "keep the in-context set small, at most 10 episodes"
    meta = pq.read_table(next((dataset_dir / "meta" / "episodes").rglob("*.parquet"))).to_pandas()
    meta = meta[meta["episode_index"] < episodes]
    assert (meta["data/file_index"] == meta["data/file_index"].iloc[0]).all(), "episodes span several data files"
    data = pq.read_table(
        dataset_dir
        / info["data_path"].format(
            chunk_index=int(meta["data/chunk_index"].iloc[0]), file_index=int(meta["data/file_index"].iloc[0])
        )
    ).to_pandas()
    video_path = dataset_dir / info["video_path"].format(
        video_key=camera,
        chunk_index=int(meta[f"videos/{camera}/chunk_index"].iloc[0]),
        file_index=int(meta[f"videos/{camera}/file_index"].iloc[0]),
    )
    fps = info["fps"]

    scene = EmptyWorldFR3Duo()
    cfg = scene.config()
    kinematics = {robot: rcs.common.Pin(*paths) for robot, paths in scene.kinematics_cfg(cfg).items()}
    tcp_offset = cfg.robot_cfgs[robots[0]].tcp_offset

    # per episode: (time in the episode, frame index in the video file) of every keyframe
    keyframes: dict[int, list[tuple[int, int]]] = {}
    for _, row in meta.iterrows():
        frame_offset = int(round(row[f"videos/{camera}/from_timestamp"] * fps))
        keyframes[int(row["episode_index"])] = [(t, frame_offset + t) for t in range(0, int(row["length"]) - 1, stride)]
    frames = decode_frames(video_path, {frame for steps in keyframes.values() for _, frame in steps}, image_size)

    def single_obs(state: np.ndarray, robot_idx: int, robot: str) -> dict:
        joints = state[robot_idx * 8 : robot_idx * 8 + 7]
        return {
            "joints": joints.round(5).tolist(),
            "gripper": float(state[robot_idx * 8 + 7]),
            "tquat": fk_tquat(joints, robot, kinematics, cfg.robot_to_shared_base_frame, tcp_offset),
        }

    out_episodes = []
    for _, row in meta.iterrows():
        episode_index, start, length = int(row["episode_index"]), int(row["dataset_from_index"]), int(row["length"])
        rows = data.iloc[start : start + length]
        states = np.stack(rows["observation.state"].to_numpy())
        actions = np.stack(rows["action"].to_numpy())
        steps = []
        for t, frame in keyframes[episode_index]:
            target = actions[min(t + stride - 1, length - 1)]
            steps.append(
                {
                    "t": t,
                    "obs": {robot: single_obs(states[t], i, robot) for i, robot in enumerate(robots)},
                    "target": {robot: single_obs(target, i, robot) for i, robot in enumerate(robots)},
                    "image": frames.get(frame),
                }
            )
        out_episodes.append({"index": episode_index, "instruction": row["tasks"][0], "length": length, "steps": steps})

    output.write_text(
        json.dumps(
            {"control_mode": control_mode, "fps": fps, "stride": stride, "camera": camera, "episodes": out_episodes}
        )
    )
    n_steps = sum(len(e["steps"]) for e in out_episodes)
    typer.echo(
        f"wrote {len(out_episodes)} episodes with {n_steps} keyframes to {output} ({output.stat().st_size / 1e6:.1f} MB)"
    )


if __name__ == "__main__":
    app()
