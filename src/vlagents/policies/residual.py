"""Residual edit policy: a small network that adjusts the VLM's action chunk at control rate.

The VLM writes a nominal chunk of absolute targets once per second. The residual runs in the environment
loop, sees the current robot and task state at every control step and adds a bounded edit (a few
centimetres and degrees) to the nominal target before it is executed. This is the EXPO edit-policy idea with
the VLM chunk as the base action. Training data: demonstrations (deviation of the demonstrated path from the
straight-line chunk the VLM would have written) and hindsight corrections the VLM gives after each chunk.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from vlagents.policies.interface import Obs, SingleAct

ROBOTS = ("left", "right")
PER_ARM_FEATURES = 8
GLOBAL_FEATURES = 3  # phase, holding, task state valid
EDIT_DIM = 6  # xyz + rotation vector per arm


@dataclass
class EditLimits:
    translation: float = 0.02
    """metres per axis"""
    rotation_deg: float = 5.0
    """degrees of rotation vector per axis"""

    @property
    def vector(self) -> np.ndarray:
        return np.array([self.translation] * 3 + [np.deg2rad(self.rotation_deg)] * 3, dtype=np.float32)


def rot6d(rot: Rotation) -> np.ndarray:
    return rot.as_matrix()[:, :2].T.reshape(-1)


def action_pose(action: np.ndarray, control_mode: str) -> tuple[np.ndarray, Rotation]:
    action = np.asarray(action, dtype=float)
    if control_mode == "tquat":
        return action[:3], Rotation.from_quat(action[3:7])
    return action[:3], Rotation.from_euler("xyz", action[3:6])


def format_pose(xyz: np.ndarray, rot: Rotation, control_mode: str) -> np.ndarray:
    if control_mode == "tquat":
        return np.concatenate([xyz, rot.as_quat()]).astype(np.float32)
    return np.concatenate([xyz, rot.as_euler("xyz")]).astype(np.float32)


def apply_edit(action: np.ndarray, edit: np.ndarray, control_mode: str) -> np.ndarray:
    xyz, rot = action_pose(action, control_mode)
    return format_pose(xyz + edit[:3], Rotation.from_rotvec(edit[3:]) * rot, control_mode)


def pose_delta(target: tuple[np.ndarray, Rotation], reference: tuple[np.ndarray, Rotation]) -> np.ndarray:
    """Edit that turns `reference` into `target`: translation and rotation vector (base frame)."""
    return np.concatenate([target[0] - reference[0], (target[1] * reference[1].inv()).as_rotvec()])


def arm_features(
    current: tuple[np.ndarray, Rotation],
    gripper: float,
    nominal: tuple[np.ndarray, Rotation],
    nominal_gripper: float,
    goal: tuple[np.ndarray, Rotation],
) -> np.ndarray:
    """Per-arm features: where the chunk goal lies relative to the nominal target and the gripper states.

    Neither the current nor the absolute nominal tool pose is used. With the current pose, behaviour cloning on
    demonstrations learns to copy the momentary deviation from the straight-line chunk, which at deployment is
    the controller's tracking error and would cancel the commanded motion. With the absolute pose the network
    memorises demonstration windows instead of generalising. Feedback enters through the task state, which for
    the ball maze contains the board pose and therefore the grasp configuration."""
    _ = current, nominal
    return np.concatenate([pose_delta(goal, nominal), [gripper, nominal_gripper]]).astype(np.float32)


def build_features(
    arms: dict[str, np.ndarray],
    phase: float,
    holding: bool,
    task_state: np.ndarray | None,
    task_state_dim: int,
) -> np.ndarray:
    state = np.zeros(task_state_dim, dtype=np.float32)
    valid = 0.0
    if task_state is not None and len(task_state) == task_state_dim:
        state[:] = task_state
        valid = 1.0
    return np.concatenate(
        [arms[robot] for robot in ROBOTS] + [np.array([phase, float(holding), valid], dtype=np.float32), state]
    )


def feature_dim(task_state_dim: int) -> int:
    return len(ROBOTS) * PER_ARM_FEATURES + GLOBAL_FEATURES + task_state_dim


class ResidualMLP:
    """Tiny torch MLP with tanh-bounded outputs; kept behind a thin wrapper so the env loop stays torch-light."""

    def __init__(self, in_dim: int, out_dim: int, limits: np.ndarray, hidden: int = 256):
        import torch
        from torch import nn

        self.torch = torch
        self.limits = torch.as_tensor(limits, dtype=torch.float32)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )
        # start as the identity edit so a warm-started policy never jumps at step 0
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.mean = torch.zeros(in_dim)
        self.std = torch.ones(in_dim)

    def forward(self, x):
        return self.torch.tanh(self.net((x - self.mean) / self.std)) * self.limits

    def zero_input_columns(self, columns: list[int]):
        """Make the given inputs inert until training data with non-zero values teaches their effect."""
        with self.torch.no_grad():
            self.net[0].weight[:, columns] = 0.0

    def predict(self, features: np.ndarray) -> np.ndarray:
        with self.torch.no_grad():
            out = self.forward(self.torch.as_tensor(features, dtype=self.torch.float32)[None])
        return out[0].numpy()

    def state(self) -> dict[str, Any]:
        return {"net": self.net.state_dict(), "mean": self.mean, "std": self.std, "limits": self.limits}

    def load(self, state: dict[str, Any]):
        self.net.load_state_dict(state["net"])
        self.mean = state["mean"]
        self.std = state["std"]
        self.limits = state["limits"]


def load_policy(path: str | Path) -> tuple[ResidualMLP, dict[str, Any]]:
    import torch

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    meta = ckpt["meta"]
    policy = ResidualMLP(meta["in_dim"], meta["out_dim"], np.asarray(meta["limits"]), meta["hidden"])
    policy.load(ckpt["state"])
    return policy, meta


@dataclass
class ChunkRecord:
    """Everything needed to relabel one executed step later."""

    chunk: int
    step: int
    holding: bool
    features: np.ndarray
    nominal: np.ndarray  # (2, action_dim) absolute targets before the edit
    edit: np.ndarray  # (12,) applied edit
    task_state: np.ndarray


@dataclass
class ResidualEditor:
    """Environment-side chunk editor. Without a checkpoint it applies zero edits but still records data."""

    control_mode: str = "xyzrpy"
    ckpt: str | None = None
    task_state_dim: int = 13
    limits: dict[str, float] = field(default_factory=dict)
    record_dir: str | None = None
    enabled: bool = True

    def __post_init__(self):
        self.edit_limits = EditLimits(**self.limits)
        self.policy: ResidualMLP | None = None
        if self.ckpt:
            self.policy, meta = load_policy(self.ckpt)
            self.task_state_dim = meta["task_state_dim"]
        self.records: list[ChunkRecord] = []
        self.chunk_index = -1
        self.chunk_edits: list[np.ndarray] = []
        self.last_summary: dict[str, Any] = {}
        self.episode = 0

    # ----- episode / chunk lifecycle -----

    def reset(self, episode: int):
        self.episode = episode
        self.records = []
        self.chunk_index = -1
        self.chunk_edits = []
        self.last_summary = {}

    def begin_chunk(self):
        self.chunk_index += 1
        self.chunk_edits = []

    def summary(self) -> dict[str, Any]:
        """Intervention statistics of the chunk executed so far (complete at the chunk's last step)."""
        if not self.chunk_edits:
            return {}
        edits = np.stack(self.chunk_edits).reshape(len(self.chunk_edits), len(ROBOTS), EDIT_DIM)
        summary: dict[str, Any] = {
            robot: {
                "max_translation_cm": float(np.abs(edits[:, r, :3]).max() * 100),
                "max_rotation_deg": float(np.rad2deg(np.abs(edits[:, r, 3:]).max())),
                "mean_translation_cm": np.round(edits[:, r, :3].mean(axis=0) * 100, 2).tolist(),
                "mean_rotation_deg": np.round(np.rad2deg(edits[:, r, 3:].mean(axis=0)), 1).tolist(),
            }
            for r, robot in enumerate(ROBOTS)
        }
        summary["steps"] = len(self.chunk_edits)
        summary["chunk"] = self.chunk_index
        return summary

    def end_chunk(self) -> dict[str, Any]:
        self.last_summary = self.summary()
        return self.last_summary

    # ----- editing -----

    def features(
        self,
        nominal: dict[str, SingleAct],
        obs: Obs,
        phase: float,
        holding: bool,
        goal: dict[str, SingleAct],
    ) -> tuple[np.ndarray, np.ndarray]:
        arms = {}
        for robot in ROBOTS:
            single = obs.obs[robot]
            current = action_pose(np.asarray(single.tquat), "tquat")
            nominal_pose = action_pose(nominal[robot].action, self.control_mode)
            goal_pose = action_pose(goal[robot].action, self.control_mode)
            arms[robot] = arm_features(
                current,
                float(single.gripper if single.gripper is not None else 1.0),
                nominal_pose,
                float(nominal[robot].gripper if nominal[robot].gripper is not None else 1.0),
                goal_pose,
            )
        first = next(iter(obs.obs.values()))
        raw_state = first.info.get("task_state")
        task_state = np.asarray(raw_state, dtype=np.float32) if raw_state is not None else None
        features = build_features(arms, phase, holding, task_state, self.task_state_dim)
        return features, (task_state if task_state is not None else np.zeros(self.task_state_dim, np.float32))

    def edit(
        self,
        nominal: dict[str, SingleAct],
        obs: Obs,
        step: int,
        n_steps: int,
        goal: dict[str, SingleAct],
        holding: bool = False,
    ) -> dict[str, SingleAct]:
        if not self.enabled or any(robot not in obs.obs for robot in ROBOTS):
            return nominal
        phase = min(step / max(n_steps - 1, 1), 1.0)
        features, task_state = self.features(nominal, obs, phase, holding, goal)
        edit = self.policy.predict(features) if self.policy is not None else np.zeros(len(ROBOTS) * EDIT_DIM, np.float32)
        edit = np.clip(edit, -np.tile(self.edit_limits.vector, len(ROBOTS)), np.tile(self.edit_limits.vector, len(ROBOTS)))
        edited = dict(nominal)
        for r, robot in enumerate(ROBOTS):
            act = nominal[robot]
            edited[robot] = SingleAct(
                action=apply_edit(act.action, edit[r * EDIT_DIM : (r + 1) * EDIT_DIM], self.control_mode),
                gripper=act.gripper,
                done=act.done,
            )
        self.chunk_edits.append(edit.astype(np.float32))
        self.records.append(
            ChunkRecord(
                chunk=self.chunk_index,
                step=step,
                holding=holding,
                features=features,
                nominal=np.stack([np.asarray(nominal[robot].action, dtype=np.float32) for robot in ROBOTS]),
                edit=edit.astype(np.float32),
                task_state=task_state,
            )
        )
        return edited

    # ----- persistence -----

    def save(self, path: str | Path | None = None) -> Path | None:
        if path is None:
            if self.record_dir is None:
                return None
            path = Path(self.record_dir) / f"episode_{self.episode:03d}.npz"
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self.records:
            return None
        np.savez_compressed(
            path,
            chunk=np.array([r.chunk for r in self.records]),
            step=np.array([r.step for r in self.records]),
            holding=np.array([r.holding for r in self.records]),
            features=np.stack([r.features for r in self.records]),
            nominal=np.stack([r.nominal for r in self.records]),
            edit=np.stack([r.edit for r in self.records]),
            task_state=np.stack([r.task_state for r in self.records]),
            meta=json.dumps(
                {
                    "control_mode": self.control_mode,
                    "task_state_dim": self.task_state_dim,
                    "limits": self.edit_limits.vector.tolist(),
                    "ckpt": self.ckpt,
                }
            ),
        )
        return path
