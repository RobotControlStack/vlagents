from typing import Any, cast

import numpy as np

from vlagents import register_env
from vlagents.envs.interface import EvalEnv
from vlagents.policies.interface import Obs, SingleAct, SingleObs

STAGE_INFO_KEYS = ("success", "stage", "max_stage", "current_subinstruction", "stage_to_subinstructions")
ROBOT_INFO_KEYS = ("collision", "ik_success", "gripper_width")


class RCSDuoBench(EvalEnv):
    """DuoBench tasks with the evaluation config from the DuoBench README (headless, absolute control, 30 Hz).

    env_kwargs:
        robot_keys: robot names, default ["left", "right"]
        control_mode: "joints", "xyzrpy" or "tquat"; also the key of the RCS action dict
        camera_resolution: (width, height) rendered by the sim, default keeps the task default (1280x720)
        max_relative_movement: per step clamp, float (rad) for joints or (m, rad) for cartesian, default None
        frequency: control frequency in Hz, default 30
    """

    def __init__(self, env_id, **env_kwargs):
        self.robot_keys: list[str] = env_kwargs.pop("robot_keys", ["left", "right"])
        self.control_mode: str = env_kwargs.pop("control_mode", "joints")
        self.camera_resolution: tuple[int, int] | None = env_kwargs.pop("camera_resolution", None)
        self.max_relative_movement = env_kwargs.pop("max_relative_movement", None)
        self.frequency: int = env_kwargs.pop("frequency", 30)
        self._instruction: str | None = None
        self._last_info: dict[str, Any] = {}
        super().__init__(env_id, **env_kwargs)

    def make_gym(self):
        import gymnasium as gym
        from rcs._core.sim import SimConfig
        from rcs.envs.base import ControlMode, RelativeTo

        control_modes = {
            "joints": ControlMode.JOINTS,
            "xyzrpy": ControlMode.CARTESIAN_TRPY,
            "tquat": ControlMode.CARTESIAN_TQuat,
        }
        cfg = cast(Any, gym.spec(self.env_id).entry_point).config()
        cfg.headless = True
        cfg.control_mode = control_modes[self.control_mode]
        cfg.relative_to = RelativeTo.NONE
        cfg.max_relative_movement = (
            tuple(self.max_relative_movement)
            if isinstance(self.max_relative_movement, list)
            else self.max_relative_movement
        )
        cfg.sim_cfg = SimConfig(async_control=True, realtime=False, frequency=self.frequency)
        cfg.wrapper_cfg.binary_gripper = True
        if self.camera_resolution is not None and cfg.camera_cfgs is not None:
            for camera_cfg in cfg.camera_cfgs.values():
                camera_cfg.resolution_width, camera_cfg.resolution_height = self.camera_resolution
        return gym.make(self.env_id, cfg=cfg, **self.env_kwargs)

    def translate_obs(self, obs: dict[str, Any]) -> Obs:
        cameras = {key: obs["frames"][key]["rgb"]["data"] for key in obs["frames"]}
        stage_info = {key: self._last_info[key] for key in STAGE_INFO_KEYS if key in self._last_info}
        return Obs(
            obs={
                robot_key: SingleObs(
                    cameras=cameras.copy(),
                    joints=np.asarray(obs[robot_key]["joints"], dtype=np.float32),
                    gripper=float(np.squeeze(obs[robot_key]["gripper"])),
                    xyzrpy=np.asarray(obs[robot_key]["xyzrpy"], dtype=np.float32),
                    tquat=np.asarray(obs[robot_key]["tquat"], dtype=np.float32),
                    info={
                        **stage_info,
                        **{
                            key: self._last_info[robot_key][key]
                            for key in ROBOT_INFO_KEYS
                            if key in self._last_info.get(robot_key, {})
                        },
                    },
                )
                for robot_key in self.robot_keys
            },
            language_instruction=self.language_instruction,
        )

    def step(self, action: dict[str, SingleAct]) -> tuple[Obs, float, bool, bool, dict]:
        env_action = {}
        for robot in self.robot_keys:
            robot_action = action[robot]
            gripper = 0.0 if robot_action.gripper is None else robot_action.gripper
            env_action[robot] = {
                self.control_mode: np.asarray(robot_action.action, dtype=np.float32),
                "gripper": np.asarray([gripper], dtype=np.float32),
            }
        obs, reward, success, truncated, info = self.env.step(env_action)
        self._last_info = info
        return self.translate_obs(obs), float(reward), success, truncated, info

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Obs, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        self._instruction = info["instruction"]
        self._last_info = info
        return self.translate_obs(obs), info

    @property
    def language_instruction(self) -> str:
        assert self._instruction is not None
        return self._instruction

    @staticmethod
    def do_import():
        import rcs  # noqa: F401
        from duobench.tasks import (  # noqa: F401
            ball_maze,
            bin_sort,
            block_balance,
            carry_pot,
            hinge_chest,
            join_blocks,
            pour_marbles,
            spring_door,
            transfer_cube,
            transfer_gate,
            transfer_reorient,
        )


for task in [
    "ball_maze",
    "bin_sort",
    "block_balance",
    "carry_pot",
    "join_blocks",
    "hinge_chest",
    "pour_marbles",
    "spring_door",
    "transfer_cube",
    "transfer_gate",
    "transfer_reorient",
]:
    register_env(f"duobench/{task}", RCSDuoBench)
