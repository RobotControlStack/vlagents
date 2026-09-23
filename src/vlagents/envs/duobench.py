from collections import deque
from typing import Any, cast

import numpy as np
from scipy.spatial.transform import Rotation

import gymnasium as gym

from vlagents import register_env
from vlagents.envs.interface import EvalEnv
from vlagents.policies.interface import Obs, SingleAct, SingleObs

STAGE_INFO_KEYS = ("success", "stage", "max_stage", "current_subinstruction", "stage_to_subinstructions")
ROBOT_INFO_KEYS = ("collision", "ik_success", "gripper_width")

BALL_MAZE_STATE_KEYS = (
    "ball_x",
    "ball_y",
    "ball_vx",
    "ball_vy",
    "goal_x",
    "goal_y",
    "board_roll",
    "board_pitch",
    "board_yaw_sin",
    "board_yaw_cos",
    "board_z",
    "board_x",
    "board_y",
)
"""Privileged ball-maze state: ball position/velocity and goal position in the board frame (m, m/s), board
roll/pitch (rad, shared base frame), board yaw as sin/cos and the board centre in the shared base frame."""
BALL_MAZE_RIM_HALF_WIDTH = 0.108
BALL_MAZE_RIM_HEIGHT = 0.037


class BallMazeState:
    """Reads the ball-maze task state from the simulator (sim only; on hardware a ball detector would supply it)."""

    keys = BALL_MAZE_STATE_KEYS

    def __init__(self, sim, base_origin_world: np.ndarray | None = None):
        self.sim = sim
        self.base_origin_world = base_origin_world

    def vector(self) -> np.ndarray:
        if self.base_origin_world is None:
            # body poses are only valid after the first reset, so the shared base frame is located lazily
            link0 = self.sim.data.body("robotleft_fr3_link0").xpos
            self.base_origin_world = np.asarray(link0 - self.robot_to_base_translation, dtype=float)
        board = self.sim.data.body("board_board")
        rot = board.xmat.reshape(3, 3)
        ball = self.sim.data.body("board_ball")
        ball_rel = rot.T @ (ball.xpos - board.xpos)
        ball_vel = rot.T @ ball.cvel[3:6]
        goal_rel = rot.T @ (self.sim.data.body("board_marker_goal").xpos - board.xpos)
        roll, pitch, yaw = Rotation.from_matrix(rot).as_euler("xyz")
        board_base = board.xpos - self.base_origin_world
        return np.array(
            [
                ball_rel[0],
                ball_rel[1],
                ball_vel[0],
                ball_vel[1],
                goal_rel[0],
                goal_rel[1],
                roll,
                pitch,
                np.sin(yaw),
                np.cos(yaw),
                board_base[2],
                board_base[0],
                board_base[1],
            ],
            dtype=np.float32,
        )

    @staticmethod
    def text(state: np.ndarray, trace: list[np.ndarray] | None = None) -> str:
        s = dict(zip(BALL_MAZE_STATE_KEYS, state.tolist()))
        yaw = np.arctan2(s["board_yaw_sin"], s["board_yaw_cos"])
        rot = Rotation.from_euler("xyz", [s["board_roll"], s["board_pitch"], yaw]).as_matrix()
        centre = np.array([s["board_x"], s["board_y"], s["board_z"]])
        sides = []
        for name, axis in (("+x", [1, 0]), ("-x", [-1, 0]), ("+y", [0, 1]), ("-y", [0, -1])):
            p = centre + rot @ np.array([axis[0] * BALL_MAZE_RIM_HALF_WIDTH, axis[1] * BALL_MAZE_RIM_HALF_WIDTH, BALL_MAZE_RIM_HEIGHT])
            bar_angle = np.rad2deg(yaw) + (90 if axis[0] != 0 else 0)
            sides.append(f"{name} side rim centre at base xyz ({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}), bar runs at {bar_angle:.0f} deg from base +x")
        lines = [
            "Maze board state (board frame: origin at the board centre, axes fixed to the board, cm): "
            f"ball at ({s['ball_x'] * 100:.1f}, {s['ball_y'] * 100:.1f}) moving ({s['ball_vx'] * 100:.1f}, "
            f"{s['ball_vy'] * 100:.1f}) cm/s, goal square at ({s['goal_x'] * 100:.1f}, {s['goal_y'] * 100:.1f}). "
            f"Board centre at base xyz ({centre[0]:.3f}, {centre[1]:.3f}, {centre[2]:.3f}), yaw {np.rad2deg(yaw):.0f} deg "
            f"(angle from the base +x axis to the board +x axis), tilt roll {np.rad2deg(s['board_roll']):.1f} deg, pitch "
            f"{np.rad2deg(s['board_pitch']):.1f} deg (base frame). The ball rolls downhill: it accelerates along the "
            "board axis that points down. " + "; ".join(sides) + "."
        ]
        if trace:
            pts = [trace[i] for i in sorted({0, len(trace) // 2, len(trace) - 1})]
            lines.append(
                "Ball trace over the last command (start, middle, end): "
                + ", ".join(f"({p[0] * 100:.1f}, {p[1] * 100:.1f})" for p in pts)
            )
        return " ".join(lines)


TASK_STATES = {"duobench/ball_maze": BallMazeState}


class RCSDuoBench(EvalEnv):
    """DuoBench tasks with the evaluation config from the DuoBench README (headless, absolute control, 30 Hz).

    env_kwargs:
        robot_keys: robot names, default ["left", "right"]
        control_mode: "joints", "xyzrpy" or "tquat"; also the key of the RCS action dict
        camera_resolution: (width, height) rendered by the sim, default keeps the task default (1280x720)
        max_relative_movement: per step clamp, float (rad) for joints or (m, rad) for cartesian, default None
        frequency: control frequency in Hz, default 30
        task_state: read privileged task state from the simulator into SingleObs.info["task_state"] (vector),
            ["task_state_text"] and ["task_state_trace"] (one entry per step of the last chunk), default False
    """

    def __init__(self, env_id, **env_kwargs):
        self.robot_keys: list[str] = env_kwargs.pop("robot_keys", ["left", "right"])
        self.control_mode: str = env_kwargs.pop("control_mode", "joints")
        self.camera_resolution: tuple[int, int] | None = env_kwargs.pop("camera_resolution", None)
        self.max_relative_movement = env_kwargs.pop("max_relative_movement", None)
        self.frequency: int = env_kwargs.pop("frequency", 30)
        self.task_state_enabled: bool = env_kwargs.pop("task_state", False)
        self._instruction: str | None = None
        self._last_info: dict[str, Any] = {}
        self._task_state: Any | None = None
        self._trace: deque[np.ndarray] = deque(maxlen=30)
        super().__init__(env_id, **env_kwargs)
        if self.task_state_enabled and self.env_id in TASK_STATES:
            cfg = cast(Any, gym.spec(self.env_id).entry_point).config()
            self._task_state = TASK_STATES[self.env_id](self.env.get_wrapper_attr("sim"))
            self._task_state.robot_to_base_translation = cfg.robot_to_shared_base_frame["left"].translation()

    def make_gym(self):
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

    def _task_info(self) -> dict[str, Any]:
        if self._task_state is None:
            return {}
        state = self._task_state.vector()
        self._trace.append(state)
        info: dict[str, Any] = {"task_state": state.tolist()}
        if self.render_next_step:
            # the agent sees this observation: add the readable summary and the trace of the last chunk
            info["task_state_keys"] = list(self._task_state.keys)
            info["task_state_text"] = self._task_state.text(state, list(self._trace))
            info["task_state_trace"] = [t.tolist() for t in self._trace]
            if self.editor is not None and hasattr(self.editor, "summary"):
                # the observation is rendered at the chunk's last step, before end_chunk(): summarise the edits so far
                summary = self.editor.summary()
                if summary:
                    info["residual_summary"] = summary
        return info

    def translate_obs(self, obs: dict[str, Any]) -> Obs:
        cameras = {key: obs["frames"][key]["rgb"]["data"] for key in obs["frames"]}
        stage_info = {key: self._last_info[key] for key in STAGE_INFO_KEYS if key in self._last_info}
        stage_info.update(self._task_info())
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
        env_action: dict[str, Any] = {"render": self.render_next_step}
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
        self._trace.clear()
        self.render_next_step = True
        translated = self.translate_obs(obs)
        self.last_obs = translated
        return translated, info

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
