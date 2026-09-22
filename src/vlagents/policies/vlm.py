"""Vision-language-model agent: a general VLM controls the robot.

Backends: any OpenAI compatible endpoint, the Anthropic API, a file mailbox served by an external pilot such as a
Claude Code subagent or a human, and a fake for tests.

Each call sends the full episode history plus the current cameras and state, and receives one JSON command
per arm. Commands are motion primitives that expand into a chunk of `chunk_size` actions (1 s at 30 Hz).
"""

import base64
import datetime
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import simplejpeg
from PIL import Image
from scipy.spatial.transform import Rotation, Slerp

from vlagents import register_agent
from vlagents.policies.interface import Act, Agent, Obs, SingleAct, SingleObs

logger = logging.getLogger(__name__)

ASSETS = Path(__file__).parent / "assets"

FR3_JOINT_LIMITS_DEG = np.rad2deg(
    np.array(
        [
            [-2.3093, -1.5133, -2.4937, -2.7478, -2.4800, 0.8521, -2.6895],
            [2.3093, 1.5133, 2.4937, -0.4461, 2.4800, 4.2094, 2.6895],
        ]
    )
)

ROBOT_DESCRIPTION = """\
You control a bimanual robot: two Franka Research 3 arms ("left" and "right") mounted side by side on a
shared base, both facing the same table (Franka FR3 Duo). Each arm has 7 joints and a Robotiq 2F-85 parallel
gripper (85 mm max opening). Gripper commands are binary: 1 = open, 0 = closed; closing takes about half a
second. Only the gripper pads can hold objects, so the fingers must be around the object before closing.
The arms can collide with each other, the table and objects; keep the tool a few centimetres above the table
unless you are grasping or placing."""

FRAME_DESCRIPTION = """\
Coordinate frame (shared base frame, metres, right-handed): the origin lies between the two robot bases at
mount height. +x points forward, away from the robots towards the far side of the table. +y points to the
left (towards the left arm, as seen from the robots). +z points up.
Orientation is roll/pitch/yaw in degrees (extrinsic rotations about the fixed x, y, z axes, in that order).
The tool pointing straight down is rpy = (180, 0, 0). The home pose of both arms is rpy = (180, -20, 0),
i.e. pointing down and tilted 20 degrees forward. Changing yaw rotates the gripper about the vertical axis;
keep roll near 180 to keep the gripper pointing down."""

WORKSPACE_DESCRIPTION = """\
Workspace: the table top is at z = -0.33 and spans x in [0.10, 0.89], y in [-0.70, 0.70]. Small objects on
the table have their centre 2 to 5 cm above the table top (z from -0.31 to -0.28); grasp them with the tool
centre point at object-centre height. Both arms start at the home pose: left tool at (0.51, 0.21, 0.20),
right tool at (0.51, -0.21, 0.20). Comfortable reach is x in [0.25, 0.80] and |y| below 0.55; the right arm
reaches the y < 0 half easily and the left arm the y > 0 half, both reach the centre strip |y| < 0.15.
Hand-overs work best in the middle above the table (around x 0.5, y 0.0, z 0.0)."""

CAMERA_DESCRIPTION = """\
Cameras (all RGB): "head" is a fixed camera above and behind the robots looking down onto the table: the
image top is +x (far side), the image left is +y (left arm side). The robot bases are below the bottom edge;
you see the arms' shadows and grippers when they are over the table. "left_wrist" and "right_wrist" are
mounted on the respective gripper and look along the fingers, the fingers are visible at the bottom."""

JOINT_DESCRIPTION = """\
Joint space: each arm is commanded with 7 absolute joint angles in degrees, listed from the base joint (1) to
the flange joint (7). Joint 1 rotates the whole arm about the vertical base axis, joints 2 and 4 are the
shoulder and elbow that move the tool forward/back and up/down, joint 6 pitches the wrist and joint 7 rotates
the gripper about its own axis. Joint limits in degrees, low: {low}, high: {high}. Because the two arms are
mounted mirrored, mirrored motions have opposite signs on joints 1, 3, 5 and 7."""

OUTPUT_RULES = """\
Every reply is exactly one JSON object (no prose outside JSON) with the keys "reasoning" (one or two short
sentences: what you see and what the next second should achieve), one command per arm named after the arm,
and "done" (true only when the task instruction is fully completed).
Your command is expanded into a chunk of {chunk_size} actions executed over {seconds:.1f} seconds at
{fps} Hz, then you are asked again with fresh images. Move in small steps (at most about 15 cm per command)
and verify progress in the next images before continuing. A "gripper" command keeps the arm still for the
whole chunk while the gripper opens or closes; use it right before lifting."""

CARTESIAN_COMMANDS = """\
Commands per arm (Cartesian mode):
  {{"type": "hold"}}                                    keep the current pose and gripper
  {{"type": "gripper", "gripper": 0}}                   keep the pose, set the gripper (0 closed, 1 open)
  {{"type": "move", "xyz": [x, y, z], "rpy_deg": [r, p, y], "gripper": 1}}
        move the tool centre point linearly to the absolute pose; "rpy_deg" is optional (keeps orientation)
  {{"type": "move_delta", "dxyz": [dx, dy, dz], "drpy_deg": [dr, dp, dy], "gripper": 1}}
        move relative to the current pose; "drpy_deg" is optional
  {{"type": "chunk", "actions": [[x, y, z, r, p, y], ...], "gripper": [g, ...]}}
        raw chunk of exactly {chunk_size} absolute poses (rpy in degrees) and gripper values
The move budget per command is {max_translation:.2f} m and {max_rotation:.0f} degrees; larger requests are
shortened along the same direction."""

JOINT_COMMANDS = """\
Commands per arm (joint mode):
  {{"type": "hold"}}                                    keep the current joints and gripper
  {{"type": "gripper", "gripper": 0}}                   keep the joints, set the gripper (0 closed, 1 open)
  {{"type": "move_joints", "joints_deg": [j1, ..., j7], "gripper": 1}}
        interpolate linearly to the absolute joint configuration in degrees
  {{"type": "move_joints_delta", "djoints_deg": [d1, ..., d7], "gripper": 1}}
        interpolate to current joints + delta (degrees)
  {{"type": "chunk", "actions": [[j1, ..., j7], ...], "gripper": [g, ...]}}
        raw chunk of exactly {chunk_size} absolute joint configurations (degrees) and gripper values
Each joint may change by at most {max_joint_delta:.0f} degrees per command; larger requests are scaled down.
Targets are clipped to the joint limits."""


def rpy_deg(rot: Rotation) -> list[int]:
    """Roll/pitch/yaw in whole degrees with roll = +180 (not -180) for a downward pointing tool."""
    rpy = rot.as_euler("xyz", degrees=True)
    if rpy[0] <= -179.5:
        rpy[0] += 360
    return np.round(rpy, 0).astype(int).tolist()


def encode_jpeg(image: np.ndarray, size: int | None = None, quality: int = 90) -> str:
    if size is not None and image.shape[:2] != (size, size):
        image = np.asarray(Image.fromarray(image).resize((size, size), Image.Resampling.BILINEAR))
    return base64.b64encode(simplejpeg.encode_jpeg(np.ascontiguousarray(image), quality=quality)).decode()


def image_part(jpeg_b64: str, detail: str) -> dict[str, Any]:
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{jpeg_b64}", "detail": detail}}


def parse_json_reply(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < 0:
        msg = "reply contains no JSON object"
        raise ValueError(msg)
    reply = json.loads(text[start : end + 1])
    if not isinstance(reply, dict):
        msg = "reply is not a JSON object"
        raise ValueError(msg)
    return reply


class CartesianSpace:
    """Tool-centre-point control. Env action key is "xyzrpy" (radians) or "tquat"."""

    def __init__(self, control_mode: str, chunk_size: int, max_translation: float, max_rotation_deg: float):
        assert control_mode in ("xyzrpy", "tquat")
        self.control_mode = control_mode
        self.chunk_size = chunk_size
        self.max_translation = max_translation
        self.max_rotation_deg = max_rotation_deg

    def describe(self) -> str:
        return "\n\n".join(
            [
                FRAME_DESCRIPTION,
                WORKSPACE_DESCRIPTION,
                CARTESIAN_COMMANDS.format(
                    chunk_size=self.chunk_size,
                    max_translation=self.max_translation,
                    max_rotation=self.max_rotation_deg,
                ),
            ]
        )

    @staticmethod
    def pose(single_obs: SingleObs) -> tuple[np.ndarray, Rotation]:
        tquat = np.asarray(single_obs.tquat, dtype=float)
        return tquat[:3], Rotation.from_quat(tquat[3:])

    def state_text(self, single_obs: SingleObs) -> str:
        xyz, rot = self.pose(single_obs)
        gripper = "open" if (single_obs.gripper or 0) > 0.5 else "closed"
        return f"tool xyz = {np.round(xyz, 3).tolist()} m, rpy = {rpy_deg(rot)} deg, gripper {gripper}"

    def target_command(self, target_obs: SingleObs) -> dict[str, Any]:
        xyz, rot = self.pose(target_obs)
        return {
            "type": "move",
            "xyz": np.round(xyz, 3).tolist(),
            "rpy_deg": rpy_deg(rot),
            "gripper": int(round(target_obs.gripper or 0)),
        }

    def expand(self, command: dict[str, Any], single_obs: SingleObs) -> tuple[np.ndarray, np.ndarray]:
        xyz0, rot0 = self.pose(single_obs)
        gripper0 = float(single_obs.gripper if single_obs.gripper is not None else 1.0)
        kind = command.get("type", "hold")
        if kind == "chunk":
            actions = np.asarray(command["actions"], dtype=float)
            if actions.shape != (self.chunk_size, 6):
                msg = f"chunk must have shape ({self.chunk_size}, 6), got {actions.shape}"
                raise ValueError(msg)
            grippers = np.broadcast_to(np.asarray(command.get("gripper", gripper0), dtype=float), (self.chunk_size,))
            rots = Rotation.from_euler("xyz", actions[:, 3:], degrees=True)
            return self._format(actions[:, :3], rots), grippers.copy()
        if kind in ("hold", "gripper"):
            xyz1, rot1 = xyz0, rot0
        elif kind == "move":
            xyz1 = np.asarray(command["xyz"], dtype=float)
            rot1 = Rotation.from_euler("xyz", command["rpy_deg"], degrees=True) if "rpy_deg" in command else rot0
        elif kind == "move_delta":
            xyz1 = xyz0 + np.asarray(command["dxyz"], dtype=float)
            drot = Rotation.from_euler("xyz", command.get("drpy_deg", [0, 0, 0]), degrees=True)
            rot1 = drot * rot0
        else:
            msg = f"unknown command type {kind!r}"
            raise ValueError(msg)
        delta = xyz1 - xyz0
        if np.linalg.norm(delta) > self.max_translation:
            xyz1 = xyz0 + delta / np.linalg.norm(delta) * self.max_translation
        rel = rot1 * rot0.inv()
        angle = np.rad2deg(rel.magnitude())
        if angle > self.max_rotation_deg:
            rel = Rotation.from_rotvec(rel.as_rotvec() * self.max_rotation_deg / angle)
            rot1 = rel * rot0
        steps = np.linspace(0, 1, self.chunk_size + 1)[1:]
        xyz = xyz0 + steps[:, None] * (xyz1 - xyz0)
        rots = Slerp([0, 1], Rotation.concatenate([rot0, rot1]))(steps)
        gripper = float(command.get("gripper", gripper0))
        return self._format(xyz, rots), np.full(self.chunk_size, gripper)

    def _format(self, xyz: np.ndarray, rots: Rotation) -> np.ndarray:
        if self.control_mode == "tquat":
            return np.concatenate([xyz, rots.as_quat()], axis=1)
        return np.concatenate([xyz, rots.as_euler("xyz")], axis=1)


class JointSpace:
    """Absolute joint control. Env action key is "joints" (radians)."""

    control_mode = "joints"

    def __init__(self, chunk_size: int, max_joint_delta_deg: float, joint_limits_deg: np.ndarray):
        self.chunk_size = chunk_size
        self.max_joint_delta_deg = max_joint_delta_deg
        self.joint_limits_deg = joint_limits_deg

    def describe(self) -> str:
        low, high = np.round(self.joint_limits_deg, 0).astype(int).tolist()
        return "\n\n".join(
            [
                FRAME_DESCRIPTION,
                WORKSPACE_DESCRIPTION,
                JOINT_DESCRIPTION.format(low=low, high=high),
                JOINT_COMMANDS.format(chunk_size=self.chunk_size, max_joint_delta=self.max_joint_delta_deg),
            ]
        )

    def state_text(self, single_obs: SingleObs) -> str:
        joints = np.round(np.rad2deg(np.asarray(single_obs.joints, dtype=float)), 0).astype(int).tolist()
        gripper = "open" if (single_obs.gripper or 0) > 0.5 else "closed"
        text = f"joints = {joints} deg, gripper {gripper}"
        if single_obs.tquat is not None:
            xyz, rot = CartesianSpace.pose(single_obs)
            text += f" (resulting tool xyz = {np.round(xyz, 3).tolist()} m, rpy = {rpy_deg(rot)} deg)"
        return text

    def target_command(self, target_obs: SingleObs) -> dict[str, Any]:
        return {
            "type": "move_joints",
            "joints_deg": np.round(np.rad2deg(np.asarray(target_obs.joints, dtype=float)), 0).astype(int).tolist(),
            "gripper": int(round(target_obs.gripper or 0)),
        }

    def expand(self, command: dict[str, Any], single_obs: SingleObs) -> tuple[np.ndarray, np.ndarray]:
        q0 = np.rad2deg(np.asarray(single_obs.joints, dtype=float))
        gripper0 = float(single_obs.gripper if single_obs.gripper is not None else 1.0)
        kind = command.get("type", "hold")
        if kind == "chunk":
            actions = np.asarray(command["actions"], dtype=float)
            if actions.shape != (self.chunk_size, 7):
                msg = f"chunk must have shape ({self.chunk_size}, 7), got {actions.shape}"
                raise ValueError(msg)
            grippers = np.broadcast_to(np.asarray(command.get("gripper", gripper0), dtype=float), (self.chunk_size,))
            return np.deg2rad(np.clip(actions, *self.joint_limits_deg)), grippers.copy()
        if kind in ("hold", "gripper"):
            q1 = q0
        elif kind == "move_joints":
            q1 = np.asarray(command["joints_deg"], dtype=float)
        elif kind == "move_joints_delta":
            q1 = q0 + np.asarray(command["djoints_deg"], dtype=float)
        else:
            msg = f"unknown command type {kind!r}"
            raise ValueError(msg)
        if q1.shape != (7,):
            msg = f"expected 7 joint values, got {q1.shape}"
            raise ValueError(msg)
        delta = q1 - q0
        scale = np.max(np.abs(delta)) / self.max_joint_delta_deg
        if scale > 1:
            delta = delta / scale
        q1 = np.clip(q0 + delta, *self.joint_limits_deg)
        steps = np.linspace(0, 1, self.chunk_size + 1)[1:]
        gripper = float(command.get("gripper", gripper0))
        return np.deg2rad(q0 + steps[:, None] * (q1 - q0)), np.full(self.chunk_size, gripper)


class OpenAIBackend:
    """Any OpenAI compatible chat-completions endpoint (OpenAI, vLLM, ...)."""

    def __init__(self, model: str | None, base_url: str | None, api_key_env: str, request_kwargs: dict[str, Any]):
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url, api_key=os.environ.get(api_key_env))
        self.model = model or "gpt-5"
        self.request_kwargs = request_kwargs

    def complete(self, messages: list[Any]) -> tuple[str, dict[str, Any]]:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            response_format={"type": "json_object"},
            **self.request_kwargs,
        )
        usage = response.usage.model_dump() if response.usage is not None else {}
        return response.choices[0].message.content or "", usage


class AnthropicBackend:
    """Claude through the Anthropic Messages API (adaptive thinking is on by default on current models)."""

    def __init__(self, model: str | None, base_url: str | None, api_key_env: str, request_kwargs: dict[str, Any]):
        import anthropic

        api_key = os.environ.get(api_key_env) if api_key_env != "ANTHROPIC_API_KEY" else None
        self.client = anthropic.Anthropic(base_url=base_url, api_key=api_key)
        self.model = model or "claude-opus-5"
        self.request_kwargs = {"max_tokens": 4096, **request_kwargs}

    @staticmethod
    def _convert(messages: list[Any]) -> tuple[list[Any], list[Any]]:
        system: list[Any] = []
        converted: list[Any] = []
        for message in messages:
            content = message["content"]
            if isinstance(content, str):
                converted.append({"role": message["role"], "content": content})
                continue
            blocks = []
            for part in content:
                if part["type"] == "text":
                    blocks.append({"type": "text", "text": part["text"]})
                else:
                    data = part["image_url"]["url"].split(",", 1)[1]
                    blocks.append(
                        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}}
                    )
            if message["role"] == "system":
                # the system prompt cannot carry images, so reference pictures open the conversation instead
                system += [block for block in blocks if block["type"] == "text"]
                images = [block for block in blocks if block["type"] == "image"]
                if images:
                    converted.append(
                        {"role": "user", "content": [*images, {"type": "text", "text": "Reference pictures."}]}
                    )
                    converted.append({"role": "assistant", "content": "Understood."})
            else:
                converted.append({"role": message["role"], "content": blocks})
        return system, converted

    def complete(self, messages: list[Any]) -> tuple[str, dict[str, Any]]:
        system, converted = self._convert(messages)
        response = self.client.messages.create(
            model=self.model, system=system, messages=converted, **self.request_kwargs
        )
        if response.stop_reason == "refusal":
            return "", {"stop_reason": "refusal"}
        text = "".join(block.text for block in response.content if block.type == "text")
        return text, response.usage.model_dump()


class MailboxBackend:
    """Hands each request to an external pilot through the file system.

    A subagent (e.g. a Claude Code agent) or a human reads `<episode>/step_XXX/request.md` and the images next
    to it and answers by writing `reply.json` into the same folder. `system.md` holds the system prompt once
    per episode; `conversation.md` accumulates the whole dialogue.
    """

    def __init__(self, mailbox_dir: str, timeout: float = 1800.0):
        self.root = Path(mailbox_dir)
        self.timeout = timeout
        self.episode_dir: Path | None = None
        self.written = 0

    def _new_episode(self):
        self.episode_dir = self.root / datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
        self.episode_dir.mkdir(parents=True)
        self.written = 0

    def _render(self, message: dict[str, Any], step_dir: Path) -> str:
        content = message["content"]
        if isinstance(content, str):
            return f"### {message['role']}\n\n{content}\n"
        lines = [f"### {message['role']}\n"]
        image_index = 0
        for part in content:
            if part["type"] == "text":
                lines.append(part["text"] + "\n")
            else:
                image_index += 1
                path = step_dir / f"image_{image_index}.jpg"
                path.write_bytes(base64.b64decode(part["image_url"]["url"].split(",", 1)[1]))
                lines.append(f"![image {image_index}]({path.relative_to(self.root)})\n")
        return "\n".join(lines)

    def complete(self, messages: list[Any]) -> tuple[str, dict[str, Any]]:
        if self.episode_dir is None or len(messages) <= self.written:
            self._new_episode()
        assert self.episode_dir is not None
        step = sum(1 for m in messages if m["role"] == "user") - 1
        step_dir = self.episode_dir / f"step_{step:03d}"
        step_dir.mkdir(exist_ok=True)
        new_messages = messages[self.written :]
        rendered = [self._render(message, step_dir) for message in new_messages]
        if self.written == 0:
            (self.episode_dir / "system.md").write_text(rendered[0])
            rendered = rendered[1:]
        with (self.episode_dir / "conversation.md").open("a") as f:
            f.write("\n".join(rendered))
        (step_dir / "request.md").write_text(
            "\n".join(rendered[-1:])
            + f"\n\nWrite your reply (one JSON object as described in system.md) to {step_dir.relative_to(self.root)}/reply.json\n"
        )
        self.written = len(messages)
        reply_path = step_dir / "reply.json"
        start = time.time()
        while not reply_path.exists():
            if time.time() - start > self.timeout:
                msg = f"no reply in {reply_path} after {self.timeout:.0f} s"
                raise TimeoutError(msg)
            time.sleep(0.5)
        time.sleep(0.2)  # let the writer finish
        return reply_path.read_text(), {"latency": time.time() - start}


class FakeBackend:
    """Cycles through canned replies; used in tests and to check the plumbing without an API key."""

    def __init__(self, replies: list[str] | None = None):
        self.replies = replies or [
            '{"reasoning": "fake", "left": {"type": "hold"}, "right": {"type": "hold"}, "done": false}'
        ]
        self.calls = 0

    def complete(self, messages: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
        reply = self.replies[self.calls % len(self.replies)]
        self.calls += 1
        return reply, {"prompt_messages": len(messages)}


class VLMAgent(Agent):
    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key_env: str | None = None,
        backend: str = "openai",
        fake_replies: list[str] | None = None,
        mailbox_dir: str | None = None,
        control_mode: str = "xyzrpy",
        chunk_size: int = 30,
        fps: int = 30,
        history: str | int = "full",
        image_size: int | None = None,
        image_detail: str = "auto",
        max_translation: float = 0.3,
        max_rotation_deg: float = 90.0,
        max_joint_delta_deg: float = 45.0,
        icl_path: str | None = None,
        icl_episodes: int = 3,
        icl_images: bool = True,
        reference_images: list[str] | None = None,
        extra_instructions: str = "",
        log_dir: str | None = None,
        request_kwargs: dict[str, Any] | None = None,
        device: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(default_checkpoint_path=model or backend, **kwargs)
        self.device = device  # set by the eval CLI, no GPU needed here
        self.model = model
        self.base_url = base_url
        self.api_key_env = api_key_env or ("ANTHROPIC_API_KEY" if backend == "anthropic" else "OPENAI_API_KEY")
        self.backend_name = backend
        self.fake_replies = fake_replies
        self.mailbox_dir = mailbox_dir
        self.control_mode = control_mode
        self.chunk_size = chunk_size
        self.fps = fps
        self.history = history
        self.image_size = image_size
        self.image_detail = image_detail
        self.icl_path = icl_path
        self.icl_episodes = icl_episodes
        self.icl_images = icl_images
        self.reference_images = reference_images
        self.extra_instructions = extra_instructions
        self.log_dir = Path(log_dir) if log_dir else None
        self.request_kwargs = request_kwargs or {}
        if control_mode == "joints":
            self.space: JointSpace | CartesianSpace = JointSpace(chunk_size, max_joint_delta_deg, FR3_JOINT_LIMITS_DEG)
        else:
            self.space = CartesianSpace(control_mode, chunk_size, max_translation, max_rotation_deg)
        self.turns: list[dict[str, Any]] = []
        self.system_message: dict[str, Any] | None = None
        self.icl_messages: list[dict[str, Any]] = []
        self.episode_dir: Path | None = None

    def initialize(self):
        self.backend: OpenAIBackend | AnthropicBackend | MailboxBackend | FakeBackend
        if self.backend_name == "fake":
            self.backend = FakeBackend(self.fake_replies)
        elif self.backend_name == "mailbox":
            self.backend = MailboxBackend(self.mailbox_dir or str(self.log_dir or "vlm_mailbox"))
        elif self.backend_name == "anthropic":
            self.backend = AnthropicBackend(self.model, self.base_url, self.api_key_env, self.request_kwargs)
        else:
            self.backend = OpenAIBackend(self.model, self.base_url, self.api_key_env, self.request_kwargs)
        reference_paths: list[str | Path] = (
            [ASSETS / "duo_frame_reference.png"] if self.reference_images is None else list(self.reference_images)
        )
        self.reference_parts = []
        for path in reference_paths:
            if Path(path).exists():
                image = np.asarray(Image.open(path).convert("RGB"))
                self.reference_parts.append(image_part(encode_jpeg(image), self.image_detail))
        if self.icl_path is not None:
            self.icl_messages = self._load_icl(json.loads(Path(self.icl_path).read_text()))

    # ----- prompt construction -----

    def _system_text(self) -> str:
        parts = [
            ROBOT_DESCRIPTION,
            self.space.describe(),
            CAMERA_DESCRIPTION,
            OUTPUT_RULES.format(chunk_size=self.chunk_size, seconds=self.chunk_size / self.fps, fps=self.fps),
        ]
        if self.reference_parts:
            parts.append(
                "The attached reference picture shows the head camera view with the coordinate axes drawn on the "
                "table and the two arms labelled."
            )
        if self.icl_messages:
            parts.append(
                "Before the current episode you will see recorded human demonstrations of the same task as pairs "
                "of observation and the command that reproduces the next second of the demonstration. Imitate them."
            )
        if self.extra_instructions:
            parts.append(self.extra_instructions)
        return "\n\n".join(parts)

    def _obs_text(self, obs: Obs, step: int) -> str:
        first = next(iter(obs.obs.values()))
        info = first.info or {}
        lines = [f"Step {step} (t = {step * self.chunk_size / self.fps:.1f} s). Task: {obs.language_instruction}"]
        if "stage" in info:
            lines.append(
                f"Task progress: stage {info['stage']} of {info['max_stage']}, current sub-goal: "
                f"{info.get('current_subinstruction')}"
            )
            if "stage_to_subinstructions" in info and step == 0:
                stages = ", ".join(f"{k}: {v}" for k, v in dict(info["stage_to_subinstructions"]).items())
                lines.append(f"All stages: {stages}")
        for robot, single_obs in obs.obs.items():
            flags = []
            if single_obs.info.get("collision"):
                flags.append("collision detected during the last second")
            if single_obs.info.get("ik_success") is False:
                flags.append("the last target was unreachable (IK failed)")
            lines.append(
                f"{robot} arm: {self.space.state_text(single_obs)}" + (f" [{'; '.join(flags)}]" if flags else "")
            )
        lines.append("Images in order: " + ", ".join(first.cameras.keys()) + ".")
        return "\n".join(lines)

    def _images(self, obs: Obs) -> list[tuple[str, str]]:
        first = next(iter(obs.obs.values()))
        return [(name, encode_jpeg(np.asarray(image), self.image_size)) for name, image in first.cameras.items()]

    def _user_message(self, text: str, images: list[tuple[str, str]], with_images: bool) -> dict[str, Any]:
        content: list[dict[str, Any]] = [{"type": "text", "text": text}]
        if with_images:
            content += [image_part(jpeg, self.image_detail) for _, jpeg in images]
        else:
            content.append({"type": "text", "text": "[images of this step omitted]"})
        return {"role": "user", "content": content}

    def _load_icl(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        if data["control_mode"] != self.space.control_mode:
            msg = f"ICL file is in {data['control_mode']} space, agent runs {self.space.control_mode}"
            raise ValueError(msg)
        messages = []
        for episode in data["episodes"][: self.icl_episodes]:
            for i, step in enumerate(episode["steps"]):
                lines = [
                    f"Demonstration {episode['index']}, step {i} (t = {step['t'] / self.fps:.1f} s). Task: {episode['instruction']}"
                ]
                for robot, single in step["obs"].items():
                    lines.append(f"{robot} arm: {self.space.state_text(SingleObs(**single))}")
                content: list[dict[str, Any]] = [{"type": "text", "text": "\n".join(lines)}]
                if self.icl_images and step.get("image"):
                    content.append(image_part(step["image"], self.image_detail))
                command = {
                    robot: self.space.target_command(SingleObs(**single)) for robot, single in step["target"].items()
                }
                reply = {"reasoning": "demonstration", **command, "done": i == len(episode["steps"]) - 1}
                messages.append({"role": "user", "content": content})
                messages.append({"role": "assistant", "content": json.dumps(reply)})
        return messages

    def _messages(self) -> list[dict[str, Any]]:
        assert self.system_message is not None
        messages = [self.system_message, *self.icl_messages]
        keep_images_from = 0 if self.history == "full" else max(0, len(self.turns) - int(self.history))
        for i, turn in enumerate(self.turns):
            messages.append(self._user_message(turn["text"], turn["images"], with_images=i >= keep_images_from))
            if "reply" in turn:
                messages.append({"role": "assistant", "content": turn["reply"]})
        return messages

    # ----- episode API -----

    def reset(self, obs: Obs, instruction: str | None = None, **kwargs) -> dict[str, Any]:
        super().reset(obs, instruction, **kwargs)
        self.turns = []
        content: list[dict[str, Any]] = [{"type": "text", "text": self._system_text()}, *self.reference_parts]
        self.system_message = {"role": "system", "content": content}
        if self.log_dir is not None:
            self.episode_dir = self.log_dir / datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
            self.episode_dir.mkdir(parents=True, exist_ok=True)
            (self.episode_dir / "system_prompt.txt").write_text(self._system_text())
        return {
            "backend": self.backend_name,
            "model": self.model,
            "control_mode": self.space.control_mode,
            "icl_messages": len(self.icl_messages),
        }

    def act(self, obs: Obs) -> Act:
        super().act(obs)
        if self.system_message is None:
            self.reset(obs)
        step = len(self.turns)
        turn = {"text": self._obs_text(obs, step), "images": self._images(obs)}
        self.turns.append(turn)
        commands, reply, usage, latency, error = self._query(obs)
        turn["reply"] = reply
        acts = self._expand(commands, obs)
        self._log(step, turn, commands, usage, latency, error)
        return Act(acts=acts)

    def _query(self, obs: Obs) -> tuple[dict[str, Any], str, dict[str, Any], float, str | None]:
        error = None
        for attempt in range(2):
            messages = self._messages()
            if error is not None:
                messages.append(
                    {
                        "role": "user",
                        "content": f"Your previous reply was invalid ({error}). Reply with one valid JSON object.",
                    }
                )
            start = time.time()
            reply, usage = self.backend.complete(messages)
            latency = time.time() - start
            try:
                commands = parse_json_reply(reply)
                for robot in obs.obs:
                    if robot not in commands:
                        msg = f"missing command for arm {robot!r}"
                        raise ValueError(msg)
                    self.space.expand(commands[robot], obs.obs[robot])
                return commands, reply, usage, latency, None
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                error = str(exc)
                logger.warning("invalid VLM reply (attempt %d): %s", attempt, error)
        hold = {robot: {"type": "hold"} for robot in obs.obs}
        return {**hold, "reasoning": f"fallback hold: {error}", "done": False}, reply, usage, latency, error

    def _expand(self, commands: dict[str, Any], obs: Obs) -> list[dict[str, SingleAct]]:
        done = bool(commands.get("done", False))
        per_robot = {robot: self.space.expand(commands[robot], single_obs) for robot, single_obs in obs.obs.items()}
        return [
            {
                robot: SingleAct(
                    action=actions[i].astype(np.float32),
                    gripper=float(grippers[i]),
                    done=done and i == self.chunk_size - 1,
                )
                for robot, (actions, grippers) in per_robot.items()
            }
            for i in range(self.chunk_size)
        ]

    def _log(
        self,
        step: int,
        turn: dict[str, Any],
        commands: dict[str, Any],
        usage: dict[str, Any],
        latency: float,
        error: str | None,
    ):
        logger.info("step %d (%.1fs): %s", step, latency, commands.get("reasoning", ""))
        if self.episode_dir is None:
            return
        step_dir = self.episode_dir / f"step_{step:03d}"
        step_dir.mkdir(exist_ok=True)
        for name, jpeg in turn["images"]:
            (step_dir / f"{name}.jpg").write_bytes(base64.b64decode(jpeg))
        record = {
            "text": turn["text"],
            "reply": turn["reply"],
            "commands": commands,
            "usage": usage,
            "latency": latency,
            "error": error,
        }
        (step_dir / "turn.json").write_text(json.dumps(record, indent=2))


register_agent("vlm", VLMAgent)
