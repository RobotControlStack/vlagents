import json

import numpy as np
import pytest

from vlagents.policies.interface import Obs, SingleObs
from vlagents.policies.vlm import (
    FR3_JOINT_LIMITS_DEG,
    CartesianSpace,
    JointSpace,
    VLMAgent,
    parse_json_reply,
)

HOME_JOINTS = np.array([-0.488, -0.572, 0.585, -2.58, -0.864, 2.053, 0.86])
HOME_TQUAT = np.array([0.508, -0.21, 0.204, 0.98481, 0.0, 0.17365, 0.0])


def _single_obs() -> SingleObs:
    return SingleObs(
        cameras={"head": np.zeros((16, 16, 3), np.uint8), "right_wrist": np.zeros((16, 16, 3), np.uint8)},
        joints=HOME_JOINTS.copy(),
        gripper=1.0,
        tquat=HOME_TQUAT.copy(),
        info={"stage": 0, "max_stage": 4, "current_subinstruction": "pick up the cube"},
    )


def _obs() -> Obs:
    return Obs(obs={"left": _single_obs(), "right": _single_obs()}, language_instruction="hand over the cube")


def test_cartesian_state_text_is_canonical():
    text = CartesianSpace("xyzrpy", 30, 0.3, 90).state_text(_single_obs())
    assert "rpy = [180, -20, 0]" in text
    assert "gripper open" in text


def test_cartesian_move_expands_to_chunk_and_reaches_target():
    space = CartesianSpace("xyzrpy", 30, 0.3, 90)
    actions, grippers = space.expand(
        {"type": "move", "xyz": [0.6, -0.3, 0.0], "rpy_deg": [180, 0, 45], "gripper": 0}, _single_obs()
    )
    assert actions.shape == (30, 6)
    np.testing.assert_allclose(actions[-1, :3], [0.6, -0.3, 0.0])
    np.testing.assert_allclose(np.rad2deg(actions[-1, 3:]), [180, 0, 45], atol=1e-6)
    assert np.all(grippers == 0)


def test_cartesian_translation_budget_is_enforced():
    space = CartesianSpace("xyzrpy", 30, 0.3, 90)
    actions, _ = space.expand({"type": "move_delta", "dxyz": [1.0, 0, 0]}, _single_obs())
    assert np.linalg.norm(actions[-1, :3] - HOME_TQUAT[:3]) == pytest.approx(0.3)


def test_tquat_output_format():
    actions, _ = CartesianSpace("tquat", 30, 0.3, 90).expand({"type": "hold"}, _single_obs())
    assert actions.shape == (30, 7)
    np.testing.assert_allclose(actions[-1], HOME_TQUAT, atol=1e-4)


def test_joint_space_clips_to_limits_and_budget():
    space = JointSpace(30, 45, FR3_JOINT_LIMITS_DEG)
    actions, _ = space.expand({"type": "move_joints_delta", "djoints_deg": [90, 0, 0, 0, 0, 0, 0]}, _single_obs())
    assert np.rad2deg(actions[-1, 0] - HOME_JOINTS[0]) == pytest.approx(45)
    actions, _ = space.expand({"type": "move_joints_delta", "djoints_deg": [0, 0, 0, -30, 0, 0, 0]}, _single_obs())
    assert np.rad2deg(actions[-1, 3]) == pytest.approx(FR3_JOINT_LIMITS_DEG[0, 3])


def test_chunk_command_requires_exact_size():
    space = JointSpace(30, 45, FR3_JOINT_LIMITS_DEG)
    with pytest.raises(ValueError, match="shape"):
        space.expand({"type": "chunk", "actions": [[0] * 7] * 5}, _single_obs())


def test_parse_json_reply_strips_fences():
    assert parse_json_reply('```json\n{"done": false}\n```') == {"done": False}


def test_agent_full_history_and_fallback(tmp_path):
    replies = [
        json.dumps(
            {
                "reasoning": "go",
                "left": {"type": "hold"},
                "right": {"type": "move_delta", "dxyz": [0.05, 0, 0], "gripper": 1},
                "done": False,
            }
        ),
        "not json",
        json.dumps(
            {"reasoning": "close", "left": {"type": "gripper", "gripper": 0}, "right": {"type": "hold"}, "done": True}
        ),
    ]
    agent = VLMAgent(backend="fake", fake_replies=replies, control_mode="xyzrpy", log_dir=str(tmp_path))
    agent.initialize()
    obs = _obs()
    info = agent.reset(obs, obs.language_instruction)
    assert info["control_mode"] == "xyzrpy"

    act = agent.act(obs)
    assert len(act.acts) == 30
    assert act.acts[-1]["right"].action[0] == pytest.approx(HOME_TQUAT[0] + 0.05)
    assert not act.acts[-1]["right"].done

    # invalid reply -> one retry -> valid reply
    act = agent.act(obs)
    assert act.acts[-1]["left"].gripper == 0.0
    assert act.acts[-1]["left"].done

    messages = agent._messages()
    assert messages[0]["role"] == "system"
    user_turns = [m for m in messages if m["role"] == "user"]
    assert len(user_turns) == 2
    assert "stage 0 of 4" in user_turns[0]["content"][0]["text"]
    assert sum(part["type"] == "image_url" for part in user_turns[0]["content"]) == 2
    assert (tmp_path / next(tmp_path.iterdir()).name / "step_001" / "turn.json").exists()


def test_agent_history_window_drops_old_images():
    agent = VLMAgent(backend="fake", control_mode="joints", history=1)
    agent.initialize()
    obs = _obs()
    agent.reset(obs)
    agent.act(obs)
    agent.act(obs)
    user_turns = [m for m in agent._messages() if m["role"] == "user"]
    assert all(part["type"] == "text" for part in user_turns[0]["content"])
    assert any(part["type"] == "image_url" for part in user_turns[1]["content"])
