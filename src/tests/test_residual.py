import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from vlagents.envs.interface import EvalEnv
from vlagents.policies.interface import Act, Obs, SingleAct, SingleObs
from vlagents.policies.residual import EDIT_DIM, ROBOTS, ResidualEditor, apply_edit, feature_dim, pose_delta
from vlagents.policies.residual_train import correction_samples, demo_samples, interpolate


def _obs(task_state=None):
    info = {"task_state": task_state} if task_state is not None else {}
    return Obs(
        obs={
            robot: SingleObs(
                tquat=np.array([0.5, 0.2 if robot == "left" else -0.2, 0.2, 0, 1, 0, 0], np.float32),
                gripper=1.0,
                info=dict(info),
            )
            for robot in ROBOTS
        }
    )


def _act(dz=0.0):
    return {
        robot: SingleAct(action=np.array([0.5, 0.2 if robot == "left" else -0.2, 0.2 + dz, np.pi, 0, 0], np.float32), gripper=1.0)
        for robot in ROBOTS
    }


def test_apply_edit_round_trips_for_both_action_formats():
    edit = np.array([0.01, -0.02, 0.005, 0.02, -0.03, 0.01])
    for mode, action in (("xyzrpy", np.array([0.5, 0.1, 0.2, np.pi, -0.3, 0.1])), ("tquat", np.array([0.5, 0.1, 0.2, 0, 1, 0, 0]))):
        edited = apply_edit(action, edit, mode)
        if mode == "tquat":
            before, after = (action[:3], Rotation.from_quat(action[3:])), (edited[:3], Rotation.from_quat(edited[3:]))
        else:
            before, after = (action[:3], Rotation.from_euler("xyz", action[3:])), (edited[:3], Rotation.from_euler("xyz", edited[3:]))
        assert np.allclose(pose_delta(after, before), edit, atol=1e-5)


def test_editor_without_checkpoint_is_identity_but_records():
    editor = ResidualEditor(control_mode="xyzrpy", task_state_dim=11)
    editor.reset(0)
    editor.begin_chunk()
    nominal = _act(0.05)
    edited = editor.edit(nominal, _obs(list(range(11))), 3, 30, nominal)
    for robot in ROBOTS:
        before = (nominal[robot].action[:3], Rotation.from_euler("xyz", nominal[robot].action[3:]))
        after = (edited[robot].action[:3], Rotation.from_euler("xyz", edited[robot].action[3:]))
        assert np.allclose(pose_delta(after, before), 0, atol=1e-5)
    summary = editor.end_chunk()
    assert summary["steps"] == 1 and summary["left"]["max_translation_cm"] == 0.0
    record = editor.records[0]
    assert record.features.shape == (feature_dim(11),)
    assert record.features[-11:].tolist() == list(range(11))  # task state is the tail of the feature vector
    assert record.features[-12] == 1.0  # valid flag
    assert editor.edit(nominal, _obs(), 0, 30, nominal)  # missing task state -> valid flag off
    assert editor.records[-1].features[-12] == 0.0


class RecordingEnv(EvalEnv):
    def __init__(self):
        self.executed = []
        super().__init__("recording")

    def make_gym(self):
        return None

    def do_import(self):
        pass

    def step(self, action):
        self.executed.append({robot: action[robot].action.copy() for robot in ROBOTS})
        return _obs(), 0.0, False, False, {}


class ConstantEditor(ResidualEditor):
    def __post_init__(self):
        super().__post_init__()
        self.calls = []

    def edit(self, nominal, obs, step, n_steps, goal, holding=False):
        self.calls.append((self.chunk_index, step, holding))
        edited = super().edit(nominal, obs, step, n_steps, goal, holding)
        # shift the left arm 1 cm up to check the edit reaches the environment
        edited["left"] = SingleAct(action=edited["left"].action + np.array([0, 0, 0.01, 0, 0, 0], np.float32), gripper=1.0)
        return edited


def test_env_applies_editor_to_chunk_and_hold_steps(tmp_path):
    env = RecordingEnv()
    env.editor = ConstantEditor(control_mode="xyzrpy", record_dir=str(tmp_path))
    env.editor.reset(0)
    env.chunk_step(Act(acts=[_act(0.01 * i) for i in range(3)]), obs=_obs())
    env.hold(2)
    assert [c[0] for c in env.editor.calls] == [0, 0, 0, 0, 0]
    assert [c[2] for c in env.editor.calls] == [False, False, False, True, True]
    assert np.allclose([e["left"][2] for e in env.executed], [0.21, 0.22, 0.23, 0.23, 0.23])
    assert np.allclose([e["right"][2] for e in env.executed], [0.20, 0.21, 0.22, 0.22, 0.22])
    path = env.editor.save()
    data = np.load(path)
    assert data["features"].shape[0] == 5 and data["holding"].tolist() == [False, False, False, True, True]


def test_demo_samples_label_the_deviation_from_the_straight_chunk():
    n = 41
    poses = {}
    for robot in ROBOTS:
        xyz = np.zeros((n, 3))
        xyz[:, 0] = np.linspace(0, 0.3, n)
        xyz[:, 2] = 0.05 * np.sin(np.linspace(0, np.pi, n))  # arc above the straight line
        poses[robot] = [(xyz[i], Rotation.identity()) for i in range(n)]
    grippers = {robot: np.ones(n) for robot in ROBOTS}
    limits = np.tile([0.02, 0.02, 0.02, 0.1, 0.1, 0.1], 2)
    feats, labels = demo_samples(poses, grippers, None, chunk=30, stride=30, limits=limits, task_state_dim=11)
    assert feats.shape == (30, feature_dim(11)) and labels.shape == (30, 12)
    assert labels[:, 2].max() > 0.01 and labels[:, 2].min() >= 0  # the arc lies above the line
    assert np.all(np.abs(labels) <= limits + 1e-6)
    assert np.allclose(labels[:, 3:6], 0)


def test_correction_samples_relabel_recorded_steps(tmp_path):
    run = tmp_path / "run"
    (run / "residual").mkdir(parents=True)
    (run / "vlm" / "ep0" / "step_000").mkdir(parents=True)
    (run / "vlm" / "ep0" / "step_001").mkdir(parents=True)
    n = 6
    np.savez(
        run / "residual" / "episode_000.npz",
        chunk=np.zeros(n, int),
        step=np.arange(n),
        holding=np.array([False] * 4 + [True] * 2),
        features=np.zeros((n, feature_dim(11)), np.float32),
        nominal=np.zeros((n, 2, 6), np.float32),
        edit=np.full((n, 12), 0.001, np.float32),
        task_state=np.zeros((n, 11), np.float32),
        meta="{}",
    )
    (run / "vlm" / "ep0" / "step_000" / "turn.json").write_text(json.dumps({"commands": {}}))
    correction = {"left": {"dxyz": [0.0, 0.0, 0.01], "drpy_deg": [0, 0, 0], "from": 0.5, "to": 1.0}, "right": None}
    (run / "vlm" / "ep0" / "step_001" / "turn.json").write_text(json.dumps({"commands": {"correction": correction}}))
    limits = np.tile([0.02, 0.02, 0.02, 0.1, 0.1, 0.1], 2)
    feats, labels, stats = correction_samples(run, limits)
    assert stats == {"chunks": 1, "corrected": 1, "steps": n}
    # steps 0,1 are before "from" (phase 0, 1/3), steps 2,3 and the holds inside -> shifted by 1 cm on left z
    assert np.allclose(labels[:, 2], [0.001, 0.001, 0.011, 0.011, 0.011, 0.011])
    assert np.allclose(labels[:, EDIT_DIM + 2], 0.001)


def test_interpolate_ends_at_goal():
    path = interpolate((np.zeros(3), Rotation.identity()), (np.ones(3), Rotation.from_euler("z", 90, degrees=True)), 10)
    assert len(path) == 10 and np.allclose(path[-1][0], 1) and np.isclose(path[-1][1].magnitude(), np.pi / 2)


@pytest.mark.parametrize("bad", ["nonsense", {"left": {"dxyz": [1, 2]}, "right": None}])
def test_vlm_agent_cleans_corrections(bad):
    from vlagents.policies.vlm import VLMAgent

    agent = VLMAgent(backend="fake", hindsight_corrections=True)
    commands = {"correction": bad}
    agent._check_correction(commands, _obs())
    assert commands["correction"] is None or commands["correction"] == {"left": None, "right": None}
    commands = {"correction": {"left": {"dxyz": [0.1, 0, 0], "drpy_deg": [0, 20, 0], "from": 0.2}, "right": None}}
    agent._check_correction(commands, _obs())
    assert commands["correction"]["left"] == {"dxyz": [0.03, 0.0, 0.0], "drpy_deg": [0.0, 8.0, 0.0], "from": 0.2, "to": 1.0}
    assert commands["correction"]["right"] is None


def test_ball_maze_state_text_uses_base_frame_heights():
    from vlagents.envs.duobench import BALL_MAZE_STATE_KEYS, BallMazeState

    state = np.zeros(len(BALL_MAZE_STATE_KEYS), np.float32)
    keys = list(BALL_MAZE_STATE_KEYS)
    state[keys.index("ball_x")], state[keys.index("ball_y")] = -0.09, 0.09
    state[keys.index("goal_x")], state[keys.index("goal_y")] = 0.09, -0.09
    state[keys.index("board_yaw_cos")] = 1.0
    state[keys.index("board_z")], state[keys.index("board_x")] = -0.332, 0.58
    text = BallMazeState.text(state, [state, state])
    assert "ball at (-9.0, 9.0)" in text and "goal square at (9.0, -9.0)" in text
    assert "+x side rim centre at base xyz (0.688, 0.000, -0.295)" in text  # centre + half width, rim height
    assert "Ball trace" in text


def test_hold_keeps_the_last_commanded_gripper():
    from vlagents.policies.vlm import VLMAgent

    agent = VLMAgent(backend="fake")
    obs = _obs()
    acts = agent._expand({"left": {"type": "gripper", "gripper": 0}, "right": {"type": "hold"}}, obs)
    assert acts[-1]["left"].gripper == 0.0 and acts[-1]["right"].gripper == 1.0
    # the observed opening of a gripper holding a thin bar is 0.5: a hold must not re-open it
    for robot in ROBOTS:
        obs.obs[robot].gripper = 0.5
    acts = agent._expand({"left": {"type": "hold"}, "right": {"type": "move_delta", "dxyz": [0, 0, 0.01]}}, obs)
    assert acts[-1]["left"].gripper == 0.0 and acts[-1]["right"].gripper == 1.0
