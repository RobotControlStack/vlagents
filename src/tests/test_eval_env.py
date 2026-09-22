import numpy as np

from vlagents.envs.interface import EvalEnv
from vlagents.policies.interface import Act, Obs, SingleAct, SingleObs


class CountingEnv(EvalEnv):
    def __init__(self):
        self.rendered = []
        self.executed = []
        super().__init__("counting")

    def make_gym(self):
        return None

    def do_import(self):
        pass

    def step(self, action):
        self.rendered.append(self.render_next_step)
        self.executed.append(float(action["arm"].action[0]))
        return Obs(obs={"arm": SingleObs()}), 0.0, False, False, {}


def _chunk(n):
    return Act(acts=[{"arm": SingleAct(action=np.array([i], dtype=np.float32))} for i in range(n)])


def test_chunk_step_renders_only_the_last_executed_action():
    env = CountingEnv()
    env.chunk_step(_chunk(5))
    assert env.rendered == [False, False, False, False, True]
    env.chunk_step(_chunk(5), max_steps=2)
    assert env.rendered[-2:] == [False, True]
    assert env.last_chunk_steps == 2


def test_hold_repeats_the_last_action_without_rendering():
    env = CountingEnv()
    assert env.hold(3) == (False, False)  # nothing executed yet
    env.chunk_step(_chunk(3))
    env.hold(4)
    assert env.executed[-4:] == [2.0, 2.0, 2.0, 2.0]
    assert env.rendered[-4:] == [False] * 4
    assert env.last_chunk_steps == 4
