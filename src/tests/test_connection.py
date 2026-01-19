import subprocess
from time import sleep

import numpy as np

from vlagents.client import RemoteAgent
from vlagents.evaluator_envs import start_server
from vlagents.policies import Act, Obs


def _test_connection(agent: RemoteAgent):
    data = np.zeros((256, 256, 3), dtype=np.uint8)
    data[2, 0, 0] = 16
    obs = Obs(cameras=dict(rgb_side=data))
    instruction = "do something"
    reset_info = agent.reset(obs, instruction)
    assert reset_info["instruction"] == instruction
    assert reset_info["shapes"] == {"rgb_side": [256, 256, 3]}
    assert reset_info["dtype"] == {"rgb_side": "uint8"}
    assert (reset_info["data"]["rgb_side"] == data).all()

    data[0, 0, 2] = 1
    a1 = agent.act(Obs(cameras=dict(rgb_side=data)))
    assert a1.info["shapes"] == {"rgb_side": [256, 256, 3]}
    assert a1.info["dtype"] == {"rgb_side": "uint8"}
    assert (a1.info["data"]["rgb_side"] == data).all()
    assert np.all(a1.action == np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float32))
    assert not a1.done

    data[0, 2, 0] = 1
    a1 = agent.act(Obs(cameras=dict(rgb_side=data)))
    assert a1.info["shapes"] == {"rgb_side": [256, 256, 3]}
    assert a1.info["dtype"] == {"rgb_side": "uint8"}
    assert (a1.info["data"]["rgb_side"] == data).all()
    assert np.all(a1.action == np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.float32))
    assert not a1.done


def _test_connection_jpeg(agent: RemoteAgent):
    data = np.zeros((256, 256, 3), dtype=np.uint8)
    obs = Obs(cameras=dict(rgb_side=data))
    instruction = "do something"
    reset_info = agent.reset(obs, instruction)
    assert reset_info["instruction"] == instruction
    assert reset_info["shapes"] == {"rgb_side": [256, 256, 3]}
    assert reset_info["dtype"] == {"rgb_side": "uint8"}
    assert (reset_info["data"]["rgb_side"] == data).all()


def test_connection_numpy_serialization():
    with start_server("test", {}, 8080, "localhost") as p:
        sleep(2)
        agent = RemoteAgent("localhost", 8080, "test")
        with agent:
            while not agent.is_initialized():
                sleep(0.1)
            _test_connection(agent)
        p.send_signal(subprocess.signal.SIGINT)


def test_connection_numpy_shm():
    with start_server("test", {}, 8080, "localhost") as p:
        sleep(2)
        agent = RemoteAgent("localhost", 8080, "test", on_same_machine=True)
        with agent:
            while not agent.is_initialized():
                sleep(0.1)
            _test_connection(agent)
        p.send_signal(subprocess.signal.SIGINT)


def test_connection_numpy_jpeg():
    with start_server("test", {}, 8080, "localhost") as p:
        sleep(2)
        agent = RemoteAgent("localhost", 8080, "test", jpeg_encoding=True)
        with agent:
            while not agent.is_initialized():
                sleep(0.1)
            _test_connection_jpeg(agent)
        p.send_signal(subprocess.signal.SIGINT)
