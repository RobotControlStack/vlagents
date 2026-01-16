import base64
import dataclasses
from dataclasses import asdict
from multiprocessing import shared_memory
from typing import Any

import json_numpy
import numpy as np
import rpyc
import simplejpeg

from vlagents.policies import Act, Agent, CameraDataType, Obs, SharedMemoryPayload


def dataclass_from_dict(klass, d):
    # https://stackoverflow.com/questions/53376099/python-dataclass-from-a-nested-dict
    try:
        fieldtypes = {f.name: f.type for f in dataclasses.fields(klass)}
        return klass(**{f: dataclass_from_dict(fieldtypes[f], d[f]) for f in d})
    except:
        return d  # Not a dataclass field


class RemoteAgent(Agent):
    def __init__(self, host: str, port: int, model: str, on_same_machine: bool = False, jpeg_encoding: bool = False):
        """Connect to a remote agent service.

        Args:
            host (str): Hostname or IP address of the remote agent service.
            port (int): Port number of the remote agent service.
            model (str): Name of the model to connect to.
            on_same_machine (bool, optional): If True, assumes the agent is running on the same machine and uses
                shared memory for more efficient communication. Defaults to False.
            jpeg_encoding (bool, optional): If True the image data is jpeg encoded for smaller transfer size.
                Defaults to False.
        """
        self.on_same_machine = on_same_machine
        self.jpeg_encoding = jpeg_encoding
        self._shm: dict[str, shared_memory.SharedMemory] = {}
        self.c = rpyc.connect(
            host, port, config={"allow_pickle": True, "allow_public_attrs": True, "sync_request_timeout": 300}
        )
        assert model == self.c.root.name()

    def _process(self, obs: Obs) -> Obs:
        if self.on_same_machine:
            camera_dict = {}
            for camera_name, camera_data in obs.cameras.items():
                assert isinstance(camera_data, np.ndarray)
                if camera_name not in self._shm:
                    self._shm[camera_name] = shared_memory.SharedMemory(create=True, size=camera_data.nbytes)
                camera_shared = np.ndarray(
                    camera_data.shape, buffer=self._shm[camera_name].buf, dtype=camera_data.dtype
                )
                camera_shared[:] = camera_data[:]
                camera_dict[camera_name] = SharedMemoryPayload(
                    shm_name=self._shm[camera_name].name,
                    shape=camera_data.shape,
                    dtype=camera_data.dtype.name,
                )
            obs.cameras = camera_dict
            obs.camera_data_type = CameraDataType.SHARED_MEMORY
        elif self.jpeg_encoding:
            camera_dict = {}
            for camera_name, camera_data in obs.cameras.items():
                assert isinstance(camera_data, np.ndarray)
                camera_dict[camera_name] = base64.urlsafe_b64encode(
                    simplejpeg.encode_jpeg(np.ascontiguousarray(camera_data))
                ).decode("utf-8")
            obs.cameras = camera_dict
            obs.camera_data_type = CameraDataType.JPEG_ENCODED
        return obs

    def act(self, obs: Obs) -> Act:
        obs = self._process(obs)
        obs = json_numpy.dumps(asdict(obs))
        # action, done, info
        return dataclass_from_dict(Act, json_numpy.loads(self.c.root.act(obs)))

    def reset(self, obs: Obs, instruction: Any, **kwargs) -> dict[str, Any]:
        obs = self._process(obs)
        obs_dict = asdict(obs)
        # info
        return json_numpy.loads(self.c.root.reset(json_numpy.dumps((obs_dict, instruction, kwargs))))

    def git_status(self) -> str:
        return json_numpy.loads(self.c.root.git_status())

    def is_initialized(self) -> bool:
        return self.c.root.is_initialized()

    def close(self):
        for shm in self._shm.values():
            shm.close()
            shm.unlink()
        self._shm = {}
        self.c.close()


if __name__ == "__main__":
    # to test the connection
    agent = RemoteAgent("localhost", 8080, "test")
    obs = Obs(cameras={"rgb_side": np.zeros((256, 256, 3), dtype=np.uint8)})
    instruction = "do something"
    agent.reset(obs, instruction)
    print(agent.act(obs))
    print(agent.act(obs))
