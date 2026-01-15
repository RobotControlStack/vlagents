from typing import Any
import io
import base64
import time

from PIL import Image
import numpy as np
from tqdm import tqdm
from agents.client import RemoteAgent
from agents.policies import Obs

class ClientAgent(RemoteAgent):
    def __init__(self, host: str= "airtower.utn-mi.de", port: int= 20997, model: str = "test_exp", on_same_machine: bool = False):
        super().__init__(host, port, model, on_same_machine)

    def get_obs(self, obs: dict, reset=False) -> Obs:

        side = obs["frames"]["side"]["rgb"]["data"]
        wrist = obs["frames"]["wrist"]["rgb"]["data"]
        if self.on_same_machine:
            return Obs(cameras=dict(rgb_side=side, rgb_wrist=wrist),
                       gripper=obs["gripper"], info=dict(joints=obs["joints"]))
        else:
            # encode to jpeg to reduce the size
            # with jpeg encoding 70 - 80 Hz transfer speed, without 17 fps
            side_bytes = io.BytesIO()
            Image.fromarray(
                side
            ).save(side_bytes, format="JPEG", quality=80)

            wrist_bytes = io.BytesIO()
            Image.fromarray(
                wrist
            ).save(wrist_bytes, format="JPEG", quality=80)

            return Obs(cameras=dict(rgb_side=base64.urlsafe_b64encode(side_bytes.getvalue()).decode("utf-8"), rgb_wrist=base64.urlsafe_b64encode(wrist_bytes.getvalue()).decode("utf-8")),
                    gripper=obs["gripper"], info=dict(joints=obs["joints"]))

    def load_obs(self, imgs_path_dict, images_size):
        obs = {}
        side = np.array(Image.open(imgs_path_dict["side"]).resize((images_size[1], images_size[0])))
        print("side shape", side.shape)
        wrist = np.array(Image.open(imgs_path_dict["wrist"]).resize((images_size[1], images_size[0])))
        print("wrist shape", wrist.shape)
        print(side.min(), side.max(), wrist.min(), wrist.max())
        # example obs
        obs = {
            "frames": {
                "side": {"rgb": {"data": side}},
                "wrist": {"rgb": {"data": wrist}},
            },
            "gripper": 0.5,
            "joints": [0.0, 0.5, 1.0, -0.5, 0.0, 0.3],
        }
        return obs

    # run round trip 1000 time and save average time max and min time
    def benchmark(self, imgs_path_dict, images_size, runs: int = 1000):
        times = []
        obs = self.load_obs(imgs_path_dict, images_size)
        obs_struct = self.get_obs(obs)
        self.reset(obs_struct, instruction="pick and place", on_same_machine=self.on_same_machine)
        for _ in tqdm(range(runs)):
            start_time = time.perf_counter()
            obs_struct = self.get_obs(obs)
            action = self.act(obs_struct)
            end_time = time.perf_counter()
            time.sleep(0.01)  # to avoid overloading the server    
            times.append(end_time - start_time)
        avg_time = sum(times) / runs
        max_time = max(times)
        min_time = min(times)
        print(f"Average time for get_obs and act() over {runs} runs: {avg_time:.4f} seconds")
        print(f"Max time for get_obs and act(): {max_time:.4f} seconds")
        print(f"Min time for get_obs and act(): {min_time:.4f} seconds")
        print("standard deviation:", np.std(np.array(times)))

if __name__ == "__main__":
    port = 20997
    local = False
    if local == True:
    # test local connection
        host = "localhost"
        model = "test"
        on_same_machine = True
    else:
    # test remote connection
        host = "airtower.utn-mi.de"
        model = "test"
        on_same_machine = False
    imgs_path_dict = {
        "side": "/home/gamal/RobotControlStack/imgs/side_observer_30.png",
        "wrist": "/home/gamal/RobotControlStack/imgs/side_right_30.png"
    }
    image_size = (224, 224, 3)
    #image_size = (720, 1280, 3)
    # Create the client agent and run the benchmark
    print("VLAgent")
    print("on_same_machine:", on_same_machine)
    print("image size:", image_size)
    client_agent = ClientAgent(host=host, port=port, model=model, on_same_machine=on_same_machine)
    client_agent.benchmark(imgs_path_dict, image_size, runs=1000)