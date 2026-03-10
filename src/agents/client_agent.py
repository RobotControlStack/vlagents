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
    def __init__(self, host: str= "airtower.utn-mi.de", 
                 port: int= 20997,
                 model: str = "test_exp",
                 on_same_machine: bool = False):
        super().__init__(host, port, model, on_same_machine)

    def get_obs(self, obs: dict, is_compressed=False) -> Obs:

        side = obs["frames"]["side"]["rgb"]["data"]
        wrist = obs["frames"]["wrist"]["rgb"]["data"]
        if self.on_same_machine:
            return Obs(cameras=dict(rgb_side=side, rgb_wrist=wrist),
                       gripper=obs["gripper"], info=dict(joints=obs["joints"]))
        else:
            if is_compressed:
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


                side_jpeg = side_bytes.getvalue()
                side_b64 = base64.urlsafe_b64encode(side_jpeg).decode("utf-8")

                # print("SIDE")
                # print("raw:", side.nbytes)
                # print("jpeg:", len(side_jpeg))
                # print("base64:", len(side_b64))

                wrist_jpeg = wrist_bytes.getvalue()
                wrist_b64 = base64.urlsafe_b64encode(wrist_jpeg).decode("utf-8")

                # print("WRIST")
                # print("raw:", wrist.nbytes)
                # print("jpeg:", len(wrist_jpeg))
                # print("base64:", len(wrist_b64))

            else:
                side_b64 = side
                wrist_b64 = wrist

            return Obs(cameras=dict(rgb_side=side_b64, rgb_wrist=wrist_b64),
                    gripper=obs["gripper"], info=dict(joints=obs["joints"], is_compressed=is_compressed))

    def load_obs(self, imgs_path_dict, images_size):
        obs = {}
        side = np.array(Image.open(imgs_path_dict["side"]).resize((images_size[1], images_size[0])))
        #print("side shape", side.shape)
        wrist = np.array(Image.open(imgs_path_dict["wrist"]).resize((images_size[1], images_size[0])))
        #print("wrist shape", wrist.shape)
        #print(side.min(), side.max(), wrist.min(), wrist.max())
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
    def benchmark(self, imgs_path_dict, images_size, runs: int = 1000, is_compressed=True):
        times = []
        obs = self.load_obs(imgs_path_dict, images_size)
        obs_struct = self.get_obs(obs, is_compressed=is_compressed)
        self.reset(obs_struct, instruction="pick and place", on_same_machine=self.on_same_machine)
        for _ in tqdm(range(runs)):
            start_time = time.perf_counter()
            obs_struct = self.get_obs(obs, is_compressed=is_compressed)
            action = self.act(obs_struct)
            end_time = time.perf_counter()
            #time.sleep(0.25)  # to avoid overloading the server    
            times.append(end_time - start_time)
        avg_time = sum(times) / runs
        max_time = max(times)
        min_time = min(times)
        print(f"avg: {avg_time:.4f} seconds")
        print(f"max: {max_time:.4f} seconds")
        print(f"min: {min_time:.4f} seconds")
        print("std:", np.std(np.array(times)))
        results = {
            "avg": avg_time,
            "max": max_time,
            "min": min_time,
            "std": np.std(np.array(times)),
            "times": times,
        }
        return results


if __name__ == "__main__":
    port = 20997
    on_same_machine = False
    is_compressed = True
    model_name = "vlagents"
    runs = 1000
    if on_same_machine == True:
    # test local connection
        host = "localhost"
        model = "test"
    else:
    # test remote connection
        #host = "airtower.utn-mi.de"
        host = "multihead.utn-mi.de"
        model = "test"
    imgs_folder_path = "/home/gamal/vlagent_benchmark/imgs"
    output_folder_path = "/home/gamal/vlagent_benchmark/outputs/vlagents"
    imgs_path_dict = {
        "side": f"{imgs_folder_path}/side_observer_30.png",
        "wrist": f"{imgs_folder_path}/side_right_30.png"
    }
    #image_size = (224, 224, 3)
    image_size = (720, 1280, 3)
    # Create the client agent and run the benchmark
    print("model:", model_name)
    print("on_same_machine:", on_same_machine)
    print("image size:", image_size)
    print("is_compressed:", is_compressed)
    print("runs:", runs)
    client_agent = ClientAgent(host=host, port=port, model=model, on_same_machine=on_same_machine)
    results_benchmark = client_agent.benchmark(imgs_path_dict, image_size, runs=runs, is_compressed=is_compressed)
    results_benchmark["model"] = model_name
    results_benchmark["on_same_machine"] = on_same_machine
    results_benchmark["is_compressed"] = is_compressed
    results_benchmark["image_size"] = image_size
    results_benchmark["runs"] = runs
    # save results to json file
    import json
    import os
    os.makedirs(output_folder_path, exist_ok=True)
    json_path = f"{output_folder_path}/benchmark_results_{model_name}_{'local' if on_same_machine else 'remote'}_{'compressed' if is_compressed else 'uncompressed'}_{image_size[0]}x{image_size[1]}.json"
    with open(json_path, "w") as f:
        json.dump(results_benchmark, f, indent=4)
    print(f"Benchmark results saved to {json_path}")