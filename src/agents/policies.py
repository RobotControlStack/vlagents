import base64
import copy
import json
import logging
import os
from collections import deque
from dataclasses import dataclass, field
from functools import partial, reduce
from multiprocessing import resource_tracker, shared_memory
from operator import getitem
from pathlib import Path
from typing import Any, Union
from torchvision.io import decode_jpeg
from torchvision.transforms import v2

import numpy as np
from PIL import Image
import torch


@dataclass(kw_only=True)
class SharedMemoryPayload:
    shm_name: str
    shape: tuple[int, ...]
    dtype: str = "uint8"


@dataclass(kw_only=True)
class Obs:
    cameras: dict[str, np.ndarray | SharedMemoryPayload] = field(default_factory=dict)
    gripper: float | None = None
    info: dict[str, Any] = field(default_factory=dict)
    camera_data_in_shared_memory: bool = False


@dataclass(kw_only=True)
class Act:
    action: np.ndarray
    done: bool = False
    info: dict[str, Any] = field(default_factory=dict)


class Agent:
    def __init__(
        self, default_checkpoint_path: str, checkpoint_path: str | None = None, checkpoint_step: int | None = None
    ) -> None:
        self.instruction = None
        self.step = -1
        self.episode = -1
        self.checkpoint_step = checkpoint_step
        self.default_checkpoint_path = default_checkpoint_path
        self.checkpoint_path = checkpoint_path
        self._shm: dict[str, shared_memory.SharedMemory] = {}

    def initialize(self):
        # heavy initialization, e.g. loading models
        pass

    def _from_shared_memory(self, obs: Obs) -> Obs:
        """transparently uses shared memory if configured and modifies obs in place"""
        if obs.camera_data_in_shared_memory:
            camera_dict = {}
            for camera_name, camera_data in obs.cameras.items():
                assert isinstance(camera_data, SharedMemoryPayload)
                if camera_data.shm_name not in self._shm:
                    self._shm[camera_data.shm_name] = shared_memory.SharedMemory(camera_data.shm_name)
                camera_dict[camera_name] = np.ndarray(
                    camera_data.shape, dtype=camera_data.dtype, buffer=self._shm[camera_data.shm_name].buf
                )
            obs.cameras = camera_dict
        return obs

    def act(self, obs: Obs) -> Act:
        assert self.instruction is not None, "forgot reset?"
        self.step += 1
        self._from_shared_memory(obs)

        return Act(action=np.zeros(7, dtype=np.float32), done=False, info={})

    def reset(self, obs: Obs, instruction: Any, **kwargs) -> dict[str, Any]:
        logging.info(f"Resetting agent, new instruction: {instruction} ###############")
        self.step = 0
        self.episode += 1
        self.instruction = instruction
        self._from_shared_memory(obs)
        # info
        return {}

    def __enter__(self):
        pass

    def __exit__(self, *args, **kwargs):
        self.close()

    def close(self, *args, **kwargs):
        for shm in self._shm.values():
            shm.close()
            resource_tracker.unregister(shm._name, "shared_memory")
        self._shm = {}


class TestAgent(Agent):

    def __init__(self, **kwargs) -> None:
        super().__init__(default_checkpoint_path="", **kwargs)
        self.i = 0

    def act(self, obs: Obs) -> Act:
        super().act(obs)
        # echo data back for testing
        if self.on_same_machine:
            logging.info("TestAgent.act using shared memory for cameras")
            info = {
                # "shapes": {k: v.shape for k, v in obs.cameras.items()},
                # "dtype": {k: v.dtype.name for k, v in obs.cameras.items()},
                #"data": {k: v for k, v in obs.cameras.items()},
            }
        else:
            side = base64.urlsafe_b64decode(obs.cameras["rgb_side"])
            side = torch.frombuffer(bytearray(side), dtype=torch.uint8)
            side = decode_jpeg(side)
            side = v2.Resize((256, 256))(side)

            wrist = base64.urlsafe_b64decode(obs.cameras["rgb_wrist"])
            wrist = torch.frombuffer(bytearray(wrist), dtype=torch.uint8)
            wrist = decode_jpeg(wrist)
            wrist = v2.Resize((256, 256))(wrist)
            info = {
            }
        self.i += 1
        a = Act(action=np.array([0, 0, 0, 0, 0, 0, self.i % 2], dtype=np.float32), done=False, info=info)
        return a

    def reset(self, obs: Obs, instruction: Any, **kwargs) -> dict[str, Any]:
        self.on_same_machine = kwargs.get("on_same_machine", False)
        if self.on_same_machine:
            logging.info("TestAgent.reset called with on_same_machine=True")
        super().reset(obs, instruction, **kwargs)
        if self.on_same_machine:
            logging.info("TestAgent.reset using shared memory for cameras")
            info = {
                # "shapes": {k: v.shape for k, v in obs.cameras.items()},
                # "dtype": {k: v.dtype.name for k, v in obs.cameras.items()},
                #"data": {k: v for k, v in obs.cameras.items()},
                # "instruction": instruction,
            }
        else:
            info = {
                # "instruction": instruction,
            }
        return info


class OpenPiModel(Agent):

    def __init__(
        self,
        model_name: str = "pi0_droid",
        default_checkpoint_path: str = "gs://openpi-assets/checkpoints/pi0_droid",
        **kwargs,
    ) -> None:
        # https://console.cloud.google.com/storage/browser/openpi-assets/checkpoints/pi0_droid
        super().__init__(default_checkpoint_path=default_checkpoint_path, **kwargs)
        from openpi.training import config

        logging.info(f"checkpoint_path: {self.checkpoint_path}, checkpoint_step: {self.checkpoint_step}")
        self.openpi_path = self.checkpoint_path.format(checkpoint_step=self.checkpoint_step)

        self.cfg = config.get_config(model_name)
        self.chunks = 20 #20 # 1
        self.s = self.chunks
        self.a = None
        self.save_chunks = None
        # self.chunk_saver = ChunkSaverMin("/home/epez82ox/repos/agents/src/agents/utils/dbg_chunks_02")
    def initialize(self):
        from openpi.policies import policy_config
        from openpi.shared import download

        checkpoint_dir = download.maybe_download(self.openpi_path)

        # Create a trained policy.
        self.policy = policy_config.create_trained_policy(self.cfg, checkpoint_dir)
        print("policy created")

    def act(self, obs: Obs) -> Act:
        # Run inference on a dummy example.
        # observation = {f"observation/{k}": v for k, v in obs.cameras.items()}
        if not self.policy.is_rtc:
            if self.s < self.chunks:
                self.s += 1
                return Act(action=self.a[self.s])
            
            else:
                self.s = 0

        side = base64.urlsafe_b64decode(obs.cameras["rgb_side"])
        side = torch.frombuffer(bytearray(side), dtype=torch.uint8)
        side = decode_jpeg(side)
        side = v2.Resize((256, 256))(side)

        wrist = base64.urlsafe_b64decode(obs.cameras["rgb_wrist"])
        wrist = torch.frombuffer(bytearray(wrist), dtype=torch.uint8)
        wrist = decode_jpeg(wrist)
        wrist = v2.Resize((256, 256))(wrist)
        
        


        # side = np.copy(obs.cameras["rgb_side"]).transpose(2, 0, 1)
        # wrist = np.copy(obs.cameras["rgb_side"]).transpose(2, 0, 1)
        # return Act(action=np.array([]))
        observation = {}
        observation.update(
            {
                "observation/image": side,
                "observation/wrist_image": wrist,
                "observation/state": np.concatenate([obs.info["joints"], [1-obs.gripper]]),
                "prompt": self.instruction,
            })
        # calculate the time it takes to run policy inference
        import time
        t1 = time.time()
        action_chunk = self.policy.infer(observation)["actions"]
        t2 = time.time()
        elapsed = t2 - t1
        # print(f"OpenPiModel.infer needed time: {elapsed}s")
        # convert gripper action
        action_chunk[:,-1] = 1 - action_chunk[:,-1]
        self.a = action_chunk
        # return Act(action=action_chunk[0])
        print("action chunk shape" ,action_chunk.shape)
        print("Inference time (s): ", elapsed)

                # ensure numpy for saving
        if hasattr(action_chunk, "detach"):
            action_chunk_np = action_chunk.detach().cpu().numpy()
        else:
            action_chunk_np = np.asarray(action_chunk)
        if self.save_chunks:
            # === SAVE RAW CHUNK FIRST (exact policy output) ===
            _, csv_path = self.chunk_saver.save(action_chunk_np)
            print(f"[chunk] RAW saved -> {csv_path}  ({elapsed:.3f}s)")

        if self.policy.is_rtc:
            return Act(action=action_chunk, info={"inference_time_s": float(elapsed)})
        return Act(action=action_chunk[0], info={"inference_time_s": float(elapsed)})

    def reset(self, obs: Obs, instruction: Any, **kwargs) -> dict[str, Any]:
        info = super().reset(obs, instruction, **kwargs)
        self.policy.is_rtc = kwargs.get("is_rtc", False)
        self.policy.previous_action = None
        self.policy.s = kwargs.get("s", None)
        self.policy.d = kwargs.get("d", None)
        self.save_chunks = kwargs.get("save_chunks", False)
        dbg_folder_name = kwargs.get("dbg_folder_name", "dbg_chunks")
        if self.save_chunks and dbg_folder_name is not None:
            from agents.utils.save_chunks import ChunkSaverMin
            self.chunk_saver = ChunkSaverMin(dbg_folder_name)
        print("RTC mode: ", self.policy.is_rtc)
        print("s: ", self.policy.s)
        print("d: ", self.policy.d)
        print("save_chunks: ", self.save_chunks)
        print("dbg_folder_name: ", dbg_folder_name)
        print("Resetting OpenPiModel with instruction:", instruction)
        return info

class LerobotPiModel(Agent):
    
    def __init__(
        self,
        checkpoint_path: str,
        **kwargs,
    ) -> None:
        super().__init__(default_checkpoint_path=checkpoint_path, **kwargs)

    def initialize(self):
        from lerobot.lerobot_inference.rtc_controller import RTCDemoConfig, LeRobotPolicy
        from lerobot.configs.policies import PreTrainedConfig
        policy_cfg = PreTrainedConfig.from_pretrained(pretrained_name_or_path=self.default_checkpoint_path)
        cfg = RTCDemoConfig(policy=policy_cfg)
        cfg.policy.pretrained_path = self.default_checkpoint_path
        print("Initializing LeRobotPolicy with path:", self.default_checkpoint_path)
        print("default_checkpoint_path:", self.default_checkpoint_path)
        self.policy = LeRobotPolicy(cfg=cfg)
        print("LeRobotPolicy initialized")

    def act(self, obs: Obs) -> Act:
        import torch

        super().act(obs)

        side = base64.urlsafe_b64decode(obs.cameras["rgb_side"])
        side = torch.frombuffer(bytearray(side), dtype=torch.uint8)
        side = decode_jpeg(side)
        side = v2.Resize((256, 256))(side)

        wrist = base64.urlsafe_b64decode(obs.cameras["rgb_wrist"])
        wrist = torch.frombuffer(bytearray(wrist), dtype=torch.uint8)
        wrist = decode_jpeg(wrist)
        wrist = v2.Resize((256, 256))(wrist)
        
        observation = {}
        observation.update(
            {
                "observation.images.image": side,
                "observation.images.image2": wrist,
                "observation/state": torch.tensor(np.concatenate([obs.info["joints"], [1-obs.gripper]])), # to torch tensor later
                "task": [self.instruction],
            })
        action = self.policy.infer(observation)
        # post, orig = self.policy.get_actions(
        #     obs, prev_chunk_left_over=prev_left, inference_delay=inference_delay
        # )
        # action = post.detach().cpu().numpy()
        return Act(action=np.array(action))

    def reset(self, obs: Obs, instruction: Any, **kwargs) -> dict[str, Any]:
        info = super().reset(obs, instruction, **kwargs)
        print("Resetting LerobotPiModel with instruction:", instruction)
        return info

class OpenVLAModel(Agent):
    # === Utilities ===
    SYSTEM_PROMPT = (
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions."
    )

    def __init__(
        self,
        attn_implementation: str,
        device: str,
        unnorm_key: str,
        default_checkpoint_path: str = "openvla/openvla-7b",
        **kwargs,
    ) -> None:
        super().__init__(default_checkpoint_path=default_checkpoint_path, **kwargs)
        self.unnorm_key = unnorm_key
        logging.info(f"Using unnorm_key: {self.unnorm_key}")
        self.attn_implementation = attn_implementation
        if self.checkpoint_step is None or self.checkpoint_path is None:
            self.openvla_path = self.default_checkpoint_path
            logging.info(f"Using default checkpoint path: {self.openvla_path}")
            if self.unnorm_key != "viola":
                logging.warning(
                    "unnorm_key should be 'viola' when using default path, ignoring unnorm_key and setting it to 'viola'"
                )
                self.unnorm_key = "viola"
        else:
            self.openvla_path = self.checkpoint_path.format(checkpoint_step=self.checkpoint_step)
            logging.info(
                f"Using custom checkpoint path: {self.openvla_path} with checkpoint step: {self.checkpoint_step}"
            )
        self.device = device
        self.attn_implementation = attn_implementation

    def initialize(self):
        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor

        self.device = torch.device(self.device) if torch.cuda.is_available() else torch.device("cpu")

        # Load VLA Model using HF AutoClasses
        self.processor = AutoProcessor.from_pretrained(self.openvla_path, trust_remote_code=True)
        self.vla = AutoModelForVision2Seq.from_pretrained(
            self.openvla_path,
            attn_implementation=self.attn_implementation,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(self.device)
        print("==========================")
        print(self.vla.norm_stats.keys())

        # [Hacky] Load Dataset Statistics from Disk (if passing a path to a fine-tuned model)
        if os.path.isdir(self.openvla_path):
            with open(Path(self.openvla_path) / "dataset_statistics.json", "r") as f:
                self.vla.norm_stats = json.load(f)

    def get_openvla_prompt(self, instruction: str, openvla_path: Union[str, Path]) -> str:
        if "v01" in openvla_path:
            return f"{self.SYSTEM_PROMPT} USER: What action should the robot take to {instruction.lower()}? ASSISTANT:"
        else:
            return f"In: What action should the robot take to {instruction.lower()}?\nOut:"

    def act(self, obs: Obs) -> Act:
        # no batch dimension here
        import torch

        super().act(obs)
        # Parse payload components
        #assert obs.cameras["rgb_side"].shape == (256, 256, 3), "wrong shape, use lanczos"
        # image = obs.cameras["rgb_side"]
        side = base64.urlsafe_b64decode(obs.cameras["rgb_side"])
        side = torch.frombuffer(bytearray(side), dtype=torch.uint8)
        side = decode_jpeg(side)
        side = v2.Resize((256, 256))(side)        
        side = side.permute(1, 2, 0).numpy()
        image = side
        assert side.shape == (256, 256, 3)

        unnorm_key = self.unnorm_key

        # Run VLA Inference
        prompt = self.get_openvla_prompt(self.instruction, self.openvla_path)
        inputs = self.processor(prompt, Image.fromarray(image).convert("RGB")).to(self.device, dtype=torch.bfloat16)
        # to use temperature use: do_sample=True, temperature=50.0
        action = self.vla.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
        # unsqueeze to add horizon dimension
        return Act(action=action)


class OctoModel(Agent):
    """
    This model is trained with a window size of 2, predicting 7 dimensional actions 4 steps into the future.
    Observations and tasks conform to the following spec:

    Observations: {
        image_primary: ('batch', 'history_window', 256, 256, 3),
        image_wrist: ('batch', 'history_window', 128, 128, 3),
    }
    Tasks: {
        image_primary: ('batch', 256, 256, 3),
        image_wrist: ('batch', 128, 128, 3),
        language_instruction: {
            attention_mask: ('batch', 16),
            input_ids: ('batch', 16),
        },
    }

    At inference, you may pass in any subset of these observation and task keys, with a history window up to 2 timesteps.
    """

    def __init__(
        self,
        horizon: int = 2,
        unnorm_key: list[str] | None = None,
        default_checkpoint_path: str = "hf://rail-berkeley/octo-base-1.5",
        **kwargs,
    ) -> None:
        # default window size is 2 in octo
        # default unnorm is viola as it used the fr3
        super().__init__(default_checkpoint_path=default_checkpoint_path, **kwargs)
        self.horizon = horizon
        if unnorm_key is None:
            self.unnorm_key = []
        else:
            self.unnorm_key = unnorm_key

        # log checkpoint path and step and kwargs
        logging.info(f"checkpoint_path: {self.checkpoint_path}, checkpoint_step: {self.checkpoint_step}")
        logging.info(f"horizon: {self.horizon}")
        logging.info(f"unnorm_key: {self.unnorm_key}")
        logging.info(f"kwargs: {kwargs}")
        if self.checkpoint_path is None:
            self.octo_path = self.default_checkpoint_path
            logging.info(f"Using default checkpoint path: {self.octo_path}")
            if self.checkpoint_step is not None:
                logging.warning(
                    "checkpoint_step should be None when using default path, ignoring checkpoint_step and setting it to None"
                )
                self.checkpoint_step = None
            if self.unnorm_key != ["viola"]:
                logging.warning(
                    "unnorm_key should be ['viola'] when using default path, ignoring unnorm_key and setting it to ['viola']"
                )
                self.unnorm_key = ["viola"]
        else:
            self.octo_path = self.checkpoint_path
            logging.info(f"Using custom checkpoint path: {self.octo_path} with checkpoint step: {self.checkpoint_step}")
        logging.info(f"Using unnorm_key: {self.unnorm_key}")


        self.chunks = 1
        self.s = self.chunks
        self.a = None

    def initialize(self):
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = (
            "false"  # disable preallocation of memory in jax (might make it less efficient)
        )
        from octo.model.octo_model import OctoModel as _OctoModel
        from octo.utils.train_callbacks import supply_rng

        self.model = _OctoModel.load_pretrained(self.octo_path, self.checkpoint_step)

        window_size = self.model.example_batch["observation"]["timestep_pad_mask"].shape[1]
        if window_size < self.horizon:
            logging.warning(
                f"Horizon {self.horizon} is greater than the model's window size {window_size} with which the model has been trained with. "
            )

        self.trained_obs = self.model.example_batch["observation"].keys()

        self.horizon = self.horizon
        self.history = deque(maxlen=self.horizon)
        self.num_obs = 0

        logging.info("==========================")
        logging.info(self.model.dataset_statistics.keys())
        self.policy_fn = supply_rng(
            partial(
                self.model.sample_actions,
                unnormalization_statistics=reduce(getitem, self.unnorm_key, self.model.dataset_statistics)["action"],
            ),
        )
        self.task = None

    def act(self, obs: Obs) -> Act:
        # from octo.model.octo_model import _verify_shapes
        import jax
        from octo.utils.gym_wrappers import stack_and_pad

        super().act(obs)
        assert self.task is not None, "forgot reset?"
        # _verify_shapes(obs, <name>, self.model.example_batch["observation"])

        self.num_obs += 1


        side = base64.urlsafe_b64decode(obs.cameras["rgb_side"])
        side = torch.frombuffer(bytearray(side), dtype=torch.uint8)
        side = decode_jpeg(side)
        side = v2.Resize((256, 256))(side)        
        side = side.permute(1, 2, 0).numpy()
        assert side.shape == (256, 256, 3)
        obs.cameras["rgb_side"] = side

        # wrist = base64.urlsafe_b64decode(obs.cameras["rgb_wrist"])
        # wrist = torch.frombuffer(bytearray(wrist), dtype=torch.uint8)
        # wrist = decode_jpeg(wrist)
        # wrist = v2.Resize((128, 128))(wrist)        
        # wrist = wrist.permute(1, 2, 0).numpy()
        # assert wrist.shape == (128, 128, 3)
        # obs.cameras["rgb_wrist"] = wrist


        # single image
        # assert obs.cameras["rgb_side"].shape == (256, 256, 3), "wrong shape, use lanczos"
        obs = {"image_primary": obs.cameras["rgb_side"]
               #,"image_wrist": obs.cameras["rgb_wrist"]
               }

        self.history.append(obs)
        assert len(self.history) == self.horizon, "forgot reset?"
        full_obs = stack_and_pad(self.history, self.num_obs)

        if self.s < self.chunks:
            self.s += 1
            return Act(action=np.array(self.a[self.s]))
        
        else:
            self.s = 0

        actions = self.policy_fn(
            jax.tree_map(
                lambda x: x[None],
                full_obs,
            ),
            self.task,
        )
        # remove the batch dimension (batch, horizon, action)
        # print(actions.shape)
        self.a = actions[0, :, :]
        return Act(action=np.array(actions[0, 0, :]))

    def reset(self, obs: Obs, instruction: Any):
        super().reset(obs, instruction)
        
        # assert obs.cameras["rgb_side"].shape == (256, 256, 3), "wrong shape"
        side = base64.urlsafe_b64decode(obs.cameras["rgb_side"])
        side = torch.frombuffer(bytearray(side), dtype=torch.uint8)
        side = decode_jpeg(side)
        side = v2.Resize((256, 256))(side)        
        side = side.permute(1, 2, 0).numpy()
        assert side.shape == (256, 256, 3)
        obs.cameras["rgb_side"] = side
        # single image

        # wrist = base64.urlsafe_b64decode(obs.cameras["rgb_wrist"])
        # wrist = torch.frombuffer(bytearray(wrist), dtype=torch.uint8)
        # wrist = decode_jpeg(wrist)
        # wrist = v2.Resize((128, 128))(wrist)        
        # wrist = wrist.permute(1, 2, 0).numpy()
        # assert wrist.shape == (128, 128, 3)
        # obs.cameras["rgb_wrist"] = wrist

        obs = {"image_primary": obs.cameras["rgb_side"]
               # ,"image_wrist": obs.cameras["rgb_wrist"]
              }
        self.task = self.model.create_tasks(texts=[instruction])
        self.num_obs = 1
        self.history.extend([obs] * self.horizon)
        return {}


class OctoActionDistribution(OctoModel):
    """
    this model does not support history window
    dont use self.step and self.episode as this model is not used sequentially
    """

    def __init__(self, **kwargs) -> None:
        assert kwargs["horizon"] == 1, "horizon must be 1 for OctoActionDistribution"
        super().__init__(**kwargs)

    def act(self, obs: Obs) -> Act:
        """
        Args:
            Obs:
                cameras:
                    rgb_side: np.ndarray[tuple[BATCH, H, W, Literal[3]], np.dtype[np.int8]]
                info:
                    num_samples: int
        Return:
            Act:
                action: None
                info:
                    means: np.ndarray[tuple[BATCH, 7], np.dtype[np.float32]]
                    stds: np.ndarray[tuple[BATCH, 7], np.dtype[np.float32]]
        """
        import jax
        import jax.numpy as jnp

        self._from_shared_memory(obs)

        batch_size = obs.cameras["rgb_side"].shape[0]
        assert obs.cameras["rgb_side"].shape == (batch_size, 256, 256, 3), "wrong shape"
        assert self.instruction is not None, "forgot reset?"
        num_samples = obs.info.get("num_samples", 1)

        x = jnp.array(obs.cameras["rgb_side"])  # BATCH, H, W, 3
        # x_expanded = x[:, None, :, :, :]
        x_expanded = jnp.expand_dims(x, 1)
        x_tiled = jnp.tile(x_expanded, (1, num_samples, 1, 1, 1))  # Shape: [BATCH, N, H, W, 3]
        x_duplicated = x_tiled.reshape(-1, x.shape[1], x.shape[2], x.shape[3])  # Shape: [BATCH*N, H, W, 3]
        full_obs = {
            "image_primary": jnp.expand_dims(x_duplicated, 1),
            "timestep_pad_mask": np.ones((batch_size * num_samples, 1)),
        }
        # full_obs = stack_and_pad(x_duplicated, 1)
        tasks = self.model.create_tasks(texts=[self.instruction] * batch_size * num_samples)
        actions = self.policy_fn(
            full_obs,
            tasks,
        )
        # actions: [num_samples x BATCH, 4, 7]
        # remove the horizon dimension and reshape to [BATCH, num_samples, 7]
        actions = actions[:, 0, :].reshape(batch_size, num_samples, 7)
        stds = jnp.std(actions, axis=1)
        means = jnp.mean(actions, axis=1)

        stds = np.asarray(stds)
        means = np.asarray(means)

        return Act(action=None, info={"means": means, "stds": stds, "actions": np.asarray(actions)})

    def reset(self, obs, instruction):
        self.instruction = instruction
        return {}


class OpenVLADistribution(OpenVLAModel):

    def act(self, obs: Obs) -> Act:
        # no batch dimension here
        import torch

        self._from_shared_memory(obs)

        assert self.instruction is not None, "forgot reset?"
        self.step += 1
        batch_size = obs.cameras["rgb_side"].shape[0]

        # Parse payload components
        images = obs.cameras["rgb_side"]
        actions = []
        unnorm_key = self.unnorm_key
        num_samples = obs.info.get("num_samples", 1)

        # time it
        import time

        t1 = time.time()
        # Run VLA Inference
        prompt = self.get_openvla_prompt(self.instruction, self.openvla_path)

        x_expanded = np.expand_dims(images, 1)
        x_tiled = np.tile(x_expanded, (1, num_samples, 1, 1, 1))  # Shape: [BATCH, N, H, W, 3]
        x_duplicated = x_tiled.reshape(
            -1, images.shape[1], images.shape[2], images.shape[3]
        )  # Shape: [BATCH*N, H, W, 3]

        for image in x_duplicated:
            inputs = self.processor(prompt, Image.fromarray(image).convert("RGB")).to(self.device, dtype=torch.bfloat16)
            # to use temperature use: do_sample=True, temperature=50.0
            action = self.vla.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
            actions.append(action)

        t2 = time.time()
        logging.info(f"needed time for {len(actions)} was {t2-t1}s")

        # unsqueeze to add horizon dimension
        actions = np.stack(actions).astype(np.float32)
        actions = actions.reshape(batch_size, num_samples, 7)
        means = np.mean(actions, axis=1).astype(np.float32)
        stds = np.std(actions, axis=1).astype(np.float32)
        return Act(action=None, info={"means": means, "stds": stds, "actions": actions})


AGENTS = dict(
    test=TestAgent,
    octo=OctoModel,
    openvla=OpenVLAModel,
    octodist=OctoActionDistribution,
    openvladist=OpenVLADistribution,
    openpi=OpenPiModel,
    lerobot_pi=LerobotPiModel
)
