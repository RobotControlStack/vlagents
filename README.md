# VLAgents
VLAgents is a python library that allows to separate next action prediction from policy networks from action execution in simulated or real environments.
It defines an interface for policies and for environments.
The policies run independent in their own virtual environment, potentially on a different computer, and can be queried for an action (in principle similar to the chatgpt api).

Why is this useful?
- Separation of dependencies by using two different python environments: Some times dependencies contradict e.g. pytorch and jax
- Some robot hardware requires a real time linux kernel which does not easily allow you to use an Nvidia GPU.
- Separate deployment and model code

This library is a byproduct of the [Refined Policy Distillation (RPD)](https://refined-policy-distillation.github.io/) paper which distilled VLAs into expert policies using Reinforcement Learning.
The work also includes a section on related engineering challenges regarding jax and pytorch.

## Installation

### Pip Installation (Recommended)
```shell
pip install vlagents
```

### Local Installation
```shell
git clone https://https://github.com/RobotControlStack/vlagents.git
cd vlagents
pip install -ve .
```


### Environment and Policy Installation
On top of vlagents you can then install a simulation environment where the agent acts.
We currently the following environments:
- [maniskill](https://github.com/haosulab/ManiSkill)
- [robot control stack](https://github.com/RobotControlStack/robot-control-stack)
- [duobench](https://github.com/RobotControlStack/duobench)
- [libero](https://github.com/Lifelong-Robot-Learning/LIBERO)


In order to avoid dependency conflicts, use a second conda/pip environment to install your policy.
We currently support the following policies:
- [octo](https://github.com/octo-models/octo)
- [openvla](https://github.com/openvla/openvla)
- [openpi](https://github.com/Physical-Intelligence/openpi)
- [vjepa2-ac](https://github.com/facebookresearch/vjepa2)
- [diffusion policy](https://github.com/real-stanford/diffusion_policy)
- [lerobot policies](https://github.com/huggingface/lerobot/tree/main/src/lerobot/policies)


### LeRobot
```shell
pip install 'lerobot[all]'
```



### Octo
To use Octo as an agent/policy you need to create a new conda environment:
```shell
conda create -n octo python=3.10
conda activate octo
conda install nvidia/label/cuda-11.8.0::cuda --no-channel-priority
conda install conda-forge::cudnn=8.9
# octo dependencies
pip install git+https://github.com/octo-models/octo.git@241fb3514b7c40957a86d869fecb7c7fc353f540
pip install -r vlagents/utils/fixed_octo_requirements.txt
# for gpu support:
pip install --upgrade "jax[cuda11_pip]==0.4.20" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

Verify that the jax installation was successful and that jax finds your gpu.
Open a python shell in the same conda env and type
```python
from jax.lib import xla_bridge
# this should output "gpu" if the gpu installation was successful
print(xla_bridge.get_backend().platform)
```


Install the vlagents library on top:
```shell
pip install git+https://github.com/juelg/vlagents.git
```

For more details, see the [Octo github page](https://github.com/octo-models/octo).

#### Troubleshooting
If pip complains about dependency issues than it might have happened that torch somehow slipped in.
Check if you have any torch packages installed by
```shell
pip freeze | grep torch
# if any, uninstall them e.g.
pip uninstall arm_pytorch_utilities
pip uninstall pytorch-seed
pip uninstall pytorch_kinematics
```

### OpenVLA
To use OpenVLA, create a new conda environment:
```shell
conda create -n openvla python=3.10 -y
conda activate openvla
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y
```

Install [flash attention](https://github.com/Dao-AILab/flash-attention):
```shell
pip install packaging ninja
ninja --version; echo $?  # Verify Ninja --> should return exit code "0"
pip install "flash-attn==2.5.5" --no-build-isolation
# if you run into issues try `pip cache remove flash_attn` first
```

Install OpenVLA
```shell
pip install git+https://github.com/openvla/openvla.git@46b752f477cc5773cc1234b2e82c0e2130e4e890
```

Install the vlagents library on top:
```shell
pip install git+https://github.com/juelg/vlagents.git
```

For more details, see the [OpenVLA github page](https://github.com/openvla/openvla).

### OpenPi / Pi0
To use OpenPi, create a new conda environment:
```shell
conda create -n openpi python=3.11 -y
conda activate openpi
```
Clone the repo and install it.
```shell
git clone --recurse-submodules git@github.com:Physical-Intelligence/openpi.git
# Or if you already cloned the repo:
git submodule update --init --recursive
# install dependencies
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```
For more details see [openpi's github](https://github.com/Physical-Intelligence/openpi).

### vjEPA2-ac
To use VJEPA2-AC, create a new conda environment:
```shell
conda create -n vjepa2 python=3.12
conda activate vjepa2
```
Clone the repo and install it.
```shell
git clone git@github.com:facebookresearch/vjepa2.git
cd vjepa2
pip install -e .

pip install git+https://github.com/juelg/vlagents.git
pip install -ve .

```

### Diffusion Policy
Currently located on the branch `diffusion_policy`.

## Usage
To start an vlagents server use the `start-server` command where `kwargs` is a dictionary of the constructor arguments of the policy you want to start e.g.
```shell
# lerobot act
python -m vlagents start-server lerobot --port 8080 --host 0.0.0.0 --kwargs '{"policy_name": "act", "checkpoint_path": "<path to pretrained_model>"}'

# lerobot pi05
python -m vlagents start-server lerobot --port 20000 --host 0.0.0.0 --kwargs '{"policy_name": "pi05", "checkpoint_path": "<path to pretrained_model>"}'

# lerobot xvla
uv run python -m vlagents start-server lerobot --port 20000 --host 0.0.0.0 --kwargs '{"policy_name": "xvla", "checkpoint_path": "<path to pretrained_model>", "rename_map": {"head": "image", "left_wrist": "image2", "right_wrist": "image3"}}'


# octo
python -m vlagents start-server octo --host localhost --port 8080 --kwargs '{"checkpoint_path": "hf://Juelg/octo-base-1.5-finetuned-maniskill", "checkpoint_step": None, "horizon": 1, "unnorm_key": []}'

# openvla
python -m vlagents start-server openvla --host localhost --port 8080 --kwargs '{"checkpoint_path": "Juelg/openvla-7b-finetuned-maniskill", "device": "cuda:0", "attn_implementation": "flash_attention_2", "unnorm_key": "maniskill_human:7.0.0", "checkpoint_step": 40000}'

# openpi
python -m vlagents start-server openpi --port=8080 --host=localhost --kwargs='{"checkpoint_path": "<path to checkpoint>/{checkpoint_step}", "model_name": "pi0_rcs", "checkpoint_step": <checkpoint_step>}' # leave "{checkpoint_step}" it will be replaced, "model_name" is the key for the training config

# vjepa2-ac
python -m vlagents start-server vjepa --port=20997 --host=0.0.0.0 --kwargs='{"cfg_path": "configs/inference/vjepa2-ac-vitg/<your_config>.yaml", "model_name": "vjepa2_ac_vit_giant", "default_checkpoint_path": "../.cache/torch/hub/checkpoints/vjepa2-ac-vitg.pt"}'

# general VLM controlling the tool pose, see "VLM agent" below
OPENAI_API_KEY=... python -m vlagents start-server vlm --port 8080 --host 0.0.0.0 --kwargs '{"backend": "openai", "model": "gpt-5", "control_mode": "xyzrpy", "log_dir": "runs/vlm"}'
ANTHROPIC_API_KEY=... python -m vlagents start-server vlm --port 8080 --host 0.0.0.0 --kwargs '{"backend": "anthropic", "model": "claude-opus-5", "control_mode": "xyzrpy", "log_dir": "runs/vlm"}'
# a Claude Code subagent or a human answers the requests written to mailbox_dir
python -m vlagents start-server vlm --port 8080 --host 0.0.0.0 --kwargs '{"backend": "mailbox", "mailbox_dir": "runs/vlm/mailbox", "control_mode": "xyzrpy"}'
```

Episodes are stateful: `RemoteAgent.reset(obs, instruction)` is called once per episode before the first `act` (the eval loop does this automatically, `examples/inference/franka.py` in RCS does it when an episode starts). Policies without memory can ignore it.


Each policy returns an `Act` action chunk. During evaluation, `EvalEnv.chunk_step` applies the chunk one environment step at a time. Configure `execution_horizon` in an evaluation config to cap how many actions from each chunk are executed before requesting a new one.

Images are resized by `RemoteAgent` before shared-memory or JPEG transport. Set `image_size` in an evaluation config to a `[width, height]` pair (default `[224, 224]`), or `null` to keep native resolution.

### VLM agent
`vlagents/policies/vlm.py` lets a general vision-language model control the robot through motion primitives. Every call sends the system prompt (robot, coordinate frame, workspace, cameras, output schema), the full episode history and the current images and state, and expects one JSON command per arm which is expanded into a chunk of `chunk_size` actions (30 actions = 1 s at 30 Hz):

- Cartesian (`control_mode` `xyzrpy` or `tquat`): `hold`, `gripper`, `move` (absolute xyz + rpy in degrees), `move_delta`, or a raw `chunk`.
- Joint space (`control_mode` `joints`): `hold`, `gripper`, `move_joints` (degrees), `move_joints_delta`, or a raw `chunk`.

Backends (`backend` kwarg): `openai` (any OpenAI compatible endpoint, default model `gpt-5`), `anthropic` (Anthropic API, default `claude-opus-5`, key from `ANTHROPIC_API_KEY`), `mailbox` (requests are written to `mailbox_dir` and answered by an external pilot such as a Claude Code subagent or a human, see [.claude/skills/robot-pilot](.claude/skills/robot-pilot/SKILL.md)) and `fake`.
Important kwargs: `model`, `base_url` (e.g. a vLLM server), `control_mode`, `history` (`"full"` or number of recent turns that keep their images), `image_size`, `icl_path`/`icl_episodes`/`icl_images` (in-context demonstrations), `log_dir` (per episode dump of prompts, images, replies and token usage), `backend: "fake"` (canned replies for tests). The env must run in the same control mode, e.g. `"env_kwargs": {"control_mode": "xyzrpy"}` for the duobench envs.

In-context examples are exported from a LeRobot dataset (at most 10 episodes), e.g. for the DuoBench transfer cube task:
```shell
python -m vlagents.policies.vlm_icl <lerobot_dataset_dir> transfer_cube_icl.json --episodes 10 --stride 30 --control-mode xyzrpy --image-size 224
```
### Residual edit policy (EXPO-style)
`vlagents/policies/residual.py` adds a small network that edits the VLM's chunk at control rate: the nominal target of every step is shifted by a bounded delta (default 2 cm / 5 deg per axis) computed from the chunk geometry, the gripper states and a privileged task state (for `duobench/ball_maze`: ball position and velocity in the board frame, board pose; `env_kwargs: {"task_state": true}`). It runs in the environment loop (`EvalConfig.editor`), also while the agent thinks when `simulate_inference_delay` is on, and records every step for training:

```shell
# warm start from demonstrations (deviation of the demonstrated path from the straight-line chunk); a DuoBench
# replay recording with simulator states adds task-state labelled samples
python -m vlagents.policies.residual_train warmstart <lerobot_dir> runs/residual/ckpt_warmstart.pt --replay-dir <replay_dir>
# evaluate with the residual and let the VLM give hindsight corrections after every command
python -m vlagents run-eval ... --agent-cfg '{..., "agent_kwargs": {"backend": "mailbox", "hindsight_corrections": true, "log_dir": "runs/maze/vlm", ...}}' \
  --eval-cfgs '[{"env_id": "duobench/ball_maze", "env_kwargs": {"control_mode": "xyzrpy", "task_state": true}, "editor": {"control_mode": "xyzrpy", "ckpt": "runs/residual/ckpt_warmstart.pt"}, ...}]'
# relabel the recorded steps with the corrections (edit + correction) and fine-tune
python -m vlagents.policies.residual_train finetune runs/residual/ckpt_warmstart.pt runs/residual/ckpt_ft.pt runs/maze/M0 runs/maze/M1
```

With `hindsight_corrections` the VLM reply carries a `"correction"` per arm: the adjustment that, knowing the outcome, should have been added to the previous command's targets (with `from`/`to` fractions of the chunk). The prompt tells the VLM how much the residual intervened during the last command.

See [docs/vlm_harness_plan.md](docs/vlm_harness_plan.md) for the design and the experiment plan.

There is also the `run-eval-during-training` command to evaluate a model during training, so a single checkpoint.
The `run-eval-post-training` command evaluates a range of checkpoints in parallel.
In both cases environment and arguments as well as policy and arguments and wandb config for logging can be passed as CLI arguments.


## Adding your own environment
```python
from vlagents import register_env
from vlagents.envs.interface import EvalEnv
from vlagents.policies.interface import Act, Obs, SingleAct
from typing import Any

class YourEnv(EvalEnv):
    # Override make_gym() when this environment is not created with gym.make().

    def translate_obs(self, obs: dict[str, Any]) -> Obs:
        # translated your observation
        return Obs()

    def step(self, action: dict[str, SingleAct]) -> tuple[Obs, float, bool, bool, dict]:
        # step your env
        obs, reward, success, truncated, info = self.env.step(action)
        return self.translate_obs(obs), reward, success, truncated, info

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Obs, dict[str, Any]]:
        obs, info = self.env.reset()
        return self.translate_obs(obs), info

    @property
    def language_instruction(self) -> str:
        # return task instruction
        return "pick up the cube"

    def do_import(self):
        # import any packages required by your env
        import libero

register_env("your-env-id", YourEnv)
```

## Adding your own policy
```python
from vlagents import register_agent
from vlagents.policies.interface import Agent
from vlagents.policies.interface import Obs, Act
from typing import Any
import numpy as np

class YourAgent(Agent):
    def initialize(self):
        # heavy initialization, e.g. loading models
        pass

    def act(self, obs: Obs) -> Act:
        # your forward pass
        return Act(action=np.zeros(7, dtype=np.float32), done=False, info={})

    def reset(self, obs: Obs, instruction: Any, **kwargs) -> dict[str, Any]:
        # reset model if it has state and return info dict
        return {}

    def close(self, *args, **kwargs):
        pass
register_agent("your-agent-id", YourAgent)
```



## Contribution

### New Policy
In order to extend the library with a new policy network, extend the `Agent` class in [policies/interface.py](src/vlagents/policies/interface.py).
It is important to only invoke policy specific imports in the class functions, as each policy can have its own dependencies.


### New Environment
In order to extend the library with a new agent environment, extend the `EvalEnv` class in [envs/interface.py](src/vlagents/envs/interface.py).


### Developer Tools
Install the following dev dependencies:
```shell
pip install 'pip>=25.1'
pip install --group dev
```

The following dev tools are provided:
```shell
# format the code
make format

# lint the code
make lint

# run tests
make test
```

## Citation
If you find the agent useful for your work, please consider citing the original works behind it:
```
@inproceedings{juelg2025refinedpolicydistillationvla,
    title={{Refined Policy Distillation}: {F}rom {VLA} Generalists to {RL} Experts}, 
    author={Tobias J{\"u}lg and Wolfram Burgard and Florian Walter},
    year={2025},
    booktitle={Proc.~of the IEEE/RSJ Int.~Conf.~on Intelligent Robots and Systems (IROS)}
}
@misc{juelg2026vlagentspolicyserverefficient,
      title={VLAgents: A Policy Server for Efficient VLA Inference}, 
      author={Tobias J{\"u}lg and Khaled Gamal and Nisarga Nilavadi and Pierre Krack and Seongjin Bien and Michael Krawez and Florian Walter and Wolfram Burgard},
      year={2026},
      howpublished={\url{https://arxiv.org/abs/2601.11250}}
}
```