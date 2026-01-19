import copy
import datetime
import json
import logging
import os
import shlex
import subprocess
from abc import ABC
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from time import sleep
from typing import Any

import gymnasium as gym
import numpy as np
from PIL import Image
from simple_slurm import Slurm
from tqdm import tqdm

from vlagents.client import RemoteAgent
from vlagents.policies import Act, Agent, Obs
from vlagents.wrappers import HumanCameraWrapper

logging.basicConfig(
    format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


class EvaluatorEnv(ABC):
    ENVS: dict[str, "EvaluatorEnv"] = {}

    def __init__(self, env_id: str, seed: int, **env_kwargs) -> None:
        self.do_import()
        self.env = gym.make(env_id, **env_kwargs)
        self.env.np_random = np.random.RandomState(seed=seed)
        self.env_id = env_id
        self.seed = seed

    def step(self, action: Act) -> tuple[Obs, float, bool, bool, dict]:
        raise NotImplementedError

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Obs, dict[str, Any]]:
        raise NotImplementedError

    @property
    def language_instruction(self) -> str:
        raise NotImplementedError

    @staticmethod
    def register(env_id: str, env: "EvaluatorEnv") -> None:
        EvaluatorEnv.ENVS[env_id] = env

    @staticmethod
    def make(env_id: str, seed: int, **env_kwargs) -> "EvaluatorEnv":
        return EvaluatorEnv.ENVS[env_id](env_id, seed, **env_kwargs)

    @staticmethod
    def do_import():
        raise NotImplementedError


class RCSPickUpCubeEval(EvaluatorEnv):
    INSTRUCTIONS = {
        "rcs/FR3SimplePickUpSim-v0": "pick the green box",
        "rcs/FR3LabPickUpSimDigitHand-v0": "pick the green box",
    }

    def translate_obs(self, obs: dict[str, Any]) -> Obs:
        # does not include history

        # side = obs["frames"]["arro"]["rgb"]["data"]
        side = obs["frames"]["side"]["rgb"]["data"]
        wrist = obs["frames"]["wrist"]["rgb"]["data"]
        # depth_side = obs["frames"]["side"]["depth"]["data"],
        return Obs(
            cameras=dict(rgb_side=side, rgb_wrist=wrist),
            # cameras=dict(rgb_side=side),
            gripper=obs["gripper"],
            info=dict(joints=obs["joints"]),
        )

    def step(self, action: Act) -> tuple[Obs, float, bool, bool, dict]:
        # includes horizon
        if action.action.shape[0] != 7:
            obs, reward, success, truncated, info = self.env.step(
                {"xyzrpy": action.action[0][:6], "gripper": action.action[0][6]}
            )
        else:
            obs, reward, success, truncated, info = self.env.step(
                {"xyzrpy": action.action[:6], "gripper": action.action[6]}
            )
        # print(action.action, obs["xyzrpy"], obs["gripper"])
        return self.translate_obs(obs), reward, success, truncated, info

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Obs, dict[str, Any]]:
        obs, info = self.env.reset()
        return self.translate_obs(obs), info

    @property
    def language_instruction(self) -> str:
        return self.INSTRUCTIONS[self.env_id]

    @staticmethod
    def do_import():
        import rcs
        import rcs_toolbox


EvaluatorEnv.register("rcs/FR3SimplePickUpSim-v0", RCSPickUpCubeEval)
EvaluatorEnv.register("rcs/FR3LabPickUpSimDigitHand-v0", RCSPickUpCubeEval)


class ManiSkill(EvaluatorEnv):
    INSTRUCTIONS = {
        "LiftPegUpright-v1": "lift the peg upright",
        "PegInsertionSide-v1": "insert the peg from the side",
        "PickCube-v1": "pick up the cube",
        "PlugCharger-v1": "plug the charger in",
        "PullCube-v1": "pull the cube towards the robot base",
        "PullCubeTool-v1": "pull the cube by using the red tool",
        "PushCube-v1": "push the cube away from the robot base",
        "PushT-v1": "align the T shape",
        "RollBall-v1": "push the ball",
        "StackCube-v1": "stack the red cube on the green cube",
        "PokeCube-v1": "push the cube by using the blue tool",
    }

    def __init__(self, env_id, seed, **env_kwargs):
        # TODO: one could save only every nth episode by adding an episode counter which steps the record env only
        # when the counter is divisible by n otherwise steps the normal env
        logging.info(f"Creating ManiSkill env {env_id}")
        output_dir = env_kwargs.pop("video_dir", None)
        super().__init__(env_id, seed, **env_kwargs)
        logging.info(f"Created ManiSkill env {env_id}")
        if "human_render_camera_configs" in env_kwargs:
            self.env = HumanCameraWrapper(self.env)

        if output_dir is not None:
            logging.info(f"Recording to {output_dir}")
            from mani_skill.utils import wrappers

            self.env = wrappers.RecordEpisode(
                self.env,
                output_dir,
                save_on_reset=True,
                save_trajectory=True,
                trajectory_name=f"eval-{env_id}",
                save_video=True,
                video_fps=30,
                record_reward=True,
            )
        logging.info(f"Done Created ManiSkill env {env_id}")

    def translate_obs(self, obs: dict[str, Any]) -> Obs:
        # does not include history
        return Obs(
            cameras=dict(rgb_side=obs["sensor_data"]["base_camera"]["rgb"].squeeze(0).numpy()),
            # gripper=float(not obs["extra"]["is_grasped"]),
        )

    def step(self, action: Act) -> tuple[Obs, float, bool, bool, dict]:
        # includes horizon
        # careful with gripper action: the model needs to be trained on [-1, 1] interval

        a = copy.copy(action.action[0])
        # a[-1] = -1.0 if a[-1] < 0.9 else 1.0
        if self.env_id == "PushT-v1":
            a = a[:-1]
        else:
            a[-1] = a[-1] * 2 - 1.0
        obs, reward, success, truncated, info = self.env.step(a)
        return self.translate_obs(obs), reward, success, truncated, info

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Obs, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        return self.translate_obs(obs), info

    @property
    def language_instruction(self) -> str:
        return self.INSTRUCTIONS[self.env_id]

    @staticmethod
    def do_import():
        import mani_skill.envs


EvaluatorEnv.register("LiftPegUpright-v1", ManiSkill)
EvaluatorEnv.register("PegInsertionSide-v1", ManiSkill)
EvaluatorEnv.register("PickCube-v1", ManiSkill)
EvaluatorEnv.register("PlugCharger-v1", ManiSkill)
EvaluatorEnv.register("PullCube-v1", ManiSkill)
EvaluatorEnv.register("PullCubeTool-v1", ManiSkill)
EvaluatorEnv.register("PushCube-v1", ManiSkill)
EvaluatorEnv.register("PushT-v1", ManiSkill)
EvaluatorEnv.register("RollBall-v1", ManiSkill)
EvaluatorEnv.register("StackCube-v1", ManiSkill)
EvaluatorEnv.register("PokeCube-v1", ManiSkill)


class Libero(EvaluatorEnv):

    def __init__(self, env_id: str, seed: int, reset_steps: int = 14, **env_kwargs) -> None:
        """
        For supported env_kwargs checkout ControlEnv class in libero.
        We add the following env_kwargs on top:
        - task_id (int): libero task id for given task suite. The number of tasks per task suite can checked with Libero.n_tasks(env_id). Defaults to 0.
        - control_mode (str): either 'relative' or 'absolute'. Defaults to 'relative'.

        """
        logging.info("Creating Libero env")
        self.env_kwargs = env_kwargs
        self.reset_steps = reset_steps
        self.control_mode = self.env_kwargs.pop("control_mode", "relative")
        self.env, self._language_instruction, self.task_name, self.task_suite, self.task_id, self.task = self._make_gym(
            env_id, seed, **self.env_kwargs
        )
        logging.info(
            f"Created Libero env, task suite: {env_id}, task id: {self.task_id}, task name {self.task_name}, instruction: {self._language_instruction}"
        )
        self.env_id = env_id
        self.seed = seed

    @staticmethod
    def n_tasks(env_id: str) -> int:
        from libero.libero import benchmark, get_libero_path

        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[env_id]()
        return task_suite.n_tasks

    @staticmethod
    def _make_gym(env_id, seed, **env_kwargs):
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        benchmark_dict = benchmark.get_benchmark_dict()

        task_suite = benchmark_dict[env_id]()
        task_id = min(max(env_kwargs.pop("task_id", 0), 0), task_suite.n_tasks - 1)
        task = task_suite.get_task(task_id)

        task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        env = OffScreenRenderEnv(
            bddl_file_name=task_bddl_file,
            **env_kwargs,
        )
        env.seed(seed)

        return env, task.language, task.name, task_suite, task_id, task

    def translate_obs(self, obs: dict[str, Any]) -> Obs:
        return Obs(
            cameras=dict(rgb_side=obs["agentview_image"][::-1], rgb_wrist=obs["robot0_eye_in_hand_image"][::-1]),
            gripper=obs["robot0_gripper_qpos"] / 0.04,  # normalize
        )

    def step(self, action: Act) -> tuple[Obs, float, bool, bool, dict]:
        # change gripper to libero format (-1, 1) where -1 is open
        act = np.copy(action.action)
        act[-1] = (1 - act[-1]) * 2 - 1.0
        obs, reward, done, info = self.env.step(act)
        success = self.env.check_success()
        return self.translate_obs(obs), reward, success, done, info

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Obs, dict[str, Any]]:
        obs = self.env.reset()
        init_states = self.task_suite.get_task_init_states(
            self.task_id
        )  # for benchmarking purpose, we fix the a set of initial states
        init_state_id = 0
        self.env.set_init_state(init_states[init_state_id])

        for robot in self.env.robots:
            robot.controller.use_delta = True
        for _ in range(self.reset_steps):
            # steps the environment to filter out falling objects
            obs, _, _, _ = self.env.step(
                np.zeros(8) if "JOINT" in self.env_kwargs.get("controller", "OSC_POSE") else np.zeros(7)
            )

        if self.control_mode == "absolute":
            for robot in self.env.robots:
                robot.controller.use_delta = False
        elif self.control_mode == "relative":
            for robot in self.env.robots:
                robot.controller.use_delta = True
        else:
            raise ValueError(f"Invalid control mode: {self.control_mode}, use 'absolute' or 'relative'.")

        return self.translate_obs(obs), {}

    @property
    def language_instruction(self) -> str:
        return self._language_instruction


EvaluatorEnv.register("libero_10", Libero)
EvaluatorEnv.register("libero_90", Libero)
EvaluatorEnv.register("libero_100", Libero)
EvaluatorEnv.register("libero_spatial", Libero)
EvaluatorEnv.register("libero_object", Libero)
EvaluatorEnv.register("libero_goal", Libero)


@dataclass
class EvalConfig:
    env_id: str
    env_kwargs: dict[str, Any]
    max_steps_per_episode: int = 100
    seed: int = 42
    same_machine: bool = False
    jpeg_encoding: bool = False


@dataclass
class AgentConfig:
    host: str
    agent_name: str
    agent_kwargs: dict[str, Any]
    python_path: str = "python"
    """modify this if you want to use a specific python environment """
    port: int = 8080


def single_eval(env: EvaluatorEnv, agent: Agent, max_steps: int, i) -> tuple[list[float], list[float], list[float]]:
    logging.debug(f"Starting evaluation")
    obs, _ = env.reset(options={})
    logging.debug(f"Reset env")
    agent.reset(obs, env.language_instruction)
    logging.debug(f"Reset agent")
    done = False
    truncated = False
    step = 0.0
    rewards = []
    im = []
    while not done and not truncated and max_steps > step:
        action = agent.act(obs)
        obs, reward, done, truncated, _ = env.step(action)
        reward = float(reward)
        done, truncated = bool(done), bool(truncated)
        step += 1
        rewards.append(reward)
        im.append(obs.cameras)

    Path(f"{os.environ['CAM_PATH']}").mkdir(exist_ok=True, parents=True)
    for camera in im[0].keys():
        imgs = []
        for img in im:
            # skip images that have timestamps closer together than 0.5s
            imgs.append(Image.fromarray(img[camera]))

        imgs[0].save(
            f"{os.environ['CAM_PATH']}/{i}_{camera}_{str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))}.gif",
            save_all=True,
            append_images=imgs[1:],
            duration=0.2 * 1000,
            loop=0,
        )

    env.reset(options={})
    logging.debug(f"Finished evaluation with {step} steps and reward {reward}, success {done}")
    # success, last reward and number of steps
    return done, rewards, step


per_process_cache = {}


def create_env_agent(agent_config: AgentConfig, cfg: EvalConfig, seed: int) -> tuple[EvaluatorEnv, RemoteAgent]:
    logging.debug(f"retrieving env {cfg.env_id} and agent")
    key = (cfg.env_id, agent_config.host, agent_config.port)
    if key not in per_process_cache:
        logging.info(f"env {cfg.env_id} not available, creating new env and agent")
        env = EvaluatorEnv.make(cfg.env_id, seed=seed, **cfg.env_kwargs)
        logging.info("done creating env")
        agent = RemoteAgent(
            agent_config.host,
            agent_config.port,
            agent_config.agent_name,
            on_same_machine=cfg.same_machine,
            jpeg_encoding=cfg.jpeg_encoding,
        )
        logging.info("done creating agent")
        per_process_cache[key] = (env, agent)
    return per_process_cache[key]


def run_episode(args: tuple[int, list[EvalConfig], int, AgentConfig]) -> tuple[float, float, float]:
    i, cfgs, episodes, agent_cfg = args
    cfg = cfgs[i // episodes]
    env, agent = create_env_agent(agent_cfg, cfg, seed=i)
    # busy wait for server to finish initialization
    while not agent.is_initialized():
        logging.info("Waiting for agent to initialize...")
        sleep(5)
    return single_eval(env, agent, cfg.max_steps_per_episode, i)


def multi_eval(
    agent_cfg: AgentConfig, cfgs: list[EvalConfig], episodes: int = 100, n_processes: int = 1
) -> tuple[np.ndarray, list[list[list[float]]]]:
    # return is [envs, episodes, 3(success, reward, steps)], [envs, episodes, rewards for all steps in the episode]
    logging.info(f"Starting evaluation with {len(cfgs)} environments and {episodes} episodes each")

    # with process
    # with Pool(n_processes) as p:
    #     args = [(i, cfgs, episodes, client_cfg) for i in range(len(cfgs) * episodes)]
    #     single_results = p.map(run_episode, args)

    # without process
    np.random.seed(cfgs[0].seed)
    args = [(i, cfgs, episodes, agent_cfg) for i in range(len(cfgs) * episodes)]
    single_results = [run_episode(arg) for arg in tqdm(args)]

    single_results_last_reward = np.array([(i[0], i[1][-1], i[2]) for i in single_results])

    # this works because row-major order
    # per_env_results = single_results.reshape(len(cfgs), episodes, 3)
    per_env_results_last_reward = single_results_last_reward.reshape(len(cfgs), episodes, 3)
    per_env_results_rewards = [
        [i[1] for i in single_results[i : i + episodes]] for i in range(0, len(single_results), episodes)
    ]
    return per_env_results_last_reward, per_env_results_rewards


@contextmanager
def start_server(
    agent_name: str, kwargs: dict[str, Any], port: int = 8080, host: str = "localhost", python_path: str = "python"
):
    """Start the agent server in a subprocess as a context manager.

    This ensures that the server is properly stopped when exiting the context and
    that all logs are printed to the console.

    Args:
        agent_name (str): Name of the agent to start.
        kwargs (dict[str, Any]): Additional keyword arguments for the agent.
        port (int): Port to start the server on. Defaults to 8080.
        host (str): Host to bind the server to. Defaults to "localhost".
        python_path (str): Path to the Python interpreter to use. If you use conda you can look up the path with `conda info --envs`.
            It can also be a format string that will be formatted with the agent_name, e.g. "conda run -n {agent_name} python".
            Defaults to "python".
    """
    cmd = [
        python_path.format(agent_name=agent_name),
        "-m",
        "vlagents",
        "start-server",
        f"{agent_name}",
        f"--port={port}",
        f"--host={host}",
        f"--kwargs={json.dumps(kwargs)}",
    ]
    logging.info("Server starting: %s", " ".join(cmd))
    p = subprocess.Popen(cmd)
    try:
        yield p
    finally:
        # Stop the server no matter how we exit the with-block (success or exception).
        try:
            p.send_signal(subprocess.signal.SIGINT)
            p.wait(timeout=5)
        except Exception:
            pass
        if p.poll() is None:
            p.terminate()
            try:
                p.wait(timeout=3)
            except subprocess.TimeoutExpired:
                p.kill()
        logging.info("Server stopped")


def evaluation(
    agent_cfg: AgentConfig,
    eval_cfgs: list[EvalConfig],
    episodes: int = 100,
    n_processes: int = 1,
):
    per_process_cache.clear()
    logging.info(f"Starting evaluation with {agent_cfg.agent_name} and {agent_cfg.agent_kwargs}")
    try:
        with start_server(
            agent_cfg.agent_name, agent_cfg.agent_kwargs, agent_cfg.port, agent_cfg.host, agent_cfg.python_path
        ):
            res = multi_eval(agent_cfg, eval_cfgs, episodes, n_processes)
    except Exception:
        # Ensures you SEE the client's stack trace and any logged errors.
        logging.exception("Client failed")
        raise

    logging.info(f"Results (success, reward, steps) for all envs: {res[0].mean(axis=1)}")
    logging.info(
        f"Mean reward for all envs: {[np.mean([np.mean(ep_rewards) for ep_rewards in env_rewards]) for env_rewards in res[1]]}"
    )
    # print indices of successful episodes
    for idx, env in enumerate(res[0]):
        logging.info(f"Env {eval_cfgs[idx].env_id} successful episodes: {np.where(env[:, 0])[0]}")
    return res


def run_eval(
    agent_cfg: AgentConfig,
    eval_cfgs: list[EvalConfig],
    wandb_entity: str,
    wandb_project: str,
    wandb_note: str,
    wandb_name: str,
    checkpoint_steps: list[int],
    slurm: Slurm,
    output_path: str,
    wandb_group: str | None = None,
    episodes: int = 100,
    n_processes: int | None = None,
    n_gpus: int = 1,
    python_path: str = "python",
):
    eval_cmd = shlex.quote(
        shlex.join(
            [
                "-m",
                "vlagents",
                "run-eval-post-training",
                f"--agent-cfg={json.dumps(asdict(agent_cfg))}",
                f"--episodes={episodes}",
                f"--n-processes={n_processes}",
                f"--eval-cfgs={json.dumps([asdict(cfg) for cfg in eval_cfgs])}",
                f"--wandb-group={wandb_group.replace(':', '_') if wandb_group else ''}",
                f"--wandb-project={wandb_project}",
                f"--wandb-entity={wandb_entity}",
                f"--wandb-note={wandb_note}",
                f"--wandb-name={wandb_name}",
                f"--n-gpus={n_gpus}",
                f"--steps={json.dumps(checkpoint_steps)}",
                f"--output-path={output_path}",
            ]
        )
    )

    python_path += eval_cmd
    slurm.sbatch(python_path)


def write_results(
    results: np.ndarray,
    rewards: list[list[list[float]]],
    eval_cfgs: list[EvalConfig],
    agent_cfg: AgentConfig,
    out: str = "",
) -> str:
    # first read json, if not exists write empty list
    path = os.path.join(out, f"results_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json")
    if not os.path.exists(path):
        with open(path, "w") as f:
            json.dump([], f)
    with open(path, "r") as f:
        prev_results = json.load(f)
    assert isinstance(prev_results, list)

    flatten_rewards = [[item for sublist in env_rewards for item in sublist] for env_rewards in rewards]
    mean_rewards = [np.mean(env_rewards) for env_rewards in flatten_rewards]

    for idx, cfg in enumerate(eval_cfgs):
        success_mean, reward_mean, steps_mean = results[idx].mean(axis=0, keepdims=False)
        success_max, reward_max, steps_max = results[idx].max(axis=0, keepdims=False)
        success_min, reward_min, steps_min = results[idx].min(axis=0, keepdims=False)
        sucess_std, reward_std, steps_std = results[idx].std(axis=0, keepdims=False)
        success_median, reward_median, steps_median = np.median(results[idx], axis=0, keepdims=False)
        prev_results.append(
            {
                "success": {
                    "mean": success_mean,
                    "max": success_max,
                    "min": success_min,
                    "std": sucess_std,
                    "median": success_median,
                    "values": results[idx, :, 0].tolist(),
                },
                "reward_last_step": {
                    "mean": reward_mean,
                    "max": reward_max,
                    "min": reward_min,
                    "std": reward_std,
                    "median": reward_median,
                    "values": results[idx, :, 1].tolist(),
                },
                "rewards": {
                    "mean": mean_rewards[idx],
                    "values": rewards[idx],
                },
                "steps": {
                    "mean": steps_mean,
                    "max": steps_max,
                    "min": steps_min,
                    "std": steps_std,
                    "median": steps_median,
                    "values": results[idx, :, 2].tolist(),
                },
                "episodes": len(results),
                "timestamp": datetime.datetime.now().isoformat(),
                "env_cfg": asdict(cfg),
                "agent_cfg": asdict(agent_cfg),
            }
        )

    with open(path, "w") as f:
        json.dump(prev_results, f, indent=2)
    return path
