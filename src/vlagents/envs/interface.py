import logging
from abc import ABC
from dataclasses import dataclass
from typing import Any

import gymnasium as gym

from vlagents import ENVS
from vlagents.policies.interface import Act, Obs, SingleAct

logging.basicConfig(
    format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


class EvalEnv(ABC):

    def __init__(self, env_id: str, execution_horizon: int | None = None, **env_kwargs) -> None:
        self.env_id = env_id
        self.env_kwargs = env_kwargs
        self.execution_horizon = execution_horizon
        self.do_import()
        self.env = self.make_gym()
        self.last_chunk_steps = 0
        self.render_next_step = True
        self.last_action: dict[str, SingleAct] | None = None

    def chunk_step(self, actions: Act, max_steps: int | None = None) -> tuple[Obs, float, bool, bool, dict[str, Any]]:
        if not actions.acts:
            raise ValueError("Agents must return at least one action")
        rewards = []
        self.last_chunk_steps = 0
        n_steps = len(actions.acts)
        if max_steps is not None:
            n_steps = min(n_steps, max_steps)
        if self.execution_horizon is not None:
            n_steps = min(n_steps, self.execution_horizon)
        for action in actions.acts[:n_steps]:
            # only the observation the agent sees next needs camera images
            self.render_next_step = self.last_chunk_steps == n_steps - 1
            obs, reward, done, truncated, info = self.step(action)
            self.last_action = action
            rewards.append(reward)
            self.last_chunk_steps += 1
            if (
                done
                or truncated
                or (self.execution_horizon is not None and self.last_chunk_steps >= self.execution_horizon)
            ):
                break
        if not rewards:
            raise ValueError("max_steps must allow at least one environment step")
        return obs, sum(rewards), done, truncated, info

    def hold(self, n_steps: int) -> tuple[bool, bool]:
        """Keep executing the last action for n_steps without rendering, e.g. to simulate the time the policy
        needed for inference. Returns the (done, truncated) flags of the last step; last_chunk_steps counts them."""
        done = truncated = False
        self.last_chunk_steps = 0
        if self.last_action is None:
            return done, truncated
        for _ in range(n_steps):
            self.render_next_step = False
            _, _, done, truncated, _ = self.step(self.last_action)
            self.last_chunk_steps += 1
            if done or truncated:
                break
        return done, truncated

    def step(self, action: dict[str, SingleAct]) -> tuple[Obs, float, bool, bool, dict]:
        raise NotImplementedError

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Obs, dict[str, Any]]:
        raise NotImplementedError

    @property
    def language_instruction(self) -> str:
        raise NotImplementedError

    def make_gym(self) -> gym.Env:
        return gym.make(self.env_id, **self.env_kwargs)

    def do_import(self):
        raise NotImplementedError

    @staticmethod
    def from_id(env_id: str, execution_horizon: int | None = None, **env_kwargs) -> "EvalEnv":
        if env_id not in ENVS:
            raise ValueError(f"Unknown environment id {env_id}. Available environments: {list(ENVS.keys())}")
        return ENVS[env_id](env_id, execution_horizon=execution_horizon, **env_kwargs)


@dataclass
class EvalConfig:
    env_id: str
    env_kwargs: dict[str, Any]
    execution_horizon: int | None = None
    max_steps_per_episode: int = 100
    seed: int = 42
    same_machine: bool = False
    jpeg_encoding: bool = False
    image_size: tuple[int, int] | None = (224, 224)
    request_timeout: float | None = 300
    """seconds the client waits for the policy before reconnecting; None for slow policies"""
    simulate_inference_delay: bool = False
    """keep executing the previous action for as long as the agent needed to reply (at control_frequency Hz)"""
    control_frequency: float = 30.0


@dataclass
class AgentConfig:
    host: str
    agent_name: str
    agent_kwargs: dict[str, Any]
    python_path: str = "python"
    """modify this if you want to use a specific python environment """
    port: int = 8080
