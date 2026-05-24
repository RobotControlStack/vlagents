import copy
import json
import logging
import os
from multiprocessing import Pool
from pathlib import Path
from typing import Annotated

import numpy as np
import rpyc
import typer
import wandb

# to use non-inline backend, necessary for the case
# when started from jupyter notebook
os.environ["MPLBACKEND"] = "Agg"

from vlagents.evaluator_envs import AgentConfig, EvalConfig, evaluation, write_results
from vlagents.policies import AGENTS
from vlagents.server import AgentService

main_app = typer.Typer(help="CLI tool for the vlagents library.")


def wandb_log_git_diff(path: str):
    path = Path(path)
    git_path = path / "git"
    git_path.mkdir(parents=True, exist_ok=True)
    log_git_diff(git_path)
    wandb.log_artifact(git_path, type="directory", name="git")


def log_git_diff(path: str):
    # git id
    git_id = os.path.join(path, "git_id.txt")
    os.system(f'git log --format="%H" -n 1 > {git_id}')

    # submodule git ids
    git_submodules = os.path.join(path, "git_submodules.txt")
    os.system(f"git submodule status > {git_submodules}")

    # get git diff
    git_diff = os.path.join(path, "git_diff.txt")
    os.system(f"git diff --submodule=diff > {git_diff}")


@main_app.command()
def start_server(
    agent_name: Annotated[str, typer.Argument(help="Agent name to run.")],
    kwargs: Annotated[str, typer.Option(help="args to start the agent.")] = "{}",
    port: Annotated[int, typer.Option(help="Port to run the server on.")] = 8080,
    host: Annotated[str, typer.Option(help="Host to run the server on.")] = "localhost",
):
    """Runs eval server."""
    agent = AGENTS[agent_name](**json.loads(kwargs))
    service = AgentService(agent, agent_name)
    with service:
        t = rpyc.ThreadedServer(
            service, port=port, hostname=host, protocol_config={"allow_pickle": True, "allow_public_attrs": True}
        )
        t.start()


def _per_process(
    args: tuple[int, AgentConfig, list[EvalConfig], int, int | None, int],
) -> tuple[np.ndarray, list[list[list[float]]], list[float], int]:
    step, _agent_cfg, eval_cfgs, episodes, n_processes, nth_gpu = args
    logging.info(f"Starting evaluation for step {step}")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(nth_gpu)
    os.environ["CAM_PATH"] = f"{os.environ['RUN_PATH']}/videos/{step}"
    agent_cfg = copy.deepcopy(_agent_cfg)
    # Each worker masks itself to a single CUDA device, so `cuda:0` resolves
    # to the GPU assigned via CUDA_VISIBLE_DEVICES.
    agent_cfg.agent_kwargs.setdefault("device", "cuda:0")
    agent_cfg.agent_kwargs["checkpoint_step"] = step

    per_env_results_last_reward, per_env_results_rewards = evaluation(
        agent_cfg=agent_cfg, eval_cfgs=eval_cfgs, episodes=episodes
    )
    logging.info(f"Finished evaluation for step {step}")
    flatten_rewards = [[item for sublist in env_rewards for item in sublist] for env_rewards in per_env_results_rewards]
    mean_rewards = [np.mean(env_rewards) if env_rewards else 0.0 for env_rewards in flatten_rewards]
    logging.info("Returning results for step %s", step)
    return per_env_results_last_reward, per_env_results_rewards, mean_rewards, step


def _eval_cfg_key(cfg: EvalConfig) -> str:
    return json.dumps({"env_id": cfg.env_id, "env_kwargs": cfg.env_kwargs}, sort_keys=True)


def _merge_env_split_results(
    results: list[tuple[np.ndarray, list[list[list[float]]], list[float], int]],
    worker_eval_cfgs: list[list[EvalConfig]],
    eval_cfgs: list[EvalConfig],
) -> tuple[np.ndarray, list[list[list[float]]], list[float], int]:
    per_cfg_results: dict[str, tuple[np.ndarray, list[list[float]], float]] = {}
    merged_step = results[0][3] if results else None

    for result, cfgs in zip(results, worker_eval_cfgs, strict=True):
        per_env_results_last_reward, per_env_results_rewards, mean_rewards, step = result
        if merged_step is None:
            merged_step = step

        assert len(cfgs) == per_env_results_last_reward.shape[0] == len(per_env_results_rewards) == len(mean_rewards)
        for idx, cfg in enumerate(cfgs):
            per_cfg_results[_eval_cfg_key(cfg)] = (
                per_env_results_last_reward[idx],
                per_env_results_rewards[idx],
                mean_rewards[idx],
            )

    ordered_last_reward = []
    ordered_rewards = []
    ordered_mean_rewards = []
    for cfg in eval_cfgs:
        key = _eval_cfg_key(cfg)
        if key not in per_cfg_results:
            raise KeyError(f"Missing worker result for evaluation config {cfg}")
        last_reward, rewards, mean_reward = per_cfg_results[key]
        ordered_last_reward.append(last_reward)
        ordered_rewards.append(rewards)
        ordered_mean_rewards.append(mean_reward)

    return np.stack(ordered_last_reward, axis=0), ordered_rewards, ordered_mean_rewards, merged_step


@main_app.command()
def run_eval(
    output_path: Annotated[str, typer.Option(help="Path to store the run results.")],
    wandb_project: Annotated[str | None, typer.Option(help="weights and biases logging project.")] = None,
    wandb_entity: Annotated[str | None, typer.Option(help="weights and biases logging entity.")] = None,
    wandb_note: Annotated[str | None, typer.Option(help="weights and biases logging note.")] = None,
    wandb_name: Annotated[str | None, typer.Option(help="weights and biases logging name.")] = None,
    wandb_group: Annotated[str | None, typer.Option(help="weights and biases logging name.")] = None,
    steps: Annotated[str | None, typer.Option(help="steps to evaluate.")] = None,
    episodes: Annotated[int, typer.Option(help="Number of episodes to run.")] = 100,
    n_processes: Annotated[int | None, typer.Option(help="Number of processes to run.")] = None,
    n_gpus: Annotated[int, typer.Option(help="Number of gpus to run.")] = 1,
    eval_cfgs: Annotated[
        str, typer.Option(help="Evaluation configurations.")
    ] = '[{"env": "rcs/SimplePickUpSim-v0", "kwargs": {}}]',
    agent_cfg: Annotated[
        str, typer.Option(help="Agent configuration.")
    ] = '{"host": "localhost", "port": 8080, "agent_name": "Test", "agent_kwargs": {}, "python_path": "python"}',
):
    """
    post training eval which goes over all checkpoints
    - each checkpoint with many envs
    """

    eval_cfgs_ = [EvalConfig(**cfg) for cfg in json.loads(eval_cfgs)]
    agent_cfg_ = AgentConfig(**json.loads(agent_cfg))
    _run_eval(
        output_path=output_path,
        eval_cfgs=eval_cfgs_,
        agent_cfg=agent_cfg_,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_note=wandb_note,
        wandb_name=wandb_name,
        wandb_group=wandb_group,
        steps=steps,
        episodes=episodes,
        n_processes=n_processes,
        n_gpus=n_gpus,
    )


def _run_eval(
    output_path: str,
    eval_cfgs: list[EvalConfig],
    agent_cfg: AgentConfig,
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
    wandb_note: str | None = None,
    wandb_name: str | None = None,
    wandb_group: str | None = None,
    steps: str | None = None,
    episodes: int = 100,
    n_processes: int | None = None,
    n_gpus: int = 1,
):
    if steps is None:
        steps = [None]
    else:
        steps = json.loads(steps)

    # TODO: make this a prober argument that is passed
    os.environ["RUN_PATH"] = output_path

    use_wandb = wandb_project is not None and wandb_entity is not None
    if use_wandb:
        wandb.init(
            entity=wandb_entity,
            resume="allow",
            project=wandb_project,
            # config=dict(agent_name=agent_name, agent_kwargs=json.loads(kwargs), eval_cfgs=json.loads(eval_cfgs)),
            notes=wandb_note,
            job_type="eval",
            name=wandb_name,
            group=wandb_group,
            config={"eval_cfgs": eval_cfgs, "agent_cfg": agent_cfg, "episodes": episodes, "n_processes": n_processes, "n_gpus": n_gpus, "steps": steps}
        )
        wandb_log_git_diff(output_path)
        wandb.run.log_code(".")

        wandb.define_metric(
            "total/success",
            step_metric="train_step",
            overwrite=False,
            step_sync=False,
            hidden=False,
            summary="max",
        )
        wandb.define_metric(
            "total/last_step_reward",
            step_metric="train_step",
            overwrite=False,
            step_sync=False,
            hidden=False,
            summary="max",
        )
        wandb.define_metric(
            "total/total_steps",
            step_metric="train_step",
            overwrite=False,
            step_sync=False,
            hidden=False,
            summary="min",
        )
        wandb.define_metric(
            "total/mean_reward",
            step_metric="train_step",
            overwrite=False,
            step_sync=False,
            hidden=False,
            summary="max",
        )
        for idx, env in enumerate(eval_cfgs):
            wandb.define_metric(
                f"{env.env_id}/success",
                step_metric="train_step",
                overwrite=False,
                step_sync=False,
                hidden=False,
                summary="max",
            )
            wandb.define_metric(
                f"{env.env_id}/last_step_reward",
                step_metric="train_step",
                overwrite=False,
                step_sync=False,
                hidden=False,
                summary="max",
            )
            wandb.define_metric(
                f"{env.env_id}/total_steps",
                step_metric="train_step",
                overwrite=False,
                step_sync=False,
                hidden=False,
                summary="min",
            )
            wandb.define_metric(
                f"{env.env_id}/mean_reward",
                step_metric="train_step",
                overwrite=False,
                step_sync=False,
                hidden=False,
                summary="max",
            )

    if len(steps) > 1:
        # Evaluate different checkpoints in parallel and keep the full env set together.
        worker_eval_cfgs = [eval_cfgs for _ in steps]
        args = []
        for idx, step in enumerate(steps):
            worker_agent_cfg = copy.deepcopy(agent_cfg)
            worker_agent_cfg.port += idx
            args.append((step, worker_agent_cfg, eval_cfgs, episodes, 1, idx % n_gpus))
        max_workers = len(args)
    else:
        # For a single checkpoint, fan out across environments so multiple GPUs
        # can each host one policy server and evaluate a subset of tasks.
        worker_eval_cfgs = [[cfg] for cfg in eval_cfgs]
        args = []
        for idx, cfg in enumerate(eval_cfgs):
            worker_agent_cfg = copy.deepcopy(agent_cfg)
            worker_agent_cfg.port += idx
            args.append((steps[0], worker_agent_cfg, [cfg], episodes, idx % n_gpus))
        max_workers = len(args)

    pool_size = min(max_workers, n_processes or max_workers)
    if pool_size > 1:
        with Pool(pool_size) as p:
            results = p.map(_per_process, args)
    else:
        results = [_per_process(arg) for arg in args]

    logging.info("Finished evaluation")

    if len(steps) > 1:
        merged_results = results
    else:
        merged_results = [_merge_env_split_results(results, worker_eval_cfgs, eval_cfgs)]

    for result in merged_results:
        per_env_results_last_reward, per_env_results_rewards, mean_rewards, step = result
        if use_wandb:
            step = step if step is not None else 0
            wandb_log_dict = {
                "total/success": per_env_results_last_reward.mean(axis=(0, 1))[0],
                "total/last_step_reward": per_env_results_last_reward.mean(axis=(0, 1))[1],
                "total/total_steps": per_env_results_last_reward.mean(axis=(0, 1))[2],
                "total/mean_reward": np.mean(mean_rewards),
                "train_step": step,
            }
            # log for each env
            for idx, env in enumerate(eval_cfgs):
                wandb_log_dict.update(
                    {
                        f"{env.env_id}/success": per_env_results_last_reward[idx].mean(axis=0)[0],
                        f"{env.env_id}/last_step_reward": per_env_results_last_reward[idx].mean(axis=0)[1],
                        f"{env.env_id}/total_steps": per_env_results_last_reward[idx].mean(axis=0)[2],
                        f"{env.env_id}/mean_reward": mean_rewards[idx],
                    }
                )
            wandb.log(wandb_log_dict, step=step, commit=True)

        path = write_results(
            per_env_results_last_reward,
            per_env_results_rewards,
            eval_cfgs,
            agent_cfg=agent_cfg,
            out=output_path,
        )
        if use_wandb:
            wandb.log_artifact(path, type="file", name="results", aliases=[f"step_{step}"])


if __name__ == "__main__":
    main_app()
