# VLM Robot-Control Harness: Plan

Goal: let a general-purpose VLM (first target: GPT via the OpenAI API) control the Franka FR3 Duo and solve
DuoBench tasks, using the existing RCS / vlagents / DuoBench stack, with the same code path for simulation
(mass evaluation through the vlagents CLI) and the real robot (RCS inference script).

## 1. How the pieces fit today

```
                 rpyc (json_numpy, jpeg or shm images)
 EvalEnv / franka.py  ── RemoteAgent.act(Obs) ──►  AgentService ── Agent.act(Obs) -> Act
 (RCS gym env)        ◄──── Act (chunk of N steps) ──┘   (policy process, own conda env)
```

* `vlagents.policies.interface.Agent` is stateless per call: `act(obs) -> Act`. `Obs` carries per-robot
  `SingleObs` (cameras, joints, gripper, optional `xyzrpy`/`tquat`, free-form `info`) and the
  `language_instruction`. `Act.acts` is a list of per-step `{robot: SingleAct(action, gripper, done)}`.
* `EvalEnv.chunk_step` executes a chunk one env step at a time (capped by `execution_horizon`).
  `vlagents run-eval` fans out envs/checkpoints over processes and logs to wandb / results json.
* `examples/inference/franka.py` in RCS is the single-rollout / real-robot loop: builds the RCS env
  (sim `EmptyWorldFR3Duo` or hardware `FrankaDuoEnv`), translates RCS obs -> `Obs`, buffers
  `n_action_steps` of each chunk, maps `SingleAct.action` -> RCS action dict.
* DuoBench registers `duobench/<task>` gym envs (`TaskStageWrapper` adds `stage`, `max_stage`,
  `current_subinstruction`, `success` to `info`; reward = normalized stage). The evaluation config in the
  DuoBench README is: headless, absolute joint control (`RelativeTo.NONE`), async 30 Hz, binary gripper.

Measured facts used for the prompts (DuoBench Vention scene, shared base frame, metres):

| item | value |
| --- | --- |
| frame | origin between the two robot bases at mount height; x forward (towards the table), y left, z up |
| table top | z = -0.33, x in [0.10, 0.89], y in [-0.70, 0.70] |
| home TCP | left (0.51, +0.21, 0.20), right (0.51, -0.21, 0.20); rpy = (180, -20, 0) deg = gripper pointing down, tilted 20 deg |
| joint limits (FR3, rad) | low [-2.31 -1.51 -2.49 -2.75 -2.48 0.85 -2.69], high [2.31 1.51 2.49 -0.45 2.48 4.21 2.69] |
| gripper | Robotiq 2F-85, binary, 0 = closed, 1 = open |
| control rate | 30 Hz; a 5 cm Cartesian target converges in ~10 steps; sustained speed ~0.2 m/s |
| head camera | image left = +y (left arm side), image top = +x (far side of table) |

RPY in RCS equals scipy `Rotation.as_euler("xyz")` (extrinsic roll-pitch-yaw). RCS returns whichever Euler
representation Eigen picks (e.g. `(0, -160, 180)` vs `(180, -20, 0)` for the same pose), so the harness
canonicalises orientation from the quaternion (`tquat`) with pitch in [-90, 90] deg before showing it to the VLM.

## 2. Library changes (all minimal)

### vlagents
1. **Stateful episodes** (re-introduce the old `reset`): `Agent.reset(obs, instruction, **kwargs) -> info`
   with a default implementation, exposed by `AgentService.reset` and `RemoteAgent.reset`. `eval.single_eval`
   calls it right after `env.reset`. Existing policies keep working unchanged.
2. **DuoBench env**: fix the import (`duobench.tasks`, the package was renamed), build the README evaluation
   config inside `make_gym` from plain kwargs (`control_mode` = `joints|xyzrpy|tquat`, `camera_resolution`,
   `max_relative_movement`, `frequency`), forward `xyzrpy`/`tquat` and the stage info (`stage`,
   `current_subinstruction`, `stage_to_subinstructions`, `success`, `collision`, `ik_success`) in
   `SingleObs.info` so any agent can see reached subgoals. The action key equals the control mode string.
3. **`vlm` agent** (`policies/vlm.py`): OpenAI-compatible chat-completions client (works with GPT and with any
   OpenAI-compatible server such as vLLM for open VLMs). One request per chunk with the full episode history.
4. **ICL exporter** (`policies/vlm_icl.py`): reads a LeRobot v3 dataset (parquet + av1 mp4), takes the first
   `n <= 10` episodes, subsamples every 30 frames and writes (observation -> command) pairs in exactly the
   agent's output schema, in joint space and, via RCS FK (`rcs.common.Pin`), in TCP space.

### RCS
1. `examples/inference/franka.py`: call `remote_agent.reset(obs, instruction)` when an episode starts;
   choose the RCS action key from `CONTROL_MODE` (joints / xyzrpy / tquat) so Cartesian policies work;
   forward per-robot `info`; allow a DuoBench task scene in simulation.
2. `MultiRobotWrapper._translate_pose` no longer mutates the caller's action dict (re-using an action dict
   across steps in Cartesian mode compounded the frame transform every step).

### DuoBench
No code change needed. The `vlagents` env builds the documented evaluation config.

## 3. The VLM agent

Interface: `python -m vlagents start-server vlm --kwargs '{...}'`; agent kwargs

| kwarg | meaning |
| --- | --- |
| `model`, `base_url`, `api_key_env` | OpenAI-compatible endpoint (default `gpt-5`, key from `OPENAI_API_KEY`) |
| `control_mode` | `joints` or `xyzrpy` (must match the env) |
| `chunk_size` / `fps` | 30 actions = 1 s |
| `history` | `full` (default), or an int: keep images only for the last N turns, older turns text-only |
| `icl_path`, `icl_episodes`, `icl_images` | in-context examples exported by `vlm_icl.py` |
| `reference_images` | extra explanatory pictures (frame convention, workspace) shown in the system prompt |
| `log_dir` | per-episode dump of prompts, images, raw replies, parsed commands, token usage |
| `backend` | `openai` or `fake` (deterministic replies for tests and plumbing checks) |

Per call the VLM receives: system prompt (robot, frames, workspace, limits, camera description, output schema,
optional ICL examples), then the whole history of (state text + 3 images, reply) pairs, then the current
state (per arm: TCP xyz + canonical rpy in degrees or joints in degrees, gripper; task instruction; stage
k/N and current sub-instruction; collision / IK flags from the last chunk) with the 3 current images.

Output: one JSON object with a short `reasoning`, a command per arm and `done`. Commands are motion
primitives that expand deterministically into a chunk of exactly 30 actions (no skill library):

* Cartesian: `move` (absolute xyz [+ rpy_deg] + gripper), `move_delta` (dxyz [+ drpy_deg] + gripper),
  `gripper` (hold pose, set gripper), `hold`, or a raw `chunk` (30 x [x,y,z,r,p,y]).
* Joints: `move_joints` (7 joints in deg + gripper), `move_joints_delta`, `gripper`, `hold`, `chunk`.

Expansion = linear interpolation (translation lerp, rotation slerp, joint lerp) from the current state to the
target over 30 steps, target clipped to a per-chunk budget (0.30 m / 90 deg / 45 deg per joint) and to the
joint limits; the RCS `max_relative_movement` clamp is the second safety layer on the env side.

## 4. Experiments on `duobench/transfer_cube`

All experiments get full history, the instruction, the reached sub-goals and the three cameras.

| id | control | prompt extras | env kwargs |
| --- | --- | --- | --- |
| E1 (priority) | Cartesian TCP (`xyzrpy`) | frame convention, workspace table, arm/gripper facts, reference pictures | `control_mode=xyzrpy` |
| E2 | joint space | joint limits, home pose, per-joint role description | `control_mode=joints` |
| E3 | Cartesian + ICL | E1 + up to 10 demonstration episodes (state -> command pairs, head images at keyframes) | `icl_path=...` |
| E4 | joints + ICL | E2 + same demonstrations in joint space | `icl_path=...` |

Ablations worth running once E1 works: history length (`history=full` vs last 3 turns), image size
(224 vs 448), primitives vs raw chunks, with/without stage feedback, `reasoning_effort`.

Metrics come from `vlagents run-eval`: success rate, mean normalised stage (reward), steps; plus tokens and
wall-clock per episode from the agent log.

Example (sim, one machine):

```shell
export OPENAI_API_KEY=...
python -m vlagents run-eval --output-path runs/e1 --episodes 10 \
  --agent-cfg '{"host": "localhost", "port": 8080, "agent_name": "vlm", "python_path": "python",
                "agent_kwargs": {"model": "gpt-5", "control_mode": "xyzrpy", "log_dir": "runs/e1/vlm"}}' \
  --eval-cfgs '[{"env_id": "duobench/transfer_cube", "env_kwargs": {"control_mode": "xyzrpy"},
                 "max_steps_per_episode": 900, "image_size": [448, 448], "jpeg_encoding": true}]'
```

Real robot: same server, `examples/inference/franka.py` with `CONTROL_MODE = ControlMode.CARTESIAN_TRPY`,
`n_action_steps = 30`, `ROBOT_INSTANCE = RobotPlatform.HARDWARE`.

## 5. Towards a general VLM harness

* **Model-agnostic**: the agent only needs an OpenAI-compatible chat endpoint; open VLMs run behind vLLM.
  Provider quirks (reasoning effort, JSON mode, image detail) stay in one small backend class.
* **Robot-agnostic prompt from the env**: the facts table above should eventually come from the env
  (`SingleObs.info` on reset: joint limits, home pose, workspace box, frame description, camera list) so the
  same agent works for a single FR3, UR5e, xArm or the duo without prompt edits.
* **Action interface as a plug-in**: `ActionSpace` objects own the output schema, the expansion into a chunk,
  the safety clipping and the textual description; joints and Cartesian are the first two, delta/velocity or
  per-step raw chunks are further variants; DuoBench-specific knowledge never enters the agent.
* **Memory strategies**: full history (baseline), sliding window with text-only older turns, and a running
  scratchpad the model updates each turn (the `reset` API makes any of these possible server-side).
* **Grounding / ICL**: the exporter turns any LeRobot dataset into (obs -> command) examples in the agent's
  own schema; retrieval by task or by visual similarity is a natural extension.
* **Diagnostics**: a privileged-info mode (object poses from the sim in the prompt) separates perception from
  control failures; the per-episode log allows replaying a prompt with another model.
* **Real-world safety**: primitive budget + RCS `LimitedAbsoluteAction` clamp + `done`; the hardware path
  is identical, only the env creator changes.

## 6. Status

Verified in this environment (CPU only, MuJoCo rendered with Mesa EGL, no OpenAI key available):

* `vlagents run-eval` with the `vlm` agent (`backend: fake`) on `duobench/transfer_cube` in `xyzrpy` and in
  `joints` mode: server start, `reset`, three 30-step chunks, `done`, camera videos and results json.
* RCS `examples/inference/franka.py` in simulation with the DuoBench transfer-cube scene against the same
  fake server: episode start calls `reset`, chunks are executed, `done` resets the environment.
* Unit tests: `src/tests/test_vlm_agent.py` (primitives, budgets, parsing, history, fallback) and the
  `reset` round trip in `src/tests/test_connection.py`.
* ICL export of the first 10 `transfer_cube/sim` episodes in both spaces (149 keyframes, 3 MB with 224 px
  head images).

Not yet run: any episode with a real GPT model (needs `OPENAI_API_KEY`). Software rendering costs ~0.4 s per
camera image (shadow map of 4096 px), so evaluations on CPU nodes are slow; on a GPU this disappears. For CPU
debugging, lowering `spec.visual.quality.shadowsize` on the composed model before `Sim(...)` gives a 5x speedup.
