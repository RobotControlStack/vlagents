# VLM Robot-Control Harness: Plan

Goal: let a general-purpose VLM (any API model, or a Claude Code agent acting as the pilot) control the Franka FR3 Duo and solve
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
3. **`vlm` agent** (`policies/vlm.py`): one request per chunk with the full episode history, served by a
   pluggable backend: OpenAI-compatible chat completions (GPT, vLLM), the Anthropic Messages API, or a file
   mailbox so that a Claude Code subagent (or a human) is the policy without any API code.
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
| `backend` | `openai` (any OpenAI-compatible endpoint), `anthropic` (Claude via the Messages API), `mailbox` (an external pilot answers through files: a Claude Code subagent following `.claude/skills/robot-pilot`, or a human), `fake` (tests) |
| `model`, `base_url`, `api_key_env` | model id and endpoint of the API backends (defaults `gpt-5` / `claude-opus-5`) |
| `mailbox_dir` | folder of the mailbox backend (`<episode>/system.md`, `step_XXX/request.md` + images, `reply.json`) |
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

## 6. Residual edit policy (EXPO-style, `policies/residual.py`)

The VLM is a good 1 Hz planner and a poor servo. Following EXPO (an expressive base policy plus a small edit
policy that adjusts its actions), a residual network edits the VLM's chunk at 30 Hz inside the environment loop:

* **Interface.** `EvalEnv.editor` receives every nominal action before execution (`edit(nominal, obs, i, n,
  goal)`), also during the simulated inference hold (`holding=True`), so the residual keeps reacting while the
  VLM thinks. Edits are bounded per axis (2 cm, 5 deg by default) and applied in the base frame
  (`target + dxyz`, `exp(drot) * R_target`). Every step is recorded (features, nominal, edit, task state).
* **Inputs.** Per arm: chunk goal relative to the nominal target (translation + rotation vector), gripper and
  nominal gripper. Global: chunk phase, holding flag, task-state validity, task state. The task state comes from
  the environment (`env_kwargs: {"task_state": true}`); for the ball maze it is the ball position and velocity
  and the goal in the board frame plus the board pose (roll, pitch, yaw, centre). Neither the current nor the
  absolute nominal tool pose is an input: with the current pose, behaviour cloning copies the tracking error and
  cancels motion; with the absolute pose it memorises demonstration windows (validated on held-out episodes).
* **Warm start.** `residual_train warmstart`: for every window of 30 frames of a demonstration the straight-line
  chunk from the start pose to the reached pose is the nominal; the label is the demonstrated deviation from it.
  On the 50 ball-maze demonstrations this explains only ~7 % of the held-out variance (the deviation of a
  human path inside one second is mostly unpredictable from the chunk geometry), so the warm start is a mild
  time-profile prior (~1 cm lead early in the chunk, zero at the end). Task-state inputs start inert (zero
  first-layer weights) because the demonstrations carry no task state; the DuoBench replay recording (one
  episode with simulator states) can be added but a single episode only gets memorised.
* **Hindsight corrections.** With `hindsight_corrections: true` the VLM adds `"correction"` to every reply: per
  arm the adjustment that should have been added to the previous command's targets over a fraction of the
  chunk. The prompt reports the residual's intervention and the ball trace of the last command so the VLM can
  attribute the outcome. `residual_train finetune` relabels each recorded step with `edit + correction`
  (dataset aggregation, DAgger-style with hindsight labels) and continues training from the warm start together
  with the prior data (lower weight).
* **Not done here.** The RL part of EXPO (a Q-function ranking several edited candidates, trained on the stage
  reward) needs many more rollouts than a pilot can produce; the harness has the pieces for it (records,
  nominal chunks, stage reward per step).

## 7. Status

Verified in this environment (CPU only, MuJoCo rendered with Mesa EGL, no OpenAI key available):

* `vlagents run-eval` with the `vlm` agent (`backend: fake`) on `duobench/transfer_cube` in `xyzrpy` and in
  `joints` mode: server start, `reset`, three 30-step chunks, `done`, camera videos and results json.
* RCS `examples/inference/franka.py` in simulation with the DuoBench transfer-cube scene against the same
  fake server: episode start calls `reset`, chunks are executed, `done` resets the environment.
* Unit tests: `src/tests/test_vlm_agent.py` (primitives, budgets, parsing, history, fallback) and the
  `reset` round trip in `src/tests/test_connection.py`.
* ICL export of the first 10 `transfer_cube/sim` episodes in both spaces (149 keyframes, 3 MB with 224 px
  head images).

* **First real pilot run** (`backend: mailbox`, a Claude Code subagent answering the requests, Cartesian
  mode, no in-context examples): `duobench/transfer_cube` solved in one episode, all 4 stages, 22 commands
  (662 env steps). The run exposed and fixed one control bug (primitives below 3 cm were dropped by the RCS
  1 mm command threshold). Observed pilot behaviour: approach from above, verify alignment in the wrist
  camera before descending, separate gripper command, rolled the right gripper sideways for the hand-over,
  used the reported pose deltas to correct by centimetres. Cost: roughly one to two minutes of wall clock per
  command with software rendering.

* **Second pilot run, harder task** (`duobench/hinge_chest`, same setup, object and lid geometry given in
  the pilot prompt): solved in 25 commands (722 env steps). The right arm pried the thin lid open with its
  closed fingertips and held it on the arc about the hinge while the left arm grasped the cube, lost it once
  during the carry, re-grasped it and dropped it into the open chest. Both arms were commanded in parallel
  in most steps.

* **Runs 3 and 4 (hinge_chest with simulated inference delay).** The eval now renders only the last step of a
  chunk (RCS `render` action flag) and keeps the robot on its previous command for as long as the policy took
  to reply (`simulate_inference_delay`). Run 3 (no demonstrations) solved the task in 26 commands, the same as
  without delay: the scene is static once the lid is held. Run 4 with three human demonstrations in context
  (head keyframes, one pair per second) solved it in 10 commands: the pilot copied the demonstrated lid sweep
  and placement pose and only read object positions from the images. Environment time per command fell from
  43 s to 11 s (now the simulated hold, physics only).

Lessons from the pilot runs, to fold into the harness next:

* The task text says "cube" but the object is a 3.2 x 3.2 x 9.6 cm upright box; four commands were lost on
  its top edge. Object dimensions (or a task hint via `extra_instructions`) belong in the prompt, ideally
  supplied by the env.
* Every move falls 0.5 to 1.8 cm and about 3 deg short within one second (controller lag); the pilot
  compensated from the reported pose. A short note in the prompt or a settle step would remove the surprise.
* The request should report the gripper width (`gripper_width` is already in the info) and optionally the
  joints, which would have revealed the stalled descent immediately.
* Finger contacts with objects do not raise the RCS collision flag, so a blocked motion is only visible from
  the unchanged pose (the agent now flags "the commanded target was not reached" itself; IK reports failures
  outside the joint limits).
* Demonstration commands must be reached states, not raw teleoperation targets (fixed in the exporter after
  run 4). A "closed" gripper is not a grasp: the state text now reports the gripper opening.
* The wrist camera mount (offset and tilt relative to the tool) should be described in the prompt; both chest
  pilots lost commands to a wrong mental model of it.

Not yet run: any episode through the API backends (needs `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`). Software rendering costs ~0.4 s per
camera image (shadow map of 4096 px), so evaluations on CPU nodes are slow; on a GPU this disappears. For CPU
debugging, lowering `spec.visual.quality.shadowsize` on the composed model before `Sim(...)` gives a 5x speedup.
