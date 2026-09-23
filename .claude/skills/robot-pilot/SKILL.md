---
name: robot-pilot
description: Drive a robot episode yourself (or through a subagent) by answering the vlagents `vlm` agent's mailbox requests, e.g. "pilot the transfer cube task", "run an episode with you as the policy", "answer the mailbox requests".
---

# Robot pilot through the vlagents mailbox

The `vlm` agent with `backend: "mailbox"` does not call a model API. It writes each request to disk and waits
for `reply.json`. That lets a Claude Code session, a subagent or a human act as the policy, using exactly the
same prompt, action primitives, history, logging and evaluation as the API backends.

## 1. Start an episode

Simulation, mass evaluation (one server process is started by the eval):

```shell
MUJOCO_GL=egl python -m vlagents run-eval --output-path runs/pilot --episodes 1 \
  --agent-cfg '{"host": "localhost", "port": 8080, "agent_name": "vlm", "python_path": "python",
                "agent_kwargs": {"backend": "mailbox", "mailbox_dir": "runs/pilot/mailbox", "control_mode": "xyzrpy"}}' \
  --eval-cfgs '[{"env_id": "duobench/transfer_cube", "env_kwargs": {"control_mode": "xyzrpy", "camera_resolution": [448, 448]},
                 "max_steps_per_episode": 900, "image_size": [448, 448], "jpeg_encoding": true}]'
```

Real robot: `python -m vlagents start-server vlm --port 20000 --host 0.0.0.0 --kwargs '{"backend": "mailbox", "mailbox_dir": "runs/pilot/mailbox", "control_mode": "xyzrpy"}'`
and run `examples/inference/franka.py` from RCS with `vlagents_model: "vlm"` and `n_action_steps: 30`.

Run it in the background; it blocks until every request is answered.

## 2. Answer requests

Layout of `mailbox_dir`:

```
<episode timestamp>/
  system.md            system prompt (robot, frame, workspace, cameras, output schema, optional demonstrations)
  conversation.md      whole dialogue so far, appended per step
  step_000/request.md  the current observation text; images referenced as image_1.jpg (head), image_2.jpg, ...
  step_000/reply.json  <- you write this
```

Loop until the episode ends (no new `step_XXX/request.md` appears for a while, or the eval process exits):

1. Wait for a new `request.md` (e.g. `until ls <mailbox>/*/step_*/request.md ...` in a background Bash, or a
   Monitor on `inotifywait`). Only steps without `reply.json` are open.
2. Read `system.md` once per episode, then the request and every image next to it (the Read tool renders
   JPEGs). The head image has +y (left arm) on the image left and +x (far side of the table) on top.
3. Decide the next second for both arms and write **one JSON object** to `reply.json`, exactly in the schema
   from `system.md`, e.g.
   `{"reasoning": "cube is right of the right gripper", "left": {"type": "hold"}, "right": {"type": "move_delta", "dxyz": [0.0, -0.08, -0.15], "gripper": 1}, "done": false}`
   Write the file atomically (write to a temp name, then `mv`), the server reads it as soon as it exists.
4. The next request shows the result (new pose, stage progress, collision or IK flags). Small moves and
   checking the wrist cameras before closing the gripper work better than large jumps.

When the agent runs with in-context demonstrations (`icl_path`), `conversation.md` starts with them: pairs of a
demonstration observation (state text plus `context_N.jpg` keyframes in the episode folder) and the command that
reproduced the next second of the recording. Read them once before answering step 0.

## 3. Hindsight corrections (residual training)

When the agent runs with `hindsight_corrections: true` (and the eval with an `editor`), a residual controller
edits your targets at 30 Hz and the request tells you how much it intervened plus, for tasks with a task state,
the trace of the object (e.g. the ball in the board frame) over the last command. From step 1 on add
`"correction"` to `reply.json`: per arm the adjustment that, knowing the outcome, should have been added to the
targets of your **previous** command, e.g.
`"correction": {"left": {"dxyz": [0, 0, 0.01], "drpy_deg": [0, -3, 0], "from": 0.5, "to": 1.0}, "right": null}`
(`from`/`to` = fraction of the command during which it should have applied; `null` = that arm was right; at most
3 cm and 8 deg). Corrections are labels for the residual, not commands: still write the next command for the
current state. Be honest and specific: "the ball rolled 4 cm past the corridor, the board should have levelled
earlier" becomes a correction on the tilting arm with `from` at the moment the ball reached the corner.

A persistent subagent (Agent tool, then SendMessage per step) keeps its own memory of the episode; a fresh
agent per step should read `conversation.md` first. Results land in the eval `results_*.json`, videos under
`videos/`, and the agent's own log under the `log_dir` kwarg if set.
