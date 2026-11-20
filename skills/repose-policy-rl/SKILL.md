---
name: repose-policy-rl
description: >-
  Running trained ONNX reinforcement-learning, imitation, diffusion, or
  vision-language-action policies on a robot arm with `repose-policy-rl`:
  observation tasks (`CuJointReachObservationTask`), single-step (`CuPolicyTask<IN, OBS, ACT>`)
  and `H`-step chunked (`CuChunkPolicyTask<OBS, H, ACT>`) ONNX inference on tract-onnx,
  zero-allocation action chunking and exponential temporal ensembling (`CuActionChunkTask<H, ACT>`),
  and action tasks (`CuJointActionTask`, `CuCartesianActionTask`, `CuGripperActionTask`).
  Use when deploying a policy exported from MuJoCo Playground, Isaac Lab, ACT,
  Diffusion Policy, or VLAs; sizing IN/OBS/H/ACT; configuring temporal ensembling,
  receding-horizon replacement, `step_decimation`, or `execution_horizon`; or pinning
  `model_hash` for replay.
---

# Repose: RL and VLA policies

`repose-policy-rl` is a separate, **`std`-only** dependency (not re-exported by
`repose`); it ships in releases after 0.8.4, so pin a tag that includes it
(check the CHANGELOG). Single-step and chunked pipelines share the same
observation and action tasks:

```text
Single-step:
joint state ─► observation task ─► CuPolicyTask ─► action task(s) ─► CuControllerTask ─► bridge
  (+ goal)     CuPolicyObservation<IN>  CuPolicyAction<ACT>   CuTrajectoryFollowOutput<N>

Chunked (ACT / Diffusion / VLA):
observation / VLA bridge ─► CuChunkPolicyTask (optional) ─► CuActionChunkTask<H, ACT> ─► action task(s)
                            CuPolicyActionChunk<H, ACT>     CuPolicyAction<ACT>
```

The model never commands joints directly: action tasks scale, limit, and
(for Cartesian actions) solve differential IK, and the standard controller
still applies its watchdog, tracking-error fault, and rate limit.

## Sizes

| Generic | Meaning |
|---|---|
| `N` | Robot joints |
| `IN` | Features from the observation task |
| `OBS` | Model input width: `IN + ACT` on `CuPolicyTask` with `"previous_action": true`, else `IN` |
| `H` | Chunk prediction horizon (`[1, H, ACT]`) |
| `ACT` | Action width per step |

`CuPolicyTask` requires one `f32` input `[1, OBS]` and one `f32` output
`[1, ACT]`; `CuChunkPolicyTask` requires `[1, OBS] -> [1, H, ACT]` (concrete or
dynamic batch dimension). Anything else is a startup error.

## Graph

```ron
(
    id: "observation",
    type: "repose_policy_rl::CuJointReachObservationTask<6, 18>",   // N, IN
    config: {"observation": [
        {"joint_positions": {"scale": 1.0}},      // width N (minus default_positions)
        {"joint_velocities": {"scale": 0.1}},     // width N
        {"goal_error": {"scale": 1.0}},           // width N
    ]},
),
(
    id: "policy",
    type: "repose_policy_rl::CuPolicyTask<18, 24, 6>",              // IN, OBS, ACT
    config: {
        "model": "policy.onnx",          // relative to the process working directory
        "model_hash": 1234567890,        // optional FNV-1a 64 of the file; mismatch fails startup
        "decimation": 1,                 // infer every k-th cycle, hold between
        "previous_action": true,         // append last raw action → OBS = IN + ACT
        "max_observation_abs": 10.0,     // optional positive finite bound on IN features
        "observation_bound_policy": "clamp", // clamp (default) | fault
        "max_action_abs": 1.0,           // optional positive finite bound on raw model outputs
        "action_bound_policy": "clamp",  // clamp (default) | fault
    },
),
(
    id: "action",
    type: "repose_policy_rl::CuJointActionTask<6, 6>",              // N, ACT
    config: {
        "chain": "igus_rebel",
        "kind": "position_delta",        // or position_absolute (uses default_positions)
        "scale": 0.05,                   // one value or one per joint (SI per action unit)
        "max_step": 0.03,                // optional per-cycle target change limit
        "period": 0.02,                  // optional; 1 / rate_target_hz, enables setpoint velocity
    },
),
```

Inputs/outputs:

- Observation: `CuJointStatePayload<N>` + goal `CuJointPositionsPayload<N>` →
  `CuPolicyObservation<IN>`. Other tasks write their own observation task that
  emits `CuPolicyObservation<IN>`; publish no payload for "no input" and `NaN`
  features for a fault.
- Single-step policy: `CuPolicyObservation<IN>` → `CuPolicyObservation<OBS>`
  (exact model input, for the log) + `CuPolicyAction<ACT>` (raw output).
  Optional `max_observation_abs` and `max_action_abs` guardrails enforce
  symmetric magnitude bounds via `CuBoundPolicy` (`"clamp"` or `"fault"`,
  configurable via `observation_bound_policy` and `action_bound_policy`).
  Non-finite features or outputs always `Fault` regardless of bound policy;
  in `clamp` mode on `CuPolicyTask`, the logged `CuPolicyObservation<OBS>`
  records the clamped features fed to the model, while `fault` mode logs the
  raw OOD features and skips inference.
- Chunked policy: `CuPolicyObservation<OBS>` → `CuPolicyActionChunk<H, ACT>`
  (or publish `CuPolicyActionChunk<H, ACT>` directly from a VLA bridge).
  Supports the same `max_observation_abs`, `observation_bound_policy`,
  `max_action_abs`, and `action_bound_policy` (`CuBoundPolicy`) fields.
- Action chunker: `CuPolicyActionChunk<H, ACT>` → `CuPolicyAction<ACT>`.
- Joint/Cartesian action: `CuPolicyAction<ACT>` + `CuJointStatePayload<N>` →
  `CuTrajectoryFollowOutput<N>` (`CuJointActionTask::last_report()` exposes a
  `CuJointActionReport` with `step_limited_mask`, `joint_limited_mask`,
  `step_limited_count`, `joint_limited_count`, and `max_requested_delta_si`;
  Cartesian also emits `CuCartesianActionReport`).
  Connect it to `CuControllerTask<N>` in place of the trajectory follower.
- Gripper action: `CuPolicyAction<ACT>` → `CuGripperCommandPayload` for
  `repose_io::CuGripperTask`.

Policy status (`CuPolicyStatus`): `Fresh` (inferred or stepped this cycle),
`Held` (between decimated inferences or chunk steps), `Idle` (no input or
starved horizon), `Fault`.

## Action chunking and temporal ensembling

ACT, Diffusion Policy, π0, and OpenVLA-OFT emit `H`-step chunks `[1, H, ACT]` at
5–50 Hz while the control loop runs at 50–500 Hz:

```ron
(
    id: "chunk_policy",
    type: "repose_policy_rl::CuChunkPolicyTask<18, 50, 7>",         // OBS, H, ACT
    config: {
        "model": "act_policy.onnx",
        "decimation": 1,                 // 1 = infer every cycle (ACT), >1 = sparse queries
        "max_observation_abs": 10.0,     // optional positive finite bound on OBS features
        "observation_bound_policy": "clamp", // clamp (default) | fault
        "max_action_abs": 1.0,           // optional positive finite bound on raw chunk outputs
        "action_bound_policy": "clamp",  // clamp (default) | fault
    },
),
(
    id: "chunker",
    type: "repose_policy_rl::CuActionChunkTask<50, 7>",             // H, ACT
    config: {
        "mode": {"temporal_ensemble": {"decay": 0.01}},             // or "replace"
        "step_decimation": 2,            // cycles per chunk step (1 = Fresh every cycle)
        "execution_horizon": 25,         // optional <= H; starve to Idle past T_a steps
    },
),
```

- `mode`:
  - `{"temporal_ensemble": {"decay": 0.01}}` (default): ACT exponential
    temporal ensembling `w_i = exp(-decay * i)` normalized across overlapping
    chunks covering the current step (`i = 0` oldest, `decay > 0` smoother,
    `decay = 0` uniform, `decay < 0` favors newer chunks).
  - `"replace"`: receding-horizon replacement for multimodal policies
    (Diffusion Policy, flow matching) where averaging two distinct modes is
    undesirable.
- `step_decimation`: cycle `0` of each chunk step emits `Fresh` and the
  remaining `step_decimation - 1` cycles emit `Held`, so downstream delta
  tasks (`CuCartesianActionTask`, `CuJointActionTask` in `position_delta`)
  apply each step's delta once over `step_decimation / rate_target_hz`.
- Upstream `None` or `Held` keeps stepping the buffered horizon until
  `execution_horizon` (default `H`) steps run out, then transitions cleanly to
  `Idle`. Upstream `Idle` or `Fault` flushes the buffered tail immediately.

## VLA end-effector actions

A typical VLA action `[dx, dy, dz, droll, dpitch, dyaw, gripper]` (`ACT = 7`):

```ron
(
    id: "arm",
    type: "repose_policy_rl::CuCartesianActionTask<6, 7>",
    config: {
        "chain": "igus_rebel",
        "period": 0.02,              // required: delta is executed over one step period
        "offset": 0,                 // reads elements [0, 6)
        "frame": "base",             // base | body
        "rotation": "rpy",           // rotation_vector (default) | rpy
        "translation_scale": 0.01,   // m per action unit
        "rotation_scale": 0.05,      // rad per action unit
        "max_translation": 0.005,    // m per cycle, optional
        "max_rotation": 0.05,        // rad per cycle, optional
        "diff_ik": {"base_damping": 0.02, "limit_margin": 0.15},
    },
),
(
    id: "gripper_action",
    type: "repose_policy_rl::CuGripperActionTask<7>",
    config: {"offset": 6, "threshold": 0.5, "close_when": "below", "hysteresis": 0.1},
),
```

Action tasks read **disjoint** slices of one action via `offset`
(`offset + width <= ACT` is checked at startup).

## Getting it right

- **Training period:** single-step policy period = `decimation / rate_target_hz`
  (chunk step period = `step_decimation / rate_target_hz`); it must equal the
  step period the policy was trained at. Set `period` on action tasks to match
  one step period (`step_decimation / rate_target_hz` for chunked deltas,
  `1 / rate_target_hz` when `step_decimation == 1`).
- **Observation layout and scales** must reproduce the trainer's exactly
  (order of terms, scales, default positions). A silent mismatch produces a
  policy that runs and misbehaves; check the `CuPolicyObservation<OBS>` in the
  log against the trainer.
- **Start conservative:** small `scale`, `max_step` / `max_translation` set,
  `execution_horizon` set when running an asynchronous VLA source, controller
  `tracking_error_limit` and `state_timeout` set, simulation first.
- **Replay:** decimation and chunk stepping count cycles; pin `model_hash` so a
  replay with a different model file fails instead of diverging.
- **Allocation:** ONNX inference via tract allocates (~14 allocations per
  inference cycle, measured); `CuActionChunkTask` and all action/observation
  tasks allocate zero bytes on the hot path. Keep ONNX tasks on a host target.
- A faulted policy publishes nothing to the gripper, so a closed gripper stays
  closed.
- **Classical transit + policy:** to plan a collision-free approach classically
  and hand over to the policy for the contact phase, feed the follower and the
  joint action task into `repose_trajectory::CuSetpointMuxTask<N>` (classical
  input first). It switches bumplessly and falls back to a hold on a policy
  `Idle` / `Fault`; see the `repose-trajectory-and-control` skill.
- **Runtime safety shield:** insert `repose_safety::CuSafetyShieldTask<N>`
  between the action task (or `CuSetpointMuxTask<N>`) and `CuControllerTask<N>`
  to enforce self/scene collisions, an axis-aligned TCP `workspace` fence,
  `max_tcp_speed`, and `braking_horizon` lookahead while logging a
  `CuSafetyReport` every cycle; see `repose-collision`.
