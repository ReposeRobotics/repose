---
name: repose-policy-rl
description: >-
  Running trained ONNX reinforcement-learning or vision-language-action policies on a
  robot arm with `repose-policy-rl`: observation tasks (`CuJointReachObservationTask`),
  pure inference (`CuPolicyTask<IN, OBS, ACT>` on tract-onnx), and action tasks that
  turn raw actions into safe setpoints (`CuJointActionTask`, `CuCartesianActionTask`
  via differential IK, `CuGripperActionTask`). Use when deploying a policy exported
  from MuJoCo Playground, Isaac Lab, or similar; sizing IN/OBS/ACT; setting decimation,
  scales, and per-cycle limits; or pinning `model_hash` for replay.
---

# Repose: RL and VLA policies

`repose-policy-rl` is a separate, **`std`-only** dependency (not re-exported by
`repose`); it ships in releases after 0.8.4, so pin a tag that includes it
(check the CHANGELOG). It splits a policy into three stages:

```text
joint state ─► observation task ─► CuPolicyTask ─► action task(s) ─► CuControllerTask ─► bridge
  (+ goal)     CuPolicyObservation<IN>  CuPolicyAction<ACT>   CuTrajectoryFollowOutput<N>
```

The model never commands joints directly: action tasks scale, limit, and
(for Cartesian actions) solve differential IK, and the standard controller
still applies its watchdog, tracking-error fault, and rate limit.

## Sizes

| Generic | Meaning |
|---|---|
| `N` | Robot joints |
| `IN` | Features from the observation task |
| `OBS` | Model input width: `IN + ACT` with `"previous_action": true`, else `IN` |
| `ACT` | Model output width |

The ONNX model must have exactly one `f32` input of shape `[1, OBS]` and one
`f32` output of shape `[1, ACT]` (a small stateless MLP); anything else is a
startup error.

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
- Policy: `CuPolicyObservation<IN>` → `CuPolicyObservation<OBS>` (exact model
  input, for the log) + `CuPolicyAction<ACT>` (raw output).
- Joint/Cartesian action: `CuPolicyAction<ACT>` + `CuJointStatePayload<N>` →
  `CuTrajectoryFollowOutput<N>` (Cartesian also emits `CuCartesianActionReport`).
  Connect it to `CuControllerTask<N>` in place of the trajectory follower.
- Gripper action: `CuPolicyAction<ACT>` → `CuGripperCommandPayload` for
  `repose_io::CuGripperTask`.

Policy status (`CuPolicyStatus`): `Fresh` (inferred this cycle), `Held`
(between decimated inferences), `Idle` (no input), `Fault`.

## VLA end-effector actions

A typical VLA action `[dx, dy, dz, droll, dpitch, dyaw, gripper]` (`ACT = 7`):

```ron
(
    id: "arm",
    type: "repose_policy_rl::CuCartesianActionTask<6, 7>",
    config: {
        "chain": "igus_rebel",
        "period": 0.02,              // required: delta is executed over one period
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

- **Training period:** policy period = `decimation / rate_target_hz`; it must
  equal the period the policy was trained at. Set `period` on action tasks to
  `1 / rate_target_hz`.
- **Observation layout and scales** must reproduce the trainer's exactly
  (order of terms, scales, default positions). A silent mismatch produces a
  policy that runs and misbehaves; check the `CuPolicyObservation<OBS>` in the
  log against the trainer.
- **Start conservative:** small `scale`, `max_step` / `max_translation` set,
  controller `tracking_error_limit` and `state_timeout` set, simulation first.
- **Replay:** decimation counts cycles; pin `model_hash` so a replay with a
  different model file fails instead of diverging.
- **Allocation:** inference via tract allocates (~14 allocations per cycle,
  measured). This crate is the documented exception to Repose's zero-allocation
  rule; keep it on a host target.
- A faulted policy publishes nothing to the gripper, so a closed gripper stays
  closed.
