---
name: repose-kinematics
description: >-
  Forward kinematics, Jacobians, damped-least-squares IK, and differential IK / twist
  servoing with `repose-kinematics` (`CuKinematicChain<N>`), as library calls and as the
  `CuForwardKinematicsTask`, `CuInverseKinematicsTask`, and `CuServoTask` Copper tasks.
  Use when computing a TCP pose, solving IK to a pose, converting a Cartesian twist to
  joint velocities, or reading `CuIkStatus` / `CuDiffIkStatus`.
---

# Repose: kinematics

All functions are methods on `CuKinematicChain<N>` (get one from
`igus_rebel_chain()` or see `repose-robot-config`). They write into
caller-owned `&mut` outputs and return `CuResult`.

## Forward kinematics

```rust
use cu_spatial_payloads::Transform3D;
use cu29::units::si::length::meter;

let mut tcp = Transform3D::<f64>::identity();
chain.forward_kinematics(&q, &mut tcp)?;          // q: [CuJointPosition; N]
let t = tcp.translation();                        // [Length; 3]
println!("x = {} m", t[0].get::<meter>());
```

- The TCP includes the chain's tool offset (`chain.set_tool_offset(...)`, or
  `"tool_offset"` in task config).
- `forward_frames(&q, &mut [Transform3D<f64>; N], &mut ee)` gives every link frame.

## Jacobians and manipulability

```rust
let mut jac = [[0.0_f64; N]; 6];
chain.geometric_jacobian(&q, &mut jac)?;   // base frame
chain.body_jacobian(&q, &mut jac)?;        // tool frame
let w2 = chain.manipulability_squared(&q)?; // det(J Jᵀ), ~0 near singularities
```

Rows `0..3` are linear velocity (m/s), rows `3..6` angular (rad/s), measured at
the tool origin. Layout is `[row][joint]`.

## Inverse kinematics (to a pose)

```rust
use repose::kinematics::{CuIkParams, CuIkStatus};

let mut solution = seed;                              // [CuJointPosition; N]
let status = chain.inverse_kinematics_dls(&target, &seed, &CuIkParams::default(), &mut solution)?;
match status {
    CuIkStatus::Converged { iterations, position_error, orientation_error } => { /* use solution */ }
    other => return Err(CuError::from(format!("IK did not converge: {other:?}"))),
}
```

- **Always check the status.** `MaxIterationsReached` still writes the best
  iterate into `solution`; using it blindly sends the robot somewhere else.
- IK is local: the result depends on the seed. Seed with the measured state
  for small moves; for a reachable-but-far pose try several seeds (that is what
  the pose planner does with `goal.attempts`).
- IK does **not** check collisions. Check the solution with
  `repose-collision` before using it, or use the pose planner
  (`repose-motion-planning`), which does both.
- `CuIkParams` fields: `solver` (`damped_least_squares` default, or
  `pseudoinverse`, which errors at singularities), `max_iterations`,
  `position_tolerance` (Length), `orientation_tolerance` (Angle), `damping`,
  `max_step` (Angle), `max_linear_step` (Length), `enforce_joint_limits`.

## Differential IK (twist → joint velocities)

```rust
use cu29::units::si::{f64::{Velocity, AngularVelocity}, velocity::meter_per_second,
                      angular_velocity::radian_per_second};
use repose::kinematics::{CuDiffIkParams, CuJointVelocity, CuTwist6D, CuTwistFrame};

let twist = CuTwist6D {
    linear: [Velocity::new::<meter_per_second>(0.05), Velocity::new::<meter_per_second>(0.0),
             Velocity::new::<meter_per_second>(0.0)],
    angular: [AngularVelocity::new::<radian_per_second>(0.0); 3],
    frame: CuTwistFrame::Base,   // or Body (tool frame)
};
let mut qd = [CuJointVelocity::default(); N];
let status = chain.differential_ik(&q, &twist, &CuDiffIkParams::default(), &mut qd)?;
```

`CuDiffIkStatus` is `Nominal`, `SingularityDamped` (damping ramped up near a
singularity), or `VelocityScaled` (all joints scaled down to respect
`max_velocity`). Integrate `qd` over the cycle period yourself or feed a
velocity-mode controller. `CuDiffIkParams`: `base_damping`,
`max_singularity_damping`, `singularity_threshold_sq`, `limit_margin`,
`linear_limit_margin`, `enforce_velocity_limits`.

## Copper tasks

| Task | Inputs | Output | Config (`Cu…Params`) |
|---|---|---|---|
| `repose_kinematics::CuForwardKinematicsTask<N>` | `CuJointPositionsPayload<N>` | `Transform3D<f64>` | `chain`, `tool_offset` |
| `repose_kinematics::CuInverseKinematicsTask<N>` | `Transform3D<f64>`, `CuJointPositionsPayload<N>` (seed) | `CuIkSolutionPayload<N>` (`positions`, `status`) | `chain`, `tool_offset`, `ik` |
| `repose_kinematics::CuServoTask<N>` | `CuTwist6D`, `CuJointPositionsPayload<N>` | `CuJointVelocitiesPayload<N>` (`velocities`, `status`) | `chain`, `tool_offset`, `diff_ik` |

```ron
(
    id: "ik",
    type: "repose_kinematics::CuInverseKinematicsTask<6>",
    config: {"chain": "igus_rebel", "ik": {"max_iterations": 200, "damping": 0.05}},
),
```

The measured robot state arrives as `CuJointStatePayload<N>` (`status`,
`positions`, `velocities`), which is the bridge contract — not the same type as
`CuJointPositionsPayload<N>`.

## Pitfalls

- Mixing degrees and radians: construct with the unit you mean
  (`Angle::new::<degree>`); never put degrees into a `radian` constructor.
- Orientation of `Transform3D` targets: build them with
  `cu_spatial_payloads` helpers or from an FK result; for XYZ-Euler typed
  poses use `repose_motion_planning::CuTcpPose`.
- Planning to a pose? Prefer `CuPosePlannerTask` (IK + collision-aware
  attempts + planning) over hand-rolled IK; see `repose-motion-planning`.
