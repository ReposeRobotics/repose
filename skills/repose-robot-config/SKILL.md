---
name: repose-robot-config
description: >-
  How to give Repose a robot: the `igus_rebel` preset, an inline serial chain and
  collision model in RON or Rust, a tool offset, or a URDF/SRDF imported at startup
  with `CuRobotDescription`. Use when a Repose task fails with a missing or mismatched
  `chain`/`collision_model`, when adding a new arm, or when converting a URDF.
---

# Repose: robot configuration

Repose supports **serial chains only**. The joint count `N` is part of every
task and payload type. No task picks a robot implicitly: FK, IK, servo,
collision, planner, and policy action tasks all require a `chain`, and
collision/planner tasks also require a `collision_model`.

## 1. The preset

```ron
config: {"chain": "igus_rebel", "collision_model": "igus_rebel"}
```

```rust
use repose::kinematics::igus_rebel_chain;                 // CuKinematicChain<6>
use repose::collision::igus_rebel_collision_model;        // CuRobotCollisionModel<6>
```

`igus_rebel` (6-DOF igus ReBeL) is currently the only built-in preset
(`CuRobotPreset::IgusRebel`).

## 2. Inline chain and collision model (RON)

Give exactly `N` joints, base to tip. Values are SI (meters, radians):

```ron
config: {
    "chain": {"joints": [
        {
            "joint_type": "revolute",            // revolute | prismatic | fixed
            "origin_translation": [0.0, 0.0, 0.1],  // URDF origin xyz
            "origin_rpy": [0.0, 0.0, 0.0],          // URDF origin rpy
            "axis": "pos_z",                        // pos_x..neg_z, or {"arbitrary": [x, y, z]}
            "limits": {"revolute": {"lower": -3.0, "upper": 3.0, "max_velocity": 1.0}},
        },
        // ... N entries
    ]},
    "collision_model": {
        "shapes": [
            {"shape": {"sphere": {"center": [0.0, 0.0, 0.0], "radius": 0.1}}, "frame": 0},
            {"shape": {"capsule": {"a": [0.0, 0.0, 0.0], "b": [0.0, 0.0, 0.3], "radius": 0.05}}, "frame": 1},
            // omit "frame" to attach a shape to the fixed base
        ],
        "safety_margin": 0.01,
    },
    "tool_offset": ...,   // optional: overrides the chain's flange-to-tool transform
}
```

- `frame` is a zero-based movable-frame index (`0..N`); without it the shape
  is on the base. Shape coordinates are in that frame.
- `CuRobotCollisionModel<N, M>` holds up to `M` shapes (`M = N` by default);
  use a larger `M` in the task type when a link has several shapes.
- Joint-limit RON variants are `revolute`, `prismatic`, `continuous`, `fixed`.

In Rust: `CuKinematicChain::<N>::new([CuJointSpec; N])` with joints from
`CuJointSpec::revolute(origin_translation, origin_rpy, axis, limits)` (and the
prismatic counterpart), and
`CuRobotCollisionModel::<N, M>::new(&[CuLinkShape], CuCollisionModelParams)?`.
Pass them to a task's `from_params`, or straight to library calls.

## 3. URDF + SRDF at startup

```rust
use cu29::units::si::{f64::Length, length::meter};
use repose::collision::CuCollisionModelParams;
use repose::kinematics::CuForwardKinematicsTask;
use repose::robot_description::{CuRobotDescription, CuSrdfImport};

let robot = CuRobotDescription::<6, 24>::from_urdf_srdf(
    &urdf_xml,
    &srdf_xml,
    CuSrdfImport::new("manipulator").with_collision(CuCollisionModelParams {
        safety_margin: Length::new::<meter>(0.005),
        skip_adjacent_links: false,
    }),
)?;
let fk = CuForwardKinematicsTask::from_params(robot.chain().clone())?;
```

Rules the importer enforces (errors, not silent fixes):

- Requires the `std` feature; parse once at startup, never in a task cycle.
- `N` counts **movable** joints; `M` (here 24) is shape capacity. Too small → error.
- Picks one base-to-tip path; fixed joints fold into origins and the tool offset.
- Joints: revolute, continuous, prismatic. Continuous joints are unwrapped
  (no shortest-angle wrapping in planning).
- Collision: spheres exact; cylinders → conservative capsules; **meshes and
  boxes are rejected** — approximate them with spheres/capsules first.
- SRDF subset: a group with exactly one `<chain>`, `<disable_collisions>`,
  complete `<group_state>`s. No mimic/passive joints, no composed groups.
- Feed expanded URDF (run xacro yourself).

Reference: `cargo run -p repose-robot-description --example serial_robot`.

## Units and metric

- Mixed revolute/prismatic arms are fine; each coordinate carries its unit.
- Planner distances/costs are dimensionless via `CuJointMetric` (default:
  1 rad ≡ 1 m). Planner keys `step_size`, `goal_threshold`, `rewire_radius`
  are in that metric.

## Common failures

| Symptom | Cause |
|---|---|
| Startup error "chain required" / "collision_model required" | No preset or inline model; nothing is implicit |
| Joint-count mismatch error | Inline `joints` length ≠ `N` in the task type |
| Unknown field error naming a key | Typo; params use `deny_unknown_fields` |
| Unit error on a joint coordinate | `Revolute` given for a prismatic joint or vice versa |
