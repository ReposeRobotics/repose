# Robot configuration

Only serial kinematic chains are supported. The joint count `N` is part of the task
and payload types; branched and closed-chain mechanisms are outside the current scope.

Task initialization never picks a robot implicitly. FK, IK, servo, collision,
and planner tasks require a `chain`, given either as a preset name or inline
with exactly `N` joint specifications:

```ron
config: {"chain": "igus_rebel", "collision_model": "igus_rebel"}
config: {"chain": {"joints": [...]}, "collision_model": {"shapes": [...], "safety_margin": 0.005}}
```

An optional `tool_offset` overrides the chain's flange-to-tool transform.
`CuRobotCollisionModel<N, M>` stores up to `M` shapes attached to `N` movable
frames or the base. The default `M = N` preserves one-shape-per-joint
construction; imported robots can have several shapes per link, missing geometry,
and shapes on fixed links. Contacts identify shape indices.

Use `from_params` to supply imported models directly.
`CuJointPosition` carries `Revolute(Angle)`, `Prismatic(Length)`, or `Fixed`;
`CuJointVelocity`, `CuJointAcceleration`, and `CuJointJerk` similarly distinguish
angular and linear quantities. Incorrect coordinate units return an error.
Planner distances and costs are dimensionless, scaled by `CuJointMetric`
(default: one radian and one meter).

## URDF and SRDF

```rust,ignore
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

Parsing requires the `repose-robot-description` crate's `std` feature and runs at
startup. The resulting kinematic and collision models remain fixed-size and
usable by the `no_std` runtime crates. `N` counts movable joints, not fixed joints;
`M` is a collision-shape capacity. Import errors report insufficient capacity or
a joint-count mismatch. No XML parser or dynamic name lookup runs in a task cycle.

URDF import selects one base-to-tip serial path, folds fixed transforms into
joint origins and the tool offset, and supports revolute, continuous, and
prismatic joints. Other source-tree branches are outside the selected model.
Continuous joints accept unbounded angles and sample a canonical `[-pi, pi]`
interval; planning uses unwrapped coordinates, without shortest-angle wrapping.

Collision spheres are exact. Cylinders become conservative capsules whose
endpoints are at the cylinder's end-face centers; the rounded ends extend beyond
the original cylinder. Meshes and boxes require explicit preprocessing and are
rejected. All collision elements on the selected links, including base and fixed
links, are retained. Links without collision elements contribute no shapes.
Imported models check all distinct rigid-body pairs unless SRDF disables them.

The SRDF subset supports a group containing exactly one `<chain>`,
`<disable_collisions>`, and complete `<group_state>` joint values. State order is
mapped by joint name. Selected-chain mimic/passive joints and composed groups
are rejected. End-effectors, virtual-joint world transforms, and other SRDF
semantics are not imported. Supply expanded URDF XML; Xacro expansion is external.
The format follows the [MoveIt URDF/SRDF guide](https://moveit.picknik.ai/main/doc/examples/urdf_srdf/urdf_srdf_tutorial.html).

Run the complete mixed revolute/prismatic example:

```bash
cargo run -p repose-robot-description --example serial_robot
```

It loads URDF/SRDF, resolves two named states, computes FK and IK, checks the IK
solution for collisions, and runs the anytime planner. It exits successfully when
all stages pass. Edge checks use discrete angular/linear interpolation; they are
not swept-volume continuous collision detection.

Existing callers must convert angular coordinates with `angle.into()` and match
joint-velocity variants. Joint-limit RON values now use `revolute`, `prismatic`,
`continuous`, or `fixed` variants. Planner `step_size`, `goal_threshold`, and
`rewire_radius` replace their former `_rad` keys and use the configured metric.

