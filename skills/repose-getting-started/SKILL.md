---
name: repose-getting-started
description: >-
  Entry point for writing code that uses Repose (deterministic, zero-allocation robot
  arm manipulation for copper-rs / cu29). Use when adding Repose to a project, picking
  which crate or task to use, or before writing any Repose call: covers the dependency
  line, the crate map, and the conventions every Repose API shares (typed SI units,
  const-generic joint counts and capacities, in-place `&mut` outputs, `CuResult`,
  explicit status enums, seeded determinism). Points to the topic skills for
  kinematics, collision, planning, trajectories/control, RON apps, and RL policies.
---

# Repose: getting started

Repose is a manipulation stack for robot arms built on
[copper-rs](https://github.com/copper-project/copper-rs) (`cu29`): kinematics,
collision checking, motion planning, trajectory timing, control, and RL policy
execution. Every algorithm is a plain library call *and* a Copper task you name
directly in RON.

## Add the dependency

Nothing is on crates.io yet; depend on a git tag (newest tag in the
[CHANGELOG](https://github.com/ReposeRobotics/repose/blob/master/CHANGELOG.md)):

```toml
[dependencies]
repose = { git = "https://github.com/ReposeRobotics/repose", tag = "v0.10.0" }
cu29 = { version = "1.3.1", features = ["units"] }
cu-spatial-payloads = "1.3.0"   # Transform3D for TCP poses
```

The umbrella crate `repose` re-exports the components as modules. RON configs
always name the **component crates** (`repose_trajectory::...`), so an app that
wires tasks in RON also depends on each component crate it names, from the same
git tag.

| Module (`repose::…`) | Crate | Use it for |
|---|---|---|
| `kinematics` | `repose-kinematics` | Chains, FK, Jacobians, DLS IK, differential IK, rigid-body dynamics, servo, joint-state payloads |
| `collision` | `repose-collision` | Sphere/capsule robot models, planning scenes, collision checks and monitor |
| `robot_description` | `repose-robot-description` | Startup URDF/SRDF import (`std`) |
| `robot_interface` | `repose-robot-interface` | Bridge channel contract and session lifecycle |
| `trajectory` | `repose-trajectory` | Timing, Cartesian blending, trajectory follower, bumpless classical/policy setpoint mux |
| `motion_planning` | `repose-motion-planning` | Planner trait, planner tasks, pose and Cartesian planning |
| `rrt_star` | `repose-joint-rrt-star` | Informed RRT* |
| `planners` | `repose` itself | `CuPlannerChoice` / `CuAnyPlanner` registry |
| `safety` | `repose-safety` | Real-time safety shield (`CuSafetyShieldTask`, `CuSafetyReport`) |
| `controller` | `repose-controller` | Position/velocity/effort joint controller |
| `io` | `repose-io` | Digital IO payloads, IO gripper |
| `mujoco_sim` (feature `mujoco-sim`) | `repose-mujoco-sim` | MuJoCo sidecar bridge |
| — | `repose-policy-rl` | ONNX RL/VLA policy tasks and OOD guardrails (`std`, separate dependency) |
| — | `repose-incident` | Black-box incident detector, root-cause blame, and checkpoint regression gating (`std`, separate dependency) |

## Conventions that apply to every call

Get these right first; most compile errors and wrong results come from them.

1. **Typed units, never bare floats.** Joint coordinates are
   `CuJointPosition::{Revolute(Angle), Prismatic(Length), Fixed}`; velocities,
   accelerations, jerks and efforts have matching enums. Build them from
   `cu29::units::si::f64::*`:

   ```rust
   use cu29::units::si::{angle::degree, f64::Angle};
   use repose::kinematics::CuJointPosition;

   fn deg(v: [f64; 6]) -> [CuJointPosition; 6] {
       v.map(|d| CuJointPosition::Revolute(Angle::new::<degree>(d)))
   }
   ```

   `Angle::new::<radian>(x).into()` also gives a revolute `CuJointPosition`.
   Read values back with `q.angle()?` / `.get::<degree>()`, or `si_value()` for
   the raw SI number. A revolute value given for a prismatic joint is an error.
   Dimensionless values (gains, weights, planner costs) stay `f64`.
2. **Const generics, not `Vec`.** `N` = joint count (6 for the igus ReBeL),
   `M` = collision-shape capacity (defaults to `N`), `MAX_*` = buffer capacity.
   If the compiler says "expected an array with a size of 6", fix the array or
   the `N`, not the call.
3. **Outputs are `&mut` arguments.** Allocate the output once, pass it in:
   `chain.forward_kinematics(&q, &mut tcp)?`. Large payloads
   (`CuTrajectoryPayload<6, 1024>`) belong in a `Box` or a long-lived struct,
   not on a small stack.
4. **`CuResult<T>` everywhere.** Propagate with `?`. Do not `unwrap()` in task
   or library code. Errors mean bad input (wrong units, invalid params);
   *algorithmic* outcomes come back as a status value instead.
5. **Check the status.** IK returns `CuIkStatus`, planners `CuPlanStatus`,
   timing `CuTimingStatus`, the follower `CuFollowStatus`, the controller
   `CuControllerStatus`. A returned `Ok(status)` is not success until you
   inspect it (`Converged`, `has_path()`, `Success`, …). Failures are
   published, not dropped.
6. **Params structs.** Every configurable thing has one `Cu<Thing>Params`
   with `Default`, `validate()`, `#[serde(deny_unknown_fields)]`. The RON
   `config:` block of a task *is* that struct, so a misspelled key is a startup
   error. Field names carry no unit suffix; RON values are SI (radians, meters,
   seconds). Programmatic construction is `from_params(required…, params)`.
7. **Determinism.** Sampling planners take every random draw from a seed
   (`begin(&request, seed)`, or a `cu_rng::CuRngBundle` resource in RON). The
   same seed replays bit for bit; never feed wall-clock time or OS randomness
   into the pipeline.
8. **`no_std`.** Runtime crates build for `no_std` with
   `default-features = false`. URDF import, the MuJoCo bridge,
   `repose-policy-rl`, and `repose-incident` are `std`-only.

## Which skill next

| Task | Skill |
|---|---|
| Choose or define a robot (preset, inline chain, URDF/SRDF) | `repose-robot-config` |
| FK, Jacobians, IK, twist servoing | `repose-kinematics` |
| Collision models, obstacles, attached objects, clearance | `repose-collision` |
| Plan joint, pose, or straight-line Cartesian motion | `repose-motion-planning` |
| Time a path, follow it, command the robot | `repose-trajectory-and-control` |
| Wire a Copper app in RON, run in MuJoCo, swap to hardware | `repose-copper-app` |
| Run an ONNX RL or VLA policy | `repose-policy-rl` |

## Working examples to copy from

In the Repose repo, these run without a simulator and check their own output:

```sh
cargo run -p repose --example kinematics   # FK, then IK back to the same pose
cargo run -p repose --example collision    # free vs colliding configuration
cargo run -p repose --example planning     # RRT-Connect around an obstacle, then timing
cargo run -p repose-robot-description --example serial_robot   # URDF/SRDF end to end
```

`examples/mujoco_rebel/` is the full goal → plan → time → follow → control loop
in RON. Rustdoc on each crate is the reference for every type named in these
skills; when a skill and the code disagree, the code wins.
