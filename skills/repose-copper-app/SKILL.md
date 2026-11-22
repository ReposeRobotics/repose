---
name: repose-copper-app
description: >-
  Building a Copper (cu29) application from Repose components: Cargo setup with
  `cu29-build`, the `#[copper_runtime]` entry point, naming Repose tasks directly in
  `copperconfig.ron` with const generics, connections and message types, the `cu_rng`
  resource, background/anytime planner nodes, the robot bridge channel contract
  (`joint_state` / `joint_command` / session), the MuJoCo simulator bridge, and moving
  from simulation to hardware. Use when creating or editing a Repose RON graph, wiring a
  plan-and-execute loop, or running in simulation.
---

# Repose: Copper applications

Applications **don't wrap** Repose tasks: every task is generic over its
joint count and capacities and is named directly in RON. Usually the only
application code is `main.rs` (plus any app-specific tasks).

## Fastest start

```sh
cargo install cargo-generate
cargo generate gh:ReposeRobotics/repose-bridges templates/robot-app --name my_robot
```

The template targets the igus ReBeL or a UR5e (selected in `robot.ron`) and can
run against URSim. **`cargo run` moves the selected robot.**

## By hand

`Cargo.toml` — depend on every crate whose types appear in the RON, all from
the same Repose tag:

```toml
[dependencies]
cu29 = { version = "1.3.1", default-features = false, features = ["std", "units"] }
cu-rng = { version = "1.3.0", features = ["std"] }
repose-kinematics      = { git = "https://github.com/ReposeRobotics/repose", tag = "v0.10.0" }
repose-motion-planning = { git = "https://github.com/ReposeRobotics/repose", tag = "v0.10.0" }
repose-joint-rrt-star  = { git = "https://github.com/ReposeRobotics/repose", tag = "v0.10.0" }
repose-trajectory      = { git = "https://github.com/ReposeRobotics/repose", tag = "v0.10.0" }
repose-controller      = { git = "https://github.com/ReposeRobotics/repose", tag = "v0.10.0" }
repose-mujoco-sim      = { git = "https://github.com/ReposeRobotics/repose", tag = "v0.10.0" }

[build-dependencies]
cu29-build = "1.3.0"
```

`build.rs`:

```rust
fn main() { cu29_build::setup(); }
```

`src/main.rs`:

```rust
use cu29::prelude::*;

#[copper_runtime(config = "copperconfig.ron")]
struct App {}

fn main() -> CuResult<()> {
    let log = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("app.copper");
    let app = App::builder().with_log_path(log, None)?.build()?;
    app.run_until_shutdown().map_err(|fault| fault.error)?;
    Ok(())
}
```

## The plan-and-execute graph

```ron
(
    runtime: (rate_target_hz: 50),
    resources: [
        (id: "planner_rng", provider: "cu_rng::CuRngBundle", config: {"seed": 42}),
    ],
    tasks: [
        (id: "goal", type: "repose_motion_planning::CuJointGoalTask<6>",
         config: {"goal": [{"revolute": 0.4}, {"revolute": -0.3}, {"revolute": 0.5},
                           {"revolute": 0.0}, {"revolute": 0.4}, {"revolute": 0.0}]}),
        (
            id: "planner",
            type: "repose_joint_rrt_star::CuJointRrtStarTask<6>",
            background: true,
            anytime: (time_budget_ms: 250.0, quality_target: 0.95),
            resources: {"rng": "planner_rng.rng"},
            config: {"chain": "igus_rebel", "collision_model": "igus_rebel"},
        ),
        (id: "timing", type: "repose_motion_planning::CuPlanToTrajectoryTask<6, 1024>",
         config: {"profile": "cubic", "sample_period": 0.02, "limits": { /* see repose-trajectory-and-control */ }}),
        (id: "follower", type: "repose_trajectory::CuFollowTrajectoryTask<6, 1024>"),
        (id: "controller", type: "repose_controller::CuControllerTask<6>",
         config: {"tracking_error_limit": 0.5, "state_timeout": 0.2}),
    ],
    bridges: [
        (
            id: "robot",
            type: "repose_mujoco_sim::CuMujocoSimBridge<6>",
            config: {"host": "127.0.0.1", "port": 55091, "read_timeout": 0.25},
            channels: [
                Tx(id: "joint_command", msg: "repose_controller::CuJointCommandPayload<6>"),
                Rx(id: "joint_state", msg: "repose_kinematics::CuJointStatePayload<6>"),
            ],
        ),
    ],
    cnx: [
        (src: "robot/joint_state", dst: "goal",       msg: "repose_kinematics::CuJointStatePayload<6>"),
        (src: "goal",              dst: "planner",    msg: "repose_motion_planning::CuJointPlanRequest<6>"),
        (src: "planner",           dst: "timing",     msg: "repose_motion_planning::CuJointPlanPath<6>"),
        (src: "timing",            dst: "follower",   msg: "repose_trajectory::CuTrajectoryPayload<6, 1024>"),
        (src: "follower",          dst: "controller", msg: "repose_trajectory::CuTrajectoryFollowOutput<6>"),
        (src: "robot/joint_state", dst: "controller", msg: "repose_kinematics::CuJointStatePayload<6>"),
        (src: "controller",        dst: "robot/joint_command", msg: "repose_controller::CuJointCommandPayload<6>"),
    ],
)
```

Complete, tested version: `examples/mujoco_rebel/` in the Repose repo (and
`examples/simplex_safety/` for a full Simplex classical/policy/safety-shield
pipeline).

## Rules for RON graphs

- `type:` and `msg:` strings are Rust paths resolved in **your** binary crate:
  use the component crate path (`repose_trajectory::…`), not `repose::trajectory::…`,
  and depend on that crate.
- Every `N`, `CAP`, `MAX_POINTS` must agree between task types and the `msg:`
  types connecting them; a mismatch is a compile error from the generated code.
- Multi-input tasks take inputs in their declared order (e.g. the controller:
  follower output, then joint state). Check `type Input` in the task's rustdoc.
- `config:` keys are the task's `Cu…Params`, SI units, no unit suffixes;
  unknown keys fail at startup.
- Every planner task (including pose and Cartesian planners) needs the `rng`
  resource from `cu_rng::CuRngBundle`; fix its `seed` for replay.
- Mark planners `background: true` with an `anytime:` budget so planning never
  blocks the control cycle.
- Keep `rate_target_hz` consistent with the timing `sample_period` and with any
  policy's training period.

## Robot bridges: one contract

Every Repose robot bridge, simulator or hardware, exposes the same channels
(`repose-robot-interface`):

| Dir | Channel | Payload |
|---|---|---|
| Rx | `joint_state` | `repose_kinematics::CuJointStatePayload<N>` |
| Tx | `joint_command` | `repose_controller::CuJointCommandPayload<N>` |
| Tx | `session` | `repose_robot_interface::CuSessionCommand` (hardware with a control session) |
| Rx | `session_state` | `repose_robot_interface::CuSessionState` |

Swapping simulation for hardware is changing the bridge entry, not the graph.
Hardware bridges (igus ReBeL CRI, Universal Robots) live in
[`repose-bridges`](https://github.com/ReposeRobotics/repose-bridges). A
hardware session follows
`AcquireControl → ClearFaults → EnableDrives → … motion … → DisableDrives → ReleaseControl`;
`CuSessionTracker` enforces one command in flight.

## MuJoCo simulation

`repose_mujoco_sim::CuMujocoSimBridge<N>` talks line-delimited JSON over TCP to
a Python sidecar (`examples/mujoco_rebel/sidecar.py`, needs `mujoco` and
`numpy`). Start the sidecar with an expanded URDF, then `cargo run`:

```sh
python sidecar.py --urdf /path/to/robot.urdf     # serves joint1..jointN
cargo run
```

Config: `host`, `port`, `read_timeout` (s). The sidecar's joint count must
equal `N`.

## Safety

- Simulation first. Treat any graph that includes a hardware bridge as able to
  move the robot as soon as it runs.
- Keep `state_timeout` and `tracking_error_limit` set on the controller.
- Collision checks in planning are discrete and against the scene snapshot of
  that job; insert `repose_safety::CuSafetyShieldTask<N>` before
  `CuControllerTask<N>` (or `CuCollisionCheckTask` + `CuCollisionMonitor`) for
  execution-time collision, workspace-fence, speed, and braking-horizon
  enforcement.
