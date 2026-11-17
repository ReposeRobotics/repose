---
name: repose-trajectory-and-control
description: >-
  Turning a path into robot motion with Repose: trajectory timing (`parameterize`,
  `CuTimingParams`, profiles trapezoidal / s_curve / minimum_jerk / cubic), Cartesian
  TCP-zone blending, `CuTrajectoryPayload` sampling, `CuFollowTrajectoryTask`,
  the bumpless classical/policy `CuSetpointMuxTask`,
  `CuPlanToTrajectoryTask`, and the `CuControllerTask` joint controller (position /
  velocity / effort modes, watchdog, tracking-error fault, rate limit) plus the IO
  gripper. Use when timing waypoints, following a trajectory, configuring the
  controller, or debugging Fault / Idle / Holding behavior.
---

# Repose: trajectories and control

Pipeline: `waypoints → timing → CuTrajectoryPayload → follower → controller → bridge`.

## Timing joint waypoints

```rust
use cu29::units::si::{f64::{AngularAcceleration, AngularVelocity, Time},
    angular_acceleration::radian_per_second_squared, angular_velocity::radian_per_second,
    time::second};
use repose::kinematics::{CuJointAcceleration, CuJointVelocity};
use repose::trajectory::{CuJointTrajectoryLimits, CuTimingOverrides, CuTimingParams,
    CuTimingStatus, CuTrajectoryPayload, CuTrajectoryProfile, parameterize};

let params = CuTimingParams {
    limits: CuJointTrajectoryLimits::new(
        [CuJointVelocity::Revolute(AngularVelocity::new::<radian_per_second>(1.0)); 6],
        [CuJointAcceleration::Revolute(
            AngularAcceleration::new::<radian_per_second_squared>(2.0)); 6],
    ),                                   // .with_jerks([...]) for s_curve jerk bounds
    profile: CuTrajectoryProfile::Trapezoidal,
    sample_period: Time::new::<second>(0.01),
};
let mut traj = Box::new(CuTrajectoryPayload::<6, 1024>::new()); // large: box it
let status = parameterize(&waypoints, &params, CuTimingOverrides::default(), &mut traj)?;
if status != CuTimingStatus::Success { /* EmptyPath, InvalidInput, CapacityExceeded, … */ }
```

- `waypoints: &[[CuJointPosition; N]]`. From a plan:
  `path.waypoints().iter().map(|w| w.positions)` into a fixed array of
  `MAX_WAYPOINTS`, then slice `[..path.len as usize]`.
- Profiles (RON names): `trapezoidal`, `s_curve` (uses jerk limits; executes
  as quintic), `minimum_jerk`, `cubic`.
- `MAX_POINTS` (here 1024) must hold `duration / sample_period` samples or
  you get `CapacityExceeded`. Lengthen `sample_period` or raise `MAX_POINTS`.
- Timing sets the payload's interpolation, so sample with
  `traj.sample(t, &mut q, &mut qd)?` (`sample_with_acceleration` adds `qdd`).
  Sampling outside `[0, duration]` holds the endpoint.
- `CuTimingOverrides` can fix `segment_durations` or `waypoint_velocities`.

## Cartesian paths with TCP zones (fly-by)

`repose_trajectory::blend_cartesian_path` takes chain, collision model, joint
waypoints, and one `Length` zone radius per waypoint (first and last must be
zero; zero = exact stop, positive = rounded corner). Time the result with
`parameterize_cartesian_path` and any profile. Statuses include
`ZoneExceeded { waypoint }` and `NoContinuousMotion { waypoint }`
(path reverses inside a zone). Details: `docs/trajectories.md` in the repo.

## Copper tasks

| Task | Input → Output | Config |
|---|---|---|
| `repose_motion_planning::CuPlanToTrajectoryTask<N, CAP>` | `CuJointPlanPath<N>` → `CuTrajectoryPayload<N, CAP>` | `CuTimingParams<N>`: `limits` (required), `profile`, `sample_period` |
| `repose_trajectory::CuFollowTrajectoryTask<N, MAX_POINTS>` | `CuTrajectoryPayload<N, MAX_POINTS>` → `CuTrajectoryFollowOutput<N>` | optional `"interpolation"`: `linear` / `cubic` / `quintic` (default: the payload's own) |
| `repose_trajectory::CuSetpointMuxTask<N>` | `CuTrajectoryFollowOutput<N>` (classical) + `CuTrajectoryFollowOutput<N>` (policy) → `CuTrajectoryFollowOutput<N>` | optional `"blend_time"` (s, default `0.2`, `0` = hard switch) |
| `repose_controller::CuControllerTask<N>` | `CuTrajectoryFollowOutput<N>` + `CuJointStatePayload<N>` → `CuJointCommandPayload<N>` | see below |

Match `CAP` / `MAX_POINTS` across the timing task, the connection `msg:` type,
and the follower.

```ron
(
    id: "timing",
    type: "repose_motion_planning::CuPlanToTrajectoryTask<6, 1024>",
    config: {
        "profile": "cubic",
        "sample_period": 0.02,
        "limits": {
            "max_velocities": [{"revolute": 0.8}, {"revolute": 0.8}, {"revolute": 0.8},
                               {"revolute": 0.8}, {"revolute": 0.8}, {"revolute": 0.8}],
            "max_accelerations": [{"revolute": 2.0}, {"revolute": 2.0}, {"revolute": 2.0},
                                  {"revolute": 2.0}, {"revolute": 2.0}, {"revolute": 2.0}],
        },
    },
),
(id: "follower", type: "repose_trajectory::CuFollowTrajectoryTask<6, 1024>"),
(
    id: "controller",
    type: "repose_controller::CuControllerTask<6>",
    config: {
        "mode": "position",                 // position | velocity | effort
        "tracking_error_limit": 0.5,        // rad (m for prismatic) before Fault
        "state_timeout": 0.2,               // s; joint-state watchdog, omit to disable
        "rate_limit": {"max_speed": 1.5, "max_acceleration": 6.0},  // position mode only
    },
),
```

## Classical ↔ policy handoff (`CuSetpointMuxTask`)

Run a collision-free classical transit (planner → timing → follower) to a
pre-grasp pose, then let a learned policy (`repose_policy_rl` action task)
do the contact-rich part, with one controller downstream:

```ron
(id: "mux", type: "repose_trajectory::CuSetpointMuxTask<6>", config: {"blend_time": 0.2}),
// cnx: follower -> mux, joint_action -> mux, mux -> controller
// (input order: classical first, policy second)
```

| From | To | When |
|---|---|---|
| `classical` | `policy` | classical was `Running` and is now `Finished`, and the policy setpoint is `Running` |
| `policy` | `classical` | classical is `Running` again (a new plan, e.g. a retract, preempts the policy) |
| `policy` | `hold` | policy setpoint missing, `Idle`, `ErrorEmptyTrajectory` (policy `Fault`), or non-finite |
| `hold` | `classical` | classical is `Running` (plan the retract from the measured state) |

- Every switch is **bumpless**: the position/velocity/acceleration offset to
  the new source decays along a quintic over `blend_time`, so the setpoint
  stays C2-continuous. Expect a small overshoot when switching at speed.
- `hold` keeps the position the mux last emitted (where the policy left the
  arm, not the stale pre-grasp pose) and reports `Finished`, so the
  controller is `Holding`. It never resumes the policy on its own: a new
  classical plan has to run first.
- Read the active source with `CuSetpointMuxTask::source()`
  (`classical` / `policy` / `hold`); it round-trips through `Freezable`.

## Follower semantics (`CuFollowStatus`)

- `Running` while sampling, `Finished` holding the final point.
- An **empty trajectory cancels**: `ErrorEmptyTrajectory` once, then `Idle`
  holding the last commanded position with zero velocity. A new non-empty
  trajectory resumes.
- `Idle` setpoints are not targets; the controller holds the measured state.

## Controller semantics (`CuControllerStatus`)

| Status | When | Command |
|---|---|---|
| `Idle` | No setpoint yet or follower `Idle` | Holds measured state |
| `Running` | Following | Mirrors setpoint |
| `Holding` | Follower `Finished` | Holds final setpoint |
| `Fault` | Tracking error > limit, stale/missing/non-finite state, non-finite setpoint, empty trajectory | Holds measured state (or last finite command) |

- The controller never emits a non-finite value; faults do not latch — good
  inputs resume normal operation next cycle.
- Effort mode: per-joint PD `effort_gains` (`kp`, `kv`; a single number means
  every joint) plus optional `effort_model` (`chain`, `inertia`, `gravity`) for
  inverse-dynamics feedforward.
- `state_timeout` compares the joint-state `Tov` to the clock: bridges must set
  `Tov` on their messages.

## Gripper on digital IO

`repose_io::CuGripperTask<IN, OUT>`: `CuGripperCommandPayload` +
`CuDigitalInputPayload<IN>` → `CuDigitalOutputPayload<OUT>` +
`CuGripperStatePayload`. Config: `close_output` (required), `open_output`,
`closed_input`, `open_input`, `settle`, `motion_timeout`.

## Pitfalls

- Timing does not recheck path constraints, and collision checks of the
  geometric path do not certify the reconstructed spline between samples;
  recheck if margins are tight.
- Don't send the planner's raw waypoints to the controller; time them first.
- Unit mix-ups in RON: `{"revolute": 0.8}` is rad/s; prismatic limits use
  `{"prismatic": …}` in m/s.
