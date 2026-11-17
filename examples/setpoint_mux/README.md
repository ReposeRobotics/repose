# Setpoint mux example

This headless Copper application tests the classical-to-policy handoff in
`repose_trajectory::CuSetpointMuxTask` in closed loop. It records the run to
a `.copper` log, and you can replay that log deterministically. It needs no
simulator: the arm is an in-process plant that tracks position commands with
a first-order lag.

```text
plant/joint_state ─► scenario ─► planner ─► timing ─► follower ──┐
        │                ▲                                       ├─► mux ─► controller ─► plant/joint_command
        ├────────► policy ─► joint_action ───────────────────────┘
```

The scenario runs in four phases:

1. **Transit (classical).** RRT* plans a path from home to the pre-grasp
   pose. `timing` adds cubic timing and `follower` executes it.
2. **Handoff to the policy.** When the transit finishes and the scripted
   policy reports `Running`, the mux blends smoothly into the
   `CuJointActionTask` setpoint. That setpoint is the contact motion.
3. **Fallback.** After 150 cycles the policy reports `Fault`. The mux holds
   the position it last emitted, where the policy left the arm, not the stale
   pre-grasp pose.
4. **Retract (classical).** Half a second later the scenario asks for a plan
   from the measured state back home. The new classical trajectory takes over
   from the hold, again blended.

| Node | Component | Role |
|---|---|---|
| `scenario` | `tasks::Scenario` (this crate) | Sends the transit request, then the retract request after the policy faults |
| `planner` | `repose_joint_rrt_star::CuJointRrtStarTask<6>` | Anytime RRT* running in the background |
| `timing` | `repose_motion_planning::CuPlanToTrajectoryTask<6, 1024>` | Times the path with a cubic profile |
| `follower` | `repose_trajectory::CuFollowTrajectoryTask<6, 1024>` | Classical setpoint |
| `policy` | `tasks::ScriptedPolicy` (this crate) | Stand-in for an ONNX or VLA policy: `Idle`, then a `Fresh` action, then `Fault` |
| `joint_action` | `repose_policy_rl::CuJointActionTask<6, 6>` | Policy setpoint (`position_delta`) |
| `mux` | `repose_trajectory::CuSetpointMuxTask<6>` | Arbitration with a 0.2 s blend on each switch |
| `controller` | `repose_controller::CuControllerTask<6>` | Tracking-error watchdog and rate limit |
| `observer` | `tasks::HandoffObserver` (this crate) | Test oracle: checks the four phases in order and the continuity of the setpoint |

## Run

From this directory:

```bash
cargo run --bin setpoint_mux_demo
```

The app writes `result.json` and exits with a nonzero status if any phase is
missing. A passing run looks like this:

```json
{"success":true,"outcome":"retracted","cycles":442,
 "phase_cycles":{"at_pregrasp":95,"policy_moved":153,"held":284,"retracted":442},
 "max_setpoint_position_step_rad":0.008000,"max_setpoint_velocity_step_rad_s":0.055995}
```

The log is `logs/setpoint_mux.copper`, with keyframes at Copper's default
interval (about one per second here). Always pass this base path to the
tools, not `_0`.

## Replay

```bash
cargo run --bin setpoint_mux_resim                    # default: logs/setpoint_mux.copper
cargo run --bin setpoint_mux_resim -- --log-base logs/setpoint_mux.copper
```

The resim injects two things from the log: the plant joint state (the
external input) and the background planner's results, which arrive on
wall-clock-dependent cycles. Every other task runs again, including the mux
and the controller. The resim writes `logs/setpoint_mux_resim.copper` and
checks that every mux and controller output is bit-identical to the
recording.

Replay is exact because the recording runs on simulated time. Copper's mock
clock advances exactly 10 ms per cycle, so every `ctx.now()` in a cycle can be
reproduced.

## Inspect

```bash
cargo run --bin setpoint_mux_logreader -- logs/setpoint_mux.copper extract-copperlists -e jsonl > cl.jsonl
cargo run --bin setpoint_mux_logreader -- logs/setpoint_mux.copper fsck
```

## View in Rerun

`setpoint_mux_to_rrd` converts the log into a [Rerun](https://rerun.io)
recording offline; the demo itself never logs to Rerun. It sits behind the
`rrd` feature so the normal build stays small.

```bash
cargo run --features rrd --bin setpoint_mux_to_rrd            # logs/setpoint_mux.copper -> logs/setpoint_mux.rrd
cargo run --features rrd --bin setpoint_mux_to_rrd -- in.copper out.rrd
rerun logs/setpoint_mux.rrd
```

The default layout shows the arm in 3D (measured in blue, controller command
in red, with the tool path) beside one plot per joint. Each joint plot
overlays the measured position, both mux inputs (follower and policy), the
mux output and the controller command, so every blend is visible. Below are
the mux setpoint velocity and a log of follower, policy, mux and controller
status changes. Timelines are `cu_time` (the Copper clock) and `copperlist`.

## Known limitations

- Decoding a copperlist with 1024-point trajectories needs a large stack.
  The logreader and the resim run on a thread with a 256 MB stack.
