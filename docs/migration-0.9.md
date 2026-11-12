# Migrating to Repose 0.9

Update all Repose crate dependencies together.

## repose-controller: per-joint effort gains

Effort-mode gains are now set per joint, and the params struct carries the
joint count. RON configs need no change: a single number for `kp` or `kv`
still applies to every joint.

| Before | After |
|---|---|
| `CuEffortGains { kp: 20.0, kv: 0.8 }` | `CuEffortGains::uniform(20.0, 0.8)`, or `CuEffortGains { kp: [..; N], kv: [..; N] }` for per-joint values |
| `CuEffortGains` | `CuEffortGains<N>` |
| `CuControllerParams` | `CuControllerParams<N>` (inferred when passed to `CuControllerTask::<N>::from_params`) |
| `CuControllerParams { .. }` written out in full | add `effort_model: None`, or end with `..CuControllerParams::default()` |
| `"effort_gains": {"kp": 20.0, "kv": 0.8}` | unchanged; `"kp": [60.0, 60.0, 40.0, 8.0, 8.0, 4.0]` sets each joint |

Keyframes of `CuControllerTask` recorded by 0.8 are not supported in 0.9: the
frozen state now holds one `kp` and one `kv` per joint, and frozen layouts are
not versioned, so thawing an old keyframe is not detected and restores wrong
state when `N > 1`. Message payloads are unchanged, so logs still decode;
replay from the start of a log instead of from a keyframe.

## repose-controller: model-based effort mode (optional)

Effort mode stays pure PD unless you add a model:

```ron
config: {
    "mode": "effort",
    "effort_gains": {"kp": [60.0, 60.0, 40.0, 8.0, 8.0, 4.0], "kv": [3.0, 3.0, 2.0, 0.3, 0.3, 0.05]},
    "effort_model": {
        "chain": "igus_rebel",
        "inertia": {"links": [ /* one CuLinkInertia per joint */ ]},
    },
}
```

With a model the commanded effort is the PD feedback plus the inverse
dynamics at the measured positions and the desired velocities and
accelerations from the follower setpoint, and the hold and fault paths add
gravity compensation. If your
bridge or drive already compensates gravity, set `"gravity": [0.0, 0.0, 0.0]`
in the model so it is not applied twice.

## repose-trajectory: follower setpoint carries acceleration

`CuTrajectoryFollowOutput<N>` has a new field, `accelerations: [CuJointAcceleration; N]`,
placed between `velocities` and `status`. `CuFollowTrajectoryTask` fills it
from the active interpolation (zero while the setpoint is held), and the
controller's model-based effort feedforward uses it. The payload encoding
changes, so recorded follower setpoints from 0.8 do not decode in 0.9.

| Before | After |
|---|---|
| `CuTrajectoryFollowOutput { positions, velocities, status }` | add `accelerations` (for example `positions.map(\|q\| CuJointAcceleration::from_coordinate(q, 0.0))`) or use `..Default::default()` |
| `traj.sample_with(kind, t, &mut q, &mut qd)` | unchanged; `traj.sample_with_acceleration(kind, t, &mut q, &mut qd, &mut qdd)` also writes accelerations |

Custom setpoint producers that feed the controller should fill `accelerations`
with the true desired acceleration (zero reproduces the previous feedforward).
With an effort model a non-finite or wrongly typed acceleration makes the
controller report `Fault`; without a model it is ignored. RON graphs need no
change.

## repose-mujoco-sim: generic over the joint count

`CuMujocoSimBridge` is now `CuMujocoSimBridge<const N: usize>`, with
`MujocoTxChannels<N>` and `MujocoRxChannels<N>`; `MUJOCO_DOF` is removed.

| Before | After |
|---|---|
| `type: "repose_mujoco_sim::CuMujocoSimBridge"` | `type: "repose_mujoco_sim::CuMujocoSimBridge<6>"` (any `N` matching the channel `msg:` types) |
| `CuMujocoSimBridge::from_params(..)` | `CuMujocoSimBridge::<6>::from_params(..)` |

Command lines gain an additive `"n"` field; a 6-joint bridge still works with a
sidecar written for the old protocol. A bridge with `N != 6` needs a sidecar
that serves `N` joints, and a joint-count mismatch is reported as an error.
