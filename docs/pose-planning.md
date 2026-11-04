# Pose goals and straight TCP moves

`CuPosePlanRequest<N>` carries a measured joint start, a TCP goal in the chain
base frame, a planning scene, and stable-ID exclusions. `CuTcpPose` uses typed
lengths for translation and typed angles for XYZ Euler rotation. The chain's
configured tool offset defines the TCP.

Two Copper pipelines accept the same request:

- `CuPosePlannerTask<P, N, M>` resolves the goal through DLS IK and passes the
  resulting joint request to any `CuJointPlanner`. The path may curve in TCP
  space and can go around obstacles according to the selected planner.
- `CuCartesianPlanTask<N, PATH, CAP, M>` builds a straight TCP line with
  shortest-arc orientation interpolation, solves IK along it, and times it with
  the configured profile. It fails when the line is blocked; it does not detour.
  `PATH` bounds geometric samples and `CAP` bounds trajectory points.

Both use the measured start as the first IK seed, then sample joint limits in
an order determined by the task's injected `CuRng` resource. `goal.attempts`
(default 8, maximum 64) bounds attempts. The first collision-free IK solution
is selected; joint planning does not retry other IK branches after a failed
search. Replay preserves seed progress.

The request's scene and exclusions apply to endpoint IK validation and planning.
Cartesian geometry and its timing spline are rechecked against that same scene,
including attached objects. Checks use discrete samples and joint-space edges;
they do not certify continuous swept volumes or the follower's reconstructed
curve between trajectory samples. Results identify the captured revision; the
application must reject stale revisions before executing a plan when its scene
changes.

`CuJointPlanPath` and `CuCartesianPlanResult` publish failures explicitly:
`NoIkSolution` for bounded IK failure, `InvalidGoal` when converged IK solutions
are in collision, `InvalidStart`, `InvalidRequest`, `CapacityExceeded`, or
`Exhausted` when no path was found. Failed results have no motion data.

## RON wiring

`CuPoseGoalTask<N>` emits once from the first ready joint-state message. Its
configuration can include a fixed `scene` and `exclusions`; applications with
changing scenes publish `CuPosePlanRequest<N>` directly.

```ron
(
    id: "goal",
    type: "repose_motion_planning::CuPoseGoalTask<6>",
    config: {
        "goal": {
            "translation": [0.25, 0.0, 0.30],
            "rotation": [0.0, 0.0, 0.0],
        },
    },
)
```

For joint planning, the planner task's config nests the existing planner params:

```ron
config: {
    "planner": {
        "chain": "igus_rebel",
        "collision_model": "igus_rebel",
        "params": {},
    },
    "goal": {"attempts": 8},
}
```

Use a task type such as
`repose_motion_planning::CuPosePlannerTask<repose_motion_planning::CuJointDirect<6>, 6>`.
Connect `CuPosePlanRequest<6>` into it and `CuJointPlanPath<6>` into the existing
`CuPlanToTrajectoryTask<6, 1024>`. The planner keeps its anytime base/refine
lifecycle and its `rng` resource binding.

For a straight move, use
`repose_motion_planning::CuCartesianPlanTask<6, 256, 1024>` with an `rng` resource:

```ron
config: {
    "chain": "igus_rebel",
    "collision_model": "igus_rebel",
    "goal": {"attempts": 8},
    "path": {"translation_step": 0.01, "rotation_step": 0.05},
    "timing": {
        "timing": {
            "limits": {
                "max_velocities": [{"revolute": 0.8}, {"revolute": 0.8}, {"revolute": 0.8}, {"revolute": 0.8}, {"revolute": 0.8}, {"revolute": 0.8}],
                "max_accelerations": [{"revolute": 2.0}, {"revolute": 2.0}, {"revolute": 2.0}, {"revolute": 2.0}, {"revolute": 2.0}, {"revolute": 2.0}],
            },
            "sample_period": 0.02,
        },
    },
}
```

Inspect `CuCartesianPlanResult<6, 1024>` for status and scene identity. Pass it
through `CuCartesianTrajectoryTask<6, 1024>` to extract successful
`CuTrajectoryPayload<6, 1024>` messages for the standard follower. Empty input or
failure clears its output; it propagates the input time of validity.
