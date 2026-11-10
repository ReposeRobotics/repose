# Motion planning

## Planners

All planners implement `repose_motion_planning::CuJointPlanner` and run in one
Copper task, `CuJointPlannerTask<P, N>`: `begin` starts a job from a
`CuJointPlanRequest<N>`, `step(budget)` does a bounded quantum of work, and
`write_best` fills a `CuJointPlanPath<N>` in place. The task owns the anytime
contract (base and refine budgets, monotone publishing, seeding, replay
state). Failures are published, not dropped: the path then carries no
waypoints and its `status` names the reason (`invalid_start`,
`invalid_goal`, `invalid_request`, `exhausted`, `capacity_exceeded`).

The planner list lives in one place, [`repose::planners`](../components/libs/repose/src/planners.rs):
`CuPlannerChoice` names every planner (RON spelling) with its behaviour, and
`CuAnyPlanner` runs any of them. Applications select planners by name from
there; adding a planner is one trait impl plus one line in that list.

The optimizers are local: they can stay stuck behind large obstacles where
sampling or search planners still succeed.

Sampling planners derive all randomness from the job seed; deterministic
planners ignore it. New planners implement the trait, share
`CuSceneValidity` for collision and clearance queries, and are registered in
`repose::planners`.

Pose goals and straight TCP moves are available through `CuPoseGoalTask`,
`CuPosePlannerTask`, and `CuCartesianPlanTask`. Both planning paths carry scene
revisions and use attached-object and exclusion checks. See
[Pose planning](pose-planning.md) for request types, failures, and RON wiring.

## Planning scene snapshots

`CuPlanningScene<O, A>` stores fixed-capacity obstacle and attached-shape
geometry with stable object IDs and a scene revision. The collision-check task
accepts joint positions, a scene snapshot, and stable-ID collision exclusions.
Every planner captures the scene and resolved exclusions once per planning job (`CuSceneValidity`), checks
obstacles and attached shapes throughout planning, and returns `scene_revision` and the captured exclusion set
with the path. Consumers must compare both with their current inputs before
accepting a plan. Refinement uses the captured snapshot until the next
base call starts a new job.

Attached shapes use link-local geometry and zero-based FK frame indices.
`CuSceneExclusions` permits explicit attached-object contact with a robot shape,
another attached object, or an obstacle. Scene objects are referenced by stable
IDs; robot shapes use their model construction indices. Invalid exclusion
references reject the query or planning job. Scene and exclusions must be published consistently.

## Path constraints

Successfully setting or clearing constraints discards the current planning job,
including its cached path and endpoint checks, even when the constraint set is
unchanged. All seven built-in planners and `CuAnyPlanner` return to their
construction state: `step` and `write_best` report `InvalidRequest`, no
waypoints are emitted, and quality is zero until the next `begin`.
An invalid or unsupported set leaves the job unchanged. Joint and pose planner
tasks also reset their job and publication bookkeeping; their next refinement
replaces any old output with a no-job failure. Start a new request with `base`
to resume planning.

Planners optionally enforce tool-frame path constraints along the whole move.
`CuOrientationConstraint` keeps a chosen tool-frame axis within a cone angle
of a base-frame reference direction ("keep the tool upright"),
`CuPositionRegionConstraint` keeps the tool centre point inside a base-frame
axis-aligned box, and `CuPathConstraintSet` combines up to
`MAX_PATH_CONSTRAINTS` of them, all of which must hold. `CuSceneValidity`
carries the set and rejects violating configurations, so every planner honours
constraints through its existing validity checks; a violating start or goal
fails with the existing `InvalidStart` / `InvalidGoal` statuses. Set them
programmatically with `CuJointPlanner::set_path_constraints` (or the
task-level setter, which forwards to the planner) or with the optional
`"constraints"` key in the planner task's node config:

```ron
config: {
    "chain": "igus_rebel",
    "collision_model": "igus_rebel",
    "params": {},
    "constraints": [
        {"orientation": {"tool_axis": "z", "reference": [0.0, 0.0, 1.0], "tolerance": 0.17}},
        {"position_region": {"min": [0.1, -0.5, 0.0], "max": [0.6, 0.5, 0.8]}},
    ],
}
```

Enforcement is rejection of violating samples at the planner's discrete
checking resolution — the same sampling that checks collisions, not a
continuous guarantee. The optimizers cannot steer toward satisfaction (their
cost only sees collisions and smoothness); they keep the best satisfying path
their descent happens through and fail with `Exhausted` when it never does.
Very tight tolerances shrink the free space the planners sample from, which
makes planning slow or lets jobs fail: a cone of a few degrees or a box of a
few millimetres rejects almost every random sample. Prefer the loosest
tolerance the task allows.

Pose-goal planning applies the same set to IK solution selection: converged IK
solutions that violate the constraints are skipped like colliding ones (see
[Pose planning](pose-planning.md)). Trajectory timing does not recheck
constraints: the timed execution curve is a spline fit through the waypoints,
not the checked joint-space segments, so blending between waypoints can leave
the constrained region.

Scene decoding validates identity, counts, and geometry. Mutations fail before
changing state if the revision cannot advance. Segment collision checks remain
discrete and depend on the configured joint resolution.
