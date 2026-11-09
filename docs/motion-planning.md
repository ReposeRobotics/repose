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

Scene decoding validates identity, counts, and geometry. Mutations fail before
changing state if the revision cannot advance. Segment collision checks remain
discrete and depend on the configured joint resolution.
