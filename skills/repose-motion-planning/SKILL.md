---
name: repose-motion-planning
description: >-
  Collision-free motion planning with Repose: the `CuJointPlanner` trait (begin / step /
  write_best), the seven built-in planners and how to choose one (`CuPlannerChoice`,
  `CuAnyPlanner`), planner tasks and their RON config, path constraints, pose goals via
  IK (`CuPosePlannerTask`), straight TCP lines (`CuCartesianPlanTask`), plan statuses,
  seeds, and scene revisions. Use when planning a move, choosing or tuning a planner,
  wiring a planner task, or debugging a failed or stale plan.
---

# Repose: motion planning

## The contract

Every planner implements `CuJointPlanner<N, M>`:

```rust
use repose::collision::{CuCollisionShape, CuSphere, igus_rebel_collision_model};
use repose::kinematics::igus_rebel_chain;
use repose::motion_planning::{CuJointPlanPath, CuJointPlanRequest, CuJointPlanner};
use repose::planners::{CuAnyPlanner, CuPlannerChoice};

let mut request = CuJointPlanRequest { start, goal, ..Default::default() };
request.scene.add_obstacle(CuCollisionShape::Sphere(ball), None)?;   // scene travels with the request

let mut planner = CuAnyPlanner::<6>::new(igus_rebel_chain(), igus_rebel_collision_model(),
                                         CuPlannerChoice::RrtConnect)?;
let mut status = planner.begin(&request, /* seed */ 1)?;
let mut steps = 0;
while !status.is_final() && steps < 400 {      // always bound the loop
    status = planner.step(64)?;                 // bounded quantum of work
    steps += 1;
}
let mut path = CuJointPlanPath::<6>::default();
let status = planner.write_best(&mut path);
if !status.has_path() {
    return Err(CuError::from(format!("planning failed: {status:?}")));
}
for wp in path.waypoints() { /* wp.positions: [CuJointPosition; 6] */ }
```

To use one planner with its own params instead of the registry, construct it
directly: `CuJointRrtConnect::<6>::new(chain, model, CuJointRrtConnectParams { … })`
(same trait).

`CuJointPlanPath<N>` holds at most `MAX_WAYPOINTS` (32) waypoints, plus
`len`, `cost` (dimensionless), `status`, `scene_revision`, and the
`exclusions` it was planned with.

## Statuses (`CuPlanStatus`)

| Status | Meaning |
|---|---|
| `Searching` | No path yet; keep stepping |
| `Improving` | Has a path, still refining (anytime) |
| `Converged` | Done, path final |
| `Exhausted` | Gave up (node/iteration budget spent, or local optimizer stuck) |
| `InvalidStart` / `InvalidGoal` | Endpoint in collision, off limits, or violating constraints |
| `InvalidRequest` | Bad request, or no job (e.g. after changing constraints) |
| `CapacityExceeded` | Path would exceed `MAX_WAYPOINTS` |
| `NoIkSolution` | Pose goal: bounded IK attempts all failed |

`has_path()` is true for `Improving` and `Converged`. Failures are published
as a path with zero waypoints and the failure status — handle them.

## Choosing a planner

RON / `CuPlannerChoice` name → task alias:

| Name | Task type | Use when |
|---|---|---|
| `direct` | `repose_motion_planning::CuJointDirectTask<N>` | Free straight joint line expected; one shot, fails if blocked |
| `rrt_connect` | `repose_motion_planning::CuJointRrtConnectTask<N>` | Default: fast feasible path in clutter |
| `rrt_star` | `repose_joint_rrt_star::CuJointRrtStarTask<N>` | Shorter paths; anytime refinement with rewiring/shortcutting |
| `prm` | `repose_motion_planning::CuJointPrmTask<N>` | Roadmap; improving paths in a static scene |
| `a_star` | `repose_motion_planning::CuJointAStarTask<N>` | Deterministic, resolution-complete on a lattice; few joints/coarse resolution |
| `stomp` | `repose_motion_planning::CuJointStompTask<N>` | Smooth paths with clearance; local, seeded rollouts |
| `path_optimizer` | `repose_motion_planning::CuJointPathOptimizerTask<N>` | Smooth CHOMP-style refinement; local, can get stuck behind big obstacles |

Optimizers are local: when they report `Exhausted` behind a large obstacle,
switch to a sampling or search planner. Sampling planners and STOMP use the
seed; the others ignore it.

## Planner task (RON)

All planner tasks share one config, `CuJointPlannerParams<P, N, M>`:

```ron
resources: [
    (id: "planner_rng", provider: "cu_rng::CuRngBundle", config: {"seed": 42}),
],
tasks: [
    (
        id: "planner",
        type: "repose_motion_planning::CuJointRrtConnectTask<6>",
        background: true,                                   // planning off the control thread
        anytime: (time_budget_ms: 250.0, quality_target: 0.95),
        resources: {"rng": "planner_rng.rng"},              // required by every planner task
        config: {
            "chain": "igus_rebel",
            "collision_model": "igus_rebel",
            "params": {"step_size": 0.3, "max_nodes": 512}, // planner's Cu…Params; omit for defaults
            "base_budget": 256,                             // optional
            "refine_budget": 64,                            // optional
            "constraints": [                                // optional
                {"orientation": {"tool_axis": "z", "reference": [0.0, 0.0, 1.0], "tolerance": 0.17}},
                {"position_region": {"min": [0.1, -0.5, 0.0], "max": [0.6, 0.5, 0.8]}},
            ],
        },
    ),
],
```

- Input `CuJointPlanRequest<N>`, output `CuJointPlanPath<N>`.
- `CuJointGoalTask<N>` (`config: {"goal": [{"revolute": 0.4}, …]}`) turns the
  measured `CuJointStatePayload<N>` into a request.
- `CuPlanToTrajectoryTask<N, CAP>` times each final path (see
  `repose-trajectory-and-control`).
- Registry from RON: `CuJointPlannerTask<repose::planners::CuAnyPlanner<6>, 6>`
  with `"params": "rrt_star"`.
- Planner param keys (`step_size`, `goal_threshold`, `rewire_radius`,
  `connection_radius`, `resolution`) are dimensionless joint-space distances
  under `metric` (default 1 rad ≡ 1 m); `collision_check_step` sets the edge
  check resolution.

## Path constraints

`CuOrientationConstraint` (tool axis within a cone of a base-frame direction)
and `CuPositionRegionConstraint` (TCP inside a base-frame box), combined in a
`CuPathConstraintSet` (≤ `MAX_PATH_CONSTRAINTS`). Set in RON as above or with
`planner.set_path_constraints(&set)?`.

- Setting or clearing constraints **discards the current job**; call `begin`
  again.
- Enforcement is by rejection at the discrete check resolution; very tight
  tolerances make planning slow or fail. Use the loosest tolerance the task
  allows.
- Trajectory timing does not recheck constraints; the timed spline can leave a
  tight region between waypoints.

## Pose goals and straight lines

Both take `CuPosePlanRequest<N>` (measured start, `CuTcpPose` goal in the base
frame with typed translation and XYZ-Euler rotation, scene, exclusions).
`CuPoseGoalTask<N>` builds one from joint state and a fixed config goal:

```ron
(id: "goal", type: "repose_motion_planning::CuPoseGoalTask<6>",
 config: {"goal": {"translation": [0.25, 0.0, 0.30], "rotation": [0.0, 0.0, 0.0]}}),
```

- **Any path to a pose:**
  `repose_motion_planning::CuPosePlannerTask<repose_motion_planning::CuJointRrtConnect<6>, 6>`
  — IK (seeded from the start, then `goal.attempts` seeded samples, default 8,
  max 64; first collision-free, constraint-satisfying solution wins) → joint
  planner → `CuJointPlanPath<N>`. Config nests the planner config:
  `{"planner": {"chain": …, "collision_model": …, "params": {}}, "goal": {"attempts": 8}}`.
  Needs the `rng` resource.
- **Straight TCP line:** `repose_motion_planning::CuCartesianPlanTask<6, 256, 1024>`
  (`PATH` geometric samples, `CAP` trajectory points) → `CuCartesianPlanResult`;
  then `CuCartesianTrajectoryTask<6, 1024>` extracts a `CuTrajectoryPayload`.
  Fails when the line is blocked — it never detours. Config:
  `chain`, `collision_model`, `goal`, `path` (`translation_step`,
  `rotation_step`), `timing` (see `docs/pose-planning.md`).

## Correctness rules

- **Bound every loop** over `step`; an unreachable goal can search until
  `Exhausted` only after its node budget.
- **Seeds:** pass a fixed seed (or the `cu_rng` resource) for replayable plans.
- **Stale scenes:** a plan is valid only for the scene it was planned against.
  Compare `path.scene_revision` (and `path.exclusions`) with the current scene
  before executing; replan if they differ.
- Collision checks along edges are discrete (`collision_check_step`), not
  swept-volume.
