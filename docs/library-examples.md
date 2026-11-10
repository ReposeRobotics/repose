# Library examples

Three simulator-free examples in `components/libs/repose/examples/` show the
stack as plain library calls on the built-in 6-DOF igus ReBeL arm: no Copper
task graph, no network, no extra files. Each one checks its own result and
exits nonzero if the check fails, and each prints the same output on every
run. CI runs all three (`just examples`); `cargo clippy --all-targets` keeps
them compiling.

| Example | Run | Shows |
|---|---|---|
| `kinematics` | `cargo run -p repose --example kinematics` | FK to a TCP pose, then DLS IK from another seed back to it |
| `collision` | `cargo run -p repose --example collision` | A one-sphere scene, one free and one colliding configuration, clearance |
| `planning` | `cargo run -p repose --example planning` | A seeded RRT-Connect plan around a blocking sphere, timed into a trajectory |

## Capacities are const generics

Joint counts and buffer sizes are compile-time constants, so the hot path
never allocates. The numbers in angle brackets are:

| Type | Parameters | Example value |
|---|---|---|
| `CuKinematicChain<N>`, `CuAnyPlanner<N>` | `N`: joint count | `6` for the igus ReBeL |
| `CuCollisionReport<MAX_CONTACTS>` | contacts stored | `16`; further contacts still count toward `total_contacts()` but are not stored |
| `CuPlanningScene<MAX_OBSTACLES, MAX_ATTACHED>` | scene capacities | defaults: 32 obstacles, 8 attached shapes |
| `CuJointPlanPath<N>` | `N`: joint count | holds `MAX_WAYPOINTS` (32) waypoints; a longer path reports `CuPlanStatus::CapacityExceeded` |
| `CuTrajectoryPayload<N, MAX_POINTS>` | joints, timed samples | `<6, 1024>`; timing past capacity reports `CuTimingStatus::CapacityExceeded` |

A bare `CuPlanningScene::new()` needs a type annotation
(`let mut scene: CuPlanningScene = ...`) because nothing else constrains its
capacities; the request-owned scene inside `CuJointPlanRequest` does not.

When a joint count mismatches, the compiler says which size it wanted:

```
error[E0308]: mismatched types
 --> src/main.rs:9:30
  |
9 |     chain.forward_kinematics(&q, &mut tcp)?;
  |           ------------------ ^^ expected an array with a size of 6, found one with a size of 5
```

Fix the array size (or the `N` on the type), not the call.

## Seeds

Sampling planners derive all randomness from the `seed` passed to `begin`, so
the same seed replays the same job bit for bit. That is what Copper's
record-and-replay relies on, and it is why the planning example prints
identical output on every run. Deterministic planners ignore the seed.
