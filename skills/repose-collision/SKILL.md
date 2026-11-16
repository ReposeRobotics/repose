---
name: repose-collision
description: >-
  Collision checking with `repose-collision`: sphere/capsule robot models, the
  `CuPlanningScene` of obstacles and attached objects with stable IDs and revisions,
  `CuSceneExclusions`, `CuCollisionReport` (contacts, signed distance / clearance),
  segment checks, the `CuCollisionCheckTask`, and `CuCollisionMonitor` for execution
  holds. Use when adding obstacles, attaching a grasped object, checking a configuration
  or a joint-space segment, or interpreting a collision report.
---

# Repose: collision checking

Geometry is **spheres and capsules only** (`CuSphere`, `CuCapsule`), with
exact signed distances. Approximate boxes and meshes with several of them.
Lengths are typed (`Length`), in meters.

## Check one configuration

```rust
use cu29::units::si::{f64::Length, length::{meter, millimeter}};
use repose::collision::{CuCollisionEnv, CuCollisionReport, CuCollisionShape,
                        CuPlanningScene, CuSphere, igus_rebel_collision_model};
use repose::kinematics::igus_rebel_chain;

let chain = igus_rebel_chain();
let model = igus_rebel_collision_model();

// Default capacities (32 obstacles, 8 attached shapes); annotate the type when
// nothing else pins them.
let mut scene: CuPlanningScene = CuPlanningScene::new();
let m = Length::new::<meter>;
let ball = CuSphere::new([m(0.4), m(0.0), m(0.3)], m(0.1))?;
let ball_id = scene.add_obstacle(CuCollisionShape::Sphere(ball), None)?; // None → auto ID
let env = scene.collision_env();          // borrowed view, no copy

let mut report = CuCollisionReport::<16>::default();   // stores ≤16 contacts
model.check_configuration(&chain, &q, &env, &mut report)?;
if report.is_collision_free() {
    println!("clearance {:.1} mm", report.min_signed_distance().get::<millimeter>());
} else {
    println!("{} contacts", report.total_contacts()); // counts even unstored contacts
}
```

- Self-collision only: pass `&CuCollisionEnv::EMPTY`.
- Boolean shortcut: `model.is_configuration_collision_free(&chain, &q, &env)?`.
- Contacts (`CuCollisionContact`): `kind` (`SelfCollision` /
  `EnvironmentCollision`), `link_index` (robot shape index), `partner_index`,
  `signed_distance` (negative = penetration).
- The model's `safety_margin` inflates robot shapes; a positive clearance is
  the distance beyond that margin.

## Check a joint-space segment

```rust
let free = model.is_segment_collision_free(&chain, &q_a, &q_b, Angle::new::<degree>(2.0), &env)?;
```

This samples the straight joint line at the given resolution. It is
**discrete**, not a swept-volume guarantee: pick a resolution small enough for
your thinnest obstacle.

## The planning scene

`CuPlanningScene<MAX_OBSTACLES = 32, MAX_ATTACHED = 8>` is a fixed-capacity,
serializable payload:

- `add_obstacle(shape, id)`, `update_obstacle(id, shape)`, `remove_obstacle(id)`;
  IDs are `CuObjectId` (`CuObjectId::new(n)`, non-zero), stable across edits.
- Attached objects (a grasped part) live on a robot frame in link-local
  coordinates: `add_attached(frame, shape, id)`, `attach(id, frame, local_shape)`
  (obstacle → attached), `detach(id, world_shape)` (attached → obstacle).
  `frame` is the zero-based FK frame index, `None` for the base.
- Every successful mutation bumps `scene.revision()`. Use
  `scene.transaction(|s| { … })` to apply several edits atomically (all or none).
- Over capacity → `Err`, never a silent drop.

## Exclusions (allowed contacts)

A grasped object touches the gripper; tell the checker that is intended:

```rust
use repose::collision::{CuSceneExclusion, CuSceneExclusionPartner, CuSceneExclusions};

let mut excl: CuSceneExclusions = CuSceneExclusions::new();
excl.add(CuSceneExclusion {
    attached_id: part_id,
    partner: CuSceneExclusionPartner::RobotShape { shape_index: 5 }, // or Attached / Obstacle
})?;
```

Exclusions reference stable IDs (and robot shape indices); an invalid
reference rejects the whole query. Publish scene and exclusions together.

## Copper tasks

`repose_collision::CuCollisionCheckTask<N, M = N, MAX_OBS = 32, MAX_ATTACHED = 8, MAX_CONTACTS, MAX_EXCL>`

- Inputs: `CuJointPositionsPayload<N>`, `CuPlanningScene<MAX_OBS, MAX_ATTACHED>`,
  `CuSceneExclusions<MAX_EXCL>`.
- Output: `CuCollisionReport<MAX_CONTACTS>`.
- Config: `{"chain": …, "collision_model": …}` (both required).

`CuCollisionMonitor` turns per-cycle results into edges:
`monitor.update(in_collision)` returns a `CuMonitorTransition`, so you raise a
hold once when a violation starts instead of every cycle.

## Pitfalls

- Planners and the pose planner snapshot the scene **once per job** and return
  the `scene_revision` they used; before executing a plan, compare it with the
  current scene revision and replan if it changed.
- `CuCollisionReport<K>` stores at most `K` contacts; `total_contacts()` still
  counts all of them. Size `K` for what you need to inspect, not for safety.
- Collision checks of a timed trajectory only cover the sampled points; see
  `repose-trajectory-and-control`.
