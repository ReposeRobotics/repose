# Changelog

All notable changes to Repose. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versioning
follows [docs/versioning.md](docs/versioning.md). Pre-1.0, every minor release
may be breaking.

## [Unreleased]

### Breaking

- `repose-controller`: effort-mode gains are per joint. `CuEffortGains` is now
  `CuEffortGains<N>` with `kp: [f64; N]` and `kv: [f64; N]`, and
  `CuControllerParams` is now `CuControllerParams<N>` (it also gains the
  `effort_model` field). Build uniform gains with
  `CuEffortGains::uniform(kp, kv)`; the defaults are unchanged (`kp = 20.0`,
  `kv = 0.8` on every joint). RON configs keep working: a single number for
  `kp` or `kv` still means that gain on every joint. The controller's frozen
  state now stores `2 * N` gain values instead of two, so keyframes of
  `CuControllerTask` written by earlier versions are not supported (frozen
  layouts are not versioned, so thawing one is not detected). See
  [docs/migration-0.9.md](docs/migration-0.9.md).

- `repose-trajectory`: `CuTrajectoryFollowOutput<N>` gains an
  `accelerations: [CuJointAcceleration; N]` field (between `velocities` and
  `status`), so the follower setpoint payload encoding changes. Code that
  builds the struct in full must set it (`CuJointAcceleration::from_coordinate`
  per joint, or `..Default::default()`); logs of the setpoint written by
  earlier versions no longer decode. See
  [docs/migration-0.9.md](docs/migration-0.9.md).

- `repose-mujoco-sim`: `CuMujocoSimBridge` is now `CuMujocoSimBridge<const N: usize>`
  (and `MujocoTxChannels<N>` / `MujocoRxChannels<N>`) instead of being fixed at
  6 joints; `MUJOCO_DOF` is removed. The joint count comes from the type, so
  RON graphs must write it explicitly. Migrate
  `type: "repose_mujoco_sim::CuMujocoSimBridge"` to
  `type: "repose_mujoco_sim::CuMujocoSimBridge<6>"` (any `N` matching the
  `CuJointCommandPayload<N>` / `CuJointStatePayload<N>` channel `msg:` types);
  Rust callers use `CuMujocoSimBridge::<6>::from_params(..)`. `N = 0` is
  rejected. Wire protocol: command lines gain an additive `"n"` field and state
  lines may carry `"n"`; a 6-joint bridge still works with a sidecar written for
  the old protocol, so the handshake is not versioned. A joint-count mismatch
  between bridge and sidecar is now a `CuError` naming both counts, and a
  sidecar may answer a rejected command with
  `{"type":"error","message":"..."}`, which the bridge reports as a `CuError`.
  The example sidecar serves every joint of the URDF (`joint1`..`jointN`).

### Added

- `repose-trajectory`: `CuFollowTrajectoryTask` fills the new setpoint
  accelerations from the active interpolation, analytically and without
  allocation: the derivative of the emitted velocity (constant per segment for
  linear, the Hermite second derivative for cubic, the quintic's acceleration
  for quintic). Held setpoints (idle, before the start, from the end, single
  point) report zero; at an interior knot the segment ending there is used.
  The same sampling is public as `CuTrajectoryPayload::sample_with_acceleration`.
- `repose-controller`: model-based effort mode. With the optional
  `effort_model` parameter (`CuEffortModel<N>`: a chain preset or inline
  chain, its `CuChainInertia<N>`, and a gravity vector), effort mode adds
  inverse-dynamics feedforward to the PD feedback: the model is evaluated at
  the measured positions and the desired velocities and accelerations of the
  follower setpoint (gravity, Coriolis/centrifugal and inertial load). A
  non-finite or wrongly typed desired acceleration faults the controller and
  falls back to the brake; without a model it is ignored. Hold and fault paths add gravity compensation at the measured
  positions to the damping brake, so the arm no longer sags. A model that
  cannot be evaluated faults the controller and falls back to the damping
  brake. Without a model, effort mode is the same PD law as before.
- `repose-controller`: per-joint effort gains in RON, as an array of `N`
  numbers per gain (`"effort_gains": {"kp": [60.0, 60.0, 40.0, 8.0, 8.0, 4.0], "kv": 0.8}`).

## [0.8.4] - 2026-11-10

### Added

- Rigid-body dynamics in `repose-kinematics`: `CuLinkInertia` and
  `CuChainInertia<N>` hold validated per-link mass, centre of mass, and
  rotational inertia; `CuKinematicChain::inverse_dynamics` computes joint
  torques/forces by recursive Newton-Euler into a caller-provided
  `[CuJointEffort; N]`; `CuKinematicChain::gravity_compensation` returns the
  holding torques via the same recursion. Inertia tensors must satisfy the
  triangle inequality on their principal moments; `CuChainInertia` validates
  on construction and on deserialization, and `inverse_dynamics` validates it
  again on every call. `CuJointEffort` (torque/force/fixed)
  joins the joint-typed units family. URDF `<inertial>` elements import into
  `CuRobotDescription::chain_inertia`, validating each link and then folding
  fixed-joint links into their owner's rigid body. Invalid or malformed inertial
  data is reported by `chain_inertia()`; kinematics and collision import
  continue to accept these descriptions.
- Simulator-free library examples in the `repose` umbrella crate, each
  runnable with `cargo run -p repose --example <name>` on the built-in
  6-DOF igus ReBeL arm: `kinematics` (FK, then IK back to the same pose),
  `collision` (one free and one colliding configuration with clearance),
  and `planning` (a seeded RRT-Connect plan around a blocking obstacle,
  timed into a trajectory). Each example checks its own result and prints
  deterministic output; `just examples` runs all three and a CI job runs
  it.
- `docs/library-examples.md`: what the simulator-free examples show, how
  to choose the const generic capacities, and how to read the compile
  errors when they mismatch.
- Path constraints for motion planning (`repose-motion-planning`):
  `CuOrientationConstraint` keeps a tool-frame axis within a cone angle of a
  base-frame reference direction, `CuPositionRegionConstraint` keeps the tool
  centre point inside a base-frame box, and `CuPathConstraintSet` combines up
  to `MAX_PATH_CONSTRAINTS` of them (all must hold). `CuSceneValidity`
  optionally carries a set and rejects violating configurations, so every
  planner — RRT-Connect, PRM, A*, STOMP, the gradient optimizer, direct, and
  RRT* — honours constraints through its existing validity checks with no
  per-planner logic; violating starts and goals fail with the existing
  `InvalidStart` / `InvalidGoal` statuses. Set them programmatically with the
  new `CuJointPlanner::set_path_constraints` / `clear_path_constraints`
  methods (forwarded by the planner and pose-planner tasks). Successfully
  setting or clearing constraints discards the current job and its cached
  result; a new request is required before planning resumes. Set constraints
  at construction with the new
  optional `"constraints"` key in the planner task's node config. Pose-goal
  planning applies the same set to IK solution selection. Enforcement is
  rejection of violating samples at the planner's discrete checking
  resolution; very tight tolerances slow planning down or fail jobs.

### Changed

- Reworked the README into a short branded landing page. The detailed
  reference sections moved unchanged into `docs/robot-configuration.md`,
  `docs/motion-planning.md`, `docs/trajectories.md`, and
  `docs/benchmarks.md`.
- Reworked the README's "Quick start" so the easiest paths come first:
  try Repose as a library (dependency lines, the three simulator-free
  example commands, and the kinematics core inline), generate a robot app
  from the template, and finally the full Copper RON and MuJoCo
  simulator pipeline.

## [0.8.3] - 2026-11-08

### Fixed

- The `trajectory` benchmark requires the `std` feature again; adding the
  `monitor` benchmark had moved its `required-features` line.
- Set the git identity inside the benchmark publish repo so the first
  `benchmark-data` publication can commit.

### Added

- Execution collision monitoring: `CuCollisionMonitor` and
  `CuMonitorTransition` in `repose-collision` track clear/in-collision
  transitions over a checked configuration, so executors raise a hold once per
  collision episode instead of reacting every cycle. Allocation-free policy
  layer over the existing check; `CuCollisionMonitor::update` is `#[must_use]`
  and the monitor is `Copy`. A `monitor` Divan benchmark measures its overhead
  against the raw check.
- Determinism check: `just determinism` runs FK, IK, collision checks, every
  built-in planner with a fixed seed, and trajectory timing and sampling in
  separate processes, and fails if any output hash differs. A `determinism`
  CI job runs it on Linux and macOS.
- Bare-metal build check: `just bare-metal` builds every library crate except
  `repose-mujoco-sim` with no operating system for `aarch64-unknown-none` and
  `thumbv7em-none-eabihf`. A `bare-metal` CI job runs it.
- Manual `Benchmarks (wall-clock)` GitHub Actions workflow running
  `cargo bench -p repose` on Linux, macOS, and Windows (x64 and ARM64)
  GitHub-hosted runners, uploading per-system results as artifacts.
- Benchmark data pipeline: the wall-clock workflow now runs weekly, converts
  Divan output to per-system JSON (`bench_to_json.py`), and publishes a merged
  `benchmarks.json` to the `benchmark-data` branch for the website graph.

### Changed

- Skip redundant endpoint collision checks in joint-space segment evaluation
  via `CuSceneValidity::is_segment_interior_free` at call sites with
  already-proven-free endpoints. Boolean-exact given free endpoints;
  deterministic paths and seeds unchanged.

## [0.8.2] - 2026-11-04

### Changed

- Reuse kinematics traversal in `CuKinematicChain::inverse_kinematics_dls` by
  combining Jacobian and pose computation into a single pass on unconverged
  iterations, while preserving the cheap forward kinematics check for
  already-converged seeds.

## [0.8.1] - 2026-11-04

### Changed

- Skip Euler trigonometry when a joint origin has an exactly zero rotation,
  preserving the constructor's matrix bits, including signed zeros.

## [0.8.0] - 2026-11-04

### Breaking

- Add `CuPlanStatus::NoIkSolution` for bounded pose-goal IK failure. Exhaustive
  status matches must handle the new variant; existing joint planning requests
  and paths keep their fields and existing enum discriminants.

### Added

- Pose-goal planning through `CuPoseGoalTask` and `CuPosePlannerTask`, with
  seeded bounded IK retries and any existing joint planner.
- `CuCartesianPlanTask` accepts TCP poses and builds collision-checked straight
  Cartesian moves with trajectory timing. Both planning paths use the request's
  scene snapshot, attached shapes, exclusions, and scene revision. Cartesian
  results preserve that metadata and expose explicit failure statuses;
  `CuCartesianTrajectoryTask` connects successful results to the follower.

- README Benchmarks section: CodSpeed methodology and scenarios (igus ReBeL
  6-DOF, 11-obstacle collision scene, anytime planner jobs, 5-waypoint
  trajectory), a local wall-clock median snapshot (Apple M4, rustc 1.99.0,
  2026-11-04), and a link to the live CodSpeed dashboard.
- CodSpeed continuous benchmarking for `repose-kinematics` inverse kinematics:
  Divan benches (`benches/ik.rs`) for DLS IK, pseudoinverse IK, and
  differential IK on the igus ReBeL chain, run by a dedicated
  `.github/workflows/codspeed.yml` workflow in simulation mode. The harness is
  a dev-dependency only, so the `no_std` library surface is unchanged.
- ClusterFuzzLite (OSS-Fuzz) fuzzing for `repose-kinematics`: a cargo-fuzz
  target (`fuzz/fuzz_targets/ik.rs`) driving FK, Jacobians, manipulability,
  differential IK, and DLS/pseudoinverse IK with bounded inputs and
  finite-output/joint-limit oracles. PR fuzzing (`.github/workflows/cflite_pr.yml`)
  and weekly batch fuzzing (`.github/workflows/cflite_batch.yml`) build from
  `.clusterfuzzlite/`. The fuzz crate is detached from the workspace and
  nightly-only, so it never affects the library builds.

### Changed

- Correct the SemVer recipe to use the checker's breaking API category for
  documented pre-1.0 breaking changes, without changing crate versions.
- Umbrella crate doc example now points at the `v0.8.0` tag.

## [0.7.1] - 2026-11-02

### Added

- `repose::planners`: the single registry of joint planners.
  `CuPlannerChoice` names every planner (RON spelling: `rrt_connect`,
  `rrt_star`, `prm`, `a_star`, `stomp`, `path_optimizer`, `direct`) and
  `CuAnyPlanner<N, M>` is a `CuJointPlanner` that runs the chosen one (directly, or in `CuJointPlannerTask` by name). Adding a
  planner means implementing the trait and adding one variant there.

- `repose-robot-interface` (re-exported as `repose::robot_interface`): the
  vendor-neutral robot bridge contract. It documents the standard channel set
  (`joint_state`, `joint_command`, `session`, `session_state`) and adds the
  session lifecycle payloads `CuSessionCommand` (acquire/release control,
  clear faults, enable/disable drives) and `CuSessionState`, plus
  `CuSessionTracker`, the protocol-independent session rules every hardware
  bridge shares (one command in flight, refuse while not in control,
  sequence-matched answers, wind-down order). `no_std`.

### Changed

- `master` is protected; changes land through squash-merged PRs from
  per-branch worktrees (`just wt`, `just wt-clean`). Releases bump the version
  in a PR and are tagged afterwards with `just tag`. CI runs the SemVer check
  on PRs; breaks are allowed once listed under `### Breaking` in `Unreleased`.

## [0.7.0] - 2026-11-01

### Breaking

- Crates renamed for the Repose rebrand: `cu-repose` → `repose`, and every
  `cu-<name>` crate → `repose-<name>` (Rust paths `cu_<name>` →
  `repose_<name>`, including type paths in RON configs). Copper's own crates
  (`cu29`, `cu-spatial-payloads`, …) are unchanged. Update dependencies and
  `use` paths; nothing else changed.

### Changed

- Versioning policy documented in `docs/versioning.md`; release notes
  consolidated into this changelog (older notes under `docs/releases/`).
- `just pr-check` runs `cargo-semver-checks`; `just release` wraps
  `cargo-release` with one shared version and `vX.Y.Z` tags.

## [0.6.0] - 2026-11-01

Runtime reflection works again, and the MuJoCo example is a RON-only graph.
Full notes: [docs/releases/0.6.0.md](docs/releases/0.6.0.md).

## [0.5.0] - 2026-10-31

Breaking API simplification across the workspace; see
[docs/migration-0.5.md](docs/migration-0.5.md).
Full notes: [docs/releases/0.5.0.md](docs/releases/0.5.0.md).

## [0.4.1] - 2026-10-31

## [0.4.0] - 2026-10-31

Planner-agnostic motion planning: sampling-based and non-sampling planners
share one contract and one Copper task.
Full notes: [docs/releases/0.4.0.md](docs/releases/0.4.0.md).

## [0.3.0] - 2026-10-30

## [0.2.3] - 2026-10-30

## [0.2.2] - 2026-10-30

## [0.2.1] - 2026-10-30

## [0.2.0] - 2026-10-30

## [0.1.0] - 2026-10-29

Initial release.

[Unreleased]: https://github.com/ReposeRobotics/repose/compare/v0.8.4...HEAD
[0.8.4]: https://github.com/ReposeRobotics/repose/compare/v0.8.3...v0.8.4
[0.8.3]: https://github.com/ReposeRobotics/repose/compare/v0.8.2...v0.8.3
[0.8.2]: https://github.com/ReposeRobotics/repose/compare/v0.8.1...v0.8.2
[0.8.1]: https://github.com/ReposeRobotics/repose/compare/v0.8.0...v0.8.1
[0.8.0]: https://github.com/ReposeRobotics/repose/compare/v0.7.1...v0.8.0
[0.7.1]: https://github.com/ReposeRobotics/repose/compare/v0.7.0...v0.7.1
[0.7.0]: https://github.com/ReposeRobotics/repose/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/ReposeRobotics/repose/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/ReposeRobotics/repose/compare/v0.4.1...v0.5.0
[0.4.1]: https://github.com/ReposeRobotics/repose/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/ReposeRobotics/repose/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/ReposeRobotics/repose/compare/v0.2.3...v0.3.0
[0.2.3]: https://github.com/ReposeRobotics/repose/compare/v0.2.2...v0.2.3
[0.2.2]: https://github.com/ReposeRobotics/repose/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/ReposeRobotics/repose/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/ReposeRobotics/repose/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/ReposeRobotics/repose/releases/tag/v0.1.0
