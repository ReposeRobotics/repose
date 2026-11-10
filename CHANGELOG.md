# Changelog

All notable changes to Repose. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versioning
follows [docs/versioning.md](docs/versioning.md). Pre-1.0, every minor release
may be breaking.

## [Unreleased]

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
  fixed-joint links into their owner's rigid body.
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
