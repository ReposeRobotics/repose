# Changelog

All notable changes to Repose. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versioning
follows [docs/versioning.md](docs/versioning.md). Pre-1.0, every minor release
may be breaking.

## [Unreleased]

### Breaking

- Add `CuPlanStatus::NoIkSolution` for bounded pose-goal IK failure. Exhaustive
  status matches must handle the new variant; existing joint planning requests
  and paths keep their fields and existing enum discriminants.

### Changed

- Correct the SemVer recipe to use the checker's breaking API category for
  documented pre-1.0 breaking changes, without changing crate versions.

### Added

- README Benchmarks section: CodSpeed methodology and scenarios (igus ReBeL
  6-DOF, 11-obstacle collision scene, anytime planner jobs, 5-waypoint
  trajectory), a local wall-clock median snapshot (Apple M4, rustc 1.99.0,
  2026-11-04), and a link to the live CodSpeed dashboard.

- Pose-goal planning through `CuPoseGoalTask` and `CuPosePlannerTask`, with
  seeded bounded IK retries and any existing joint planner.
- `CuCartesianPlanTask` accepts TCP poses and builds collision-checked straight
  Cartesian moves with trajectory timing. Both planning paths use the request's
  scene snapshot, attached shapes, exclusions, and scene revision. Cartesian
  results preserve that metadata and expose explicit failure statuses;
  `CuCartesianTrajectoryTask` connects successful results to the follower.

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

- Umbrella crate doc example now points at the `v0.7.1` tag.

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

[Unreleased]: https://github.com/ReposeRobotics/repose/compare/v0.7.1...HEAD
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
