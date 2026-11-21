# Repose Agent Guidance

**Repose** (this repository, umbrella crate `repose`) is a Copper-idiomatic (`cu29`) manipulation, kinematics, collision checking, and motion planning workspace built in strict alignment with `copper-project/copper-rs`.

## Skills

Before writing or reviewing code, read the applicable `SKILL.md` under `.agents/skills/`:

| Work | Skill |
|---|---|
| Architecture / reading apps | `.agents/skills/copper-arch/SKILL.md` |
| Rust / documentation / console or UI style | `.agents/skills/copper-coding-style/SKILL.md` |
| Core / runtime constraints | `.agents/skills/copper-core-dev/SKILL.md` |
| Public traits, adapters, config policies | `.agents/skills/copper-api-flavor/SKILL.md` |
| Sources, tasks, sinks, bridges | `.agents/skills/copper-component-design/SKILL.md` |
| RON config | `.agents/skills/copper-ron-config/SKILL.md` |
| Build, checks, justfiles, releases | `.agents/skills/copper-workflow/SKILL.md` |

The user-facing skills in `skills/` (for agents that use Repose, not develop it) document the public API: a change to a public name, config key, default, or behavior they describe updates the affected `skills/*/SKILL.md` in the same PR.

Public API shape (params structs, constructors, units, visibility) follows `docs/api-conventions.md`.
Versioning, breaking-change rules, and releases follow `docs/versioning.md`; record user-visible changes under `Unreleased` in `CHANGELOG.md`.

## Non-Negotiable Copper House Rules

1. **Static over dynamic:** Encode degrees of freedom (`const N: usize`) and frame/chain topology at compile time; zero heap allocations on the real-time path.
2. **In-place `&mut` outputs (`copper-api-flavor`):** Hot-path methods write into caller-provided `&mut` buffers or `&mut Self::Output<'o>`. Never return payloads by value on the hot path.
3. **`std` / `no_std` discipline (`copper-coding-style`):** Shared crates start with `#![cfg_attr(not(feature = "std"), no_std)]`. Never use `std::` outside `#[cfg(feature = "std")]`.
4. **Physical units (`cu29::units`):** Public APIs and payloads use `cu29::units::si::f64::{Angle, AngularVelocity, Length}` rather than bare floats whose unit exists only in comments.
5. **Error handling:** Fallible operations return `CuResult<T>`. Never `unwrap()`, `expect()`, or `panic!` in library or task code.
6. **Payload & Task contracts:**
   - Payloads derive `#[derive(Default, Debug, Clone, Encode, Decode, Serialize, Deserialize, Reflect)]`.
   - `Tov` stays on the `CuMsg` envelope, never inside payload structs.
   - Tasks derive `Reflect` and implement `Freezable` + `CuTask` (or `CuAnytimeTask`).
7. **Workflow:** Follow the Git, PR and Release Workflow below. Remove generated `Cargo.lock` before a work pass if needed, run `just fmt` after edits, and run `just pr-check` before committing.

## Git, PR and Release Workflow

These rules apply to every contributor, human or AI agent, and to every tool
(Claude Code, Codex, Copilot, Cursor, T3, …). They are not optional: `master`
is protected on GitHub (PR required, `pr-check` and `semver-check` must pass,
linear history, enforced for admins), so violating them fails anyway.
Background: `CONTRIBUTING.md` and `docs/versioning.md`.

**Never**

- Commit to, push to, force-push, or reset `master`, even if asked to "commit
  and push". On `master`, branch first.
- Create, move, or push `v*` tags, or run `just release --execute`, unless
  the user explicitly asks for a release.
- Merge a PR, change branch protection or repository settings, or delete
  someone else's branch or worktree without explicit approval.
- Bypass checks (`--no-verify`, skipping `just pr-check`, disabling CI jobs).

**Every change**

1. Start a branch in its own worktree: `just wt <user>/<kind>/<description>`
   (`kind`: `feat`, `fix`, `refactor`, `docs`, `chore`, `perf`, `test`), then
   work in the printed `../repose-wt/<description>` directory. Tool-managed
   worktrees (e.g. `t3/...`) are fine. Never check out one branch in two
   worktrees.
2. Keep one focused change per branch. Rebase on `origin/master`; don't merge
   it in.
3. Run `just fmt` and `just pr-check` (fmt, clippy std/no_std/reflect, nextest,
   doctests, SemVer). All must pass before you push.
4. Record user-visible changes under `## [Unreleased]` in `CHANGELOG.md`.
   Intentional breaking changes (API, payload encoding, RON config, `cu29`
   minor, MSRV) go under `### Breaking`. That is what allows the SemVer
   check to pass, so never work around it any other way.
5. Push with `git push -u origin <branch>` and open a PR with `gh pr create`.
   The title becomes the squash commit message; the body says what changed,
   why, and how it was tested. Follow the attribution rules the user gives you.
6. Address review comments and failing checks on the same branch. PRs are
   squash-merged and the branch is deleted on merge. Afterwards, `just wt-clean`
   in the main checkout removes merged worktrees.

**Releases** (only when asked): follow the release process in
`docs/versioning.md`. The version bump lands through a release PR via
`just release <patch|minor> --execute`; after the merge, `just tag` on a clean,
up-to-date `master` tags it.

**Release cadence:** Prefer grouped patch releases over releasing every merged
PR. Small behavior-preserving fixes, proven performance improvements, tests,
and compatible additions accumulate under `Unreleased` and ship together as a
patch release once the bundle has enough user value—or sooner for an important
fix. Do not bump the version merely because another focused PR merged. Reserve
minor releases for a substantial, coherent theme, such as a meaningful new
capability, broad verified improvement, public API expansion, or an intentional
breaking change documented under `### Breaking`.

## Roadmap

See [ROADMAP.md](ROADMAP.md) for shipped capabilities (`v0.10.x`) and upcoming milestones (scene-revision execution contract, per-feature OOD envelopes and action-chatter detection, hardware presets and servo hardening, and `1.0` stabilization). Current workspace crates:

1. `components/libs/repose` (`repose`): Umbrella crate re-exporting runtime crates, `repose::planners` registry, and library examples.
2. `components/libs/repose_kinematics` (`repose-kinematics`): Serial-chain FK, space/body Jacobians, DLS & differential IK, `CuServoTask<N>`, serial-chain `inverse_dynamics` / `gravity_compensation` (`CuChainInertia<N>`), and 6-DOF igus ReBeL preset (`no_std`).
3. `components/libs/repose_collision` (`repose-collision`): Fixed-capacity capsule/sphere `CuRobotCollisionModel<N, M>`, exact signed distance, versioned `CuPlanningScene`, `CuCollisionCheckTask`, and `CuCollisionMonitor` (`no_std`).
4. `components/libs/repose_robot_description` (`repose-robot-description`): Startup URDF/SRDF import into fixed-size kinematic, collision, and inertial models (`std`).
5. `components/libs/repose_robot_interface` (`repose-robot-interface`): Vendor-neutral robot bridge channel contract and `CuSessionTracker` (`no_std`).
6. `components/libs/repose_trajectory` (`repose-trajectory`): Fixed-capacity trajectories, timing profiles (`Trapezoidal`, `Cubic`, `MinimumJerk`, `SCurve`), Cartesian TCP-zone blending, `CuFollowTrajectoryTask`, and `CuSetpointMuxTask<N>` / `CuAuditedSetpointMuxTask<N>` bumpless arbitration (`no_std`).
7. `components/tasks/repose_motion_planning` (`repose-motion-planning`): `CuJointPlanner` contract, anytime `CuJointPlannerTask`, built-in planners (Direct, RRT-Connect, PRM, A*, PathOptimizer, STOMP), `CuPosePlannerTask`, `CuCartesianPlanTask`, and `CuPathConstraintSet` (`no_std`).
8. `components/tasks/repose_joint_rrt_star` (`repose-joint-rrt-star`): Deterministic `CuRng`-seeded RRT* (`CuJointRrtStar` / `CuJointRrtStarTask`, `no_std`).
9. `components/tasks/repose_policy_rl` (`repose-policy-rl`): Host (`std`) ONNX inference (`tract-onnx`) with `CuJointReachObservationTask`, `CuPolicyTask`, `CuChunkPolicyTask`, `CuActionChunkTask` (ACT temporal ensembling / receding-horizon), observation/action `CuBoundPolicy` guardrails, `CuJointActionTask` (`CuJointActionReport`), `CuCartesianActionTask`, and `CuGripperActionTask`.
10. `components/tasks/repose_safety` (`repose-safety`): Real-time safety shield (`CuSafetyShieldTask<N, M>`, `CuSafetyReport`) enforcing joint limits, self/scene collisions, `CuWorkspaceBounds` TCP fences, `max_tcp_speed`, and `braking_horizon` lookahead (`no_std`).
11. `components/tasks/repose_controller` (`repose-controller`): `CuControllerTask<N>` with position, velocity, PD effort, and inverse-dynamics effort modes, per-joint gains, gravity-compensated hold brake, tracking watchdog, and rate limiting (`no_std`).
12. `components/libs/repose_incident` (`repose-incident`): Host (`std`) black-box incident detector (`CuIncidentDetector`, `CuCycleSample`), windowed `CuIncidentRecord` / `CuIncidentSummary` JSON manifests, deterministic `CuIncidentBlame` root-cause attribution, and `CuCheckpointDiff` counterfactual regression gating.
13. `components/tasks/repose_io` (`repose-io`): Digital IO payloads and RON-configured `CuGripperTask` (`no_std`).
14. `components/bridges/repose_mujoco_sim` (`repose-mujoco-sim`): Generic `CuMujocoSimBridge<N>` TCP/JSON `CuBridge` to a MuJoCo sidecar (`std`). External hardware bridges (e.g. `repose-igus-cri`) live in `ReposeRobotics/repose-bridges`.
