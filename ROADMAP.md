# Repose Roadmap

## Purpose & Positioning

**Repose** is a deterministic, real-time manipulation and Physical AI safety stack
for [`copper-rs` (`cu29`)](https://github.com/copper-project/copper-rs). It
targets both classical plan-and-execute robot arm manipulation and safe
deployment of learned RL, ACT, Diffusion, and VLA policies behind deterministic
Simplex fallback and black-box incident replay.

Design invariants enforced across the workspace:

- **Static topology, zero hot-path allocation:** Joint counts (`const N: usize`)
  and capacities are compile-time constants; library and control paths write into
  caller-provided `&mut` buffers without heap allocation (host ONNX inference in
  `repose-policy-rl` is the single bounded, measured exception).
- **`no_std` core, safety, and control:** Shared library and task crates build
  without `std`; every non-simulator, non-ONNX, non-host-only library crate
  (excluding `repose-mujoco-sim`, `repose-policy-rl`, and `repose-incident`) is
  verified on bare-metal Arm (`aarch64-unknown-none`) and Cortex-M
  (`thumbv7em-none-eabihf`) targets.
- **Typed physical units:** Public APIs and payloads use `cu29::units` SI types
  (`Angle`, `AngularVelocity`, `Length`, `Time`, `Torque`, `Force`) rather than
  untyped floats.
- **Deterministic record-and-replay:** Tasks derive `Reflect` and implement
  `Freezable` with fixed-size snapshots; sampling planners draw randomness from
  explicit per-job `CuRng` seeds; every setpoint, observation, action, and
  controller output is captured in cycle-exact `.copper` logs.

## Shipped Capabilities (Built — through `v0.10.x`)

All 14 workspace crates share one synchronized workspace version (`v0.10.0` on
crates tagged from `master`) and are tested under `std`, `no_std`, and
`cu29/reflect`.

| Crate | Status | Shipped Capabilities |
|---|---|---|
| `repose` | Shipped | Umbrella re-exports of runtime crates (`mujoco-sim` feature-gated), `repose::planners` registry (`CuPlannerChoice`, `CuAnyPlanner`), simulator-free library examples (`kinematics`, `collision`, `planning`, `determinism`). |
| `repose-kinematics` | Shipped | `CuKinematicChain<N>`, forward kinematics, space/body Jacobians, manipulability, damped least-squares IK, singularity/limit-damped differential IK, `CuServoTask<N>`, serial-chain rigid-body `inverse_dynamics` & `gravity_compensation` (`CuChainInertia<N>`), 6-DOF igus ReBeL preset. |
| `repose-collision` | Shipped | Fixed-capacity sphere/capsule `CuRobotCollisionModel<N, M>`, exact signed-distance queries (`check_frames_with_closest`), self/environment/attached-object checks, versioned `CuPlanningScene` (`CuSceneRevision`) with stable object IDs & exclusions, `CuCollisionCheckTask`, `CuCollisionMonitor` (`CuMonitorTransition`). |
| `repose-robot-description` | Shipped | Startup URDF (`<joint>`, `<link>`, `<inertial>`) and SRDF (groups, named states, link-pair exclusions) import into fixed-size kinematic, collision, and inertial models (`std`). |
| `repose-motion-planning` | Shipped | Planner-agnostic `CuJointPlanner` trait, anytime `CuJointPlannerTask`, `CuJointDirect`, `CuJointRrtConnect`, `CuJointPrm`, `CuJointAStar`, gradient `CuJointPathOptimizer`, `CuJointStomp`, pose-goal `CuPosePlannerTask`, Cartesian straight-line `CuCartesianPlanTask`, and `CuPathConstraintSet` (orientation & position-region constraints). |
| `repose-joint-rrt-star` | Shipped | Deterministic `CuRng`-seeded RRT* (`CuJointRrtStar` / `CuJointRrtStarTask`) with informed pruning, rewiring, and shortcutting. |
| `repose-trajectory` | Shipped | Fixed-capacity trajectory payloads, profile timing (`Trapezoidal`, `Cubic`, `MinimumJerk`, `SCurve` with quintic jerk bounds), Cartesian TCP-zone blending (`blend_cartesian_path`, `parameterize_cartesian_path`), `CuFollowTrajectoryTask` with fixed-size `Freezable` keyframes, `CuSetpointMuxTask<N>` bumpless quintic arbitration (`CuSetpointSource`: classical, policy, hold), and `CuAuditedSetpointMuxTask<N>` with `CuSetpointMuxReport` (`CuSetpointSwitchReason`) attribution telemetry. |
| `repose-policy-rl` | Shipped | Host (`std`) ONNX execution (`tract-onnx`) with `model_hash` pinning, decimation, and observation/raw-action magnitude guardrails (`CuBoundPolicy`: `Clamp` or `Fault`): `CuJointReachObservationTask`, single-step `CuPolicyTask`, chunked `CuChunkPolicyTask`, zero-allocation `CuActionChunkTask` (ACT exponential temporal ensembling & receding-horizon replacement), `CuJointActionTask` (`CuJointActionReport`), `CuCartesianActionTask` (VLA pose-delta via damped differential IK + `CuCartesianActionReport`), `CuGripperActionTask`. |
| `repose-safety` | Shipped | Real-time `no_std`, zero-allocation `CuSafetyShieldTask<N, M>` and `CuSafetyReport` (`CuSafetyVerdict`, `CuSafetyWitness`) enforcing joint limits, self/environment/attached-object collision margins via `CuCollisionMonitor`, axis-aligned TCP `CuWorkspaceBounds` fences, Cartesian `max_tcp_speed` limits, and velocity-extrapolated `braking_horizon` lookahead with safe stationary holds. |
| `repose-controller` | Shipped | `CuControllerTask<N>` supporting position, velocity, PD effort, and model-based inverse-dynamics effort modes, per-joint `CuEffortGains<N>`, gravity-compensated hold brake, tracking-error fault, `Tov` freshness watchdog, and command rate limiting. |
| `repose-incident` | Shipped | Host (`std`) black-box per-cycle incident detector (`CuIncidentDetector`, `CuCycleSample` with payload adapters), windowed episode manifests (`CuIncidentRecord`, `CuIncidentWindow`, `CuIncidentSummary` with `to_json_pretty()`), deterministic root-cause blame attribution (`CuIncidentBlame`), and counterfactual checkpoint regression gating (`CuCheckpointDiff`). |
| `repose-io` | Shipped | Digital IO payloads (`CuDigitalInputPayload`, `CuDigitalOutputPayload`) and RON-configured `CuGripperTask`. |
| `repose-robot-interface` | Shipped | Vendor-neutral robot bridge channel contract (`joint_command` / `joint_state`) and session lifecycle tracking (`CuSessionTracker`). |
| `repose-mujoco-sim` | Shipped | Generic `CuMujocoSimBridge<N>` TCP/JSON `CuBridge` to a MuJoCo sidecar (`std`), closed-loop `examples/mujoco_rebel`, `examples/setpoint_mux` (transit -> policy -> fault hold -> retract with `.copper` resim and offline Rerun `.rrd` export), and `examples/simplex_safety` (closed-loop Simplex safety shield, incident triage, and counterfactual replay). |

**Verification & Tooling (Shipped):**

- **Benchmarks:** 45 Divan/CodSpeed benchmarks in `repose` (including `safety`
  bench for `CuSafetyShieldTask`, plus 3 IK benches in `repose-kinematics`)
  tracked on every PR via CodSpeed.
- **Determinism & Embedded Gates:** Cross-process `just determinism` harness and
  bare-metal builds (`just bare-metal`) for `aarch64-unknown-none` and
  `thumbv7em-none-eabihf`.
- **API & Fuzzing:** `cargo-semver-checks` gated in CI, ClusterFuzzLite IK
  fuzzing (`fuzz_ik`), and user-facing Agent Skills under `skills/repose-*`.

## In Progress & Next Up (Not Built Yet)

Work after `v0.10.x` prioritizes scene-revision execution contracts, richer
per-feature policy distribution guardrails, hardware presets and servo
hardening, and pre-`1.0` stabilization, ordered by priority.

### Milestone 1 — Scene-Revision-During-Execution Contract (`repose-trajectory` & `repose-safety`)

`CuSafetyShieldTask` and `CuSafetyReport` shipped in `v0.10.0`. The remaining
safety-shield item is end-to-end scene-revision continuity across planning and
trajectory following:

- **Scene-revision-during-execution contract:** Propagate `CuSceneRevision` from
  planned trajectories into `CuTrajectoryPayload` and `CuFollowTrajectoryTask`,
  and define deterministic stale-scene policies in `CuSafetyShieldTask` when
  the live `CuPlanningScene` mutates mid-trajectory.

### Milestone 2 — Per-Feature OOD Envelopes & Action-Chatter Guardrails (`repose-policy-rl`)

`CuSetpointMuxReport` (`CuAuditedSetpointMuxTask`), scalar observation/action
magnitude guardrails (`CuBoundPolicy`), `CuJointActionReport`,
`repose-incident`, and `examples/simplex_safety` shipped in `v0.10.0`. The
remaining policy-guardrail item is:

- **Per-feature OOD envelopes & action-chatter detection (`repose-policy-rl`):**
  Per-feature min/max or z-score out-of-distribution (OOD) envelopes pinned in
  RON (extending scalar `max_observation_abs` / `max_action_abs`), plus
  action-chatter and oscillation detection before actions reach the mux or
  safety shield.

### Milestone 3 — Hardware Presets & Servo Hardening

- **Validated physical 6-DOF igus ReBeL preset (Issue #1):** Identify and verify
  inertial parameters (`CuChainInertia<6>`) and effort-mode `CuEffortGains<6>`
  on physical hardware.
- **Production `CuServoTask<N>` hardening:** Enforce latency, allocation,
  singularity, and joint-limit budgets under continuous load once external
  vision or teleoperation pose-stream consumers require it.
- **External hardware bridges:** Hardware-specific `CuBridge` crates (such as
  `repose-igus-cri`) live in `ReposeRobotics/repose-bridges` implementing
  `repose-robot-interface` and stay outside this workspace unless the shared
  interface contract changes.

### Milestone 4 — Pre-1.0 Stabilization & Optional Generality

- **API soak & `v1.0.0` release:** Soak public Rust APIs, payload wire layouts,
  and RON schemas across `0.10+` minor releases; complete rustdoc and doctests
  on every public item, document the RON compatibility policy, baseline
  `just api-check`, and flip `publish = false` to `true` in workspace
  `Cargo.toml` for lockstep crates.io publishing.
- **Deferred mechanism generality:** Branched/tree and closed-chain kinematics
  and dynamics start only when concrete robot targets demand them, without
  compromising existing serial-chain hot-path guarantees.

## Non-Goals & House Constraints

### Non-Goals

- **No ROS / ROS 2 runtime interop:** No ROS nodes, messages, actions,
  `ros2_control` adapters, or MoveIt plugin compatibility. Startup URDF/SRDF
  file parsing (`repose-robot-description`) is a file-format importer, not a ROS
  runtime dependency.
- **No generic perception stack:** No point-cloud pipelines, camera drivers,
  object recognition, or grasp synthesis. External perception publishes typed
  Copper payloads for the planning scene or policies to consume.
- **No GUI product:** No planning-scene editor or RViz equivalent. Visualization
  uses offline `.copper` -> `.rrd` exports (Rerun).
- **No dynamic runtime topology:** No runtime-loaded task graphs, plugin
  registries, or stringly-typed hot-path configuration; Copper's compile-time
  RON graph and typed configs are the model.
- **No early generality:** No branched or closed-chain mechanism abstractions
  until concrete robot targets require them.

### Validation Gate

A milestone or user-visible change ships only when all applicable checks hold:

- `just pr-check` passes (`cargo fmt --check`, clippy with denied warnings on
  `std`, `no_std`, and `cu29/reflect`, nextest on `std` and `no_std`, doctests,
  and `cargo-semver-checks`).
- `just determinism` and `just bare-metal` pass for behavioral or core changes.
- User-visible changes are recorded under `## [Unreleased]` in `CHANGELOG.md`
  (with `### Breaking` entries for API, payload, or RON changes).
- Affected user-facing Agent Skills under `skills/repose-*` are updated in the
  same PR whenever public names, RON config keys, defaults, or behaviors change.
