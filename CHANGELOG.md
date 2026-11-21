# Changelog

All notable changes to Repose. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versioning
follows [docs/versioning.md](docs/versioning.md). Pre-1.0, every minor release
may be breaking.

## [Unreleased]

### Breaking

- `repose-policy-rl`: `CuPolicyParams` and `CuChunkPolicyParams` gain four
  `pub` fields for observation and raw-action magnitude guardrails
  (`max_observation_abs`, `observation_bound_policy`, `max_action_abs`,
  `action_bound_policy`). Rust callers constructing either struct with an
  exhaustive literal must set them or use `..Default::default()`; existing RON
  configurations remain backwards-compatible via `#[serde(default)]`
  (defaults: `None` and `CuBoundPolicy::Clamp`).

### Added

- `repose-incident`: new host (`std`) crate providing black-box per-cycle
  incident detection (`CuIncidentDetector`, `CuCycleSample`, plus
  `CuCycleSample::with_joint_state`, `with_policy_action`,
  `with_joint_action_report`, `with_follow_output`, `with_safety_report`, and
  `with_controller_command` adapters), windowed episode manifests
  (`CuIncidentRecord`, `CuIncidentWindow`, `CuIncidentSummary` with
  `to_json_pretty()`), deterministic root-cause blame attribution
  (`CuIncidentBlame`: `LearnedPolicy`, `EnvironmentChange`,
  `TrajectoryOrCollision`, `ControllerOrTracking`, `HardwareOrBridge`), and
  counterfactual checkpoint regression gating (`CuCheckpointDiff::compare`,
  `passed_gate()`).
- `examples/simplex_safety`: end-to-end closed-loop Physical AI Simplex safety
  demonstration wiring `CuJointRrtStarTask`, `CuAuditedSetpointMuxTask`,
  `CuJointActionTask`, `CuSafetyShieldTask`, and `CuControllerTask` on a 6-DOF
  igus ReBeL arm with offline `.copper` incident triage
  (`simplex_safety_triage`), bit-identical replay (`simplex_safety_resim`), and
  a 1080p side-by-side comparison video renderer (`render_video.py`).
- `repose-safety` (re-exported as `repose::safety`): added the `no_std`,
  zero-allocation [`CuSafetyShieldTask`] and [`CuSafetyReport`] real-time safety
  shield sitting between setpoint sources (`CuFollowTrajectoryTask`,
  `CuSetpointMuxTask`, or `repose-policy-rl` action tasks) and
  `CuControllerTask`, enforcing joint limits, self/environment/attached
  collision margins, axis-aligned TCP `CuWorkspaceBounds` fences, Cartesian TCP
  speed limits (`max_tcp_speed`), and velocity-extrapolated
  `braking_horizon` lookahead with safe stationary holds and fixed-size
  `Freezable` state.
- `repose-collision`: derived `Default`, `Serialize`, `Deserialize`, `Encode`,
  `Decode`, and `Reflect` on `CuMonitorTransition` (and `Encode`, `Decode`,
  `Reflect` on `CuCollisionMonitor`), and added
  `CuRobotCollisionModel::check_frames_with_closest`.
- `repose-trajectory`: added simplex arbitration and switch-reason attribution
  telemetry (`CuSetpointSwitchReason`, `CuSetpointMuxReport`,
  `CuSetpointMuxTask::last_report`, `CuSetpointMuxTask::evaluate`, and
  dual-output `CuAuditedSetpointMuxTask<N>`) without changing
  `CuSetpointMuxTask<N>`'s `CuTask` interface, RON config, or `Freezable`
  keyframe layout.
- `repose-policy-rl`: `CuBoundPolicy` (`Clamp` or `Fault`) and optional
  observation and raw-action magnitude guardrails (`max_observation_abs`,
  `observation_bound_policy`, `max_action_abs`, `action_bound_policy`) on
  `CuPolicyParams` and `CuChunkPolicyParams`, enforcing clamping or faults
  after non-finite checks (observation faults skip inference, while action
  faults reject the inferred output and publish zeros); and
  `CuJointActionReport`
  exposed via `CuJointActionTask::last_report()` with `step_limited_mask`,
  `joint_limited_mask`, `step_limited_count`, `joint_limited_count`,
  `max_requested_delta_si`, and `is_step_limited` / `is_joint_limited` /
  `any_limited` helpers without changing task outputs or `Freezable` state.
- `repose`: expanded Divan / CodSpeed benchmarks covering serial-chain
  rigid-body dynamics (`inverse_dynamics`, `gravity_compensation`), trajectory
  following, bumpless setpoint multiplexing, Cartesian path blending and
  parameterization, pose-goal / Cartesian / path-constrained motion planning,
  `CuControllerTask` position and PD / model-based effort modes,
  `repose-policy-rl` observation, single-step and chunked ONNX inference, ACT
  temporal ensembling, joint / VLA Cartesian / gripper action tasks, and
  `CuSafetyShieldTask` (`safety` bench).

### Changed

- `repose-motion-planning`: `CuJointPrm` now filters candidate neighbours by
  squared joint-space distance before taking `sqrt`, skips segment collision
  checks for already-full-degree neighbours without changing candidate selection,
  caches edge costs at link time, and runs Dijkstra over a fixed-capacity
  indexed binary min-heap with identical tie-breaking, RNG sequence, public API,
  and `no_std` / zero-allocation guarantees.
- `repose-joint-rrt-star`: `CuJointRrtStar` now uses squared joint-space
  distance pre-filtering for nearest-neighbor and rewiring neighborhood checks,
  reuses a single-pass stack-allocated neighbor list between parent selection
  and rewiring, and caches `parent_dist` per node so rewire cost propagation
  runs without recomputing `joint_distance`. Public API, RNG consumption, and
  deterministic tree construction are unchanged.
- `repose-trajectory`: `parameterize` now advances a monotonic segment cursor
  with a stack-allocated per-segment parameter cache instead of scanning
  segments and recomputing per-joint shapes on every resampled knot, and folds
  Hermite velocity-limit checks into the initial segment pass. Public API,
  timing, and `no_std` / zero-allocation guarantees are unchanged.
- `repose-trajectory`: `CuTrajectoryPayload` sampling (`sample_linear`,
  `sample_cubic`, `sample_quintic`, and `sample_with_acceleration`) now uses
  binary search (`O(log n)`) for knot lookup and hoists Hermite basis /
  reciprocal-duration evaluation across joints without changing the public API
  or boundary semantics.
- `repose-motion-planning`: `CuJointPathOptimizer` and `CuJointStomp` now
  short-circuit `record_if_better` on path cost before collision checking,
  hoist per-joint scales/limits/fixed-joint flags out of inner waypoint and
  rollout loops, and update/smooth active `..count` waypoints in place without
  copying unused tail slots. Public API, RNG order, and determinism are
  unchanged.
- `repose-motion-planning`: `CuJointAStar` now uses a fixed-capacity indexed
  binary min-heap for its open set with exact deterministic tie-breaking by node
  index, power-of-two hash slot masking, and single-joint neighbor updates,
  replacing the `O(V²)` linear open-set scan without changing the public API,
  search order, or `no_std` / zero-allocation guarantees.

## [0.9.0] - 2026-11-17

### Breaking

- `repose-trajectory`: the frozen state of `CuFollowTrajectoryTask` has a new
  fixed-size layout (see `### Fixed`), so follower keyframes written by
  earlier versions are not supported. Frozen layouts are not versioned, so
  thawing an old keyframe is not detected and restores wrong state; replay
  such logs from the start instead of from a keyframe.

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

- `repose-controller`: `CuJointEffort` is now the same type as
  `repose_kinematics::CuJointEffort`; the controller's own duplicate is
  removed and re-exported from `repose-kinematics`, so a bridge or controller
  payload effort is exactly what `inverse_dynamics` produces, with no
  conversion. Variants (`Revolute(Torque)`, `Prismatic(Force)`, `Fixed`),
  `from_position`, `si_value`, derives and the bincode/serde encoding are
  unchanged, so the `CuJointCommandPayload::efforts` wire format and RON are
  identical (now pinned by a test). `From<Torque>` and `From<Force>` are
  gained. `repose_controller::CuJointEffort` still resolves, but it is a
  re-export; code that implemented traits for it or named it through other
  paths needs no change except where listed in
  [docs/migration-0.9.md](docs/migration-0.9.md).

### Added

- `repose-trajectory`: `CuSetpointMuxTask<N>` arbitrates between a classical
  setpoint (e.g. `CuFollowTrajectoryTask` running a collision-free transit)
  and a policy setpoint (e.g. `repose_policy_rl::CuJointActionTask`), both
  `CuTrajectoryFollowOutput<N>`. It hands over to the policy when the
  classical trajectory finishes and the policy is `Running`, lets a new
  classical plan preempt the policy, and falls back to holding the last
  emitted position when the policy turns `Idle`, faults, or goes non-finite.
  Every switch is bumpless (quintic position/velocity/acceleration offset
  decay over `blend_time`, default 0.2 s). Adds `CuSetpointMuxParams`,
  `CuSetpointSource`, and `DEFAULT_MUX_BLEND_TIME_S`. The new
  `examples/setpoint_mux` app runs transit → policy → fault hold → retract
  closed loop on an in-process plant, records a `.copper` log with keyframes,
  replays it bit-exactly (`setpoint_mux_resim`), and converts it to a Rerun
  recording (`setpoint_mux_to_rrd`, `rrd` feature).

- Agent Skills for Repose users in `skills/`: getting started and shared
  conventions, robot configuration, kinematics, collision, motion planning,
  trajectories and control, Copper apps, and RL policies. Copy them into your
  agent's skills folder; see [skills/README.md](skills/README.md).

- `repose-policy-rl`: new `std`-only crate running a stateless feed-forward
  ONNX policy in the graph. `CuPolicyTask<IN, OBS, ACT>` is pure inference:
  it consumes one flat `CuPolicyObservation<IN>` feature vector from an
  upstream observation task, so the observation is defined by the RL task and
  not by the runtime. It infers with `tract-onnx` and publishes the
  `CuPolicyObservation<OBS>` model input of the cycle plus the raw
  `CuPolicyAction<ACT>` with a `CuPolicyStatus`
  (`Idle`/`Fresh`/`Held`/`Fault`), so every observation and raw action is in
  the log. With `previous_action: true` the task appends its own last raw
  action after the features (`OBS == IN + ACT`); otherwise `OBS == IN`. No
  input payload is `Idle`; a non-finite input, a failed inference, or a
  non-finite output is `Fault`. Cycle-exact decimation, an optional pinned
  `model_hash`, and real `Freezable` state keep replay deterministic.
  Inference is the workspace's one documented hot-path allocation exception:
  14 allocations per cycle with the `linear_24_6` fixture, measured by
  `tests/alloc.rs`, which asserts at most 32 on every cycle. Other models
  allocate differently; measure yours the same way.
- `repose-policy-rl`: `CuJointReachObservationTask<N, IN>` builds the
  joint-reach features from the measured joint state and a joint goal, with a
  RON-configured layout of `joint_positions`, `joint_velocities`, and
  `goal_error` terms whose widths sum to `IN`. It emits no payload without a
  state, a goal, or a `Ready` robot (the policy reports `Idle`) and a
  `NaN`-filled payload on a joint unit-kind mismatch (the policy reports
  `Fault`). Other RL tasks supply their own observation task emitting a
  `CuPolicyObservation<IN>`.
- `repose-policy-rl`: `CuJointActionTask<N, ACT>` turns `N` elements of a raw
  `CuPolicyAction<ACT>`, starting at the configured `offset` (default 0), into
  the `CuTrajectoryFollowOutput<N>` the controller already consumes, so
  several action tasks can share one action vector (`offset + N > ACT` is a
  startup error). It applies `position_delta` (an offset from
  the measured position) or `position_absolute` (from typed default positions)
  actions with a dimensionless scalar or per-joint `scale`, an optional
  per-cycle `max_step`, and clamping to the configured chain's joint limits.
  Idle, faulted, and non-finite inputs, a robot that is not `Ready`, and a
  joint state whose units do not match the chain drop the target and hold
  the measured position with `Idle` or `ErrorEmptyTrajectory` status, so
  motion after any hold restarts from where the arm is. The previous target
  round-trips through `Freezable`. With the optional `period` (the control
  cycle period) the setpoint velocity is the target change per cycle, which
  a controller in `velocity` mode executes; without it the setpoint velocity
  is zero and the controller must run in `position` or `effort` mode.
- `repose-policy-rl`: `CuCartesianActionTask<N, ACT>` runs end-effector
  actions, the output of most vision-language-action models. It reads a pose
  delta `[dx, dy, dz, r0, r1, r2]` from `CuPolicyAction<ACT>` at `offset`, in
  the `base` or `body` frame, with the rotation as a rotation vector or as
  roll/pitch/yaw. Translation and rotation scales are typed (`Length`,
  `Angle`), and optional `max_translation` and `max_rotation` limit each
  cycle's delta. The delta becomes a twist over the required `period`, which
  `CuKinematicChain::differential_ik` solves at the measured joint positions:
  damped least squares with adaptive singularity damping, joint-limit margin
  damping, and scaling to the chain's joint velocity limits. The task emits
  the `CuTrajectoryFollowOutput<N>` the controller consumes, clamped to the
  joint limits, plus a `CuCartesianActionReport` that logs the IK status and
  whether a Cartesian limit shortened the delta. Hold and fault behavior,
  `Held` actions, and `Freezable` state match `CuJointActionTask`. Nothing on
  the hot path allocates.
- `repose-policy-rl`: `CuGripperActionTask<ACT>` turns one action element into
  the `CuGripperCommandPayload` that `repose_io::CuGripperTask` consumes. It
  takes a `threshold`, `close_when: above | below` (RT-1 closedness or
  Bridge/Open X-Embodiment openness), and an optional `hysteresis` dead band.
  An idle, faulted, non-finite, or dead-band value publishes no command, so
  the gripper keeps its last action and a faulted policy never drops what it
  holds. `repose-policy-rl` now depends on `repose-io`.
- `repose-policy-rl`: action chunking and exponential temporal ensembling for
  ACT, Diffusion Policy, and VLA models. `CuPolicyActionChunk<H, ACT>` carries
  an `H`-step raw action horizon (round-tripping `serde` and `bincode` at any
  `H` and `ACT`); `CuChunkPolicyTask<OBS, H, ACT>` runs `[1, OBS] -> [1, H, ACT]`
  ONNX models on `tract-onnx` with `decimation`, `model_hash` pinning, and
  `Freezable` state (external VLA bridges can also publish
  `CuPolicyActionChunk<H, ACT>` directly); and `CuActionChunkTask<H, ACT>`
  turns incoming chunks into one `CuPolicyAction<ACT>` per control cycle with
  zero hot-path heap allocations (`tests/chunk_alloc.rs`), so downstream
  `CuJointActionTask`, `CuCartesianActionTask`, and `CuGripperActionTask`
  connect unchanged. Supports `CuActionChunkMode::TemporalEnsemble { decay }`
  (`w_i = exp(-decay * i)` in an `O(H * ACT)` ring buffer, default `0.01`) and
  `CuActionChunkMode::Replace` receding-horizon execution, multi-rate
  `step_decimation` (`Fresh` on step entry, `Held` between steps), optional
  `execution_horizon` starvation to `Idle`, immediate buffer flush on upstream
  `Idle` or `Fault`, and full `Freezable` keyframe restore.
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
  falls back to the brake. Without a model a finite desired acceleration is
  ignored and a non-finite one faults. Hold and fault paths add gravity
  compensation at the measured positions to the damping brake, so the arm no longer sags. A model that
  cannot be evaluated faults the controller and falls back to the damping
  brake. Without a model, effort mode is the same PD law as before.
- `repose-controller`: per-joint effort gains in RON, as an array of `N`
  numbers per gain (`"effort_gains": {"kp": [60.0, 60.0, 40.0, 8.0, 8.0, 4.0], "kv": 0.8}`).

### Fixed

- `repose-trajectory`: `CuFollowTrajectoryTask` keyframes no longer fail with
  `UnexpectedEnd`. Its frozen state used to grow once it adopted a trajectory,
  but Copper sizes the keyframe buffer from the first snapshot, so any app
  with keyframe logging on failed its first keyframe after a plan. The
  snapshot is now fixed-size (presence flags, little-endian integers, and the
  full `MAX_POINTS` capacity). See `### Breaking` for keyframe compatibility.

- `repose-trajectory`: the `Cubic` timing profile no longer returns
  `InvalidInput` for long rest-to-rest segments. Its default segment duration
  was the trapezoidal minimum, at which a cubic Hermite segment peaks at
  `1.5 * distance / duration`, above the velocity limit once a joint travels
  more than `2 * v^2 / a`. This rejected most long two-waypoint plans, such
  as RRT* shortcuts. Without fixed durations or non-zero waypoint velocities,
  the duration is now stretched to keep that peak at the limit; shorter
  segments keep their timing. Fixed durations that are too short are still
  rejected.

- `repose-controller`: `CuControllerTask::process` never emits a non-finite
  field (`positions`, `velocities`, `efforts`, `max_tracking_error`) in
  `CuJointCommandPayload` or stores non-finite state across cycles (`last_command`,
  rate-limiter state, or frozen state). Measured states with any non-finite
  position or velocity are treated as unusable (`Fault`, holding the last finite
  command with zero effort without echoing the bad state); active setpoints
  with any non-finite position, velocity, or acceleration fault across all
  modes (holding the finite measured state with the effort brake, or the last
  finite command otherwise); tracking-error and effort arithmetic guards against
  overflow; and subsequent good cycles resume normal operation without latching
  `NaN` or `±∞` in the rate limiter.

- `repose-trajectory` / `repose-controller`: cancelling a trajectory no longer
  commands joint zero. After an empty trajectory the follower used to emit
  `Idle` with zero positions, and the controller chased them whenever zero was
  within `tracking_error_limit` of the measured state. Now the follower holds
  the last commanded positions (zero velocity and acceleration) after a
  cancel, reporting `ErrorEmptyTrajectory` once and then `Idle`; a new
  non-empty trajectory releases the cancel. The controller treats an `Idle`
  follower setpoint as no setpoint: it holds the measured state with status
  `Idle` (also before any trajectory was received, when the follower's
  positions are zero placeholders). The follower's frozen state gains the held
  positions, so keyframes written by earlier versions do not thaw; this
  rides with the other breaking frozen-state changes above.

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

[Unreleased]: https://github.com/ReposeRobotics/repose/compare/v0.9.0...HEAD
[0.9.0]: https://github.com/ReposeRobotics/repose/compare/v0.8.4...v0.9.0
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
