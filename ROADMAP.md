# Repose Roadmap

## Goal

Build Repose, a general manipulation stack comparable in scope to MoveIt, for
[copper-rs (`cu29`)](https://github.com/copper-project/copper-rs) — Copper-native,
not ported from ROS. It covers kinematics, collision checking, motion planning,
trajectory generation, execution, servoing, and robot bridges for Copper apps,
in that order.

## Success Criteria

- Every phase below is closed only when all of its done-criteria hold
  (see Validation Plan).
- The stack runs end-to-end in a local demonstration: robot description in,
  planned and timed trajectory out, executed under collision monitoring.
- Public APIs are documented, semver-stable in intent, and crates.io-ready;
  no placeholder surfaces ship as "done".

## Current State (done)

- `repose-kinematics`: `CuKinematicChain<N>` FK, space/body Jacobians,
  manipulability, DLS and differential IK with singularity/joint-limit damping,
  FK/IK/servo `CuTask`s, igus ReBeL preset. Serial-chain rigid-body dynamics
  (`v0.8.4`): `CuChainInertia<N>`, recursive Newton-Euler `inverse_dynamics`,
  and `gravity_compensation`.
- `repose-collision`: fixed-capacity sphere/capsule model, signed-distance queries,
  self/environment checking, `CuCollisionCheckTask<N>`, the versioned
  `CuPlanningScene`, and the `CuCollisionMonitor` transition policy (`v0.8.3`;
  not yet wired into execution, see Step 3).
- `repose-robot-description`: startup URDF serial-chain and SRDF group/exclusion/
  state import into fixed-size kinematic and collision models; URDF
  `<inertial>` import into `CuChainInertia<N>` (`v0.8.4`).
- `repose-motion-planning`: planner-agnostic `CuJointPlanner` trait, request/path
  payloads with explicit `CuPlanStatus`, shared scene validity checks, the
  generic anytime `CuJointPlannerTask<P, N>`, and built-in planners:
  RRT-Connect and PRM (sampling), A* lattice (search), gradient path
  optimizer and STOMP (optimization), direct joint line. Pose-goal and
  Cartesian planning tasks (`v0.8.0`) and path constraints honoured by every
  planner (`v0.8.4`).
- `repose-joint-rrt-star`: sampling-based RRT* implementing `CuJointPlanner`, with
  deterministic `CuRng` seeding, informed pruning, rewiring, and shortcutting.
- `repose-trajectory`: fixed-capacity trajectory payloads, profile timing, per-waypoint
  Cartesian TCP zones, C² spline geometry, quintic interpolation, and S-curve
  joint jerk bounds; follower preemption and empty-trajectory cancellation.
- `repose-controller`: `CuControllerTask<N>` with position, velocity, and
  effort modes, tracking-error fault, state watchdog, and rate limiting
  (see Release 0.3).
- `repose-io`: digital IO payloads and the IO-driven `CuGripperTask<IN, OUT>`.
- `repose-robot-interface`: vendor-neutral bridge contract, with the standard
  channel set and session lifecycle payloads (`v0.7.1`).
- `repose-mujoco-sim`: `CuBridge` to a MuJoCo sidecar over TCP/JSON carrying
  `CuJointCommandPayload<6>` out and `CuJointStatePayload<6>` back (first
  released in `v0.2.1`); `examples/mujoco_rebel` and the local playground
  close the plan → follow → sim → feedback loop.
- `repose`: umbrella crate re-exporting every crate, with the
  `repose::planners` registry (`v0.7.1`), simulator-free library examples
  (`v0.8.4`), and Divan benchmarks tracked by CodSpeed.
- Checks beyond `just pr-check`: cross-process determinism, bare-metal
  builds, and IK fuzzing (`v0.8.0`–`v0.8.3`).
- All crates pass `just pr-check` and are published at
  https://github.com/ReposeRobotics/repose.

## Approach

Phases are sequential; each phase builds only on the crates and tasks of the
phases before it. Serial chains are the only mechanism in Phases 1–4; general
mechanisms are a deliberate later phase, never a drive-by addition. Copper
house rules apply throughout: `const N: usize` static topology, zero hot-path
allocation, in-place `&mut` outputs, `no_std`-compatible shared crates,
`cu29::units` physical units, `CuResult<T>` errors, `Reflect` + `Freezable`
tasks, deterministic replay-compatible behaviour.

## Release 0.3 (released in `v0.3.0`)

Controller-interface completion, building on `v0.2.2`:

- **Effort/torque commands (landed in `v0.2.3`).** `CuJointCommandPayload<N>`
  carries position, velocity, and effort modes, with typed torque/force per
  joint kind. `CuControllerTask<N>` computes PD effort; bridges apply raw
  torque/force. Position-mode velocity feedforward landed with it.
- **Gripper and digital IO (implemented).** `repose-io` defines the
  robot-agnostic `digital_in`/`digital_out` payloads
  (`CuDigitalInputPayload<N>` with device status, `CuDigitalOutputPayload<N>`
  with a drive mask). A gripper is IO, not a payload of its own:
  `CuGripperTask<IN, OUT>` maps open/close commands onto RON-configured
  output bits and optional sensor inputs, reporting open/closed/moving/fault.
  Bridge support (sim and hardware) follows with each bridge.
- **Robot-side status/fault reporting (implemented).** `CuJointStatePayload<N>` gains a
  bridge-reported status (ok, fault with code, estop, not-ready), distinct
  from the controller's own status; the controller surfaces it and faults
  instead of commanding a robot that reports a fault.
- **Rate limiting and watchdog (implemented).** `CuControllerTask<N>` gains configurable
  command rate limiting (max joint velocity/acceleration of emitted
  commands) and a time-based watchdog: stale measured state (by `Tov`)
  beyond a configured age faults and holds, replacing today's purely
  per-cycle checks. (Pulled forward from Step 3.)

## Steps

1. **Trajectory generation and simulated execution (implemented; closeout pending).**
   Fixed-capacity trajectory payloads, profile timing, per-waypoint Cartesian
   TCP-zone blending, and `CuFollowTrajectoryTask<N>` interpolation, status,
   preemption, and cancellation are implemented. Cartesian S-curve joint jerk
   bounds require quintic execution. The git-excluded `playground/` demonstrates
   local planning, timing, and simulated playback. Closeout reviews the public
   API, packaging, and cancellation semantics against the Validation Plan.
   Collision monitoring and collision-triggered abort during execution are
   deferred to Step 3; they do not block Step 1.
2. **Planning and simulated robot execution (implemented; see v0.2.2, v0.8.0).**
   The planning scene, scene snapshots and revision semantics, motion planning
   and planners, simulated execution, the unified robot controller, pose-goal
   planning, and Cartesian planning with TCP zones are all shipped (latest:
   `v0.8.0`). Timing benchmarks are shipped; the success-rate and path-cost
   report is not. Historical build order:
   - A versioned, fixed-capacity planning scene shared by collision checking and
     planner tasks: stable object IDs, obstacle add/update/remove, attached
     objects with link-relative transforms, and explicit collision exclusions.
   - Consistent scene snapshots for each planning request, with scene revisions
     carried into results and deterministic restart/invalidation when the scene
     changes during anytime planning.
   - Connect a physics simulator through a Copper joint-state/command interface.
     Read measured joint states, plan from the current state, time the path,
     send follower setpoints to the simulator, and observe returned joint states
     and execution status. Use a local, git-excluded demonstration to exercise
     the complete loop; kinematic animation alone does not satisfy this step.
     Define the joint command/state contract here so it can also support the
     hardware bridge in Step 4. (Implemented: `repose-mujoco-sim` `CuBridge` over
     `CuJointPositionsPayload<6>`, released in `v0.2.1`; the local playground
     closes the loop and the Trajectory Studio executes in the sim.)
   - **Unified robot controller (implemented in `v0.2.2`).** Upstream `cu29` has no robot
     controller abstraction — only the generic `CuBridge` trait and task
     roles — so this contract is defined here, robot-agnostically:
     a `CuControllerTask<N>` between the trajectory follower and a `CuBridge`,
     consuming the follower setpoint plus measured joint state and emitting one
     unified joint command (positions, optional velocities) with an explicit
     status (idle/running/holding/fault) and tracking error. Shared
     `CuJointStatePayload<N>` / `CuJointCommandPayload<N>` payloads (extending
     today's `CuJointPositionsPayload<N>`) are the only interface a bridge
     sees: simulators (`repose-mujoco-sim`) and hardware (`repose-igus-cri`, Step 4)
     implement the same `joint_command`/`joint_state` channels, so swapping
     sim for a real robot is a RON graph change, not an app change. Generalize
     `repose-mujoco-sim` from fixed `N = 6` as the first consumer (not done:
     the bridge is still fixed at `N = 6`).
     Shape (per Copper component design): a new crate
     `components/tasks/repose_controller` holding the command payload and a
     `CuControllerTask<N>` (`CuJointStatePayload<N>` lives in
     `repose-kinematics`) — a multi-input `CuTask` (follower setpoint +
     measured state in, unified command out), `Reflect` + `Freezable`
     (last command, status, watchdog state round-trip for replay), zero
     hot-path allocation, `Tov` propagated from the state input. Bridges stay
     pure transports: `CuBridge` with `joint_command`/`joint_state` channels,
     polling in `preprocess`, dispatch in `send`/`receive`. Done: the
     playground sim demo runs through the controller unchanged
     (`v0.2.2`).
   - Pose-goal planning using the existing IK and joint-space RRT*, with explicit
     no-IK-solution, invalid-start/goal, capacity, and planning-failure outcomes.
   - Integrate existing Cartesian paths and TCP zones with the same scene and
     request/result contracts. Recheck the reconstructed execution curve for
     joint limits, collisions, and zone containment before accepting a plan.
   - A seeded benchmark harness reporting success rate, geometric path cost,
     planning effort, and elapsed time. Add planners such as PRM/BIT* only where
     measured results justify them.
   Completion requires a local physics-simulator demonstration that reads joint
   state, plans and times a trajectory, commands the simulated robot, and
   observes state feedback and execution status. It also updates obstacles,
   attaches an object, and plans joint, pose-goal, and Cartesian requests
   consistently against the selected scene revision, plus the Validation Plan
   checks.
3. **Execution collision monitoring (policy landed in `v0.8.3`; integration
   next).** Monitor the follower's setpoints against the live planning scene
   during execution and abort/hold with defined stop/hold reporting when a
   violation appears. This serves the existing
   plan → execute stack and is distinct from geometric validation before
   execution: plans are still checked upfront; monitoring runs while they
   execute. It defines the scene-revision-during-execution semantics (the
   roadmap's highest-risk item).
   Done: `CuCollisionMonitor` / `CuMonitorTransition` in `repose-collision`,
   an allocation-free policy that turns a caller's per-cycle check result into
   one event per collision episode, plus its overhead benchmark.
   Remaining: a task that checks follower setpoints against the live scene and
   feeds the monitor; abort/hold on a violation with a reported status; the
   scene-revision-during-execution semantics (trajectories carry no scene
   revision today); and an end-to-end demonstration. No task, RON graph, or
   example uses the monitor yet.
4. **Real-time servoing (deferred).** Harden `CuServoTask<N>` into a production
   servo loop — singularity/limit behaviour under load, latency/allocation
   budgets enforced by tests — only when a pose-stream consumer exists
   (vision, teleop, an external servo source). Until then the machinery has no
   user; revisit together with Step 6's dynamics work.
5. **Robot bridges (satisfied externally).** The igus ReBeL CRI bridge lives in
   a separate repository, not in this workspace; it implements the joint
   command/state and session contract defined here in
   `repose-robot-interface`. This roadmap tracks no further bridge work in
   this repo unless that contract changes.
6. **Generality (deliberately last).** Tree/branched kinematic structures,
   then closed-chain mechanisms, then dynamics and torque-level control. No
   Phase 1–4 API may be designed so these become impossible; none of this
   starts before Step 4 is done.
   Already shipped for serial chains, ahead of this order: PD effort commands
   in `CuControllerTask<N>` (`v0.2.3`) and rigid-body inverse dynamics with
   gravity compensation in `repose-kinematics` (`v0.8.4`), which no task uses
   yet. In progress: model-based effort control in `CuControllerTask<N>`
   (per-joint gains, inverse-dynamics feedforward, gravity-compensated hold;
   issue #1). Not started: tree/branched and closed-chain structures and
   their dynamics.
7. **1.0 stabilization and publishing.** Rustdoc and doctests on every public
   item; semver commitment; a stated compatibility policy for RON
   configuration schemas. Publishing is pre-wired so this step is a flip, not
   a restructure: the workspace already carries one shared version, versioned
   path dependencies between crates, correct `repository` metadata, and
   `just api-check` / `just release-dry-run`. To publish: flip the single
   `publish = false` switch in the workspace `Cargo.toml` to `true`, publish
   all crates to crates.io in lockstep (dependency order) at the shared
   version, publish the `repose` umbrella crate that re-exports them, and tag the release `v1.0.0`. Until then, releases are
   git tags (`vX.Y.Z`, workspace-wide, never per-crate) with GitHub Releases,
   and consumers use git dependencies.

## Validation Plan

A phase counts as done only when all four hold:

1. `just pr-check` passes: fmt, `std`, `no_std`, and reflect clippy with
   denied warnings, nextest, doctests, and the SemVer check.
2. The phase's public API is documented and semver-stable in intent.
3. Packaging metadata (repository, license, keywords, categories) is correct
   and crates.io-ready.
4. A runnable local demonstration (in `playground/`, never committed)
   demonstrates the phase end-to-end.

Passing tests alone does not close a phase.

## Non-Goals — What We Will Not Do

- **No ROS / ROS 2 interop.** No ROS nodes, messages, actions, `ros2_control`
  adapters, or MoveIt plugin/config compatibility. URDF/SRDF *file* import at
  startup (already in `repose-robot-description`) is the sole exception — a file
  format, not a ROS runtime dependency.
- **No generic perception.** No point-cloud pipelines, camera drivers, object
  recognition, or grasp synthesis. External perception may publish Copper
  payloads for the planning scene to consume; building it is not this project.
- **No GUI product.** No planning-scene editor or RViz equivalent as a
  deliverable. The local playground is an unpublished development aid and does
  not set product direction.
- **No dynamic runtime topology.** No runtime-loaded task graphs, plugin
  registries, or stringly-typed hot-path configuration; Copper's compile-time
  graph and RON policies are the model.
- **No early generality.** Branched/closed chains and dynamics appear only in
  Step 6, on that step's terms — not as additions to earlier phases.
  (Serial-chain dynamics shipped early in `v0.8.4`; see Step 6.)

## Risks / Open Questions

- Highest risk: scene revision and invalidation semantics across anytime planning
  and later execution monitoring. Define ownership and stale-result handling
  before freezing the scene/request/result contracts. Planning requests and
  results carry the scene revision; execution does not yet (Step 3).
- Undecided: additional robot presets beyond the igus ReBeL (added only when a
  bridge or user needs one); the second bridge target after `repose-igus-cri`
  (decided by hardware access); whether the planner success-rate and
  path-cost report ships as a report crate or as local demonstrations
  (timing results ship through CodSpeed and the `benchmark-data` branch).
