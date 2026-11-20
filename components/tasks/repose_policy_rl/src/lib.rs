//! ONNX reinforcement-learning and action-chunking policy execution for `copper-rs`.
//!
//! The crate splits a policy into observation, inference (single-step or
//! `H`-step chunked), optional chunk buffering / temporal ensembling, and
//! safe action execution:
//!
//! - an **observation task** turns task-specific inputs into one flat,
//!   dimensionless [`CuPolicyObservation<IN>`] feature vector.
//!   [`CuJointReachObservationTask`] is the joint-reaching one: it builds the
//!   features from the measured joint state and a joint goal, following an
//!   ordered layout of [`CuObservationTerm`]s. Other RL or VLA tasks supply
//!   their own observation task emitting `CuPolicyObservation<IN>`.
//! - [`CuPolicyTask`] runs a stateless single-step ONNX policy (`[1, OBS] ->
//!   [1, ACT]`) with `tract-onnx`, optionally appending its own last raw
//!   action to the features, and publishes both the [`CuPolicyObservation`]
//!   model input built this cycle and the raw [`CuPolicyAction`] before any
//!   scaling or limiting.
//! - [`CuChunkPolicyTask`] runs a stateless chunked ONNX policy (`[1, OBS] ->
//!   [1, H, ACT]`) and publishes a [`CuPolicyActionChunk<H, ACT>`] (external
//!   VLA bridges or custom tasks can also publish [`CuPolicyActionChunk`]
//!   directly). [`CuActionChunkTask`] turns incoming chunks into one
//!   [`CuPolicyAction<ACT>`] per control cycle with zero heap allocation,
//!   supporting both **exponential temporal ensembling** (ACT, Zhao et al.)
//!   and **receding-horizon replacement** (diffusion and flow-matching
//!   policies), multi-rate `step_decimation`, and `execution_horizon`
//!   starvation bounds.
//!
//! On a `Fresh` cycle of [`CuPolicyTask`] the published observation is
//! exactly what the model was fed; `Held` and `Idle` cycles run no inference,
//! and a `Fault` cycle may or may not have. Every observation and every raw
//! action is in the log, so a recorded run shows whether the model or the
//! downstream execution layer caused a given motion. An observation task
//! signals "no input" by publishing no payload (the policy reports `Idle`)
//! and a fault by publishing `NaN` features (the policy reports `Fault`).
//! An action task turns the raw [`CuPolicyAction`] into the limited joint
//! setpoint that `repose_controller::CuControllerTask` consumes:
//!
//! - [`CuJointActionTask`] for joint-space actions (most RL and ACT policies),
//! - [`CuCartesianActionTask`] for end-effector pose deltas (most
//!   vision-language-action models): the delta is solved with
//!   singularity- and joint-limit-aware differential IK on the configured
//!   chain, so the model never commands joints directly,
//! - [`CuGripperActionTask`] for an open/close element, feeding
//!   `repose_io::CuGripperTask`.
//!
//! Action tasks read disjoint slices of one action through their `offset`,
//! so a typical VLA action `[dx, dy, dz, droll, dpitch, dyaw, gripper]` is a
//! Cartesian task at offset 0 with `"rotation": "rpy"` and a gripper task
//! at offset 6.
//!
//! This crate is `std`-only: a policy runtime is a host component. There is
//! no `no_std` surface and no `#![no_std]` attribute; the `std` feature
//! exists and forwards to the dependencies so workspace-wide `--features std`
//! and `--no-default-features` invocations keep working. (With
//! `--no-default-features` the crate still builds for the host: `tract`
//! requires `std` either way.)
//!
//! # Allocation on the hot path
//!
//! House rules forbid heap allocation on the real-time path; this crate is
//! the one documented exception while [`CuPolicyTask`] and
//! [`CuChunkPolicyTask`] use `tract`, which allocates output tensors per run.
//! [`CuJointReachObservationTask`], [`CuActionChunkTask`], and all action
//! tasks allocate nothing (`tests/alloc.rs` and `tests/chunk_alloc.rs`).
//! Measured with the `linear_24_6` fixture: **14 allocations per cycle**, all
//! in the policy task's `process` (asserted bound: 32). Model loading at
//! startup allocates freely and runs once.
//!
//! # Determinism
//!
//! Decimation and chunk stepping count cycles, not wall time, and all task
//! state round-trips through `Freezable`, so replay is cycle-exact. Pin
//! `model_hash` in RON so a replay against a different model file fails at
//! startup instead of silently diverging. The single-step policy period is
//! `decimation / rate_target_hz` (for chunked policies, each step's period is
//! `step_decimation / rate_target_hz`) and must equal the period the policy
//! was trained at. Model paths resolve relative to the process working
//! directory.
//!
//! ```ron
//! (
//!     id: "observation",
//!     type: "repose_policy_rl::CuJointReachObservationTask<6, 18>",
//!     config: {
//!         "observation": [
//!             {"joint_positions": {"scale": 1.0}},
//!             {"joint_velocities": {"scale": 0.1}},
//!             {"goal_error": {"scale": 1.0}},
//!         ],
//!     },
//! ),
//! (
//!     id: "policy",
//!     type: "repose_policy_rl::CuPolicyTask<18, 24, 6>",
//!     config: {
//!         "model": "policy.onnx",
//!         "decimation": 1,
//!         "previous_action": true,
//!     },
//! ),
//! ```

mod action;
mod backend;
mod cartesian;
mod chunk;
mod gripper;
mod observation;
mod payload;
mod policy;

pub use action::CuJointActionKind;
pub use action::CuJointActionParams;
pub use action::CuJointActionReport;
pub use action::CuJointActionTask;
pub use cartesian::CARTESIAN_ACTION_WIDTH;
pub use cartesian::CuCartesianActionParams;
pub use cartesian::CuCartesianActionReport;
pub use cartesian::CuCartesianActionTask;
pub use cartesian::CuRotationEncoding;
pub use chunk::CuActionChunkMode;
pub use chunk::CuActionChunkParams;
pub use chunk::CuActionChunkTask;
pub use chunk::CuChunkPolicyParams;
pub use chunk::CuChunkPolicyTask;
pub use chunk::DEFAULT_ENSEMBLE_DECAY;
pub use gripper::CuGripperActionParams;
pub use gripper::CuGripperActionTask;
pub use gripper::CuGripperCloseWhen;
pub use observation::CuJointReachObservationParams;
pub use observation::CuJointReachObservationTask;
pub use observation::CuObservationTerm;
pub use payload::CuPolicyAction;
pub use payload::CuPolicyActionChunk;
pub use payload::CuPolicyObservation;
pub use payload::CuPolicyStatus;
pub use policy::CuBoundPolicy;
pub use policy::CuPolicyParams;
pub use policy::CuPolicyTask;

#[cfg(test)]
mod tests;
