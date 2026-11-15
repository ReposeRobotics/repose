//! ONNX reinforcement-learning policy execution for `copper-rs`.
//!
//! The crate splits a policy into two Copper tasks:
//!
//! - an **observation task** turns task-specific inputs into one flat,
//!   dimensionless [`CuPolicyObservation<IN>`] feature vector.
//!   [`CuJointReachObservationTask`] is the joint-reaching one: it builds the
//!   features from the measured joint state and a joint goal, following an
//!   ordered layout of [`CuObservationTerm`]s. Other RL tasks supply their
//!   own observation task emitting `CuPolicyObservation<IN>`.
//! - [`CuPolicyTask`] is pure inference and knows nothing about joints. It
//!   runs a small stateless feed-forward policy (an MLP exported to ONNX
//!   from MuJoCo Playground, Isaac Lab, or a similar trainer) with the
//!   pure-Rust `tract-onnx` runtime, optionally appending its own last raw
//!   action to the features, and publishes two payloads: the
//!   [`CuPolicyObservation`] model input built this cycle and the raw
//!   [`CuPolicyAction`] before any scaling or limiting.
//!
//! On a `Fresh` cycle the published observation is exactly what the model
//! was fed; `Held` and `Idle` cycles run no inference, and a `Fault` cycle
//! may or may not have. Every observation and every raw action is in the
//! log, so a recorded run shows whether the model or the downstream
//! execution layer caused a given motion. An observation task signals "no
//! input" by publishing no payload (the policy reports `Idle`) and a fault
//! by publishing `NaN` features (the policy reports `Fault`).
//! [`CuJointActionTask`] turns the raw action into the limited joint
//! setpoint that `repose_controller::CuControllerTask` consumes.
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
//! the one documented exception while [`CuPolicyTask`] uses `tract`, which
//! allocates output tensors per run. [`CuJointReachObservationTask`]
//! allocates nothing. The bound is measured, not assumed: `tests/alloc.rs`
//! counts allocations over 1000 cycles of the observation task feeding the
//! policy task after warm-up and asserts an upper bound per cycle. Measured
//! with the `linear_24_6` fixture: **14 allocations per cycle**, all in the
//! policy task's `process` (asserted bound: 32). Model loading at startup
//! allocates freely and runs once.
//!
//! # Determinism
//!
//! Decimation counts cycles, not wall time, and the last action, cycle
//! counter, and last status round-trip through `Freezable`, so replay is
//! cycle-exact. Pin `model_hash` in RON so a replay against a different
//! model file fails at startup instead of silently diverging. The policy
//! period is `decimation / rate_target_hz` and must equal the period the
//! policy was trained at. Model paths resolve relative to the process
//! working directory.
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
mod observation;
mod payload;
mod policy;

pub use action::CuJointActionKind;
pub use action::CuJointActionParams;
pub use action::CuJointActionTask;
pub use observation::CuJointReachObservationParams;
pub use observation::CuJointReachObservationTask;
pub use observation::CuObservationTerm;
pub use payload::CuPolicyAction;
pub use payload::CuPolicyObservation;
pub use payload::CuPolicyStatus;
pub use policy::CuPolicyParams;
pub use policy::CuPolicyTask;

#[cfg(test)]
mod tests;
