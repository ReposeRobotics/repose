//! ONNX reinforcement-learning policy execution for `copper-rs`.
//!
//! [`CuPolicyTask`] runs a small stateless feed-forward policy (an MLP
//! exported to ONNX from MuJoCo Playground, Isaac Lab, or a similar trainer)
//! inside a Copper graph. It consumes the measured joint state and a joint
//! goal, builds the configured observation, infers with the pure-Rust
//! `tract-onnx` runtime, and publishes two payloads: the
//! [`CuPolicyObservation`] built this cycle and the raw [`CuPolicyAction`]
//! before any scaling or limiting. On a `Fresh` cycle the observation is
//! exactly what the model was fed; `Held` and `Idle` cycles run no
//! inference, and a `Fault` cycle may or may not have. Every observation and every raw action is in the
//! log, so a recorded run shows whether
//! the model or the downstream execution layer caused a given motion.
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
//! the one documented exception while it uses `tract`, which allocates
//! output tensors per run. The bound is measured, not assumed:
//! `tests/alloc.rs` counts allocations over 1000 `process` calls after
//! warm-up and asserts an upper bound per call. Measured with the
//! `linear_24_6` fixture: **14 allocations per `process` call** (asserted
//! bound: 32). Model loading at startup allocates
//! freely and runs once.
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
//!     id: "policy",
//!     type: "repose_policy_rl::CuPolicyTask<6, 24, 6>",
//!     config: {
//!         "model": "policy.onnx",
//!         "decimation": 1,
//!         "observation": [
//!             {"joint_positions": {"scale": 1.0}},
//!             {"joint_velocities": {"scale": 0.1}},
//!             {"goal_error": {"scale": 1.0}},
//!             "previous_action",
//!         ],
//!     },
//! )
//! ```

mod action;
mod backend;
mod observation;
mod payload;
mod policy;

pub use action::CuJointActionKind;
pub use action::CuJointActionParams;
pub use action::CuJointActionTask;
pub use observation::CuObservationTerm;
pub use payload::CuPolicyAction;
pub use payload::CuPolicyObservation;
pub use payload::CuPolicyStatus;
pub use policy::CuPolicyParams;
pub use policy::CuPolicyTask;

#[cfg(test)]
mod tests;
