//! Vendor-neutral contract for robot bridges in `copper-rs`.
//!
//! Every Repose robot bridge, whether simulator or hardware and whatever the
//! vendor, exposes the same channels, so an application swaps robots by
//! changing one bridge entry in its RON config:
//!
//! | Dir | Channel | Payload | Required |
//! |---|---|---|---|
//! | Rx | `joint_state` | `repose_kinematics::CuJointStatePayload<N>` | always |
//! | Tx | `joint_command` | `repose_controller::CuJointCommandPayload<N>` | bridges that move the robot |
//! | Tx | `session` | [`CuSessionCommand`] | bridges with a control session |
//! | Rx | `session_state` | [`CuSessionState`] | bridges with a control session |
//!
//! Bridges may add vendor channels (raw diagnostics, controller-reported
//! poses) under vendor-prefixed names such as `cri_status`.
//!
//! # Session lifecycle
//!
//! A hardware controller has to grant control and energise its drives before
//! it accepts motion. The [`CuSessionCommand`] vocabulary covers that
//! lifecycle in vendor-neutral terms, and each bridge maps it to its protocol:
//!
//! ```text
//! AcquireControl → ClearFaults → EnableDrives → … motion … → DisableDrives → ReleaseControl
//! ```
//!
//! [`CuSessionTracker`] holds the protocol-independent rules, so every bridge
//! enforces them the same way:
//! - one command in flight; a request while one is pending is refused and
//!   counted,
//! - every command except [`CuSessionCommand::AcquireControl`] is refused
//!   locally while the controller reports this client as not in control,
//! - acknowledgements and rejections only count for the sequence number that
//!   was sent,
//! - shutdown disables drives, then releases control, but only what this
//!   session itself changed.
//!
//! ```rust
//! use repose_robot_interface::{CuSessionCommand, CuSessionDecision, CuSessionResult, CuSessionTracker};
//!
//! let mut tracker = CuSessionTracker::default();
//! tracker.on_control_report(false);
//! assert_eq!(tracker.request(CuSessionCommand::EnableDrives), CuSessionDecision::Refused);
//!
//! assert_eq!(tracker.request(CuSessionCommand::AcquireControl), CuSessionDecision::Send);
//! let command = tracker.take_outgoing().unwrap();
//! tracker.mark_sent(command, 7);
//! tracker.on_control_report(true); // completes AcquireControl
//! assert_eq!(tracker.state().result, CuSessionResult::Acknowledged);
//! assert!(tracker.state().control_acquired);
//! ```
#![cfg_attr(not(feature = "std"), no_std)]

mod session;

pub use session::CuSessionCommand;
pub use session::CuSessionDecision;
pub use session::CuSessionDetail;
pub use session::CuSessionRejection;
pub use session::CuSessionResult;
pub use session::CuSessionState;
pub use session::CuSessionTracker;
