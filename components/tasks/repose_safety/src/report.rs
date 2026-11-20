//! Structured safety audit payloads emitted by [`CuSafetyShieldTask`](crate::CuSafetyShieldTask).

use bincode::Decode;
use bincode::Encode;
use cu29::prelude::*;
use cu29::units::si::f64::Length;
use cu29::units::si::f64::Velocity;
use cu29::units::si::length::meter;
use repose_collision::CuCollisionContact;
use repose_collision::CuContactKind;
use repose_collision::CuMonitorTransition;
use repose_collision::CuSceneRevision;
use serde::Deserialize;
use serde::Serialize;

/// Safety verdict produced on every cycle of [`CuSafetyShieldTask`](crate::CuSafetyShieldTask).
#[derive(
    Default, Debug, Clone, Copy, PartialEq, Eq, Encode, Decode, Serialize, Deserialize, Reflect,
)]
#[serde(rename_all = "snake_case")]
pub enum CuSafetyVerdict {
    /// Setpoint passed through unchanged (or upstream is `Idle` / cancelled).
    #[default]
    Clear,
    /// Current configuration violates robot self, environment, or attached collision margin.
    HeldCollision,
    /// Current configuration is clear, but `q + v * braking_horizon` violates collision or
    /// workspace bounds.
    HeldBrakingLookahead,
    /// TCP pose of the current configuration lies outside `workspace` bounds (`min..=max`).
    HeldWorkspaceFence,
    /// TCP linear speed from `(q, v)` exceeds `max_tcp_speed`.
    HeldTcpSpeed,
    /// Setpoint positions lie outside chain joint limits, or a setpoint value is non-finite or
    /// carries the wrong unit kind.
    HeldJointLimits,
    /// `require_scene` is `true` and no [`CuPlanningScene`](repose_collision::CuPlanningScene)
    /// payload was provided.
    HeldMissingScene,
    /// Scene or exclusions failed validation against the robot model.
    HeldInvalidScene,
}

impl CuSafetyVerdict {
    /// Returns `true` when the verdict is [`CuSafetyVerdict::Clear`].
    pub const fn is_clear(self) -> bool {
        matches!(self, Self::Clear)
    }
}

/// Tightest or offending contact, workspace plane, or joint index recorded in [`CuSafetyReport`].
#[derive(
    Default, Debug, Clone, Copy, PartialEq, Eq, Encode, Decode, Serialize, Deserialize, Reflect,
)]
#[serde(rename_all = "snake_case")]
pub enum CuSafetyWitness {
    /// No contact, fence plane, or joint witness applies.
    #[default]
    None,
    /// Robot self-collision between two link shapes.
    SelfCollision { shape_a: u16, shape_b: u16 },
    /// Collision between a robot link shape and a scene obstacle.
    Environment {
        robot_shape: u16,
        obstacle_index: u16,
    },
    /// Collision between an attached shape and a robot link shape.
    AttachedSelf {
        attached_index: u16,
        robot_shape: u16,
    },
    /// Collision between two attached shapes.
    AttachedAttached { attached_a: u16, attached_b: u16 },
    /// Collision between an attached shape and a scene obstacle.
    AttachedEnvironment {
        attached_index: u16,
        obstacle_index: u16,
    },
    /// Workspace keep-in box plane (`axis`: `0=x`, `1=y`, `2=z`; `upper`: `false=min`, `true=max`).
    WorkspaceAxis { axis: u8, upper: bool },
    /// Offending joint index (`0..N`).
    Joint { index: u16 },
}

impl CuSafetyWitness {
    pub(crate) const fn from_contact(
        contact: CuCollisionContact,
        robot_shape_count: usize,
    ) -> Self {
        let rc = robot_shape_count as u16;
        match contact.kind {
            CuContactKind::SelfCollision => {
                if contact.link_index < rc && contact.partner_index < rc {
                    Self::SelfCollision {
                        shape_a: contact.link_index,
                        shape_b: contact.partner_index,
                    }
                } else if contact.link_index >= rc && contact.partner_index < rc {
                    Self::AttachedSelf {
                        attached_index: contact.link_index - rc,
                        robot_shape: contact.partner_index,
                    }
                } else {
                    Self::AttachedAttached {
                        attached_a: contact.link_index.saturating_sub(rc),
                        attached_b: contact.partner_index.saturating_sub(rc),
                    }
                }
            }
            CuContactKind::EnvironmentCollision => {
                if contact.link_index < rc {
                    Self::Environment {
                        robot_shape: contact.link_index,
                        obstacle_index: contact.partner_index,
                    }
                } else {
                    Self::AttachedEnvironment {
                        attached_index: contact.link_index - rc,
                        obstacle_index: contact.partner_index,
                    }
                }
            }
        }
    }
}

/// Structured black-box safety audit record emitted every cycle by
/// [`CuSafetyShieldTask`](crate::CuSafetyShieldTask).
#[derive(Debug, Clone, PartialEq, Encode, Decode, Serialize, Deserialize, Reflect)]
pub struct CuSafetyReport {
    /// Cycle safety verdict (`Clear` or `Held*`).
    pub verdict: CuSafetyVerdict,
    /// Edge transition from the internal [`CuCollisionMonitor`](repose_collision::CuCollisionMonitor)
    /// (`in_violation = verdict != CuSafetyVerdict::Clear`).
    pub transition: CuMonitorTransition,
    /// `true` when `verdict == Clear` and `min_clearance < model.safety_margin() + near_miss_margin`.
    pub near_miss: bool,
    /// Minimum signed clearance across collision checks and workspace fence planes
    /// (`f64::INFINITY` meters when idle or unchecked).
    pub min_clearance: Length,
    /// Computed TCP linear speed in the base frame for active setpoints (`0.0` otherwise).
    pub tcp_speed: Velocity,
    /// Tightest or offending contact, workspace plane, or joint index.
    pub witness: CuSafetyWitness,
    /// Scene revision used for this check (`CuSceneRevision::INITIAL` when no scene was given).
    pub scene_revision: CuSceneRevision,
}

impl Default for CuSafetyReport {
    fn default() -> Self {
        Self {
            verdict: CuSafetyVerdict::default(),
            transition: CuMonitorTransition::default(),
            near_miss: false,
            min_clearance: Length::new::<meter>(f64::INFINITY),
            tcp_speed: Velocity::default(),
            witness: CuSafetyWitness::default(),
            scene_revision: CuSceneRevision::INITIAL,
        }
    }
}
