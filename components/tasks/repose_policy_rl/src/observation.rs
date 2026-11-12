//! The ordered observation layout: a list of terms whose widths must sum to
//! `OBS`, filled in place from the joint state, the goal, and the last raw
//! action.

use cu29::prelude::*;
use repose_kinematics::CuJointPositionsPayload;
use repose_kinematics::CuJointStatePayload;
use serde::Deserialize;
use serde::Serialize;

fn default_scale() -> f64 {
    1.0
}

/// One term of the observation layout, in the order it appears in the model
/// input vector.
///
/// In RON the layout is a list of terms; `scale` is dimensionless and
/// defaults to `1.0`:
///
/// ```ron
/// "observation": [
///     {"joint_positions": {"scale": 1.0}},
///     {"joint_velocities": {"scale": 0.1}},
///     {"goal_error": {"scale": 1.0}},
///     "previous_action",
/// ]
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Reflect)]
#[serde(rename_all = "snake_case")]
pub enum CuObservationTerm {
    /// Measured position minus `default_positions`, times `scale`. Width `N`.
    JointPositions {
        /// Dimensionless multiplier.
        #[serde(default = "default_scale")]
        scale: f64,
    },
    /// Measured velocity times `scale`. Width `N`.
    JointVelocities {
        /// Dimensionless multiplier.
        #[serde(default = "default_scale")]
        scale: f64,
    },
    /// Goal position minus measured position, times `scale`. Width `N`.
    GoalError {
        /// Dimensionless multiplier.
        #[serde(default = "default_scale")]
        scale: f64,
    },
    /// Last raw action of the policy task. Width `ACT`.
    PreviousAction,
}

impl CuObservationTerm {
    /// Width of this term for a robot with `n` joints and `act` actions.
    pub const fn width(&self, n: usize, act: usize) -> usize {
        match self {
            Self::JointPositions { .. } | Self::JointVelocities { .. } | Self::GoalError { .. } => {
                n
            }
            Self::PreviousAction => act,
        }
    }

    /// Checks that the term's scale (when it has one) is finite.
    pub fn validate(&self) -> CuResult<()> {
        let scale = match self {
            Self::JointPositions { scale }
            | Self::JointVelocities { scale }
            | Self::GoalError { scale } => *scale,
            Self::PreviousAction => return Ok(()),
        };
        if !scale.is_finite() {
            return Err(CuError::from(
                "repose-policy-rl: observation term 'scale' must be finite",
            ));
        }
        Ok(())
    }
}

/// Total width of a layout for a robot with `n` joints and `act` actions.
pub(crate) fn layout_width(terms: &[CuObservationTerm], n: usize, act: usize) -> usize {
    terms.iter().map(|term| term.width(n, act)).sum()
}

/// Checks every term and that the layout width equals `obs`.
pub(crate) fn validate_layout(
    terms: &[CuObservationTerm],
    n: usize,
    obs: usize,
    act: usize,
) -> CuResult<()> {
    for term in terms {
        term.validate()?;
    }
    let width = layout_width(terms, n, act);
    if width != obs {
        return Err(CuError::from(
            "repose-policy-rl: observation layout width does not equal OBS",
        ));
    }
    Ok(())
}

/// Fills `out` in place, term by term, in layout order.
///
/// Positions and velocities are read in SI base units of each joint kind
/// (radians / meters, radians per second / meters per second), so the
/// observation is dimensionless after scaling. The caller guarantees the
/// layout width equals `OBS` (see [`validate_layout`]); an unchecked layout
/// leaves the tail of `out` at its previous values, so `out` is zeroed first.
pub(crate) fn fill<const N: usize, const OBS: usize, const ACT: usize>(
    terms: &[CuObservationTerm],
    state: &CuJointStatePayload<N>,
    goal: &CuJointPositionsPayload<N>,
    default_positions_si: &[f64; N],
    previous_action: &[f32; ACT],
    out: &mut [f32; OBS],
) {
    *out = [0.0; OBS];
    let mut offset = 0usize;
    for term in terms {
        match *term {
            CuObservationTerm::JointPositions { scale } => {
                for j in 0..N {
                    out[offset + j] =
                        ((state.positions[j].si_value() - default_positions_si[j]) * scale) as f32;
                }
            }
            CuObservationTerm::JointVelocities { scale } => {
                for j in 0..N {
                    out[offset + j] = (state.velocities[j].si_value() * scale) as f32;
                }
            }
            CuObservationTerm::GoalError { scale } => {
                for j in 0..N {
                    out[offset + j] = ((goal.positions[j].si_value()
                        - state.positions[j].si_value())
                        * scale) as f32;
                }
            }
            CuObservationTerm::PreviousAction => {
                out[offset..offset + ACT].copy_from_slice(previous_action);
            }
        }
        offset += term.width(N, ACT);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cu29::units::si::angle::radian;
    use cu29::units::si::angular_velocity::radian_per_second;
    use cu29::units::si::f64::Angle;
    use cu29::units::si::f64::AngularVelocity;
    use repose_kinematics::CuJointPosition;
    use repose_kinematics::CuJointVelocity;
    use repose_kinematics::CuRobotStatus;

    fn pos(rad: f64) -> CuJointPosition {
        CuJointPosition::Revolute(Angle::new::<radian>(rad))
    }

    fn vel(rad_s: f64) -> CuJointVelocity {
        CuJointVelocity::Revolute(AngularVelocity::new::<radian_per_second>(rad_s))
    }

    #[test]
    fn layout_width_sums_terms() {
        let terms = [
            CuObservationTerm::JointPositions { scale: 1.0 },
            CuObservationTerm::JointVelocities { scale: 0.1 },
            CuObservationTerm::GoalError { scale: 1.0 },
            CuObservationTerm::PreviousAction,
        ];
        assert_eq!(layout_width(&terms, 6, 6), 24);
        assert!(validate_layout(&terms, 6, 24, 6).is_ok());
        assert!(validate_layout(&terms, 6, 23, 6).is_err());
    }

    #[test]
    fn non_finite_scale_is_rejected() {
        let terms = [CuObservationTerm::JointPositions { scale: f64::NAN }];
        assert!(validate_layout(&terms, 2, 2, 2).is_err());
    }

    #[test]
    fn fill_writes_terms_in_order() {
        let terms = [
            CuObservationTerm::JointPositions { scale: 1.0 },
            CuObservationTerm::JointVelocities { scale: 0.1 },
            CuObservationTerm::GoalError { scale: 2.0 },
            CuObservationTerm::PreviousAction,
        ];
        let state = CuJointStatePayload::<2> {
            status: CuRobotStatus::Ready,
            positions: [pos(0.5), pos(-0.25)],
            velocities: [vel(10.0), vel(-20.0)],
        };
        let goal = CuJointPositionsPayload::<2> {
            positions: [pos(1.5), pos(0.75)],
        };
        let mut out = [9.0f32; 8];
        fill(&terms, &state, &goal, &[0.25, -0.25], &[3.0, 4.0], &mut out);
        assert_eq!(out, [0.25, 0.0, 1.0, -2.0, 2.0, 2.0, 3.0, 4.0]);
    }
}
