//! [`CuJointReachObservationTask`]: the joint-reach observation builder. It
//! turns the measured joint state and a joint goal into the flat,
//! dimensionless feature vector [`crate::CuPolicyTask`] consumes, following
//! an ordered layout of terms whose widths must sum to `IN`.

use crate::payload::CuPolicyObservation;
use bincode::de::Decoder;
use bincode::enc::Encoder;
use bincode::error::DecodeError;
use bincode::error::EncodeError;
use cu29::prelude::*;
use repose_kinematics::CuJointPosition;
use repose_kinematics::CuJointPositionsPayload;
use repose_kinematics::CuJointStatePayload;
use repose_kinematics::CuJointVelocity;
use repose_kinematics::CuRobotStatus;
use serde::Deserialize;
use serde::Serialize;

fn default_scale() -> f64 {
    1.0
}

/// One term of the joint-reach observation layout, in the order it appears
/// in the feature vector.
///
/// In RON the layout is a list of terms; `scale` is dimensionless and
/// defaults to `1.0`:
///
/// ```ron
/// "observation": [
///     {"joint_positions": {"scale": 1.0}},
///     {"joint_velocities": {"scale": 0.1}},
///     {"goal_error": {"scale": 1.0}},
/// ]
/// ```
///
/// The previous action is not a term here: [`crate::CuPolicyTask`] appends
/// it itself when `previous_action` is set, because Copper graphs are
/// acyclic.
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
}

impl CuObservationTerm {
    /// Width of this term for a robot with `n` joints.
    pub const fn width(&self, n: usize) -> usize {
        match self {
            Self::JointPositions { .. } | Self::JointVelocities { .. } | Self::GoalError { .. } => {
                n
            }
        }
    }

    /// Dimensionless multiplier of this term.
    pub const fn scale(&self) -> f64 {
        match *self {
            Self::JointPositions { scale }
            | Self::JointVelocities { scale }
            | Self::GoalError { scale } => scale,
        }
    }

    /// Checks that the term's scale is finite.
    pub fn validate(&self) -> CuResult<()> {
        if !self.scale().is_finite() {
            return Err(CuError::from(
                "repose-policy-rl: observation term 'scale' must be finite",
            ));
        }
        Ok(())
    }
}

/// Total width of a layout for a robot with `n` joints.
pub(crate) fn layout_width(terms: &[CuObservationTerm], n: usize) -> usize {
    terms.iter().map(|term| term.width(n)).sum()
}

/// Checks every term and that the layout width equals `width`.
pub(crate) fn validate_layout(terms: &[CuObservationTerm], n: usize, width: usize) -> CuResult<()> {
    for term in terms {
        term.validate()?;
    }
    if layout_width(terms, n) != width {
        return Err(CuError::from(
            "repose-policy-rl: observation layout width does not equal IN",
        ));
    }
    Ok(())
}

/// Fills `out` in place, term by term, in layout order.
///
/// Positions and velocities are read in SI base units of each joint kind
/// (radians / meters, radians per second / meters per second), so the
/// features are dimensionless after scaling. The caller guarantees the
/// layout width equals `IN` (see [`validate_layout`]). `out` is zeroed
/// first, so nothing from an earlier cycle can remain in it.
pub(crate) fn fill<const N: usize, const IN: usize>(
    terms: &[CuObservationTerm],
    state: &CuJointStatePayload<N>,
    goal: &CuJointPositionsPayload<N>,
    default_positions_si: &[f64; N],
    out: &mut [f32; IN],
) {
    *out = [0.0; IN];
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
        }
        offset += term.width(N);
    }
}

/// Parameters of [`CuJointReachObservationTask`]; the node `config:` block.
///
/// ```ron
/// config: {
///     "default_positions": [{"revolute": 0.0}, {"revolute": 0.0}],
///     "observation": [
///         {"joint_positions": {"scale": 1.0}},
///         {"joint_velocities": {"scale": 0.1}},
///         {"goal_error": {"scale": 1.0}},
///     ],
/// }
/// ```
///
/// `N` is the joint count and `IN` the feature width; both come from the
/// task type in the RON graph (for example
/// `repose_policy_rl::CuJointReachObservationTask<6, 18>`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Reflect)]
#[reflect(no_field_bounds, from_reflect = false)]
// Field-level defaults: serde does not support container `default` on const-generic structs.
#[serde(deny_unknown_fields)]
pub struct CuJointReachObservationParams<const N: usize, const IN: usize> {
    /// Reference joint positions subtracted by the `joint_positions` term.
    #[serde(
        default = "default_positions",
        with = "repose_kinematics::serde_arrays::position_array"
    )]
    #[reflect(ignore)]
    pub default_positions: [CuJointPosition; N],
    /// Ordered observation layout; the term widths must sum to `IN`.
    #[serde(default)]
    pub observation: Vec<CuObservationTerm>,
}

fn default_positions<const N: usize>() -> [CuJointPosition; N] {
    [CuJointPosition::default(); N]
}

impl<const N: usize, const IN: usize> Default for CuJointReachObservationParams<N, IN> {
    /// Zero default positions and an empty layout. Rejected by
    /// [`Self::validate`] until `observation` is set (unless `IN` is 0).
    fn default() -> Self {
        Self {
            default_positions: default_positions(),
            observation: Vec::new(),
        }
    }
}

impl<const N: usize, const IN: usize> CuJointReachObservationParams<N, IN> {
    /// Checks the default positions and that the layout width equals `IN`.
    pub fn validate(&self) -> CuResult<()> {
        if self
            .default_positions
            .iter()
            .any(|position| !position.si_value().is_finite())
        {
            return Err(CuError::from(
                "repose-policy-rl: 'default_positions' must be finite on every joint",
            ));
        }
        validate_layout(&self.observation, N, IN)
    }
}

/// Builds the joint-reach policy input from the joint state and a goal.
///
/// Inputs are `(joint state, goal)`: a [`CuJointStatePayload`] and a
/// [`CuJointPositionsPayload`]. The output is the [`CuPolicyObservation`]
/// feature vector of width `IN`, carrying the joint state input's `Tov`.
/// The node `config:` block is a [`CuJointReachObservationParams`].
///
/// This is one observation task among many: another RL task supplies its
/// own task that emits a `CuPolicyObservation<IN>` and feeds the same
/// [`crate::CuPolicyTask`].
///
/// The output contract with the policy task, which has no status field to
/// read:
///
/// - No joint state, no goal, or a robot status other than
///   [`CuRobotStatus::Ready`]: **no payload**. The policy task reports
///   `Idle`.
/// - A joint whose unit kind differs between the values a term subtracts
///   (state and default positions, state and goal, or a joint's position
///   and velocity; radians minus meters is not a feature): a payload filled
///   with **`f32::NAN`**. The policy task reports `Fault`, as it does for
///   any non-finite input.
///
/// The task is stateless and allocates nothing in `process`.
#[derive(Reflect)]
#[reflect(no_field_bounds, from_reflect = false)]
pub struct CuJointReachObservationTask<const N: usize, const IN: usize> {
    #[reflect(ignore)]
    params: CuJointReachObservationParams<N, IN>,
    #[reflect(ignore)]
    default_positions_si: [f64; N],
}

impl<const N: usize, const IN: usize> CuJointReachObservationTask<N, IN> {
    /// Creates an observation task from validated params.
    pub fn from_params(params: CuJointReachObservationParams<N, IN>) -> CuResult<Self> {
        params.validate()?;
        let default_positions_si = params.default_positions.map(|position| position.si_value());
        Ok(Self {
            params,
            default_positions_si,
        })
    }

    /// Params this task runs with.
    pub fn params(&self) -> &CuJointReachObservationParams<N, IN> {
        &self.params
    }

    /// Whether every joint has the same unit kind in the values the layout
    /// subtracts from each other. Radians minus meters is not a feature.
    fn kinds_match(
        &self,
        state: &CuJointStatePayload<N>,
        goal: &CuJointPositionsPayload<N>,
    ) -> bool {
        let same = |a: &CuJointPosition, b: &CuJointPosition| {
            core::mem::discriminant(a) == core::mem::discriminant(b)
        };
        self.params.observation.iter().all(|term| match term {
            CuObservationTerm::JointPositions { .. } => state
                .positions
                .iter()
                .zip(&self.params.default_positions)
                .all(|(a, b)| same(a, b)),
            CuObservationTerm::GoalError { .. } => state
                .positions
                .iter()
                .zip(&goal.positions)
                .all(|(a, b)| same(a, b)),
            CuObservationTerm::JointVelocities { .. } => state
                .positions
                .iter()
                .zip(&state.velocities)
                .all(|(position, velocity)| {
                    matches!(
                        (position, velocity),
                        (CuJointPosition::Revolute(_), CuJointVelocity::Revolute(_))
                            | (CuJointPosition::Prismatic(_), CuJointVelocity::Prismatic(_))
                            | (CuJointPosition::Fixed, CuJointVelocity::Fixed)
                    )
                }),
        })
    }
}

impl<const N: usize, const IN: usize> Freezable for CuJointReachObservationTask<N, IN> {
    /// Stateless: nothing to freeze.
    fn freeze<E: Encoder>(&self, _encoder: &mut E) -> Result<(), EncodeError> {
        Ok(())
    }

    fn thaw<D: Decoder>(&mut self, _decoder: &mut D) -> Result<(), DecodeError> {
        Ok(())
    }
}

impl<const N: usize, const IN: usize> CuTask for CuJointReachObservationTask<N, IN> {
    type Resources<'r> = ();
    type Input<'m> = input_msg!('m, CuJointStatePayload<N>, CuJointPositionsPayload<N>);
    type Output<'m> = output_msg!(CuPolicyObservation<IN>);

    fn new(cfg: Option<&ComponentConfig>, _res: Self::Resources<'_>) -> CuResult<Self>
    where
        Self: Sized,
    {
        let params = cfg
            .map(ComponentConfig::deserialize_into::<CuJointReachObservationParams<N, IN>>)
            .transpose()?
            .unwrap_or_default();
        Self::from_params(params)
    }

    fn process<'i, 'o>(
        &mut self,
        _ctx: &CuContext,
        input: &Self::Input<'i>,
        output: &mut Self::Output<'o>,
    ) -> CuResult<()> {
        output.tov = input.0.tov;
        let usable = match (input.0.payload(), input.1.payload()) {
            (Some(state), Some(goal)) if state.status == CuRobotStatus::Ready => {
                Some((state, goal))
            }
            _ => None,
        };
        let Some((state, goal)) = usable else {
            output.clear_payload();
            return Ok(());
        };
        let mut observation = CuPolicyObservation::<IN>::default();
        if self.kinds_match(state, goal) {
            fill(
                &self.params.observation,
                state,
                goal,
                &self.default_positions_si,
                &mut observation.values,
            );
        } else {
            observation.values = [f32::NAN; IN];
        }
        output.set_payload(observation);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cu29::units::si::angle::radian;
    use cu29::units::si::angular_velocity::radian_per_second;
    use cu29::units::si::f64::Angle;
    use cu29::units::si::f64::AngularVelocity;

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
        ];
        assert_eq!(layout_width(&terms, 6), 18);
        assert!(validate_layout(&terms, 6, 18).is_ok());
        assert!(validate_layout(&terms, 6, 17).is_err());
    }

    #[test]
    fn non_finite_scale_is_rejected() {
        let terms = [CuObservationTerm::JointPositions { scale: f64::NAN }];
        assert!(validate_layout(&terms, 2, 2).is_err());
    }

    #[test]
    fn fill_writes_terms_in_order() {
        let terms = [
            CuObservationTerm::JointPositions { scale: 1.0 },
            CuObservationTerm::JointVelocities { scale: 0.1 },
            CuObservationTerm::GoalError { scale: 2.0 },
        ];
        let state = CuJointStatePayload::<2> {
            status: CuRobotStatus::Ready,
            positions: [pos(0.5), pos(-0.25)],
            velocities: [vel(10.0), vel(-20.0)],
        };
        let goal = CuJointPositionsPayload::<2> {
            positions: [pos(1.5), pos(0.75)],
        };
        let mut out = [9.0f32; 6];
        fill(&terms, &state, &goal, &[0.25, -0.25], &mut out);
        assert_eq!(out, [0.25, 0.0, 1.0, -2.0, 2.0, 2.0]);
    }
}
