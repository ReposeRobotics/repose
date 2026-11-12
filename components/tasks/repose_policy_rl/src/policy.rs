//! [`CuPolicyTask`]: builds the observation, runs a stateless feed-forward
//! ONNX policy, and publishes both the observation and the raw action so a
//! recorded run shows what the model saw and produced.

use crate::backend::CuPolicyBackend;
use crate::observation::CuObservationTerm;
use crate::observation::fill;
use crate::observation::validate_layout;
use crate::payload::CuPolicyAction;
use crate::payload::CuPolicyObservation;
use crate::payload::CuPolicyStatus;
use bincode::Decode;
use bincode::Encode;
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
use std::path::Path;

/// Parameters of [`CuPolicyTask`]; the node `config:` block.
///
/// ```ron
/// config: {
///     "model": "policy.onnx",
///     "model_hash": 1234567890,
///     "decimation": 1,
///     "default_positions": [{"revolute": 0.0}, {"revolute": 0.0}],
///     "observation": [
///         {"joint_positions": {"scale": 1.0}},
///         {"joint_velocities": {"scale": 0.1}},
///         {"goal_error": {"scale": 1.0}},
///         "previous_action",
///     ],
/// }
/// ```
///
/// `N` is the joint count, `OBS` the model input width, and `ACT` the model
/// output width; all three come from the task type in the RON graph (for
/// example `repose_policy_rl::CuPolicyTask<6, 24, 6>`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Reflect)]
#[reflect(no_field_bounds, from_reflect = false)]
// Field-level defaults: serde does not support container `default` on const-generic structs.
#[serde(deny_unknown_fields)]
pub struct CuPolicyParams<const N: usize, const OBS: usize, const ACT: usize> {
    /// ONNX model path, resolved relative to the process working directory.
    /// Required: an empty path is rejected by [`Self::validate`].
    #[serde(default)]
    pub model: String,
    /// Optional FNV-1a 64-bit hash of the model file bytes. When set, a
    /// mismatch is a startup error, so a replay with a different model file
    /// fails instead of silently diverging.
    #[serde(default)]
    pub model_hash: Option<u64>,
    /// Infer every `decimation`-th cycle and hold the action in between.
    /// Counts cycles, not wall time, so replay is cycle-exact. The policy
    /// period is `decimation / rate_target_hz` and must equal the period the
    /// policy was trained at. An `Idle` cycle restarts the count, so the
    /// first usable cycle after a gap always infers.
    #[serde(default = "default_decimation")]
    pub decimation: u32,
    /// Reference joint positions subtracted by the `joint_positions`
    /// observation term.
    #[serde(
        default = "default_positions",
        with = "repose_kinematics::serde_arrays::position_array"
    )]
    #[reflect(ignore)]
    pub default_positions: [CuJointPosition; N],
    /// Ordered observation layout; the term widths must sum to `OBS`.
    #[serde(default)]
    pub observation: Vec<CuObservationTerm>,
}

fn default_decimation() -> u32 {
    1
}

fn default_positions<const N: usize>() -> [CuJointPosition; N] {
    [CuJointPosition::default(); N]
}

impl<const N: usize, const OBS: usize, const ACT: usize> Default for CuPolicyParams<N, OBS, ACT> {
    /// Empty model path, no hash pin, decimation 1, zero default positions,
    /// empty observation layout. Rejected by [`Self::validate`] until
    /// `model` and `observation` are set.
    fn default() -> Self {
        Self {
            model: String::new(),
            model_hash: None,
            decimation: default_decimation(),
            default_positions: default_positions(),
            observation: Vec::new(),
        }
    }
}

impl<const N: usize, const OBS: usize, const ACT: usize> CuPolicyParams<N, OBS, ACT> {
    /// Checks the model path, decimation, default positions, and that the
    /// observation layout width equals `OBS`.
    pub fn validate(&self) -> CuResult<()> {
        if self.model.is_empty() {
            return Err(CuError::from(
                "repose-policy-rl: missing required 'model' path",
            ));
        }
        if self.decimation == 0 {
            return Err(CuError::from(
                "repose-policy-rl: 'decimation' must be at least 1",
            ));
        }
        if self
            .default_positions
            .iter()
            .any(|position| !position.si_value().is_finite())
        {
            return Err(CuError::from(
                "repose-policy-rl: 'default_positions' must be finite on every joint",
            ));
        }
        validate_layout(&self.observation, N, OBS, ACT)
    }
}

/// Runs a stateless feed-forward ONNX policy inside the Copper graph.
///
/// Inputs are `(joint state, goal)`: a
/// [`CuJointStatePayload`](repose_kinematics::CuJointStatePayload) and a
/// [`CuJointPositionsPayload`](repose_kinematics::CuJointPositionsPayload).
/// Outputs are `(observation, action)`: the [`CuPolicyObservation`] built
/// this cycle (what the model was fed when the action is `Fresh`) and the
/// raw [`CuPolicyAction`]. Both outputs carry
/// the joint state input's `Tov`. The node `config:` block is a
/// [`CuPolicyParams`].
///
/// Without a usable joint state or goal the action is
/// [`CuPolicyStatus::Idle`] with zero values, and the previous action and
/// the decimation count are reset. A non-finite observation, a joint whose
/// unit kind differs between the state and the goal or default positions it
/// is compared with (or between a joint's position and velocity), a failed
/// inference, or a non-finite model output is [`CuPolicyStatus::Fault`]
/// with zero values. The inputs are checked on every cycle, so bad inputs
/// fault a held cycle too. A fault lasts until the next inference: the
/// cycles in between report `Fault`, never `Held`. Neither
/// case is a `CuError`, because the downstream action task turns the status
/// into a held setpoint.
///
/// Frozen state (last action, decimation cycle counter, last status)
/// round-trips through `Freezable`, so deterministic replay resumes
/// identically. The previous action is kept here, not fed back through the
/// graph, because Copper graphs are acyclic.
#[derive(Reflect)]
#[reflect(no_field_bounds, from_reflect = false)]
pub struct CuPolicyTask<const N: usize, const OBS: usize, const ACT: usize> {
    #[reflect(ignore)]
    params: CuPolicyParams<N, OBS, ACT>,
    #[reflect(ignore)]
    default_positions_si: [f64; N],
    #[reflect(ignore)]
    backend: CuPolicyBackend,
    #[reflect(ignore)]
    last_action: [f32; ACT],
    cycle: u64,
    last_status: CuPolicyStatus,
}

impl<const N: usize, const OBS: usize, const ACT: usize> CuPolicyTask<N, OBS, ACT> {
    /// Creates a policy task from validated params, loading and optimizing
    /// the ONNX model. The model must have one `f32` input of shape
    /// `[1, OBS]` and one `f32` output of shape `[1, ACT]`; when
    /// `model_hash` is set it must match the model file.
    pub fn from_params(params: CuPolicyParams<N, OBS, ACT>) -> CuResult<Self> {
        params.validate()?;
        let backend = CuPolicyBackend::load(Path::new(&params.model), OBS, ACT, params.model_hash)?;
        let default_positions_si = params.default_positions.map(|position| position.si_value());
        Ok(Self {
            params,
            default_positions_si,
            backend,
            last_action: [0.0; ACT],
            cycle: 0,
            last_status: CuPolicyStatus::Idle,
        })
    }

    /// Params this task runs with.
    pub fn params(&self) -> &CuPolicyParams<N, OBS, ACT> {
        &self.params
    }

    /// Whether every joint has the same unit kind in the values the layout
    /// subtracts from each other. Radians minus meters is not an observation.
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
            CuObservationTerm::PreviousAction => true,
        })
    }
}

impl<const N: usize, const OBS: usize, const ACT: usize> Freezable for CuPolicyTask<N, OBS, ACT> {
    fn freeze<E: Encoder>(&self, encoder: &mut E) -> Result<(), EncodeError> {
        Encode::encode(&self.last_action, encoder)?;
        Encode::encode(&self.cycle, encoder)?;
        Encode::encode(&self.last_status, encoder)
    }

    fn thaw<D: Decoder>(&mut self, decoder: &mut D) -> Result<(), DecodeError> {
        self.last_action = Decode::decode(decoder)?;
        self.cycle = Decode::decode(decoder)?;
        self.last_status = Decode::decode(decoder)?;
        Ok(())
    }
}

impl<const N: usize, const OBS: usize, const ACT: usize> CuTask for CuPolicyTask<N, OBS, ACT> {
    type Resources<'r> = ();
    type Input<'m> = input_msg!('m, CuJointStatePayload<N>, CuJointPositionsPayload<N>);
    type Output<'m> = output_msg!(CuPolicyObservation<OBS>, CuPolicyAction<ACT>);

    fn new(cfg: Option<&ComponentConfig>, _res: Self::Resources<'_>) -> CuResult<Self>
    where
        Self: Sized,
    {
        let params = cfg
            .map(ComponentConfig::deserialize_into::<CuPolicyParams<N, OBS, ACT>>)
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
        let (observation_out, action_out) = output;
        observation_out.tov = input.0.tov;
        action_out.tov = input.0.tov;

        let usable = match (input.0.payload(), input.1.payload()) {
            (Some(state), Some(goal)) if state.status == CuRobotStatus::Ready => {
                Some((state, goal))
            }
            _ => None,
        };
        let Some((state, goal)) = usable else {
            self.last_status = CuPolicyStatus::Idle;
            self.last_action = [0.0; ACT];
            self.cycle = 0;
            observation_out.set_payload(CuPolicyObservation::default());
            action_out.set_payload(CuPolicyAction {
                values: [0.0; ACT],
                status: CuPolicyStatus::Idle,
            });
            return Ok(());
        };

        let mut observation = CuPolicyObservation::<OBS>::default();
        fill(
            &self.params.observation,
            state,
            goal,
            &self.default_positions_si,
            &self.last_action,
            &mut observation.values,
        );
        observation_out.set_payload(observation.clone());

        // Checked on every cycle, not only when inferring: a held action
        // must not outlive inputs the policy could not have acted on.
        let input_bad = !self.kinds_match(state, goal)
            || observation.values.iter().any(|value| !value.is_finite());
        let infer_now = self.cycle.is_multiple_of(u64::from(self.params.decimation));
        self.cycle = self.cycle.wrapping_add(1);
        let (values, status) = if infer_now {
            let mut fresh = [0.0f32; ACT];
            let failed = input_bad
                || self.backend.infer(&observation.values, &mut fresh).is_err()
                || fresh.iter().any(|value| !value.is_finite());
            if failed {
                self.last_action = [0.0; ACT];
                ([0.0; ACT], CuPolicyStatus::Fault)
            } else {
                self.last_action = fresh;
                (fresh, CuPolicyStatus::Fresh)
            }
        } else if input_bad || self.last_status == CuPolicyStatus::Fault {
            // A fault lasts until the next inference; its zero values are
            // never presented as a usable held action.
            ([0.0; ACT], CuPolicyStatus::Fault)
        } else {
            (self.last_action, CuPolicyStatus::Held)
        };
        self.last_status = status;
        action_out.set_payload(CuPolicyAction { values, status });
        Ok(())
    }
}
