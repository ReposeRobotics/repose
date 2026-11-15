//! [`CuPolicyTask`]: runs a stateless feed-forward ONNX policy on a flat
//! feature vector produced upstream, and publishes both the vector the model
//! was fed and the raw action so a recorded run shows what the model saw and
//! produced.

use crate::backend::CuPolicyBackend;
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
///     "previous_action": true,
/// }
/// ```
///
/// `IN` is the width of the upstream feature vector, `OBS` the model input
/// width, and `ACT` the model output width; all three come from the task
/// type in the RON graph (for example
/// `repose_policy_rl::CuPolicyTask<18, 24, 6>`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Reflect)]
#[reflect(no_field_bounds, from_reflect = false)]
// Field-level defaults: serde does not support container `default` on const-generic structs.
#[serde(deny_unknown_fields)]
pub struct CuPolicyParams<const IN: usize, const OBS: usize, const ACT: usize> {
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
    /// Append this task's last raw action after the input features, so the
    /// model input is `[features (IN) | last action (ACT)]` and
    /// `OBS == IN + ACT`. When `false`, the model input is the features
    /// alone and `OBS == IN`.
    #[serde(default)]
    pub previous_action: bool,
}

fn default_decimation() -> u32 {
    1
}

impl<const IN: usize, const OBS: usize, const ACT: usize> Default for CuPolicyParams<IN, OBS, ACT> {
    /// Empty model path, no hash pin, decimation 1, no previous action.
    /// Rejected by [`Self::validate`] until `model` is set.
    fn default() -> Self {
        Self {
            model: String::new(),
            model_hash: None,
            decimation: default_decimation(),
            previous_action: false,
        }
    }
}

impl<const IN: usize, const OBS: usize, const ACT: usize> CuPolicyParams<IN, OBS, ACT> {
    /// Checks the model path, decimation, and that `OBS` is `IN + ACT` with
    /// `previous_action` or `IN` without it.
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
        if self.previous_action {
            if IN.checked_add(ACT) != Some(OBS) {
                return Err(CuError::from(
                    "repose-policy-rl: with 'previous_action: true' the model input width OBS must equal IN + ACT",
                ));
            }
        } else if OBS != IN {
            return Err(CuError::from(
                "repose-policy-rl: with 'previous_action: false' the model input width OBS must equal IN",
            ));
        }
        Ok(())
    }
}

/// Runs a stateless feed-forward ONNX policy inside the Copper graph.
///
/// The input is one flat, dimensionless [`CuPolicyObservation<IN>`] feature
/// vector produced by an upstream observation task (for joint reaching,
/// [`crate::CuJointReachObservationTask`]); this task knows nothing about
/// joints. Outputs are `(observation, action)`: the [`CuPolicyObservation`]
/// model input of this cycle (the features, followed by the last raw action
/// when `previous_action` is set; what the model was fed when the action is
/// `Fresh`) and the raw [`CuPolicyAction`]. Both outputs carry the input's
/// `Tov`. The node `config:` block is a [`CuPolicyParams`].
///
/// The upstream contract, since the input has no status field:
///
/// - **No input payload** (the observation task cannot build features, for
///   example the robot is not ready): [`CuPolicyStatus::Idle`] with zero
///   values and a zero observation; the previous action and the decimation
///   count are reset.
/// - **A non-finite feature** (an observation task signals a fault by
///   filling its payload with `NaN`), a failed inference, or a non-finite
///   model output: [`CuPolicyStatus::Fault`] with zero values.
///
/// The model input is checked on every cycle, so a bad input faults a held
/// cycle too and clears the previous action. A fault lasts until the next
/// inference: the cycles in between report `Fault`, never `Held`. Neither
/// case is a `CuError`, because the downstream action task turns the status
/// into a held setpoint.
///
/// Frozen state (last action, decimation cycle counter, last status)
/// round-trips through `Freezable`, so deterministic replay resumes
/// identically. The previous action is kept here, not fed back through the
/// graph, because Copper graphs are acyclic.
#[derive(Reflect)]
#[reflect(no_field_bounds, from_reflect = false)]
pub struct CuPolicyTask<const IN: usize, const OBS: usize, const ACT: usize> {
    #[reflect(ignore)]
    params: CuPolicyParams<IN, OBS, ACT>,
    #[reflect(ignore)]
    backend: CuPolicyBackend,
    #[reflect(ignore)]
    last_action: [f32; ACT],
    cycle: u64,
    last_status: CuPolicyStatus,
}

impl<const IN: usize, const OBS: usize, const ACT: usize> CuPolicyTask<IN, OBS, ACT> {
    /// Creates a policy task from validated params, loading and optimizing
    /// the ONNX model. The model must have one `f32` input of shape
    /// `[1, OBS]` and one `f32` output of shape `[1, ACT]`; when
    /// `model_hash` is set it must match the model file.
    pub fn from_params(params: CuPolicyParams<IN, OBS, ACT>) -> CuResult<Self> {
        params.validate()?;
        let backend = CuPolicyBackend::load(Path::new(&params.model), OBS, ACT, params.model_hash)?;
        Ok(Self {
            params,
            backend,
            last_action: [0.0; ACT],
            cycle: 0,
            last_status: CuPolicyStatus::Idle,
        })
    }

    /// Params this task runs with.
    pub fn params(&self) -> &CuPolicyParams<IN, OBS, ACT> {
        &self.params
    }

    /// Writes the model input: the features, then the last raw action when
    /// `previous_action` is set. Widths were checked by
    /// [`CuPolicyParams::validate`].
    fn build_input(&self, features: &[f32; IN], out: &mut [f32; OBS]) {
        *out = [0.0; OBS];
        out[..IN].copy_from_slice(features);
        if self.params.previous_action {
            out[IN..].copy_from_slice(&self.last_action);
        }
    }
}

impl<const IN: usize, const OBS: usize, const ACT: usize> Freezable for CuPolicyTask<IN, OBS, ACT> {
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

impl<const IN: usize, const OBS: usize, const ACT: usize> CuTask for CuPolicyTask<IN, OBS, ACT> {
    type Resources<'r> = ();
    type Input<'m> = input_msg!(CuPolicyObservation<IN>);
    type Output<'m> = output_msg!(CuPolicyObservation<OBS>, CuPolicyAction<ACT>);

    fn new(cfg: Option<&ComponentConfig>, _res: Self::Resources<'_>) -> CuResult<Self>
    where
        Self: Sized,
    {
        let params = cfg
            .map(ComponentConfig::deserialize_into::<CuPolicyParams<IN, OBS, ACT>>)
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
        observation_out.tov = input.tov;
        action_out.tov = input.tov;

        let Some(features) = input.payload() else {
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
        self.build_input(&features.values, &mut observation.values);
        observation_out.set_payload(observation.clone());

        // Checked on every cycle, not only when inferring: a held action
        // must not outlive inputs the policy could not have acted on.
        let input_bad = observation.values.iter().any(|value| !value.is_finite());
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
            // never presented as a usable held action. The previous action
            // is cleared so the appended previous action matches what was
            // published.
            self.last_action = [0.0; ACT];
            ([0.0; ACT], CuPolicyStatus::Fault)
        } else {
            (self.last_action, CuPolicyStatus::Held)
        };
        self.last_status = status;
        action_out.set_payload(CuPolicyAction { values, status });
        Ok(())
    }
}
