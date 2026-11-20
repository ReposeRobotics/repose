//! Action-chunk execution (`CuActionChunkTask`) and chunked ONNX inference
//! (`CuChunkPolicyTask`).
//!
//! Modern imitation, diffusion, and vision-language-action policies predict an
//! `H`-step horizon chunk (`[1, H, ACT]`) at 5–50 Hz rather than a single
//! step. [`CuChunkPolicyTask`] runs such an ONNX model on `tract-onnx` (and an
//! external VLA bridge or custom task can publish [`CuPolicyActionChunk`]
//! directly); [`CuActionChunkTask`] buffers or temporally ensembles overlapping
//! chunks with zero hot-path allocations and emits one [`CuPolicyAction`] per
//! control cycle for [`crate::CuJointActionTask`],
//! [`crate::CuCartesianActionTask`], and [`crate::CuGripperActionTask`].

use crate::backend::CuPolicyBackend;
use crate::payload::CuPolicyAction;
use crate::payload::CuPolicyActionChunk;
use crate::payload::CuPolicyObservation;
use crate::payload::CuPolicyStatus;
use crate::policy::CuBoundPolicy;
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

/// Canonical exponential temporal-ensembling decay (`0.01`, Zhao et al., ACT).
pub const DEFAULT_ENSEMBLE_DECAY: f64 = 0.01;

fn default_ensemble_decay() -> f64 {
    DEFAULT_ENSEMBLE_DECAY
}

fn default_decimation() -> u32 {
    1
}

/// How [`CuActionChunkTask`] combines incoming `Fresh` chunks across cycles.
///
/// In RON:
///
/// ```ron
/// "mode": "replace"
/// // or
/// "mode": {"temporal_ensemble": {"decay": 0.01}}
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Reflect)]
#[serde(rename_all = "snake_case")]
pub enum CuActionChunkMode {
    /// Receding-horizon execution: each `Fresh` chunk discards any remaining
    /// buffered steps and starts immediately from its step `0`. Use for
    /// multimodal policies (diffusion, flow matching) where averaging distinct
    /// action modes across chunks is undesirable.
    Replace,
    /// Exponential temporal ensembling (Zhao et al., ACT): overlapping chunks
    /// covering the same target step are combined with normalized weights
    /// `w_i = exp(-decay * i)`, where `i = 0` is the oldest chunk covering
    /// that step and larger `i` are newer chunks.
    ///
    /// - `decay > 0.0` weights older committed predictions higher for smooth
    ///   execution (ACT default: [`DEFAULT_ENSEMBLE_DECAY`] = `0.01`).
    /// - `decay == 0.0` averages all overlapping predictions uniformly.
    /// - `decay < 0.0` weights newer predictions higher for faster reaction.
    TemporalEnsemble {
        /// Exponential decay per overlapping chunk (dimensionless); defaults
        /// to [`DEFAULT_ENSEMBLE_DECAY`] (`0.01`).
        #[serde(default = "default_ensemble_decay")]
        decay: f64,
    },
}

impl Default for CuActionChunkMode {
    fn default() -> Self {
        Self::TemporalEnsemble {
            decay: DEFAULT_ENSEMBLE_DECAY,
        }
    }
}

/// Parameters of [`CuActionChunkTask`]; the node `config:` block.
///
/// ```ron
/// config: {
///     "mode": {"temporal_ensemble": {"decay": 0.01}}, // or "replace"
///     "step_decimation": 1,                           // cycles per chunk step
///     "execution_horizon": 8,                         // optional; <= H
/// }
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Reflect)]
#[serde(default, deny_unknown_fields)]
pub struct CuActionChunkParams {
    /// How incoming `Fresh` chunks update the horizon; defaults to
    /// `temporal_ensemble` with `decay = 0.01`.
    pub mode: CuActionChunkMode,
    /// Control cycles each chunk step spans (`>= 1`); defaults to `1`.
    ///
    /// Cycle `0` of a step emits [`CuPolicyStatus::Fresh`]; the remaining
    /// `step_decimation - 1` cycles repeat that step's action with
    /// [`CuPolicyStatus::Held`], so downstream delta action tasks apply the
    /// step once and hold between steps. The chunk step period is
    /// `step_decimation / rate_target_hz` and must match the step spacing the
    /// policy was trained at.
    pub step_decimation: u32,
    /// Maximum number of chunk steps executed without a new `Fresh` chunk
    /// before starving to [`CuPolicyStatus::Idle`] (`1..=H`). `None` allows
    /// all `H` steps of a chunk to execute.
    pub execution_horizon: Option<usize>,
}

impl Default for CuActionChunkParams {
    fn default() -> Self {
        Self {
            mode: CuActionChunkMode::default(),
            step_decimation: default_decimation(),
            execution_horizon: None,
        }
    }
}

impl CuActionChunkParams {
    /// Checks `step_decimation`, `execution_horizon`, and finiteness of any
    /// temporal-ensemble decay. Horizon-dependent bounds (`execution_horizon <= H`
    /// and exponential overflow over `H`) are checked by
    /// [`CuActionChunkTask::from_params`].
    pub fn validate(&self) -> CuResult<()> {
        if self.step_decimation == 0 {
            return Err(CuError::from(
                "repose-policy-rl: 'step_decimation' must be at least 1",
            ));
        }
        if self.execution_horizon == Some(0) {
            return Err(CuError::from(
                "repose-policy-rl: 'execution_horizon' must be at least 1",
            ));
        }
        if let CuActionChunkMode::TemporalEnsemble { decay } = self.mode
            && !decay.is_finite()
        {
            return Err(CuError::from(
                "repose-policy-rl: temporal ensemble 'decay' must be finite",
            ));
        }
        Ok(())
    }
}

/// Running weighted accumulator for one future step in the ring buffer.
#[derive(Clone, Copy, Debug, PartialEq, Encode, Decode)]
struct ChunkSlot<const ACT: usize> {
    weighted_sum: [f64; ACT],
    weight_total: f64,
    count: u32,
}

impl<const ACT: usize> Default for ChunkSlot<ACT> {
    fn default() -> Self {
        Self {
            weighted_sum: [0.0; ACT],
            weight_total: 0.0,
            count: 0,
        }
    }
}

/// Buffers or temporally ensembles [`CuPolicyActionChunk<H, ACT>`] inputs into
/// one [`CuPolicyAction<ACT>`] per control cycle, with zero heap allocation on
/// the hot path.
///
/// Downstream [`crate::CuJointActionTask`], [`crate::CuCartesianActionTask`],
/// and [`crate::CuGripperActionTask`] connect unchanged.
///
/// # Example
///
/// ```rust
/// use cu29::prelude::*;
/// use repose_policy_rl::{
///     CuActionChunkMode, CuActionChunkParams, CuActionChunkTask, CuPolicyAction,
///     CuPolicyActionChunk, CuPolicyStatus,
/// };
///
/// let mut task = CuActionChunkTask::<3, 2>::from_params(CuActionChunkParams {
///     mode: CuActionChunkMode::TemporalEnsemble { decay: 0.0 },
///     ..CuActionChunkParams::default()
/// })?;
/// let (ctx, _clock) = CuContext::new_mock_clock();
///
/// let c0 = CuMsg::new(Some(CuPolicyActionChunk {
///     values: [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
///     status: CuPolicyStatus::Fresh,
/// }));
/// let mut out = CuMsg::<CuPolicyAction<2>>::default();
/// task.process(&ctx, &c0, &mut out)?;
/// assert_eq!(out.payload().unwrap().values, [1.0, 2.0]);
///
/// // Next cycle: a second chunk overlaps step 1 ([3.0, 4.0] and [5.0, 8.0]).
/// let c1 = CuMsg::new(Some(CuPolicyActionChunk {
///     values: [[5.0, 8.0], [7.0, 10.0], [9.0, 12.0]],
///     status: CuPolicyStatus::Fresh,
/// }));
/// task.process(&ctx, &c1, &mut out)?;
/// assert_eq!(out.payload().unwrap().values, [4.0, 6.0]);
/// # Ok::<(), CuError>(())
/// ```
#[derive(Reflect)]
#[reflect(no_field_bounds, from_reflect = false)]
pub struct CuActionChunkTask<const H: usize, const ACT: usize> {
    params: CuActionChunkParams,
    #[reflect(ignore)]
    weights: [f64; H],
    max_steps: usize,
    #[reflect(ignore)]
    slots: [ChunkSlot<ACT>; H],
    head: usize,
    step_cycle: u32,
    steps_since_fresh: usize,
    #[reflect(ignore)]
    last_action: [f32; ACT],
    faulted: bool,
}

impl<const H: usize, const ACT: usize> CuActionChunkTask<H, ACT> {
    /// Creates an action-chunk task from validated params.
    pub fn from_params(params: CuActionChunkParams) -> CuResult<Self> {
        if H == 0 {
            return Err(CuError::from(
                "repose-policy-rl: chunk horizon H must be at least 1",
            ));
        }
        if ACT == 0 {
            return Err(CuError::from(
                "repose-policy-rl: action width ACT must be at least 1",
            ));
        }
        params.validate()?;
        let max_steps = match params.execution_horizon {
            Some(limit) if limit > H => {
                return Err(CuError::from(
                    "repose-policy-rl: 'execution_horizon' must not exceed the chunk horizon H",
                ));
            }
            Some(limit) => limit,
            None => H,
        };
        let mut weights = [1.0f64; H];
        if let CuActionChunkMode::TemporalEnsemble { decay } = params.mode {
            let mut sum = 0.0f64;
            for (i, slot) in weights.iter_mut().enumerate() {
                let w = (-decay * (i as f64)).exp();
                *slot = w;
                sum += w;
            }
            if !weights.iter().all(|w| w.is_finite()) || !sum.is_finite() || sum <= 0.0 {
                return Err(CuError::from(
                    "repose-policy-rl: temporal ensemble 'decay' overflows over horizon H",
                ));
            }
        }
        Ok(Self {
            params,
            weights,
            max_steps,
            slots: [ChunkSlot::default(); H],
            head: 0,
            step_cycle: 0,
            steps_since_fresh: 0,
            last_action: [0.0; ACT],
            faulted: false,
        })
    }

    /// Params this task runs with.
    pub fn params(&self) -> &CuActionChunkParams {
        &self.params
    }

    fn reset(&mut self) {
        self.slots = [ChunkSlot::default(); H];
        self.head = 0;
        self.step_cycle = 0;
        self.steps_since_fresh = 0;
        self.last_action = [0.0; ACT];
        self.faulted = false;
    }

    fn fault(&mut self) -> CuPolicyAction<ACT> {
        self.reset();
        self.faulted = true;
        CuPolicyAction {
            values: [0.0; ACT],
            status: CuPolicyStatus::Fault,
        }
    }

    fn idle_action() -> CuPolicyAction<ACT> {
        CuPolicyAction {
            values: [0.0; ACT],
            status: CuPolicyStatus::Idle,
        }
    }

    fn weight_for(&self, count: u32) -> f64 {
        let idx = count as usize;
        if let Some(&w) = self.weights.get(idx) {
            w
        } else if let CuActionChunkMode::TemporalEnsemble { decay } = self.params.mode {
            (-decay * f64::from(count)).exp()
        } else {
            1.0
        }
    }

    /// Ingests a validated finite `Fresh` chunk; returns `false` if ensemble
    /// weights overflowed `f64`.
    fn ingest_fresh(&mut self, values: &[[f32; ACT]; H]) -> bool {
        self.faulted = false;
        self.steps_since_fresh = 0;
        match self.params.mode {
            CuActionChunkMode::Replace => {
                self.slots = [ChunkSlot::default(); H];
                self.head = 0;
                self.step_cycle = 0;
                for (slot, step) in self.slots.iter_mut().zip(values.iter()) {
                    slot.weighted_sum = step.map(f64::from);
                    slot.weight_total = 1.0;
                    slot.count = 1;
                }
                true
            }
            CuActionChunkMode::TemporalEnsemble { .. } => {
                for (s, step) in values.iter().enumerate() {
                    let idx = (self.head + s) % H;
                    let w = self.weight_for(self.slots[idx].count);
                    if !w.is_finite() {
                        return false;
                    }
                    let slot = &mut self.slots[idx];
                    for (acc, &val) in slot.weighted_sum.iter_mut().zip(step.iter()) {
                        *acc += w * f64::from(val);
                    }
                    slot.weight_total += w;
                    if !slot.weight_total.is_finite() {
                        return false;
                    }
                    slot.count = slot.count.saturating_add(1);
                }
                true
            }
        }
    }

    fn advance_step_if_complete(&mut self) {
        if self.step_cycle >= self.params.step_decimation {
            self.slots[self.head] = ChunkSlot::default();
            self.head = (self.head + 1) % H;
            self.step_cycle = 0;
        }
    }
}

impl<const H: usize, const ACT: usize> Freezable for CuActionChunkTask<H, ACT> {
    fn freeze<E: Encoder>(&self, encoder: &mut E) -> Result<(), EncodeError> {
        Encode::encode(&self.slots, encoder)?;
        Encode::encode(&self.head, encoder)?;
        Encode::encode(&self.step_cycle, encoder)?;
        Encode::encode(&self.steps_since_fresh, encoder)?;
        Encode::encode(&self.last_action, encoder)?;
        Encode::encode(&self.faulted, encoder)
    }

    fn thaw<D: Decoder>(&mut self, decoder: &mut D) -> Result<(), DecodeError> {
        let slots: [ChunkSlot<ACT>; H] = Decode::decode(decoder)?;
        let head: usize = Decode::decode(decoder)?;
        let step_cycle: u32 = Decode::decode(decoder)?;
        let steps_since_fresh: usize = Decode::decode(decoder)?;
        let last_action: [f32; ACT] = Decode::decode(decoder)?;
        let faulted: bool = Decode::decode(decoder)?;
        if head >= H || step_cycle >= self.params.step_decimation {
            return Err(DecodeError::Other(
                "repose-policy-rl: thawed CuActionChunkTask indices out of bounds",
            ));
        }
        self.slots = slots;
        self.head = head;
        self.step_cycle = step_cycle;
        self.steps_since_fresh = steps_since_fresh;
        self.last_action = last_action;
        self.faulted = faulted;
        Ok(())
    }
}

impl<const H: usize, const ACT: usize> CuTask for CuActionChunkTask<H, ACT> {
    type Resources<'r> = ();
    type Input<'m> = input_msg!(CuPolicyActionChunk<H, ACT>);
    type Output<'m> = output_msg!(CuPolicyAction<ACT>);

    fn new(cfg: Option<&ComponentConfig>, _res: Self::Resources<'_>) -> CuResult<Self>
    where
        Self: Sized,
    {
        let params = cfg
            .map(ComponentConfig::deserialize_into::<CuActionChunkParams>)
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
        output.tov = input.tov;

        if let Some(chunk) = input.payload() {
            match chunk.status {
                CuPolicyStatus::Idle => {
                    self.reset();
                    output.set_payload(Self::idle_action());
                    return Ok(());
                }
                CuPolicyStatus::Fault => {
                    output.set_payload(self.fault());
                    return Ok(());
                }
                CuPolicyStatus::Fresh => {
                    if chunk.values.iter().flatten().any(|v| !v.is_finite())
                        || !self.ingest_fresh(&chunk.values)
                    {
                        output.set_payload(self.fault());
                        return Ok(());
                    }
                }
                CuPolicyStatus::Held => {}
            }
        }

        if self.faulted {
            output.set_payload(CuPolicyAction {
                values: [0.0; ACT],
                status: CuPolicyStatus::Fault,
            });
            return Ok(());
        }

        if self.step_cycle > 0 {
            let held = CuPolicyAction {
                values: self.last_action,
                status: CuPolicyStatus::Held,
            };
            self.step_cycle += 1;
            self.advance_step_if_complete();
            output.set_payload(held);
            return Ok(());
        }

        let slot = &self.slots[self.head];
        if self.steps_since_fresh >= self.max_steps || slot.count == 0 || slot.weight_total <= 0.0 {
            self.reset();
            output.set_payload(Self::idle_action());
            return Ok(());
        }

        let inv_weight = 1.0 / slot.weight_total;
        let mut values = [0.0f32; ACT];
        for (dst, &acc) in values.iter_mut().zip(slot.weighted_sum.iter()) {
            *dst = (acc * inv_weight) as f32;
        }
        if values.iter().any(|v| !v.is_finite()) {
            output.set_payload(self.fault());
            return Ok(());
        }

        self.last_action = values;
        self.steps_since_fresh = self.steps_since_fresh.saturating_add(1);
        self.step_cycle = 1;
        self.advance_step_if_complete();
        output.set_payload(CuPolicyAction {
            values,
            status: CuPolicyStatus::Fresh,
        });
        Ok(())
    }
}

/// Parameters of [`CuChunkPolicyTask`]; the node `config:` block.
///
/// ```ron
/// config: {
///     "model": "act_policy.onnx",
///     "model_hash": 1234567890,
///     "decimation": 1,
///     "max_observation_abs": 10.0,
///     "observation_bound_policy": "clamp",
///     "max_action_abs": 1.0,
///     "action_bound_policy": "clamp",
/// }
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Reflect)]
#[serde(default, deny_unknown_fields)]
pub struct CuChunkPolicyParams {
    /// ONNX model path, resolved relative to the process working directory.
    /// Required: an empty path is rejected by [`Self::validate`].
    pub model: String,
    /// Optional FNV-1a 64-bit hash of the model file bytes. When set, a
    /// mismatch is a startup error, so a replay with a different model file
    /// fails instead of silently diverging.
    pub model_hash: Option<u64>,
    /// Infer a new chunk every `decimation`-th cycle and emit
    /// [`CuPolicyStatus::Held`] on the cycles in between. Counts cycles, not
    /// wall time, so replay is cycle-exact. Defaults to `1`.
    pub decimation: u32,
    /// Optional symmetric magnitude bound on observation features
    /// (dimensionless; must be positive and finite when `Some`). Checked on
    /// every usable cycle after finiteness; behavior when exceeded is
    /// governed by `observation_bound_policy`. `None` disables the check.
    pub max_observation_abs: Option<f32>,
    /// Policy when a finite observation feature exceeds `max_observation_abs`;
    /// defaults to `clamp`.
    pub observation_bound_policy: CuBoundPolicy,
    /// Optional symmetric magnitude bound on raw ONNX action chunk elements
    /// (dimensionless; must be positive and finite when `Some`). Behavior when
    /// exceeded is governed by `action_bound_policy`. `None` disables the
    /// check.
    pub max_action_abs: Option<f32>,
    /// Policy when a finite raw chunk element exceeds `max_action_abs`;
    /// defaults to `clamp`.
    pub action_bound_policy: CuBoundPolicy,
}

impl Default for CuChunkPolicyParams {
    fn default() -> Self {
        Self {
            model: String::new(),
            model_hash: None,
            decimation: default_decimation(),
            max_observation_abs: None,
            observation_bound_policy: CuBoundPolicy::default(),
            max_action_abs: None,
            action_bound_policy: CuBoundPolicy::default(),
        }
    }
}

impl CuChunkPolicyParams {
    /// Checks the model path, decimation, and optional observation/action bounds.
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
        if let Some(limit) = self.max_observation_abs
            && (!limit.is_finite() || limit <= 0.0)
        {
            return Err(CuError::from(
                "repose-policy-rl: 'max_observation_abs' must be positive and finite",
            ));
        }
        if let Some(limit) = self.max_action_abs
            && (!limit.is_finite() || limit <= 0.0)
        {
            return Err(CuError::from(
                "repose-policy-rl: 'max_action_abs' must be positive and finite",
            ));
        }
        Ok(())
    }
}

/// Runs a stateless chunked ONNX policy (`[1, OBS] -> [1, H, ACT]`) inside the
/// Copper graph and feeds [`CuActionChunkTask`].
///
/// Input is a flat [`CuPolicyObservation<OBS>`] feature vector from an upstream
/// observation task; output is the raw [`CuPolicyActionChunk<H, ACT>`] carrying
/// the input's `Tov`:
///
/// - **No input payload**: [`CuPolicyStatus::Idle`], resetting the decimation
///   counter so the first usable observation after a gap always infers.
/// - **A non-finite input**, a failed inference, or a non-finite model output:
///   [`CuPolicyStatus::Fault`], latched across between-query cycles until the
///   next inference or `Idle`.
/// - **Inference cycle** (`cycle % decimation == 0`): [`CuPolicyStatus::Fresh`]
///   with the inferred `[H, ACT]` chunk.
/// - **Between-query cycle**: [`CuPolicyStatus::Held`] with zero values and no
///   inference, telling [`CuActionChunkTask`] to keep stepping its buffered
///   horizon while still checking every cycle's observation for faults.
///
/// When `max_observation_abs` is set, finite observation elements exceeding
/// `[-limit, limit]` are either clamped before inference
/// (`CuBoundPolicy::Clamp`) or faulted without running inference
/// (`CuBoundPolicy::Fault`). When `max_action_abs` is set, finite raw chunk
/// elements exceeding `[-limit, limit]` are either clamped
/// (`CuBoundPolicy::Clamp`) or faulted (`CuBoundPolicy::Fault`).
#[derive(Reflect)]
#[reflect(no_field_bounds, from_reflect = false)]
pub struct CuChunkPolicyTask<const OBS: usize, const H: usize, const ACT: usize> {
    params: CuChunkPolicyParams,
    #[reflect(ignore)]
    backend: CuPolicyBackend,
    cycle: u64,
    last_status: CuPolicyStatus,
}

impl<const OBS: usize, const H: usize, const ACT: usize> CuChunkPolicyTask<OBS, H, ACT> {
    /// Creates a chunked policy task from validated params, loading and
    /// optimizing the `[1, OBS] -> [1, H, ACT]` ONNX model.
    pub fn from_params(params: CuChunkPolicyParams) -> CuResult<Self> {
        if H == 0 {
            return Err(CuError::from(
                "repose-policy-rl: chunk horizon H must be at least 1",
            ));
        }
        if ACT == 0 {
            return Err(CuError::from(
                "repose-policy-rl: action width ACT must be at least 1",
            ));
        }
        params.validate()?;
        let backend =
            CuPolicyBackend::load_chunk(Path::new(&params.model), OBS, H, ACT, params.model_hash)?;
        Ok(Self {
            params,
            backend,
            cycle: 0,
            last_status: CuPolicyStatus::Idle,
        })
    }

    /// Params this task runs with.
    pub fn params(&self) -> &CuChunkPolicyParams {
        &self.params
    }
}

impl<const OBS: usize, const H: usize, const ACT: usize> Freezable
    for CuChunkPolicyTask<OBS, H, ACT>
{
    fn freeze<E: Encoder>(&self, encoder: &mut E) -> Result<(), EncodeError> {
        Encode::encode(&self.cycle, encoder)?;
        Encode::encode(&self.last_status, encoder)
    }

    fn thaw<D: Decoder>(&mut self, decoder: &mut D) -> Result<(), DecodeError> {
        self.cycle = Decode::decode(decoder)?;
        self.last_status = Decode::decode(decoder)?;
        Ok(())
    }
}

impl<const OBS: usize, const H: usize, const ACT: usize> CuTask for CuChunkPolicyTask<OBS, H, ACT> {
    type Resources<'r> = ();
    type Input<'m> = input_msg!(CuPolicyObservation<OBS>);
    type Output<'m> = output_msg!(CuPolicyActionChunk<H, ACT>);

    fn new(cfg: Option<&ComponentConfig>, _res: Self::Resources<'_>) -> CuResult<Self>
    where
        Self: Sized,
    {
        let params = cfg
            .map(ComponentConfig::deserialize_into::<CuChunkPolicyParams>)
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
        output.tov = input.tov;

        let Some(observation) = input.payload() else {
            self.last_status = CuPolicyStatus::Idle;
            self.cycle = 0;
            output.set_payload(CuPolicyActionChunk::default());
            return Ok(());
        };

        let mut obs_values = observation.values;
        let non_finite_input = obs_values.iter().any(|v| !v.is_finite());
        let mut input_bad = non_finite_input;
        if !non_finite_input && let Some(limit) = self.params.max_observation_abs {
            match self.params.observation_bound_policy {
                CuBoundPolicy::Clamp => {
                    for value in &mut obs_values {
                        *value = value.clamp(-limit, limit);
                    }
                }
                CuBoundPolicy::Fault => {
                    if obs_values.iter().any(|v| v.abs() > limit) {
                        input_bad = true;
                    }
                }
            }
        }
        let infer_now = self.cycle.is_multiple_of(u64::from(self.params.decimation));
        self.cycle = self.cycle.wrapping_add(1);

        let (values, status) = if infer_now {
            let mut fresh = [[0.0f32; ACT]; H];
            let mut failed = input_bad
                || self.backend.infer_chunk(&obs_values, &mut fresh).is_err()
                || fresh.iter().flatten().any(|v| !v.is_finite());
            if !failed && let Some(limit) = self.params.max_action_abs {
                match self.params.action_bound_policy {
                    CuBoundPolicy::Clamp => {
                        for value in fresh.iter_mut().flatten() {
                            *value = value.clamp(-limit, limit);
                        }
                    }
                    CuBoundPolicy::Fault => {
                        if fresh.iter().flatten().any(|v| v.abs() > limit) {
                            failed = true;
                        }
                    }
                }
            }
            if failed {
                ([[0.0; ACT]; H], CuPolicyStatus::Fault)
            } else {
                (fresh, CuPolicyStatus::Fresh)
            }
        } else if input_bad || self.last_status == CuPolicyStatus::Fault {
            ([[0.0; ACT]; H], CuPolicyStatus::Fault)
        } else {
            ([[0.0; ACT]; H], CuPolicyStatus::Held)
        };

        self.last_status = status;
        output.set_payload(CuPolicyActionChunk { values, status });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CuCartesianActionParams;
    use crate::CuCartesianActionTask;
    use crate::CuJointActionKind;
    use crate::CuJointActionParams;
    use crate::CuJointActionTask;
    use cu29::units::si::angle::radian;
    use cu29::units::si::angular_velocity::radian_per_second;
    use cu29::units::si::f64::Angle;
    use cu29::units::si::f64::AngularVelocity;
    use cu29::units::si::f64::Length;
    use cu29::units::si::f64::Time;
    use cu29::units::si::length::meter;
    use cu29::units::si::time::second;
    use repose_kinematics::CuChainSpec;
    use repose_kinematics::CuJointPosition;
    use repose_kinematics::CuJointStatePayload;
    use repose_kinematics::CuJointVelocity;
    use repose_kinematics::CuRobotPreset;
    use repose_kinematics::CuRobotStatus;
    use repose_trajectory::CuFollowStatus;

    fn step<const H: usize, const ACT: usize>(
        task: &mut CuActionChunkTask<H, ACT>,
        chunk: Option<CuPolicyActionChunk<H, ACT>>,
    ) -> CuPolicyAction<ACT> {
        let (ctx, _clock) = CuContext::new_mock_clock();
        let mut msg = CuMsg::new(chunk);
        msg.tov = Tov::Time(CuTime::from_nanos(123_456));
        let mut out = CuMsg::<CuPolicyAction<ACT>>::default();
        task.process(&ctx, &msg, &mut out).unwrap();
        assert_eq!(out.tov, Tov::Time(CuTime::from_nanos(123_456)));
        out.payload().unwrap().clone()
    }

    fn fresh<const H: usize, const ACT: usize>(
        values: [[f32; ACT]; H],
    ) -> Option<CuPolicyActionChunk<H, ACT>> {
        Some(CuPolicyActionChunk {
            values,
            status: CuPolicyStatus::Fresh,
        })
    }

    #[test]
    fn act_temporal_ensembling_matches_reference_matrix_exactly() {
        // Verify online ring-buffer accumulation against ACT's explicit
        // `all_time_actions[t, t:t+H]` history table for H = 4, ACT = 2 across
        // dense, sparse, and draining steps.
        const H: usize = 4;
        const STEPS: usize = 9;
        let decay = 0.25f64;

        let mut task = CuActionChunkTask::<H, 2>::from_params(CuActionChunkParams {
            mode: CuActionChunkMode::TemporalEnsemble { decay },
            ..CuActionChunkParams::default()
        })
        .unwrap();

        let arrivals: [Option<[[f32; 2]; H]>; STEPS] = [
            Some([[1.0, -1.0], [2.0, -2.0], [3.0, -3.0], [4.0, -4.0]]),
            Some([[2.5, -1.5], [3.5, -2.5], [4.5, -3.5], [5.5, -4.5]]),
            None, // sparse gap: no new chunk at t = 2
            Some([[4.2, -3.2], [5.2, -4.2], [6.2, -5.2], [7.2, -6.2]]),
            Some([[6.0, -5.0], [7.0, -6.0], [8.0, -7.0], [9.0, -8.0]]),
            None,
            None,
            None,
            None, // drains past horizon at t = 8
        ];

        // Reference table: populated[gen_t][target_t]
        let mut table = [[[0.0f64; 2]; STEPS + H]; STEPS];
        let mut populated = [[false; STEPS + H]; STEPS];

        for t in 0..STEPS {
            let chunk = arrivals[t].and_then(fresh);
            if let Some(raw) = arrivals[t] {
                for s in 0..H {
                    table[t][t + s] = raw[s].map(f64::from);
                    populated[t][t + s] = true;
                }
            }
            let actual = step(&mut task, chunk);

            let mut preds: Vec<[f64; 2]> = Vec::new();
            for gen_t in 0..=t {
                if populated[gen_t][t] {
                    preds.push(table[gen_t][t]);
                }
            }
            if preds.is_empty() {
                assert_eq!(actual.status, CuPolicyStatus::Idle);
                assert_eq!(actual.values, [0.0, 0.0]);
            } else {
                assert_eq!(actual.status, CuPolicyStatus::Fresh);
                let weights: Vec<f64> = (0..preds.len())
                    .map(|i| (-decay * (i as f64)).exp())
                    .collect();
                let w_sum: f64 = weights.iter().sum();
                for d in 0..2 {
                    let expected = (preds
                        .iter()
                        .zip(weights.iter())
                        .map(|(p, w)| p[d] * w)
                        .sum::<f64>()
                        / w_sum) as f32;
                    assert!(
                        (actual.values[d] - expected).abs() <= 1e-6,
                        "step {t} dim {d}: actual {} vs expected {expected}",
                        actual.values[d]
                    );
                }
            }
        }
    }

    #[test]
    fn decay_sign_orders_older_vs_newer_predictions() {
        let older = fresh([[0.0], [0.0]]);
        let newer = fresh([[10.0], [10.0]]);

        let run_pair = |decay: f64| -> f32 {
            let mut task = CuActionChunkTask::<2, 1>::from_params(CuActionChunkParams {
                mode: CuActionChunkMode::TemporalEnsemble { decay },
                ..CuActionChunkParams::default()
            })
            .unwrap();
            let _ = step(&mut task, older.clone());
            step(&mut task, newer.clone()).values[0]
        };

        let favor_old = run_pair(1.0);
        let uniform = run_pair(0.0);
        let favor_new = run_pair(-1.0);
        assert!((uniform - 5.0).abs() < 1e-6);
        assert!(favor_old < uniform, "{favor_old} should be < {uniform}");
        assert!(favor_new > uniform, "{favor_new} should be > {uniform}");
    }

    #[test]
    fn replace_mode_discards_old_tail_and_respects_execution_horizon() {
        let mut task = CuActionChunkTask::<4, 1>::from_params(CuActionChunkParams {
            mode: CuActionChunkMode::Replace,
            step_decimation: 1,
            execution_horizon: Some(2),
        })
        .unwrap();

        // Execute 1 step of chunk A, then replace with chunk B before exhaustion.
        assert_eq!(
            step(&mut task, fresh([[1.0], [2.0], [3.0], [4.0]])),
            CuPolicyAction {
                values: [1.0],
                status: CuPolicyStatus::Fresh
            }
        );
        assert_eq!(
            step(&mut task, fresh([[10.0], [20.0], [30.0], [40.0]])),
            CuPolicyAction {
                values: [10.0],
                status: CuPolicyStatus::Fresh
            }
        );
        // Step 2 of chunk B executes (Held upstream == no new chunk).
        let held = Some(CuPolicyActionChunk {
            values: [[99.0]; 4],
            status: CuPolicyStatus::Held,
        });
        assert_eq!(
            step(&mut task, held),
            CuPolicyAction {
                values: [20.0],
                status: CuPolicyStatus::Fresh
            }
        );
        // Step 3 of chunk B is blocked by execution_horizon = 2 -> Idle starvation!
        assert_eq!(
            step(&mut task, None),
            CuPolicyAction {
                values: [0.0],
                status: CuPolicyStatus::Idle
            }
        );
        // A later chunk resumes cleanly.
        assert_eq!(
            step(&mut task, fresh([[7.0], [8.0], [9.0], [10.0]])),
            CuPolicyAction {
                values: [7.0],
                status: CuPolicyStatus::Fresh
            }
        );
    }

    #[test]
    fn step_decimation_holds_between_steps_with_cartesian_and_joint_deltas() {
        // 3 control cycles per chunk step: cycle 0 is Fresh (applies delta),
        // cycles 1 and 2 are Held (downstream delta tasks hold target without
        // re-applying the delta!).
        let mut chunker = CuActionChunkTask::<2, 6>::from_params(CuActionChunkParams {
            mode: CuActionChunkMode::Replace,
            step_decimation: 3,
            execution_horizon: None,
        })
        .unwrap();

        let mut joint_task = CuJointActionTask::<6, 6>::from_params(CuJointActionParams {
            chain: Some(CuChainSpec::Preset(CuRobotPreset::IgusRebel)),
            kind: CuJointActionKind::PositionDelta,
            scale: [0.1; 6],
            ..CuJointActionParams::default()
        })
        .unwrap();

        let mut cart_task = CuCartesianActionTask::<6, 6>::from_params(CuCartesianActionParams {
            chain: Some(CuChainSpec::Preset(CuRobotPreset::IgusRebel)),
            period: Some(Time::new::<second>(0.06)),
            translation_scale: Length::new::<meter>(0.01),
            rotation_scale: Angle::new::<radian>(0.05),
            ..CuCartesianActionParams::default()
        })
        .unwrap();

        let state = CuMsg::new(Some(CuJointStatePayload::<6> {
            status: CuRobotStatus::Ready,
            positions: [0.0, 0.4, 0.8, 0.0, 0.6, 0.0]
                .map(|r| CuJointPosition::Revolute(Angle::new::<radian>(r))),
            velocities: [CuJointVelocity::Revolute(AngularVelocity::new::<radian_per_second>(0.0));
                6],
        }));
        let (ctx, _clock) = CuContext::new_mock_clock();

        let incoming = fresh([
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]);
        let mut statuses = Vec::new();
        let mut joint0_targets = Vec::new();
        let mut cart_targets = Vec::new();

        for cycle in 0..7 {
            let action = step(
                &mut chunker,
                if cycle == 0 { incoming.clone() } else { None },
            );
            statuses.push(action.status);
            let act_msg = CuMsg::new(Some(action));

            let mut j_out = CuMsg::default();
            joint_task
                .process(&ctx, &(&act_msg, &state), &mut j_out)
                .unwrap();
            joint0_targets.push((
                j_out.payload().unwrap().status,
                j_out.payload().unwrap().positions[0].si_value(),
            ));

            let mut c_out = (CuMsg::default(), CuMsg::default());
            cart_task
                .process(&ctx, &(&act_msg, &state), &mut c_out)
                .unwrap();
            cart_targets.push((
                c_out.0.payload().unwrap().status,
                c_out.0.payload().unwrap().positions,
            ));
        }

        use CuPolicyStatus::{Fresh, Held, Idle};
        assert_eq!(statuses, vec![Fresh, Held, Held, Fresh, Held, Held, Idle]);
        // Joint 0 target on cycles 0, 1, 2 is +0.1 rad (held, NOT +0.2/+0.3!),
        // on cycles 3, 4, 5 is +0.2 rad, and on cycle 6 (Idle) holds measured 0.0.
        for c in 0..3 {
            assert_eq!(joint0_targets[c].0, CuFollowStatus::Running);
            assert!((joint0_targets[c].1 - 0.1).abs() < 1e-12);
            assert_eq!(cart_targets[c].0, CuFollowStatus::Running);
            assert_eq!(cart_targets[c].1, cart_targets[0].1);
        }
        for c in 3..6 {
            assert_eq!(joint0_targets[c].0, CuFollowStatus::Running);
            assert!((joint0_targets[c].1 - 0.2).abs() < 1e-12);
            assert_eq!(cart_targets[c].0, CuFollowStatus::Running);
            assert_eq!(cart_targets[c].1, cart_targets[3].1);
        }
        assert_eq!(joint0_targets[6].0, CuFollowStatus::Idle);
        assert!((joint0_targets[6].1 - 0.0).abs() < 1e-12);
        assert_eq!(cart_targets[6].0, CuFollowStatus::Idle);
    }

    #[test]
    fn fault_and_idle_flush_buffered_horizon_immediately() {
        let mut task =
            CuActionChunkTask::<3, 2>::from_params(CuActionChunkParams::default()).unwrap();

        assert_eq!(
            step(&mut task, fresh([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])).status,
            CuPolicyStatus::Fresh
        );
        // Non-finite value in a future step of a new chunk faults immediately,
        // clears the buffer, and stays latched across None / Held cycles.
        assert_eq!(
            step(&mut task, fresh([[1.0, 2.0], [3.0, 4.0], [f32::NAN, 6.0]])),
            CuPolicyAction {
                values: [0.0, 0.0],
                status: CuPolicyStatus::Fault
            }
        );
        assert_eq!(step(&mut task, None).status, CuPolicyStatus::Fault);
        assert_eq!(
            step(
                &mut task,
                Some(CuPolicyActionChunk {
                    values: [[0.0; 2]; 3],
                    status: CuPolicyStatus::Held
                })
            )
            .status,
            CuPolicyStatus::Fault
        );

        // Explicit Idle clears the latched fault.
        assert_eq!(
            step(
                &mut task,
                Some(CuPolicyActionChunk {
                    values: [[0.0; 2]; 3],
                    status: CuPolicyStatus::Idle
                })
            ),
            CuPolicyAction {
                values: [0.0, 0.0],
                status: CuPolicyStatus::Idle
            }
        );
        assert_eq!(step(&mut task, None).status, CuPolicyStatus::Idle);

        // Upstream Idle mid-chunk flushes remaining buffered steps on the spot.
        assert_eq!(
            step(&mut task, fresh([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])).values,
            [1.0, 2.0]
        );
        assert_eq!(
            step(
                &mut task,
                Some(CuPolicyActionChunk {
                    values: [[0.0; 2]; 3],
                    status: CuPolicyStatus::Idle
                })
            )
            .status,
            CuPolicyStatus::Idle
        );
        assert_eq!(step(&mut task, None).status, CuPolicyStatus::Idle);
    }

    #[test]
    fn freezable_round_trips_mid_ensemble_and_mid_step() {
        let params = CuActionChunkParams {
            mode: CuActionChunkMode::TemporalEnsemble { decay: 0.05 },
            step_decimation: 2,
            execution_horizon: Some(3),
        };
        let mut live = CuActionChunkTask::<4, 2>::from_params(params).unwrap();

        let _ = step(
            &mut live,
            fresh([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]),
        );
        let _ = step(&mut live, None); // completes step 0
        let _ = step(
            &mut live,
            fresh([[2.0, 3.0], [4.0, 5.0], [6.0, 7.0], [8.0, 9.0]]),
        ); // mid-step (step_cycle == 1) with overlapping ensemble weights!

        let mut buf = [0u8; 512];
        let mut enc = bincode::enc::EncoderImpl::new(
            bincode::enc::write::SliceWriter::new(&mut buf),
            bincode::config::standard(),
        );
        live.freeze(&mut enc).unwrap();
        let written = enc.into_writer().bytes_written();

        let mut restored = CuActionChunkTask::<4, 2>::from_params(params).unwrap();
        let mut dec = bincode::de::DecoderImpl::new(
            bincode::de::read::SliceReader::new(&buf[..written]),
            bincode::config::standard(),
            (),
        );
        restored.thaw(&mut dec).unwrap();

        for _ in 0..8 {
            assert_eq!(step(&mut live, None), step(&mut restored, None));
        }
    }

    #[test]
    fn invalid_params_and_capacities_are_rejected() {
        assert!(CuActionChunkTask::<0, 2>::from_params(CuActionChunkParams::default()).is_err());
        assert!(CuActionChunkTask::<2, 0>::from_params(CuActionChunkParams::default()).is_err());
        for bad in [
            CuActionChunkParams {
                step_decimation: 0,
                ..CuActionChunkParams::default()
            },
            CuActionChunkParams {
                execution_horizon: Some(0),
                ..CuActionChunkParams::default()
            },
            CuActionChunkParams {
                execution_horizon: Some(5),
                ..CuActionChunkParams::default()
            },
            CuActionChunkParams {
                mode: CuActionChunkMode::TemporalEnsemble { decay: f64::NAN },
                ..CuActionChunkParams::default()
            },
            CuActionChunkParams {
                mode: CuActionChunkMode::TemporalEnsemble { decay: -1000.0 },
                ..CuActionChunkParams::default()
            },
        ] {
            assert!(
                CuActionChunkTask::<4, 2>::from_params(bad).is_err(),
                "{bad:?}"
            );
        }
    }

    #[test]
    fn params_read_from_ron_and_reject_typos() {
        let cfg: ComponentConfig = ron::from_str(
            r#"#![enable(unwrap_newtypes)] {
                "mode": {"temporal_ensemble": {"decay": 0.02}},
                "step_decimation": 4,
                "execution_horizon": 8,
            }"#,
        )
        .unwrap();
        let task = CuActionChunkTask::<16, 7>::new(Some(&cfg), ()).unwrap();
        assert_eq!(
            task.params(),
            &CuActionChunkParams {
                mode: CuActionChunkMode::TemporalEnsemble { decay: 0.02 },
                step_decimation: 4,
                execution_horizon: Some(8),
            }
        );

        let replace_cfg: ComponentConfig =
            ron::from_str(r#"#![enable(unwrap_newtypes)] {"mode": "replace"}"#).unwrap();
        let replace_task = CuActionChunkTask::<16, 7>::new(Some(&replace_cfg), ()).unwrap();
        assert_eq!(replace_task.params().mode, CuActionChunkMode::Replace);

        let typo: ComponentConfig =
            ron::from_str(r#"#![enable(unwrap_newtypes)] {"step_decimaton": 2}"#).unwrap();
        assert!(CuActionChunkTask::<16, 7>::new(Some(&typo), ()).is_err());
    }

    #[test]
    fn mid_step_async_arrival_holds_current_step_and_ensembles_next() {
        let mut task = CuActionChunkTask::<3, 1>::from_params(CuActionChunkParams {
            mode: CuActionChunkMode::TemporalEnsemble { decay: 0.0 },
            step_decimation: 3,
            execution_horizon: Some(1),
        })
        .unwrap();

        // Cycle 0: chunk A arrives -> emits step 0 (Fresh).
        assert_eq!(
            step(&mut task, fresh([[10.0], [20.0], [30.0]])),
            CuPolicyAction {
                values: [10.0],
                status: CuPolicyStatus::Fresh
            }
        );
        // Cycle 1 (mid-step, step_cycle == 1): async chunk B arrives! Current
        // 3-cycle step keeps Held (never double-applies a delta), and resets
        // execution_horizon budget so the next step boundary executes cleanly.
        assert_eq!(
            step(&mut task, fresh([[100.0], [40.0], [50.0]])),
            CuPolicyAction {
                values: [10.0],
                status: CuPolicyStatus::Held
            }
        );
        // Cycle 2: finishes holding step 0.
        assert_eq!(
            step(&mut task, None),
            CuPolicyAction {
                values: [10.0],
                status: CuPolicyStatus::Held
            }
        );
        // Cycle 3: emits step 1 with uniform average of A[1]=20 and B[1]=40 -> 30.0!
        assert_eq!(
            step(&mut task, None),
            CuPolicyAction {
                values: [30.0],
                status: CuPolicyStatus::Fresh
            }
        );
        assert_eq!(step(&mut task, None).status, CuPolicyStatus::Held);
        assert_eq!(step(&mut task, None).status, CuPolicyStatus::Held);
        // Cycle 6: execution_horizon = 1 after chunk B is now reached -> Idle!
        assert_eq!(step(&mut task, None).status, CuPolicyStatus::Idle);
    }
}
