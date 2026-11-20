//! [`CuJointActionTask`]: raw policy action in, limited joint setpoint out.

use crate::payload::CuPolicyAction;
use crate::payload::CuPolicyStatus;
use bincode::Decode;
use bincode::Encode;
use bincode::de::Decoder;
use bincode::enc::Encoder;
use bincode::error::DecodeError;
use bincode::error::EncodeError;
use cu29::prelude::*;
use cu29::units::si::f64::Time;
use cu29::units::si::time::second;
use repose_kinematics::CuChainSpec;
use repose_kinematics::CuJointAcceleration;
use repose_kinematics::CuJointPosition;
use repose_kinematics::CuJointStatePayload;
use repose_kinematics::CuJointVelocity;
use repose_kinematics::CuKinematicChain;
use repose_kinematics::CuRobotStatus;
use repose_trajectory::CuFollowStatus;
use repose_trajectory::CuTrajectoryFollowOutput;
use serde::Deserialize;
use serde::Serialize;

/// Serde for `scale`: one dimensionless number, or one per joint.
mod scale_array {
    use core::fmt;
    use serde::Deserializer;
    use serde::Serializer;
    use serde::de::Error as DeError;
    use serde::de::SeqAccess;
    use serde::de::Visitor;
    use serde::ser::SerializeTuple;

    pub fn serialize<S, const N: usize>(value: &[f64; N], serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut tuple = serializer.serialize_tuple(N)?;
        for item in value {
            tuple.serialize_element(item)?;
        }
        tuple.end()
    }

    pub fn deserialize<'de, D, const N: usize>(deserializer: D) -> Result<[f64; N], D::Error>
    where
        D: Deserializer<'de>,
    {
        struct ScaleVisitor<const M: usize>;
        impl<'de, const M: usize> Visitor<'de> for ScaleVisitor<M> {
            type Value = [f64; M];
            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(
                    formatter,
                    "an array of {M} per-joint scales, or one scale for every joint"
                )
            }
            fn visit_f64<E: DeError>(self, v: f64) -> Result<Self::Value, E> {
                Ok([v; M])
            }
            fn visit_i64<E: DeError>(self, v: i64) -> Result<Self::Value, E> {
                Ok([v as f64; M])
            }
            fn visit_u64<E: DeError>(self, v: u64) -> Result<Self::Value, E> {
                Ok([v as f64; M])
            }
            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let mut out = [0.0; M];
                let mut i = 0;
                while i < M {
                    out[i] = seq
                        .next_element()?
                        .ok_or_else(|| DeError::invalid_length(i, &self))?;
                    i += 1;
                }
                if seq.next_element::<serde::de::IgnoredAny>()?.is_some() {
                    return Err(DeError::invalid_length(M + 1, &self));
                }
                Ok(out)
            }
        }
        deserializer.deserialize_any(ScaleVisitor::<N>)
    }
}

fn default_scale<const N: usize>() -> [f64; N] {
    [1.0; N]
}

fn default_positions<const N: usize>() -> [CuJointPosition; N] {
    [CuJointPosition::default(); N]
}

/// How a raw action becomes a joint target (dimensionless action, scaled).
#[derive(
    Default, Debug, Clone, Copy, PartialEq, Eq, Encode, Decode, Serialize, Deserialize, Reflect,
)]
#[serde(rename_all = "snake_case")]
pub enum CuJointActionKind {
    /// `target = measured_position + scale * action`: an offset from where
    /// the arm is, the usual meaning of a relative joint action in RL
    /// trainers.
    #[default]
    PositionDelta,
    /// `target = default_positions + scale * action`.
    PositionAbsolute,
}

/// Parameters of [`CuJointActionTask`]; the node `config:` block.
///
/// ```ron
/// config: {
///     "chain": "igus_rebel",
///     "offset": 0,                       // optional; first action element this task reads
///     "kind": "position_delta",          // position_delta | position_absolute
///     "scale": 0.05,                     // dimensionless, or one value per joint
///     "default_positions": [{"revolute": 0.0}], // used by position_absolute
///     "max_step": 0.03,                  // optional; largest target change per cycle
///     "period": 0.02,                    // optional; s, enables the setpoint velocity
/// }
/// ```
///
/// `scale` and `max_step` are in the SI unit of each joint (radians for
/// revolute, meters for prismatic): the action is dimensionless, the scaled
/// step carries the joint's unit via [`CuJointPosition::with_si_value`], so
/// revolute and prismatic joints in one chain are both handled.
///
/// `offset` selects which part of the policy action this task consumes:
/// joint `j` reads action element `offset + j`, so the task reads elements
/// `[offset, offset + N)` of a `CuPolicyAction<ACT>`. Several action tasks
/// can share one action message by reading disjoint slices (for example an
/// arm in elements `0..6` and a gripper in element `6`). The bound
/// `offset + N <= ACT` depends on the task's `ACT` and is checked by
/// [`CuJointActionTask::from_params`], not by [`Self::validate`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Reflect)]
#[reflect(no_field_bounds, from_reflect = false)]
// Field-level defaults: serde does not support container `default` on const-generic structs.
#[serde(deny_unknown_fields)]
pub struct CuJointActionParams<const N: usize> {
    /// Required robot chain: a preset name (`"igus_rebel"`) or an inline
    /// chain. Joint limits come from the chain.
    #[serde(default)]
    #[reflect(ignore)]
    pub chain: Option<CuChainSpec<N>>,
    /// Index of the first action element this task reads; defaults to `0`.
    #[serde(default)]
    pub offset: usize,
    /// Action kind; defaults to `position_delta`.
    #[serde(default)]
    pub kind: CuJointActionKind,
    /// Dimensionless scale: one value for every joint, or one per joint.
    #[serde(default = "default_scale", with = "scale_array")]
    pub scale: [f64; N],
    /// Rest positions used by `position_absolute` (typed joint coordinates).
    #[serde(
        default = "default_positions",
        with = "repose_kinematics::serde_arrays::position_array"
    )]
    pub default_positions: [CuJointPosition; N],
    /// Largest target change per cycle, in each joint's SI unit. `None`
    /// disables the step limit (the controller's own limits still apply).
    #[serde(default)]
    pub max_step: Option<f64>,
    /// Control cycle period (`1 / rate_target_hz`). When set, the setpoint
    /// velocity is the target change of the cycle divided by this period,
    /// which is what a controller in `velocity` mode executes. `None` emits
    /// zero setpoint velocity (`position` and `effort` modes only).
    #[serde(default)]
    pub period: Option<Time>,
}

impl<const N: usize> Default for CuJointActionParams<N> {
    /// No chain (rejected by [`Self::validate`]), offset `0`, `position_delta`, scale
    /// `1.0` on every joint, zero default positions, no step limit.
    fn default() -> Self {
        Self {
            chain: None,
            offset: 0,
            kind: CuJointActionKind::default(),
            scale: default_scale(),
            default_positions: [CuJointPosition::default(); N],
            max_step: None,
            period: None,
        }
    }
}

impl<const N: usize> CuJointActionParams<N> {
    /// Resolves the configured chain and validates it.
    pub fn resolve_chain(&self) -> CuResult<CuKinematicChain<N>> {
        let spec = self.chain.as_ref().ok_or_else(|| {
            CuError::from("repose-policy-rl: missing required 'chain' configuration")
        })?;
        spec.resolve()
    }

    /// Checks the chain, scale, step limit, and (for `position_absolute`)
    /// that the default positions carry each joint's unit kind and lie
    /// inside its limits.
    pub fn validate(&self) -> CuResult<()> {
        if N == 0 {
            return Err(CuError::from(
                "repose-policy-rl: joint count N must be at least 1",
            ));
        }
        let chain = self.resolve_chain()?;
        if !self.scale.iter().all(|s| s.is_finite()) {
            return Err(CuError::from(
                "repose-policy-rl: 'scale' must be finite on every joint",
            ));
        }
        if let Some(max_step) = self.max_step
            && (!max_step.is_finite() || max_step <= 0.0)
        {
            return Err(CuError::from(
                "repose-policy-rl: 'max_step' must be positive and finite",
            ));
        }
        if let Some(period) = self.period {
            let s = period.get::<second>();
            if !s.is_finite() || s <= 0.0 {
                return Err(CuError::from(
                    "repose-policy-rl: 'period' must be positive and finite",
                ));
            }
        }
        if self.kind == CuJointActionKind::PositionAbsolute {
            for (j, pos) in self.default_positions.iter().enumerate() {
                let limits = chain.joints()[j].limits;
                limits.validate(*pos)?;
                if !limits.contains(*pos) {
                    return Err(CuError::from(
                        "repose-policy-rl: 'default_positions' must lie inside the joint limits",
                    ));
                }
            }
        }
        Ok(())
    }
}

/// What [`CuJointActionTask`] did with the action on its most recent cycle.
///
/// Exposed via [`CuJointActionTask::last_report`] without changing the task's
/// `CuTrajectoryFollowOutput<N>` output contract, so telemetry and incident
/// triage can inspect step-clamping, joint-limit clamping, and requested delta
/// magnitude.
#[derive(
    Default, Debug, Clone, Copy, PartialEq, Encode, Decode, Serialize, Deserialize, Reflect,
)]
pub struct CuJointActionReport {
    /// Bitmask of joints (`1u64 << j`, for `j < 64`) whose target change from
    /// the reference position was clamped by `max_step`.
    pub step_limited_mask: u64,
    /// Bitmask of joints (`1u64 << j`, for `j < 64`) whose target was clamped
    /// by the chain's joint limits.
    pub joint_limited_mask: u64,
    /// Number of joints clamped by `max_step` on the cycle.
    pub step_limited_count: u16,
    /// Number of joints clamped by the chain's joint limits on the cycle.
    pub joint_limited_count: u16,
    /// Maximum `|raw_target[j] - reference[j]|` across all joints on the
    /// cycle, in each joint's SI unit (radians or meters), before `max_step`
    /// and joint-limit clamping. Zero on hold or held-delta cycles.
    pub max_requested_delta_si: f64,
}

impl CuJointActionReport {
    /// Returns `true` if joint `j` was clamped by `max_step` (`false` when
    /// `j >= 64`).
    pub const fn is_step_limited(&self, j: usize) -> bool {
        j < 64 && (self.step_limited_mask & (1u64 << j)) != 0
    }

    /// Returns `true` if joint `j` was clamped by the chain's joint limits
    /// (`false` when `j >= 64`).
    pub const fn is_joint_limited(&self, j: usize) -> bool {
        j < 64 && (self.joint_limited_mask & (1u64 << j)) != 0
    }

    /// Returns `true` if any joint was clamped by `max_step` or joint limits.
    pub const fn any_limited(&self) -> bool {
        self.step_limited_count > 0 || self.joint_limited_count > 0
    }
}

/// Turns a raw policy action into a limited joint setpoint.
///
/// Inputs are `(action, state)`: a [`CuPolicyAction<ACT>`] and a
/// [`CuJointStatePayload<N>`]. The task reads action elements
/// `[offset, offset + N)` (see [`CuJointActionParams::offset`]), so one flat
/// policy action can feed several action tasks, each owning its own slice;
/// `offset + N <= ACT` is checked at construction. Output is a
/// [`CuTrajectoryFollowOutput<N>`] for `repose_controller::CuControllerTask`,
/// with zero velocity and acceleration; its `Tov` is propagated from the
/// joint-state input.
///
/// Without `period` the setpoint carries a position target only, so the
/// controller must run in `position` or `effort` mode: in `velocity` mode
/// the zero setpoint velocity is what the bridge would execute, and the arm
/// would not move. With `period` set to the control cycle period, the
/// setpoint velocity is the target change of the cycle divided by the
/// period, so a controller in `velocity` mode moves the arm by exactly that
/// change. Every hold case, and a held delta, has zero velocity. With a
/// policy `decimation` above 1 a delta is therefore executed in the one
/// cycle it is inferred in, as in `position` mode; bound its speed with
/// `max_step` (`max_step / period`).
///
/// Per cycle, for a usable (`Fresh` or `Held`) action and a `Ready` robot:
///
/// 1. `position_delta`: `target = measured_position + scale * action`. The
///    offset is taken from the measured position, not from the previous
///    target, so a target cannot run ahead of an arm that lags behind it.
///    A zero action therefore commands the measured position.
///    `position_absolute`: `target = default_positions + scale * action`.
/// 2. The change from the previous target (from the measured position when
///    there is no previous target yet) is limited to `max_step`.
/// 3. The target is clamped to the chain's joint limits. The clamp wins
///    over `max_step`: a target that starts outside the limits (a measured
///    position beyond them) moves to the limit in one cycle.
/// 4. The target is emitted with [`CuFollowStatus::Running`].
///
/// A `Held` action repeats the last inferred action (policy decimation). A
/// delta is applied once, on the cycle it was inferred: while it is `Held`
/// the existing target is kept with [`CuFollowStatus::Running`], which is
/// how trainers hold a target across control substeps. With no target yet
/// (the first action seen is already `Held`) the delta is applied once. A
/// `Held` absolute action is re-applied, so a `max_step`-limited target
/// keeps approaching it.
///
/// Hold cases drop the target and emit the measured position with zero
/// velocity:
///
/// | Situation | Status emitted |
/// |---|---|
/// | No action yet, action status `Idle`, or robot not `Ready` | `Idle` |
/// | Action status `Fault`, a non-finite value in this task's action slice or a non-finite setpoint velocity, or a joint state that is not finite or whose units do not match the chain | `ErrorEmptyTrajectory` |
///
/// The controller holds the measured position during a hold, and a robot
/// that is not `Ready` (fault, emergency stop) may be moved while it is
/// down. Dropping the target makes the first active cycle afterwards start
/// from where the arm is: `max_step` and the setpoint velocity are then
/// measured from the measured position, not from a target the arm never
/// reached. With no joint state at all the previous target is kept and
/// emitted (zeros before the first state) with `Idle`.
///
/// Only the slice this task reads is checked for finiteness: a non-finite
/// value outside it belongs to another consumer and does not fault this
/// task (the policy task already faults the whole action on any non-finite
/// output). The action status is shared, so a `Fault` or `Idle` action
/// holds every task reading it.
///
/// `ErrorEmptyTrajectory` is the plan's naming compromise so the unchanged
/// controller maps it to its `Fault` status.
///
/// Previous target and whether a target exists round-trip through
/// `Freezable`, so deterministic replay resumes identically. Parameters are
/// not frozen: they come from the configuration and cannot change at run
/// time.
#[derive(Reflect)]
#[reflect(no_field_bounds, from_reflect = false)]
pub struct CuJointActionTask<const N: usize, const ACT: usize> {
    #[reflect(ignore)]
    params: CuJointActionParams<N>,
    #[reflect(ignore)]
    chain: CuKinematicChain<N>,
    #[reflect(ignore)]
    previous_target: [CuJointPosition; N],
    has_target: bool,
    #[reflect(ignore)]
    last_report: CuJointActionReport,
}

impl<const N: usize, const ACT: usize> CuJointActionTask<N, ACT> {
    /// Creates an action task from validated params. Also rejects an
    /// `offset` whose slice `[offset, offset + N)` does not fit in the
    /// `ACT`-wide action.
    pub fn from_params(params: CuJointActionParams<N>) -> CuResult<Self> {
        params.validate()?;
        match params.offset.checked_add(N) {
            Some(end) if end <= ACT => {}
            _ => {
                return Err(CuError::from(
                    "repose-policy-rl: 'offset' + joint count N exceeds the policy action width ACT",
                ));
            }
        }
        let chain = params.resolve_chain()?;
        Ok(Self {
            params,
            chain,
            previous_target: [CuJointPosition::default(); N],
            has_target: false,
            last_report: CuJointActionReport::default(),
        })
    }

    /// Params this task runs with.
    pub fn params(&self) -> &CuJointActionParams<N> {
        &self.params
    }

    /// Diagnostic report of clamping and requested delta magnitude from the
    /// most recent `process` call.
    pub fn last_report(&self) -> &CuJointActionReport {
        &self.last_report
    }

    /// Setpoint at rest: zero velocity and acceleration carrying each
    /// joint's unit kind.
    fn setpoint(
        positions: [CuJointPosition; N],
        status: CuFollowStatus,
    ) -> CuTrajectoryFollowOutput<N> {
        CuTrajectoryFollowOutput {
            positions,
            velocities: positions.map(|q| CuJointVelocity::from_coordinate(q, 0.0)),
            accelerations: positions.map(|q| CuJointAcceleration::from_coordinate(q, 0.0)),
            status,
        }
    }

    fn held_positions(&self, state: Option<&CuJointStatePayload<N>>) -> [CuJointPosition; N] {
        if self.has_target {
            self.previous_target
        } else if let Some(state) = state {
            state.positions
        } else {
            [CuJointPosition::default(); N]
        }
    }
}

impl<const N: usize, const ACT: usize> Freezable for CuJointActionTask<N, ACT> {
    fn freeze<E: Encoder>(&self, encoder: &mut E) -> Result<(), EncodeError> {
        Encode::encode(&self.previous_target, encoder)?;
        Encode::encode(&self.has_target, encoder)
    }

    fn thaw<D: Decoder>(&mut self, decoder: &mut D) -> Result<(), DecodeError> {
        self.previous_target = Decode::decode(decoder)?;
        self.has_target = Decode::decode(decoder)?;
        Ok(())
    }
}

impl<const N: usize, const ACT: usize> CuTask for CuJointActionTask<N, ACT> {
    type Resources<'r> = ();
    type Input<'m> = input_msg!('m, CuPolicyAction<ACT>, CuJointStatePayload<N>);
    type Output<'m> = output_msg!(CuTrajectoryFollowOutput<N>);

    fn new(cfg: Option<&ComponentConfig>, _res: Self::Resources<'_>) -> CuResult<Self>
    where
        Self: Sized,
    {
        let params = cfg
            .map(ComponentConfig::deserialize_into::<CuJointActionParams<N>>)
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
        output.tov = input.1.tov;
        self.last_report = CuJointActionReport::default();
        let action = input.0.payload();
        let state = input.1.payload();

        let hold = |task: &Self, status: CuFollowStatus| {
            Self::setpoint(task.held_positions(state), status)
        };

        // No state at all: hold, whatever the action.
        let Some(state) = state else {
            output.set_payload(hold(self, CuFollowStatus::Idle));
            return Ok(());
        };
        // Every hold below drops the target: the controller holds the
        // measured position during a hold, so the next active cycle has to
        // start from where the arm is, not from a target it never reached.
        //
        // Robot not ready (the arm may be moved while it is down).
        if state.status != CuRobotStatus::Ready {
            self.has_target = false;
            output.set_payload(hold(self, CuFollowStatus::Idle));
            return Ok(());
        }
        // A joint state whose units do not match the chain, or that is not
        // finite, is a fault of the cycle, not of the task.
        let mut measured = state.positions;
        if self.chain.clamp_to_limits(&mut measured).is_err() {
            self.has_target = false;
            output.set_payload(hold(self, CuFollowStatus::ErrorEmptyTrajectory));
            return Ok(());
        }
        // No action yet, or an idle action.
        let Some(action) = action.filter(|a| a.status != CuPolicyStatus::Idle) else {
            self.has_target = false;
            output.set_payload(hold(self, CuFollowStatus::Idle));
            return Ok(());
        };
        // This task's slice; the bound was checked at construction, so
        // `None` is unreachable but still holds rather than panicking.
        let offset = self.params.offset;
        let Some(values) = action.values.get(offset..offset.saturating_add(N)) else {
            self.has_target = false;
            output.set_payload(hold(self, CuFollowStatus::ErrorEmptyTrajectory));
            return Ok(());
        };
        // A faulted action, or a non-finite value in this task's slice.
        if action.status == CuPolicyStatus::Fault || !values.iter().all(|v| v.is_finite()) {
            self.has_target = false;
            output.set_payload(hold(self, CuFollowStatus::ErrorEmptyTrajectory));
            return Ok(());
        }

        // A held delta was already applied on the cycle it was inferred:
        // keep that target instead of stepping again.
        if self.params.kind == CuJointActionKind::PositionDelta
            && action.status == CuPolicyStatus::Held
            && self.has_target
        {
            output.set_payload(Self::setpoint(
                self.previous_target,
                CuFollowStatus::Running,
            ));
            return Ok(());
        }

        // Usable action (`Fresh` or `Held`): build the raw target in each
        // joint's SI unit, keeping the joint kind of the base coordinate.
        let base: [CuJointPosition; N] = match self.params.kind {
            CuJointActionKind::PositionDelta => state.positions,
            CuJointActionKind::PositionAbsolute => self.params.default_positions,
        };
        let reference: [CuJointPosition; N] = if self.has_target {
            self.previous_target
        } else {
            state.positions
        };
        let mut target = [CuJointPosition::default(); N];
        let mut report = CuJointActionReport::default();
        for j in 0..N {
            let raw = base[j].si_value() + self.params.scale[j] * f64::from(values[j]);
            if !raw.is_finite() {
                self.has_target = false;
                output.set_payload(hold(self, CuFollowStatus::ErrorEmptyTrajectory));
                return Ok(());
            }
            let ref_si = reference[j].si_value();
            let requested_delta = raw - ref_si;
            let delta_abs = requested_delta.abs();
            if delta_abs > report.max_requested_delta_si {
                report.max_requested_delta_si = delta_abs;
            }
            let stepped = match self.params.max_step {
                Some(max_step) => {
                    if delta_abs > max_step {
                        if j < 64 {
                            report.step_limited_mask |= 1u64 << j;
                        }
                        report.step_limited_count = report.step_limited_count.saturating_add(1);
                    }
                    let change = requested_delta.clamp(-max_step, max_step);
                    ref_si + change
                }
                None => raw,
            };
            target[j] = base[j].with_si_value(stepped);
        }
        let pre_limit = target;
        if self.chain.clamp_to_limits(&mut target).is_err() {
            self.has_target = false;
            output.set_payload(hold(self, CuFollowStatus::ErrorEmptyTrajectory));
            return Ok(());
        }
        for j in 0..N {
            if target[j].si_value() != pre_limit[j].si_value() {
                if j < 64 {
                    report.joint_limited_mask |= 1u64 << j;
                }
                report.joint_limited_count = report.joint_limited_count.saturating_add(1);
            }
        }
        let mut setpoint = Self::setpoint(target, CuFollowStatus::Running);
        if let Some(period) = self.params.period {
            let period = period.get::<second>();
            for j in 0..N {
                let velocity = (target[j].si_value() - reference[j].si_value()) / period;
                if !velocity.is_finite() {
                    self.has_target = false;
                    output.set_payload(hold(self, CuFollowStatus::ErrorEmptyTrajectory));
                    return Ok(());
                }
                setpoint.velocities[j] = CuJointVelocity::from_coordinate(target[j], velocity);
            }
        }
        self.previous_target = target;
        self.has_target = true;
        self.last_report = report;
        output.set_payload(setpoint);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cu29::units::si::angle::radian;
    use cu29::units::si::f64::Angle;
    use cu29::units::si::f64::Length;
    use cu29::units::si::length::meter;
    use repose_controller::CuControllerStatus;
    use repose_controller::CuControllerTask;
    use repose_controller::CuJointCommandPayload;
    use repose_kinematics::CuJointAxis;
    use repose_kinematics::CuJointLimits;
    use repose_kinematics::CuJointSpec;
    use repose_kinematics::CuJointVelocity as KinVelocity;

    fn rev(value: f64) -> CuJointPosition {
        CuJointPosition::Revolute(Angle::new::<radian>(value))
    }

    fn rev_spec(lo: f64, hi: f64) -> CuJointSpec {
        CuJointSpec::revolute(
            [Length::new::<meter>(0.0); 3],
            [Angle::new::<radian>(0.0); 3],
            CuJointAxis::PosZ,
            CuJointLimits::revolute(
                Angle::new::<radian>(lo),
                Angle::new::<radian>(hi),
                cu29::units::si::f64::AngularVelocity::new::<
                    cu29::units::si::angular_velocity::radian_per_second,
                >(2.0),
            ),
        )
    }

    fn chain2(lo: f64, hi: f64) -> CuKinematicChain<2> {
        CuKinematicChain::new([rev_spec(lo, hi), rev_spec(lo, hi)])
    }

    fn params<const N: usize>(
        chain: CuKinematicChain<N>,
        kind: CuJointActionKind,
    ) -> CuJointActionParams<N> {
        CuJointActionParams {
            chain: Some(chain.into()),
            offset: 0,
            kind,
            scale: [1.0; N],
            default_positions: [CuJointPosition::default(); N],
            max_step: None,
            period: None,
        }
    }

    fn state<const N: usize>(pos: f64) -> CuJointStatePayload<N> {
        CuJointStatePayload {
            status: CuRobotStatus::Ready,
            positions: [rev(pos); N],
            velocities: [KinVelocity::default(); N],
        }
    }

    fn action<const N: usize>(value: f32, status: CuPolicyStatus) -> CuPolicyAction<N> {
        CuPolicyAction {
            values: [value; N],
            status,
        }
    }

    fn run<const N: usize, const ACT: usize>(
        task: &mut CuJointActionTask<N, ACT>,
        act: Option<CuPolicyAction<ACT>>,
        st: Option<CuJointStatePayload<N>>,
    ) -> CuTrajectoryFollowOutput<N> {
        let (ctx, _clock) = CuContext::new_mock_clock();
        let action_msg = CuMsg::new(act);
        let state_msg = CuMsg::new(st);
        let mut output = CuMsg::<CuTrajectoryFollowOutput<N>>::default();
        task.process(&ctx, &(&action_msg, &state_msg), &mut output)
            .unwrap();
        output.payload().unwrap().clone()
    }

    fn si<const N: usize>(out: &CuTrajectoryFollowOutput<N>) -> [f64; N] {
        out.positions.map(|p| p.si_value())
    }

    fn assert_si<const N: usize>(out: &CuTrajectoryFollowOutput<N>, expected: f64) {
        assert!(
            si(out).iter().all(|v| (*v - expected).abs() < 1e-9),
            "{:?} != {expected}",
            si(out)
        );
    }

    #[test]
    fn delta_is_an_offset_from_the_measured_position() {
        let mut p = params(chain2(-2.0, 2.0), CuJointActionKind::PositionDelta);
        p.scale = [0.05; 2];
        let mut task = CuJointActionTask::<2, 2>::from_params(p).unwrap();
        // Measured start is 0.10 even though no target exists yet.
        let out = run(
            &mut task,
            Some(action(1.0, CuPolicyStatus::Fresh)),
            Some(state(0.10)),
        );
        assert_eq!(out.status, CuFollowStatus::Running);
        assert!(
            si(&out).iter().all(|v| (*v - 0.15).abs() < 1e-12),
            "{:?}",
            si(&out)
        );
        // The measured position has not moved: the next delta is taken from
        // the measurement again and does not accumulate.
        let out = run(
            &mut task,
            Some(action(1.0, CuPolicyStatus::Fresh)),
            Some(state(0.10)),
        );
        assert!(
            si(&out).iter().all(|v| (*v - 0.15).abs() < 1e-12),
            "{:?}",
            si(&out)
        );
        // The arm moved: the same delta now lands further along.
        let out = run(
            &mut task,
            Some(action(1.0, CuPolicyStatus::Fresh)),
            Some(state(0.15)),
        );
        assert!(
            si(&out).iter().all(|v| (*v - 0.20).abs() < 1e-12),
            "{:?}",
            si(&out)
        );
        // A held delta was already applied: the target stays where it is.
        let out = run(
            &mut task,
            Some(action(1.0, CuPolicyStatus::Held)),
            Some(state(0.10)),
        );
        assert_eq!(out.status, CuFollowStatus::Running);
        assert!(
            si(&out).iter().all(|v| (*v - 0.20).abs() < 1e-12),
            "{:?}",
            si(&out)
        );
    }

    #[test]
    fn position_absolute_uses_default_positions() {
        let mut p = params(chain2(-2.0, 2.0), CuJointActionKind::PositionAbsolute);
        p.scale = [0.05; 2];
        p.default_positions = [rev(0.30); 2];
        let mut task = CuJointActionTask::<2, 2>::from_params(p).unwrap();
        let out = run(
            &mut task,
            Some(action(2.0, CuPolicyStatus::Fresh)),
            Some(state(0.10)),
        );
        assert_eq!(out.status, CuFollowStatus::Running);
        assert!(
            si(&out).iter().all(|v| (*v - 0.40).abs() < 1e-12),
            "{:?}",
            si(&out)
        );
        // Repeating the same absolute action does not accumulate.
        let out = run(
            &mut task,
            Some(action(2.0, CuPolicyStatus::Fresh)),
            Some(state(0.10)),
        );
        assert!(
            si(&out).iter().all(|v| (*v - 0.40).abs() < 1e-12),
            "{:?}",
            si(&out)
        );
    }

    #[test]
    fn max_step_limits_a_large_action() {
        let mut p = params(chain2(-2.0, 2.0), CuJointActionKind::PositionDelta);
        p.max_step = Some(0.03);
        let mut task = CuJointActionTask::<2, 2>::from_params(p).unwrap();
        let out = run(
            &mut task,
            Some(action(10.0, CuPolicyStatus::Fresh)),
            Some(state(0.0)),
        );
        assert!(
            si(&out).iter().all(|v| (*v - 0.03).abs() < 1e-12),
            "{:?}",
            si(&out)
        );
        let out = run(
            &mut task,
            Some(action(-10.0, CuPolicyStatus::Fresh)),
            Some(state(0.0)),
        );
        assert!(si(&out).iter().all(|v| v.abs() < 1e-12), "{:?}", si(&out));
    }

    #[test]
    fn joint_limits_clamp_the_target() {
        let p = params(chain2(-0.5, 0.5), CuJointActionKind::PositionDelta);
        let mut task = CuJointActionTask::<2, 2>::from_params(p).unwrap();
        let out = run(
            &mut task,
            Some(action(10.0, CuPolicyStatus::Fresh)),
            Some(state(0.0)),
        );
        assert!(
            si(&out).iter().all(|v| (*v - 0.5).abs() < 1e-12),
            "{:?}",
            si(&out)
        );
        let out = run(
            &mut task,
            Some(action(-10.0, CuPolicyStatus::Fresh)),
            Some(state(0.0)),
        );
        assert!(
            si(&out).iter().all(|v| (*v + 0.5).abs() < 1e-12),
            "{:?}",
            si(&out)
        );
    }

    #[test]
    fn hold_cases_emit_the_previous_target_with_documented_status() {
        let mut p = params(chain2(-2.0, 2.0), CuJointActionKind::PositionDelta);
        p.scale = [0.05; 2];
        let mut task = CuJointActionTask::<2, 2>::from_params(p).unwrap();
        let out = run(
            &mut task,
            Some(action(1.0, CuPolicyStatus::Fresh)),
            Some(state(0.10)),
        );
        assert_si(&out, 0.15);

        // No joint state at all: the previous target is kept and emitted.
        let out = run(&mut task, Some(action(1.0, CuPolicyStatus::Fresh)), None);
        assert_eq!(out.status, CuFollowStatus::Idle);
        assert_si(&out, 0.15);

        // No action yet: the target is dropped, the measured position held.
        let out = run(&mut task, None, Some(state(0.10)));
        assert_eq!(out.status, CuFollowStatus::Idle);
        assert_si(&out, 0.10);
        assert!(out.velocities.iter().all(|v| v.si_value() == 0.0));
        assert!(out.accelerations.iter().all(|v| v.si_value() == 0.0));

        // Action status Idle.
        let out = run(
            &mut task,
            Some(action(1.0, CuPolicyStatus::Idle)),
            Some(state(0.10)),
        );
        assert_eq!(out.status, CuFollowStatus::Idle);
        assert_si(&out, 0.10);

        // Action status Fault.
        let out = run(
            &mut task,
            Some(action(1.0, CuPolicyStatus::Fault)),
            Some(state(0.10)),
        );
        assert_eq!(out.status, CuFollowStatus::ErrorEmptyTrajectory);
        assert_si(&out, 0.10);
        assert!(out.velocities.iter().all(|v| v.si_value() == 0.0));

        // Robot not Ready (every non-Ready status): the target is dropped
        // and the measured position is held.
        for status in [
            CuRobotStatus::NotReady,
            CuRobotStatus::Fault(7),
            CuRobotStatus::EmergencyStop,
        ] {
            let mut st = state::<2>(0.10);
            st.status = status;
            let out = run(
                &mut task,
                Some(action(1.0, CuPolicyStatus::Fresh)),
                Some(st),
            );
            assert_eq!(out.status, CuFollowStatus::Idle, "{status:?}");
            assert_si(&out, 0.10);
        }
    }

    #[test]
    fn recovery_after_not_ready_starts_from_the_measured_position() {
        let mut p = params(chain2(-2.0, 2.0), CuJointActionKind::PositionDelta);
        p.scale = [0.05; 2];
        let mut task = CuJointActionTask::<2, 2>::from_params(p).unwrap();
        let out = run(
            &mut task,
            Some(action(1.0, CuPolicyStatus::Fresh)),
            Some(state(0.10)),
        );
        assert_si(&out, 0.15);

        // Emergency stop; the arm is moved by hand to 0.60 while it is down.
        let mut down = state::<2>(0.60);
        down.status = CuRobotStatus::EmergencyStop;
        let out = run(
            &mut task,
            Some(action(1.0, CuPolicyStatus::Fresh)),
            Some(down),
        );
        assert_eq!(out.status, CuFollowStatus::Idle);
        assert_si(&out, 0.60);

        // Ready again: the delta starts from 0.60, not from the stale 0.15.
        let out = run(
            &mut task,
            Some(action(1.0, CuPolicyStatus::Fresh)),
            Some(state(0.60)),
        );
        assert_eq!(out.status, CuFollowStatus::Running);
        assert_si(&out, 0.65);
    }

    #[test]
    fn hold_before_any_target_uses_the_measured_position() {
        let p = params(chain2(-2.0, 2.0), CuJointActionKind::PositionDelta);
        let mut task = CuJointActionTask::<2, 2>::from_params(p).unwrap();
        let out = run(&mut task, None, Some(state(0.30)));
        assert_eq!(out.status, CuFollowStatus::Idle);
        assert_si(&out, 0.30);
        let out = run(&mut task, None, None);
        assert_eq!(out.status, CuFollowStatus::Idle);
        assert_si(&out, 0.0);
    }

    #[test]
    fn non_finite_action_value_holds_and_faults() {
        let mut p = params(chain2(-2.0, 2.0), CuJointActionKind::PositionDelta);
        p.scale = [0.05; 2];
        let mut task = CuJointActionTask::<2, 2>::from_params(p).unwrap();
        run(
            &mut task,
            Some(action(1.0, CuPolicyStatus::Fresh)),
            Some(state(0.10)),
        );
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut act = action::<2>(1.0, CuPolicyStatus::Fresh);
            act.values[1] = bad;
            let out = run(&mut task, Some(act), Some(state(0.10)));
            assert_eq!(out.status, CuFollowStatus::ErrorEmptyTrajectory, "{bad}");
            assert_si(&out, 0.10);
        }
        // A valid action resumes from the measured position.
        let out = run(
            &mut task,
            Some(action(2.0, CuPolicyStatus::Fresh)),
            Some(state(0.10)),
        );
        assert_eq!(out.status, CuFollowStatus::Running);
        assert_si(&out, 0.20);
    }

    #[test]
    fn tov_is_propagated_from_the_joint_state() {
        let p = params(chain2(-2.0, 2.0), CuJointActionKind::PositionDelta);
        let mut task = CuJointActionTask::<2, 2>::from_params(p).unwrap();
        let (ctx, _clock) = CuContext::new_mock_clock();
        let mut action_msg = CuMsg::new(Some(action::<2>(0.0, CuPolicyStatus::Fresh)));
        action_msg.tov = Tov::Time(CuTime::from_secs(10));
        let mut state_msg = CuMsg::new(Some(state::<2>(0.0)));
        state_msg.tov = Tov::Time(CuTime::from_secs(42));
        let mut output = CuMsg::<CuTrajectoryFollowOutput<2>>::default();
        task.process(&ctx, &(&action_msg, &state_msg), &mut output)
            .unwrap();
        assert_eq!(output.tov, Tov::Time(CuTime::from_secs(42)));
    }

    #[test]
    fn freeze_thaw_resumes_identically() {
        let mut p = params(chain2(-2.0, 2.0), CuJointActionKind::PositionDelta);
        p.scale = [0.05; 2];
        p.max_step = Some(0.04);
        let mut original = CuJointActionTask::<2, 2>::from_params(p).unwrap();
        run(
            &mut original,
            Some(action(1.0, CuPolicyStatus::Fresh)),
            Some(state(0.10)),
        );
        run(
            &mut original,
            Some(action(1.0, CuPolicyStatus::Fresh)),
            Some(state(0.10)),
        );

        let mut bytes = [0u8; 8192];
        let mut encoder = bincode::enc::EncoderImpl::new(
            bincode::enc::write::SliceWriter::new(&mut bytes),
            bincode::config::standard(),
        );
        original.freeze(&mut encoder).unwrap();
        let len = encoder.into_writer().bytes_written();
        let mut decoder = bincode::de::DecoderImpl::new(
            bincode::de::read::SliceReader::new(&bytes[..len]),
            bincode::config::standard(),
            (),
        );
        let mut restored = CuJointActionTask::<2, 2>::from_params(p).unwrap();
        restored.thaw(&mut decoder).unwrap();

        for value in [1.0f32, -0.5, 0.25] {
            let a = run(
                &mut original,
                Some(action(value, CuPolicyStatus::Fresh)),
                Some(state(0.10)),
            );
            let b = run(
                &mut restored,
                Some(action(value, CuPolicyStatus::Fresh)),
                Some(state(0.10)),
            );
            assert_eq!(a, b, "value {value}");
        }
    }

    #[test]
    fn prismatic_joint_keeps_its_unit() {
        use cu29::units::si::f64::Velocity;
        use cu29::units::si::velocity::meter_per_second;
        let chain = CuKinematicChain::new([
            rev_spec(-2.0, 2.0),
            CuJointSpec::prismatic(
                [Length::new::<meter>(0.0); 3],
                [Angle::new::<radian>(0.0); 3],
                CuJointAxis::PosZ,
                CuJointLimits::prismatic(
                    Length::new::<meter>(0.0),
                    Length::new::<meter>(1.0),
                    Velocity::new::<meter_per_second>(0.2),
                ),
            ),
        ]);
        let mut p = params(chain, CuJointActionKind::PositionAbsolute);
        p.default_positions = [
            rev(0.10),
            CuJointPosition::Prismatic(Length::new::<meter>(0.20)),
        ];
        let mut task = CuJointActionTask::<2, 2>::from_params(p).unwrap();
        let mut st = state::<2>(0.0);
        st.positions[1] = CuJointPosition::Prismatic(Length::new::<meter>(0.20));
        let mut act = action::<2>(0.05, CuPolicyStatus::Fresh);
        act.values = [0.05, -0.05];
        let (ctx, _clock) = CuContext::new_mock_clock();
        let mut output = CuMsg::<CuTrajectoryFollowOutput<2>>::default();
        task.process(
            &ctx,
            &(&CuMsg::new(Some(act)), &CuMsg::new(Some(st))),
            &mut output,
        )
        .unwrap();
        let out = output.payload().unwrap();
        assert!(matches!(out.positions[0], CuJointPosition::Revolute(_)));
        assert!(matches!(out.positions[1], CuJointPosition::Prismatic(_)));
        assert!((out.positions[0].si_value() - 0.15).abs() < 1e-6);
        assert!((out.positions[1].si_value() - 0.15).abs() < 1e-6);
        // The zero velocity and acceleration carry each joint's unit kind.
        assert!(matches!(out.velocities[0], CuJointVelocity::Revolute(_)));
        assert!(matches!(out.velocities[1], CuJointVelocity::Prismatic(_)));
        assert!(matches!(
            out.accelerations[1],
            CuJointAcceleration::Prismatic(_)
        ));
    }

    #[test]
    fn position_absolute_ramps_to_a_far_target_with_max_step() {
        let mut p = params(chain2(-2.0, 2.0), CuJointActionKind::PositionAbsolute);
        p.default_positions = [rev(1.0); 2];
        p.max_step = Some(0.25);
        let mut task = CuJointActionTask::<2, 2>::from_params(p).unwrap();
        // Target 1.0 from a measured 0.0: one step per cycle, and a held
        // action keeps approaching.
        let out = run(
            &mut task,
            Some(action(0.0, CuPolicyStatus::Fresh)),
            Some(state(0.0)),
        );
        assert_si(&out, 0.25);
        let out = run(
            &mut task,
            Some(action(0.0, CuPolicyStatus::Held)),
            Some(state(0.0)),
        );
        assert_si(&out, 0.50);
        for _ in 0..4 {
            run(
                &mut task,
                Some(action(0.0, CuPolicyStatus::Held)),
                Some(state(0.0)),
            );
        }
        let out = run(
            &mut task,
            Some(action(0.0, CuPolicyStatus::Held)),
            Some(state(0.0)),
        );
        assert_si(&out, 1.0);
    }

    #[test]
    fn period_sets_the_setpoint_velocity_from_the_target_change() {
        let mut p = params(chain2(-2.0, 2.0), CuJointActionKind::PositionDelta);
        p.scale = [0.05; 2];
        p.period = Some(Time::new::<second>(0.02));
        let mut task = CuJointActionTask::<2, 2>::from_params(p).unwrap();
        // 0.05 rad in one 0.02 s cycle is 2.5 rad/s.
        let out = run(
            &mut task,
            Some(action(1.0, CuPolicyStatus::Fresh)),
            Some(state(0.10)),
        );
        assert_si(&out, 0.15);
        assert!(
            out.velocities
                .iter()
                .all(|v| (v.si_value() - 2.5).abs() < 1e-9)
        );
        // A held delta keeps the target: no change, zero velocity.
        let out = run(
            &mut task,
            Some(action(1.0, CuPolicyStatus::Held)),
            Some(state(0.10)),
        );
        assert_si(&out, 0.15);
        assert!(out.velocities.iter().all(|v| v.si_value() == 0.0));
        // A hold case has zero velocity too.
        let out = run(&mut task, None, Some(state(0.10)));
        assert!(out.velocities.iter().all(|v| v.si_value() == 0.0));
        // After the hold the reference is the measured position; integrating
        // the velocity over the cycles then reproduces the target.
        let mut integrated = 0.10;
        for _ in 0..5 {
            let out = run(
                &mut task,
                Some(action(-0.5, CuPolicyStatus::Fresh)),
                Some(state(0.10)),
            );
            integrated += out.velocities[0].si_value() * 0.02;
            assert!((integrated - out.positions[0].si_value()).abs() < 1e-9);
        }
    }

    #[test]
    fn absolute_action_after_a_hold_moves_from_the_measured_position() {
        let mut p = params(chain2(-2.0, 2.0), CuJointActionKind::PositionAbsolute);
        p.default_positions = [rev(0.5); 2];
        p.period = Some(Time::new::<second>(0.02));
        let mut task = CuJointActionTask::<2, 2>::from_params(p).unwrap();
        let out = run(
            &mut task,
            Some(action(0.0, CuPolicyStatus::Fresh)),
            Some(state(0.0)),
        );
        assert_si(&out, 0.5);
        // A hold: the controller holds the measured position, still 0.0.
        run(&mut task, None, Some(state(0.0)));
        // The same absolute action again must command motion, not zero
        // velocity toward a target the arm never reached.
        let out = run(
            &mut task,
            Some(action(0.0, CuPolicyStatus::Fresh)),
            Some(state(0.0)),
        );
        assert_si(&out, 0.5);
        assert!(
            out.velocities
                .iter()
                .all(|v| (v.si_value() - 25.0).abs() < 1e-9)
        );
    }

    #[test]
    fn overflowing_setpoint_velocity_faults() {
        let mut p = params(chain2(-2.0, 2.0), CuJointActionKind::PositionDelta);
        // The smallest positive (subnormal) period: any target change overflows.
        p.period = Some(Time::new::<second>(f64::from_bits(1)));
        let mut task = CuJointActionTask::<2, 2>::from_params(p).unwrap();
        let out = run(
            &mut task,
            Some(action(1.0, CuPolicyStatus::Fresh)),
            Some(state(0.0)),
        );
        assert_eq!(out.status, CuFollowStatus::ErrorEmptyTrajectory);
        assert!(out.velocities.iter().all(|v| v.si_value() == 0.0));
    }

    #[test]
    fn mismatched_measured_units_fault_every_action_kind() {
        for kind in [
            CuJointActionKind::PositionDelta,
            CuJointActionKind::PositionAbsolute,
        ] {
            let mut task =
                CuJointActionTask::<2, 2>::from_params(params(chain2(-2.0, 2.0), kind)).unwrap();
            let mut st = state::<2>(0.0);
            st.positions[1] = CuJointPosition::Prismatic(Length::new::<meter>(0.0));
            let out = run(
                &mut task,
                Some(action(0.1, CuPolicyStatus::Fresh)),
                Some(st),
            );
            assert_eq!(out.status, CuFollowStatus::ErrorEmptyTrajectory, "{kind:?}");
        }
    }

    #[test]
    fn period_must_be_positive_and_finite() {
        for bad in [0.0, -0.02, f64::NAN] {
            let mut p = params(chain2(-2.0, 2.0), CuJointActionKind::PositionDelta);
            p.period = Some(Time::new::<second>(bad));
            assert!(CuJointActionTask::<2, 2>::from_params(p).is_err(), "{bad}");
        }
    }

    #[test]
    fn default_positions_outside_the_limits_are_rejected() {
        let mut p = params(chain2(-2.0, 2.0), CuJointActionKind::PositionAbsolute);
        p.default_positions = [rev(3.0); 2];
        assert!(CuJointActionTask::<2, 2>::from_params(p).is_err());
        // Unused in delta mode, so not checked there.
        let mut p = params(chain2(-2.0, 2.0), CuJointActionKind::PositionDelta);
        p.default_positions = [rev(3.0); 2];
        assert!(CuJointActionTask::<2, 2>::from_params(p).is_ok());
    }

    #[cfg(feature = "std")]
    #[test]
    fn params_ron_round_trip_and_typo_rejection() {
        let cfg: ComponentConfig = ron::from_str(
            r#"#![enable(unwrap_newtypes)] {"chain": "igus_rebel", "kind": "position_absolute",
            "scale": [0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
            "default_positions": [{"revolute": 0.1}, {"revolute": 0.2}, {"revolute": 0.3},
                {"revolute": 0.4}, {"revolute": 0.5}, {"revolute": 0.6}],
            "max_step": 0.03}"#,
        )
        .unwrap();
        let task = CuJointActionTask::<6, 6>::new(Some(&cfg), ()).unwrap();
        let p = task.params();
        assert_eq!(p.kind, CuJointActionKind::PositionAbsolute);
        assert_eq!(p.scale, [0.05; 6]);
        assert_eq!(p.max_step, Some(0.03));
        assert_eq!(
            p.default_positions.map(|q| q.si_value()),
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        );

        // One scale for every joint, delta kind, no step limit.
        let cfg: ComponentConfig = ron::from_str(
            r#"#![enable(unwrap_newtypes)] {"chain": "igus_rebel", "kind": "position_delta", "scale": 0.05}"#,
        )
        .unwrap();
        let task = CuJointActionTask::<6, 6>::new(Some(&cfg), ()).unwrap();
        assert_eq!(task.params().kind, CuJointActionKind::PositionDelta);
        assert_eq!(task.params().scale, [0.05; 6]);
        assert_eq!(task.params().max_step, None);
        assert_eq!(task.params().offset, 0, "offset defaults to 0");

        // An explicit offset is read, and checked against ACT.
        let cfg: ComponentConfig = ron::from_str(
            r#"#![enable(unwrap_newtypes)] {"chain": "igus_rebel", "offset": 1, "scale": 0.05}"#,
        )
        .unwrap();
        let task = CuJointActionTask::<6, 7>::new(Some(&cfg), ()).unwrap();
        assert_eq!(task.params().offset, 1);
        assert!(CuJointActionTask::<6, 6>::new(Some(&cfg), ()).is_err());

        // Serializing params with an inline chain and reading them back
        // gives the same params (a preset serializes as a bare name, which
        // only the config path above reads).
        let mut inline = params(chain2(-2.0, 2.0), CuJointActionKind::PositionAbsolute);
        inline.scale = [0.05, 0.25];
        inline.default_positions = [rev(0.1), rev(-0.2)];
        inline.max_step = Some(0.03);
        inline.offset = 4;
        let text = ron::to_string(&inline).unwrap();
        let back: CuJointActionParams<2> = ron::from_str(&text).unwrap();
        assert_eq!(back, inline);

        for bad in [
            // Typo: unknown key is a startup error, not a silent default.
            r#"{"chain": "igus_rebel", "scalle": 0.05}"#,
            r#"{"chain": "igus_rebel", "kind": "delta"}"#,
            r#"{"chain": "igus_rebel", "scale": [0.05, 0.05]}"#,
            r#"{"chain": "igus_rebel", "max_step": -0.03}"#,
            r#"{"kind": "position_delta"}"#,
            r#"{"chain": "nope", "kind": "position_delta"}"#,
        ] {
            let cfg: ComponentConfig =
                ron::from_str(&format!("#![enable(unwrap_newtypes)] {bad}")).unwrap();
            assert!(
                CuJointActionTask::<6, 6>::new(Some(&cfg), ()).is_err(),
                "{bad}"
            );
        }
        assert!(CuJointActionTask::<6, 6>::new(None, ()).is_err());
    }

    fn run_values<const N: usize, const ACT: usize>(
        task: &mut CuJointActionTask<N, ACT>,
        values: [f32; ACT],
        st: CuJointStatePayload<N>,
    ) -> CuTrajectoryFollowOutput<N> {
        run(
            task,
            Some(CuPolicyAction {
                values,
                status: CuPolicyStatus::Fresh,
            }),
            Some(st),
        )
    }

    #[test]
    fn offset_selects_its_slice_of_a_wider_action() {
        let mut p = params(chain2(-2.0, 2.0), CuJointActionKind::PositionAbsolute);
        p.offset = 3;
        let mut task = CuJointActionTask::<2, 6>::from_params(p).unwrap();
        let out = run_values(&mut task, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6], state(0.0));
        assert_eq!(out.status, CuFollowStatus::Running);
        let got = si(&out);
        assert!(
            (got[0] - 0.4).abs() < 1e-6 && (got[1] - 0.5).abs() < 1e-6,
            "{got:?}"
        );
    }

    #[test]
    fn two_tasks_share_one_action_through_disjoint_slices() {
        let arm = params(chain2(-2.0, 2.0), CuJointActionKind::PositionAbsolute);
        let mut gripper = params(
            CuKinematicChain::new([rev_spec(-2.0, 2.0)]),
            CuJointActionKind::PositionAbsolute,
        );
        gripper.offset = 2;
        let mut arm_task = CuJointActionTask::<2, 3>::from_params(arm).unwrap();
        let mut gripper_task = CuJointActionTask::<1, 3>::from_params(gripper).unwrap();
        let (ctx, _clock) = CuContext::new_mock_clock();
        let action_msg = CuMsg::new(Some(CuPolicyAction::<3> {
            values: [0.1, -0.2, 0.7],
            status: CuPolicyStatus::Fresh,
        }));
        let arm_state = CuMsg::new(Some(state::<2>(0.0)));
        let gripper_state = CuMsg::new(Some(state::<1>(0.0)));
        let mut arm_out = CuMsg::<CuTrajectoryFollowOutput<2>>::default();
        let mut gripper_out = CuMsg::<CuTrajectoryFollowOutput<1>>::default();
        arm_task
            .process(&ctx, &(&action_msg, &arm_state), &mut arm_out)
            .unwrap();
        gripper_task
            .process(&ctx, &(&action_msg, &gripper_state), &mut gripper_out)
            .unwrap();
        let arm_si = si(arm_out.payload().unwrap());
        let gripper_si = si(gripper_out.payload().unwrap());
        assert!((arm_si[0] - 0.1).abs() < 1e-6 && (arm_si[1] + 0.2).abs() < 1e-6);
        assert!((gripper_si[0] - 0.7).abs() < 1e-6);
    }

    #[test]
    fn offset_past_the_action_width_is_rejected() {
        for (offset, ok) in [(0, true), (4, true), (5, false), (usize::MAX, false)] {
            let mut p = params(chain2(-2.0, 2.0), CuJointActionKind::PositionDelta);
            p.offset = offset;
            assert_eq!(
                CuJointActionTask::<2, 6>::from_params(p).is_ok(),
                ok,
                "offset {offset}"
            );
        }
        // ACT smaller than N cannot fit any slice.
        let p = params(chain2(-2.0, 2.0), CuJointActionKind::PositionDelta);
        assert!(CuJointActionTask::<2, 1>::from_params(p).is_err());
    }

    #[test]
    fn non_finite_value_outside_the_slice_does_not_fault() {
        let mut p = params(chain2(-2.0, 2.0), CuJointActionKind::PositionDelta);
        p.offset = 1;
        let mut task = CuJointActionTask::<2, 4>::from_params(p).unwrap();
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let out = run_values(&mut task, [bad, 0.5, 0.5, bad], state(0.0));
            assert_eq!(out.status, CuFollowStatus::Running, "{bad}");
            assert_si(&out, 0.5);
            // Inside the slice it still faults.
            let out = run_values(&mut task, [0.0, 0.5, bad, 0.0], state(0.0));
            assert_eq!(out.status, CuFollowStatus::ErrorEmptyTrajectory, "{bad}");
            assert_si(&out, 0.0);
        }
    }

    #[test]
    fn action_task_drives_the_controller_to_the_goal() {
        // position_absolute: default 0 + scale 1 * action, so a constant
        // synthetic action of 0.4 commands 0.4 rad every cycle.
        let mut p = params(
            CuKinematicChain::new([rev_spec(-2.0, 2.0)]),
            CuJointActionKind::PositionAbsolute,
        );
        p.default_positions = [rev(0.0)];
        let mut action_task = CuJointActionTask::<1, 1>::from_params(p).unwrap();
        let mut controller =
            CuControllerTask::<1>::from_params(repose_controller::CuControllerParams::default())
                .unwrap();
        let (ctx, _clock) = CuContext::new_mock_clock();

        let mut q = 0.0f64;
        let mut last_status = CuControllerStatus::Idle;
        for _ in 0..200 {
            let st = CuJointStatePayload::<1> {
                status: CuRobotStatus::Ready,
                positions: [rev(q)],
                velocities: [KinVelocity::default(); 1],
            };
            let state_msg = CuMsg::new(Some(st));
            let action_msg = CuMsg::new(Some(action::<1>(0.4, CuPolicyStatus::Fresh)));
            let mut follow_out = CuMsg::<CuTrajectoryFollowOutput<1>>::default();
            action_task
                .process(&ctx, &(&action_msg, &state_msg), &mut follow_out)
                .unwrap();
            let mut cmd_out = CuMsg::<CuJointCommandPayload<1>>::default();
            controller
                .process(&ctx, &(&follow_out, &state_msg), &mut cmd_out)
                .unwrap();
            let cmd = cmd_out.payload().unwrap();
            last_status = cmd.status;
            assert_ne!(last_status, CuControllerStatus::Fault);
            // Simulated first-order joint chasing the commanded position.
            q += 0.3 * (cmd.positions[0].si_value() - q);
        }
        assert!((q - 0.4).abs() < 1e-6, "q = {q}");
        assert_ne!(last_status, CuControllerStatus::Fault);
    }

    #[test]
    fn last_report_records_step_and_joint_clamping_and_resets_on_holds() {
        let mut p = params(chain2(-0.5, 0.5), CuJointActionKind::PositionDelta);
        p.max_step = Some(0.2);
        let mut task = CuJointActionTask::<2, 2>::from_params(p).unwrap();
        assert_eq!(*task.last_report(), CuJointActionReport::default());

        // Joint 0 requests +0.1 (neither step nor limit clamped).
        // Joint 1 requests +0.35 from 0.30 -> delta 0.35 > max_step 0.2 (step clamped to 0.50),
        // then 0.50 is inside [-0.5, 0.5] so not joint-limited.
        let mut st = state::<2>(0.0);
        st.positions = [rev(0.0), rev(0.30)];
        let out = run_values(&mut task, [0.1, 0.35], st.clone());
        assert_eq!(out.status, CuFollowStatus::Running);
        let report = *task.last_report();
        assert_eq!(report.step_limited_mask, 0b10);
        assert_eq!(report.step_limited_count, 1);
        assert_eq!(report.joint_limited_mask, 0);
        assert_eq!(report.joint_limited_count, 0);
        assert!(report.is_step_limited(1));
        assert!(!report.is_step_limited(0));
        assert!(!report.is_joint_limited(1));
        assert!(report.any_limited());
        assert!((report.max_requested_delta_si - 0.35).abs() < 1e-7);

        // Held delta keeps previous target and resets last_report.
        let out = run(
            &mut task,
            Some(action(0.35, CuPolicyStatus::Held)),
            Some(st),
        );
        assert_eq!(out.status, CuFollowStatus::Running);
        assert_eq!(*task.last_report(), CuJointActionReport::default());
        assert!(!task.last_report().any_limited());

        // Joint 0 at 0.45 requests +0.15 -> delta 0.15 <= max_step 0.2 (not step-limited),
        // target 0.60 clamped to 0.50 (joint-limited).
        // Joint 1 at 0.45 requests +0.30 -> delta 0.30 > max_step 0.2 (step-limited to 0.65),
        // then 0.65 clamped to 0.50 (also joint-limited!).
        let mut st2 = state::<2>(0.45);
        st2.positions = [rev(0.45), rev(0.45)];
        // First clear previous target via Idle so reference is measured st2 (0.45).
        run(&mut task, None, Some(st2.clone()));
        assert_eq!(*task.last_report(), CuJointActionReport::default());

        let out = run_values(&mut task, [0.15, 0.30], st2.clone());
        assert_eq!(out.status, CuFollowStatus::Running);
        let report = *task.last_report();
        assert_eq!(report.step_limited_mask, 0b10);
        assert_eq!(report.step_limited_count, 1);
        assert_eq!(report.joint_limited_mask, 0b11);
        assert_eq!(report.joint_limited_count, 2);
        assert!(!report.is_step_limited(0));
        assert!(report.is_step_limited(1));
        assert!(report.is_joint_limited(0));
        assert!(report.is_joint_limited(1));
        assert!(report.any_limited());
        assert!((report.max_requested_delta_si - 0.30).abs() < 1e-7);

        // Fault action resets last_report.
        let out = run(
            &mut task,
            Some(action(0.1, CuPolicyStatus::Fault)),
            Some(st2),
        );
        assert_eq!(out.status, CuFollowStatus::ErrorEmptyTrajectory);
        assert_eq!(*task.last_report(), CuJointActionReport::default());

        // CuJointActionReport round-trips serde and bincode.
        let text = ron::to_string(&report).unwrap();
        let back: CuJointActionReport = ron::from_str(&text).unwrap();
        assert_eq!(back, report);
        let mut bytes = [0u8; 256];
        let len =
            bincode::encode_into_slice(report, &mut bytes, bincode::config::standard()).unwrap();
        let (decoded, _): (CuJointActionReport, usize) =
            bincode::decode_from_slice(&bytes[..len], bincode::config::standard()).unwrap();
        assert_eq!(decoded, report);
    }
}
