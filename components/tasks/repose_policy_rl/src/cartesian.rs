//! [`CuCartesianActionTask`]: an end-effector pose delta in, a limited joint
//! setpoint out, through singularity- and limit-aware differential IK.

use crate::payload::CuPolicyAction;
use crate::payload::CuPolicyStatus;
use bincode::Decode;
use bincode::Encode;
use bincode::de::Decoder;
use bincode::enc::Encoder;
use bincode::error::DecodeError;
use bincode::error::EncodeError;
use cu29::prelude::*;
use cu29::units::si::angle::radian;
use cu29::units::si::angular_velocity::radian_per_second;
use cu29::units::si::f64::Angle;
use cu29::units::si::f64::AngularVelocity;
use cu29::units::si::f64::Length;
use cu29::units::si::f64::Time;
use cu29::units::si::f64::Velocity;
use cu29::units::si::length::meter;
use cu29::units::si::time::second;
use cu29::units::si::velocity::meter_per_second;
use repose_kinematics::CuChainSpec;
use repose_kinematics::CuDiffIkParams;
use repose_kinematics::CuDiffIkStatus;
use repose_kinematics::CuJointAcceleration;
use repose_kinematics::CuJointPosition;
use repose_kinematics::CuJointStatePayload;
use repose_kinematics::CuJointVelocity;
use repose_kinematics::CuKinematicChain;
use repose_kinematics::CuRobotStatus;
use repose_kinematics::CuTwist6D;
use repose_kinematics::CuTwistFrame;
use repose_trajectory::CuFollowStatus;
use repose_trajectory::CuTrajectoryFollowOutput;
use serde::Deserialize;
use serde::Serialize;

/// Number of action elements a Cartesian action reads: `[dx, dy, dz, r0, r1, r2]`.
pub const CARTESIAN_ACTION_WIDTH: usize = 6;

/// How the three rotation elements of a Cartesian action encode the
/// rotation delta (after `rotation_scale`, in radians).
#[derive(
    Default, Debug, Clone, Copy, PartialEq, Eq, Encode, Decode, Serialize, Deserialize, Reflect,
)]
#[serde(rename_all = "snake_case")]
pub enum CuRotationEncoding {
    /// Rotation vector (axis times angle).
    #[default]
    RotationVector,
    /// Roll, pitch, yaw: rotations about the fixed `x`, `y`, `z` axes of the
    /// delta frame, in that order (`R = Rz(yaw) * Ry(pitch) * Rx(roll)`),
    /// the convention of most VLA datasets (Bridge, Open X-Embodiment).
    Rpy,
}

fn default_length_scale() -> Length {
    Length::new::<meter>(1.0)
}

fn default_angle_scale() -> Angle {
    Angle::new::<radian>(1.0)
}

/// Parameters of [`CuCartesianActionTask`]; the node `config:` block.
///
/// ```ron
/// config: {
///     "chain": "igus_rebel",
///     "period": 0.02,                 // s, required: the control cycle period
///     "offset": 0,                    // optional; first action element this task reads
///     "frame": "base",                // base | body: frame the delta is expressed in
///     "rotation": "rpy",              // rotation_vector | rpy
///     "translation_scale": 0.01,      // m per unit of action
///     "rotation_scale": 0.05,         // rad per unit of action
///     "max_translation": 0.005,       // optional; m per cycle
///     "max_rotation": 0.05,           // optional; rad per cycle
///     "diff_ik": {"base_damping": 0.02, "limit_margin": 0.15}, // optional
/// }
/// ```
///
/// The task reads action elements `[offset, offset + 6)`: a translation
/// `[dx, dy, dz]` and a rotation `[r0, r1, r2]` in `rotation` encoding.
/// `offset + 6 <= ACT` is checked by [`CuCartesianActionTask::from_params`].
/// A gripper element of the same action is read by a separate
/// [`crate::CuGripperActionTask`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Reflect)]
#[reflect(no_field_bounds, from_reflect = false)]
// Field-level defaults: serde does not support container `default` on const-generic structs.
#[serde(deny_unknown_fields)]
pub struct CuCartesianActionParams<const N: usize> {
    /// Required robot chain: a preset name (`"igus_rebel"`) or an inline
    /// chain, including its tool offset. Joint limits and joint velocity
    /// limits come from the chain.
    #[serde(default)]
    #[reflect(ignore)]
    pub chain: Option<CuChainSpec<N>>,
    /// Required control cycle period (`1 / rate_target_hz`). The pose delta
    /// is executed over one period: it becomes a twist of `delta / period`.
    #[serde(default)]
    pub period: Option<Time>,
    /// Index of the first action element this task reads; defaults to `0`.
    #[serde(default)]
    pub offset: usize,
    /// Frame the delta is expressed in: `base` (the robot base frame, the
    /// usual VLA convention) or `body` (the tool frame). Defaults to `base`.
    #[serde(default)]
    pub frame: CuTwistFrame,
    /// Rotation encoding; defaults to `rotation_vector`.
    #[serde(default)]
    pub rotation: CuRotationEncoding,
    /// Translation per unit of action; defaults to 1 m.
    #[serde(default = "default_length_scale")]
    pub translation_scale: Length,
    /// Rotation per unit of action; defaults to 1 rad.
    #[serde(default = "default_angle_scale")]
    pub rotation_scale: Angle,
    /// Largest tool translation per cycle. A longer translation is shortened
    /// along its direction. `None` disables the limit.
    #[serde(default)]
    pub max_translation: Option<Length>,
    /// Largest tool rotation angle per cycle. A larger rotation is reduced
    /// about the same axis. `None` disables the limit.
    #[serde(default)]
    pub max_rotation: Option<Angle>,
    /// Differential IK: damping, singularity damping, joint-limit margins,
    /// and joint velocity limiting.
    #[serde(default)]
    #[reflect(ignore)]
    pub diff_ik: CuDiffIkParams,
}

impl<const N: usize> Default for CuCartesianActionParams<N> {
    /// No chain and no period (both rejected by [`Self::validate`]), offset
    /// `0`, `base` frame, rotation vector, unit scales, no Cartesian limits,
    /// default differential IK.
    fn default() -> Self {
        Self {
            chain: None,
            period: None,
            offset: 0,
            frame: CuTwistFrame::Base,
            rotation: CuRotationEncoding::default(),
            translation_scale: default_length_scale(),
            rotation_scale: default_angle_scale(),
            max_translation: None,
            max_rotation: None,
            diff_ik: CuDiffIkParams::default(),
        }
    }
}

impl<const N: usize> CuCartesianActionParams<N> {
    /// Resolves the configured chain and validates it.
    pub fn resolve_chain(&self) -> CuResult<CuKinematicChain<N>> {
        let spec = self.chain.as_ref().ok_or_else(|| {
            CuError::from("repose-policy-rl: missing required 'chain' configuration")
        })?;
        let chain = spec.resolve()?;
        chain.validate()?;
        Ok(chain)
    }

    /// Checks the chain, period, scales, Cartesian limits, and differential
    /// IK parameters.
    pub fn validate(&self) -> CuResult<()> {
        if N == 0 {
            return Err(CuError::from(
                "repose-policy-rl: joint count N must be at least 1",
            ));
        }
        self.resolve_chain()?;
        let period = self.period.ok_or_else(|| {
            CuError::from("repose-policy-rl: missing required 'period' configuration")
        })?;
        let positive = |v: f64| v.is_finite() && v > 0.0;
        if !positive(period.get::<second>()) {
            return Err(CuError::from(
                "repose-policy-rl: 'period' must be positive and finite",
            ));
        }
        if !self.translation_scale.get::<meter>().is_finite()
            || !self.rotation_scale.get::<radian>().is_finite()
        {
            return Err(CuError::from(
                "repose-policy-rl: 'translation_scale' and 'rotation_scale' must be finite",
            ));
        }
        if let Some(max) = self.max_translation
            && !positive(max.get::<meter>())
        {
            return Err(CuError::from(
                "repose-policy-rl: 'max_translation' must be positive and finite",
            ));
        }
        if let Some(max) = self.max_rotation
            && !positive(max.get::<radian>())
        {
            return Err(CuError::from(
                "repose-policy-rl: 'max_rotation' must be positive and finite",
            ));
        }
        self.diff_ik.validate()
    }
}

/// What [`CuCartesianActionTask`] did with the action this cycle, published
/// next to the setpoint so a log shows how much of the model's request was
/// executed.
#[derive(Default, Debug, Clone, PartialEq, Encode, Decode, Serialize, Deserialize, Reflect)]
pub struct CuCartesianActionReport {
    /// Status of this cycle's differential IK solve: `nominal`,
    /// `singularity_damped` (with the manipulability and the effective
    /// damping), or `velocity_scaled` (with the scale applied to respect
    /// joint velocity limits). `None` when no solve ran: a hold, or a held
    /// delta keeping its target.
    pub ik_status: Option<CuDiffIkStatus>,
    /// The translation was shortened to `max_translation`.
    pub translation_limited: bool,
    /// The rotation was reduced to `max_rotation`.
    pub rotation_limited: bool,
}

/// Turns an end-effector pose delta (the action of most vision-language-action
/// models) into a limited joint setpoint.
///
/// Inputs are `(action, state)`: a [`CuPolicyAction<ACT>`] and a
/// [`CuJointStatePayload<N>`]. Outputs are the [`CuTrajectoryFollowOutput<N>`]
/// that `repose_controller::CuControllerTask` consumes and a
/// [`CuCartesianActionReport`]; both carry the joint state's `Tov`.
///
/// Per cycle, for a usable (`Fresh` or `Held`) action and a `Ready` robot:
///
/// 1. Action elements `[offset, offset + 3)` times `translation_scale` are
///    the translation, `[offset + 3, offset + 6)` times `rotation_scale` the
///    rotation in `rotation` encoding, both in `frame`.
/// 2. The translation is shortened to `max_translation` and the rotation
///    reduced to `max_rotation`, each keeping its direction.
/// 3. The delta becomes the twist `delta / period`, and
///    [`CuKinematicChain::differential_ik`] solves it at the **measured**
///    joint positions with damped least squares: adaptive damping near
///    singularities, damping of motion toward a joint limit inside the limit
///    margin, and uniform scaling to the chain's joint velocity limits.
/// 4. `target = measured + joint_velocity * period`, clamped to the joint
///    limits, is emitted with [`CuFollowStatus::Running`] and a setpoint
///    velocity of `(target - measured) / period`.
///
/// The delta is applied to the measured pose, as VLA datasets record it, so
/// a target cannot run ahead of an arm that lags behind it, and the
/// linearization error of one small step does not accumulate. Near a
/// singularity or a joint limit the arm executes less than the model asked
/// for instead of producing large joint motion; the report says when.
///
/// A `Held` action (policy decimation) keeps the target its delta produced
/// with [`CuFollowStatus::Running`] and zero velocity; with no target yet it
/// is applied once.
///
/// Hold cases drop the target and emit the measured position with zero
/// velocity, as in [`crate::CuJointActionTask`]:
///
/// | Situation | Status emitted |
/// |---|---|
/// | No action yet, action status `Idle`, or robot not `Ready` | `Idle` |
/// | Action status `Fault`, a non-finite value in this task's action slice, a joint state that is not finite or whose units do not match the chain, or a failed IK solve | `ErrorEmptyTrajectory` |
///
/// With no joint state at all the previous target is kept and emitted (zeros
/// before the first state) with `Idle`. The previous target and whether one
/// exists round-trip through `Freezable`; nothing on the hot path allocates.
#[derive(Reflect)]
#[reflect(no_field_bounds, from_reflect = false)]
pub struct CuCartesianActionTask<const N: usize, const ACT: usize> {
    #[reflect(ignore)]
    params: CuCartesianActionParams<N>,
    #[reflect(ignore)]
    chain: CuKinematicChain<N>,
    period_s: f64,
    #[reflect(ignore)]
    previous_target: [CuJointPosition; N],
    has_target: bool,
}

impl<const N: usize, const ACT: usize> CuCartesianActionTask<N, ACT> {
    /// Creates the task from validated params. Also rejects an `offset`
    /// whose slice `[offset, offset + 6)` does not fit in the `ACT`-wide
    /// action.
    pub fn from_params(params: CuCartesianActionParams<N>) -> CuResult<Self> {
        params.validate()?;
        match params.offset.checked_add(CARTESIAN_ACTION_WIDTH) {
            Some(end) if end <= ACT => {}
            _ => {
                return Err(CuError::from(
                    "repose-policy-rl: 'offset' + 6 exceeds the policy action width ACT",
                ));
            }
        }
        let chain = params.resolve_chain()?;
        let period_s = params.period.map_or(0.0, |p| p.get::<second>());
        Ok(Self {
            params,
            chain,
            period_s,
            previous_target: [CuJointPosition::default(); N],
            has_target: false,
        })
    }

    /// Params this task runs with.
    pub fn params(&self) -> &CuCartesianActionParams<N> {
        &self.params
    }

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

    /// Positions emitted when there is no joint state at all.
    fn held_positions(&self) -> [CuJointPosition; N] {
        if self.has_target {
            self.previous_target
        } else {
            [CuJointPosition::default(); N]
        }
    }

    /// The scaled and limited delta of one action slice, as a twist over
    /// one period. Returns `None` for a non-finite result.
    fn twist(&self, values: &[f32]) -> Option<(CuTwist6D, bool, bool)> {
        let t_scale = self.params.translation_scale.get::<meter>();
        let r_scale = self.params.rotation_scale.get::<radian>();
        let mut translation = [0.0; 3];
        let mut rotation = [0.0; 3];
        for i in 0..3 {
            translation[i] = t_scale * f64::from(*values.get(i)?);
            rotation[i] = r_scale * f64::from(*values.get(i + 3)?);
        }
        let mut rotation_vector = match self.params.rotation {
            CuRotationEncoding::RotationVector => rotation,
            CuRotationEncoding::Rpy => {
                rotation_vector_of(&rpy_matrix(rotation[0], rotation[1], rotation[2]))
            }
        };
        let translation_limited = self
            .params
            .max_translation
            .is_some_and(|max| limit_norm(&mut translation, max.get::<meter>()));
        let rotation_limited = self
            .params
            .max_rotation
            .is_some_and(|max| limit_norm(&mut rotation_vector, max.get::<radian>()));
        let p = self.period_s;
        let linear = translation.map(|v| v / p);
        let angular = rotation_vector.map(|v| v / p);
        if !linear.iter().chain(angular.iter()).all(|v| v.is_finite()) {
            return None;
        }
        let twist = CuTwist6D::new(
            linear.map(Velocity::new::<meter_per_second>),
            angular.map(AngularVelocity::new::<radian_per_second>),
            self.params.frame,
        );
        Some((twist, translation_limited, rotation_limited))
    }
}

/// Shortens `v` to `max` along its direction; returns whether it did.
fn limit_norm(v: &mut [f64; 3], max: f64) -> bool {
    let norm = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if norm > max {
        let s = max / norm;
        for x in v.iter_mut() {
            *x *= s;
        }
        true
    } else {
        false
    }
}

/// `Rz(yaw) * Ry(pitch) * Rx(roll)`.
fn rpy_matrix(roll: f64, pitch: f64, yaw: f64) -> [[f64; 3]; 3] {
    let (sr, cr) = roll.sin_cos();
    let (sp, cp) = pitch.sin_cos();
    let (sy, cy) = yaw.sin_cos();
    [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]
}

/// Rotation vector (matrix logarithm) of a rotation matrix.
fn rotation_vector_of(r: &[[f64; 3]; 3]) -> [f64; 3] {
    let w = [r[2][1] - r[1][2], r[0][2] - r[2][0], r[1][0] - r[0][1]];
    let cos = ((r[0][0] + r[1][1] + r[2][2] - 1.0) * 0.5).clamp(-1.0, 1.0);
    let angle = cos.acos();
    if angle < 1e-9 {
        return w.map(|x| 0.5 * x);
    }
    if core::f64::consts::PI - angle > 1e-6 {
        let k = angle / (2.0 * angle.sin());
        return w.map(|x| k * x);
    }
    // Near a half turn `w` vanishes: take the axis from the symmetric part,
    // using the largest diagonal element for conditioning.
    let mut i = 0;
    for d in 1..3 {
        if r[d][d] > r[i][i] {
            i = d;
        }
    }
    let a_i = ((r[i][i] + 1.0) * 0.5).max(0.0).sqrt();
    let mut axis = [0.0; 3];
    for (j, a) in axis.iter_mut().enumerate() {
        *a = if j == i {
            a_i
        } else {
            (r[i][j] + r[j][i]) / (4.0 * a_i)
        };
    }
    let norm = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
    axis.map(|a| a / norm * angle)
}

impl<const N: usize, const ACT: usize> Freezable for CuCartesianActionTask<N, ACT> {
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

impl<const N: usize, const ACT: usize> CuTask for CuCartesianActionTask<N, ACT> {
    type Resources<'r> = ();
    type Input<'m> = input_msg!('m, CuPolicyAction<ACT>, CuJointStatePayload<N>);
    type Output<'m> = output_msg!(CuTrajectoryFollowOutput<N>, CuCartesianActionReport);

    fn new(cfg: Option<&ComponentConfig>, _res: Self::Resources<'_>) -> CuResult<Self>
    where
        Self: Sized,
    {
        let params = cfg
            .map(ComponentConfig::deserialize_into::<CuCartesianActionParams<N>>)
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
        let (setpoint_out, report_out) = output;
        setpoint_out.tov = input.1.tov;
        report_out.tov = input.1.tov;
        let action = input.0.payload();

        // No state at all: keep the previous target, whatever the action.
        let Some(state) = input.1.payload() else {
            let held = self.held_positions();
            setpoint_out.set_payload(Self::setpoint(held, CuFollowStatus::Idle));
            report_out.set_payload(CuCartesianActionReport::default());
            return Ok(());
        };

        // Every hold drops the target, so the next active cycle starts from
        // where the arm is (see `CuJointActionTask`).
        let mut emit = |task: &mut Self,
                        positions: [CuJointPosition; N],
                        status: CuFollowStatus,
                        report: CuCartesianActionReport| {
            if status != CuFollowStatus::Running {
                task.has_target = false;
            }
            setpoint_out.set_payload(Self::setpoint(positions, status));
            report_out.set_payload(report);
        };
        let measured_hold = state.positions;
        if state.status != CuRobotStatus::Ready {
            emit(
                self,
                measured_hold,
                CuFollowStatus::Idle,
                CuCartesianActionReport::default(),
            );
            return Ok(());
        }
        let mut measured = state.positions;
        if self.chain.clamp_to_limits(&mut measured).is_err()
            || !measured.iter().all(|q| q.si_value().is_finite())
        {
            emit(
                self,
                measured_hold,
                CuFollowStatus::ErrorEmptyTrajectory,
                CuCartesianActionReport::default(),
            );
            return Ok(());
        }
        let Some(action) = action.filter(|a| a.status != CuPolicyStatus::Idle) else {
            emit(
                self,
                measured_hold,
                CuFollowStatus::Idle,
                CuCartesianActionReport::default(),
            );
            return Ok(());
        };
        let offset = self.params.offset;
        let Some(values) = action
            .values
            .get(offset..offset.saturating_add(CARTESIAN_ACTION_WIDTH))
        else {
            emit(
                self,
                measured_hold,
                CuFollowStatus::ErrorEmptyTrajectory,
                CuCartesianActionReport::default(),
            );
            return Ok(());
        };
        if action.status == CuPolicyStatus::Fault || !values.iter().all(|v| v.is_finite()) {
            emit(
                self,
                measured_hold,
                CuFollowStatus::ErrorEmptyTrajectory,
                CuCartesianActionReport::default(),
            );
            return Ok(());
        }

        // A held delta was already applied on the cycle it was inferred.
        if action.status == CuPolicyStatus::Held && self.has_target {
            emit(
                self,
                self.previous_target,
                CuFollowStatus::Running,
                CuCartesianActionReport::default(),
            );
            return Ok(());
        }

        let Some((twist, translation_limited, rotation_limited)) = self.twist(values) else {
            emit(
                self,
                measured_hold,
                CuFollowStatus::ErrorEmptyTrajectory,
                CuCartesianActionReport::default(),
            );
            return Ok(());
        };
        let mut joint_velocities = [CuJointVelocity::default(); N];
        let Ok(ik_status) = self.chain.differential_ik(
            &measured,
            &twist,
            &self.params.diff_ik,
            &mut joint_velocities,
        ) else {
            emit(
                self,
                measured_hold,
                CuFollowStatus::ErrorEmptyTrajectory,
                CuCartesianActionReport::default(),
            );
            return Ok(());
        };

        let period = self.period_s;
        let mut target = measured;
        for j in 0..N {
            let q = measured[j].si_value() + joint_velocities[j].si_value() * period;
            target[j] = measured[j].with_si_value(q);
        }
        let mut setpoint = Self::setpoint(target, CuFollowStatus::Running);
        let clamped = self.chain.clamp_to_limits(&mut target);
        let mut finite = clamped.is_ok();
        for j in 0..N {
            let velocity = (target[j].si_value() - measured[j].si_value()) / period;
            finite &= velocity.is_finite() && target[j].si_value().is_finite();
            setpoint.velocities[j] = CuJointVelocity::from_coordinate(target[j], velocity);
        }
        if !finite {
            emit(
                self,
                measured_hold,
                CuFollowStatus::ErrorEmptyTrajectory,
                CuCartesianActionReport::default(),
            );
            return Ok(());
        }
        setpoint.positions = target;
        self.previous_target = target;
        self.has_target = true;
        setpoint_out.set_payload(setpoint);
        report_out.set_payload(CuCartesianActionReport {
            ik_status: Some(ik_status),
            translation_limited,
            rotation_limited,
        });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cu_spatial_payloads::Transform3D;
    use repose_kinematics::CuJointVelocity as KinVelocity;
    use repose_kinematics::igus_rebel_chain;

    const PERIOD: f64 = 0.02;
    /// A well-conditioned igus ReBeL configuration (wrist not straight).
    const NOMINAL: [f64; 6] = [0.3, 0.5, 1.0, 0.2, 0.8, 0.1];

    fn rev(value: f64) -> CuJointPosition {
        CuJointPosition::Revolute(Angle::new::<radian>(value))
    }

    fn params() -> CuCartesianActionParams<6> {
        CuCartesianActionParams {
            chain: Some(igus_rebel_chain().into()),
            period: Some(Time::new::<second>(PERIOD)),
            translation_scale: Length::new::<meter>(0.01),
            rotation_scale: Angle::new::<radian>(0.01),
            ..CuCartesianActionParams::default()
        }
    }

    fn state(q: [f64; 6]) -> CuJointStatePayload<6> {
        CuJointStatePayload {
            status: CuRobotStatus::Ready,
            positions: q.map(rev),
            velocities: [KinVelocity::default(); 6],
        }
    }

    fn action<const ACT: usize>(values: [f32; ACT], status: CuPolicyStatus) -> CuPolicyAction<ACT> {
        CuPolicyAction { values, status }
    }

    fn run<const ACT: usize>(
        task: &mut CuCartesianActionTask<6, ACT>,
        act: Option<CuPolicyAction<ACT>>,
        st: Option<CuJointStatePayload<6>>,
    ) -> (CuTrajectoryFollowOutput<6>, CuCartesianActionReport) {
        let (ctx, _clock) = CuContext::new_mock_clock();
        let action_msg = CuMsg::new(act);
        let state_msg = CuMsg::new(st);
        let mut output = (
            CuMsg::<CuTrajectoryFollowOutput<6>>::default(),
            CuMsg::<CuCartesianActionReport>::default(),
        );
        task.process(&ctx, &(&action_msg, &state_msg), &mut output)
            .unwrap();
        (
            output.0.payload().unwrap().clone(),
            output.1.payload().unwrap().clone(),
        )
    }

    fn pose(q: &[CuJointPosition; 6]) -> [[f64; 4]; 4] {
        let mut ee = Transform3D::<f64>::identity();
        igus_rebel_chain().forward_kinematics(q, &mut ee).unwrap();
        ee.to_matrix()
    }

    fn translation(m: &[[f64; 4]; 4]) -> [f64; 3] {
        [m[0][3], m[1][3], m[2][3]]
    }

    fn rotation(m: &[[f64; 4]; 4]) -> [[f64; 3]; 3] {
        core::array::from_fn(|i| core::array::from_fn(|j| m[i][j]))
    }

    fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
        [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
    }

    fn norm(v: [f64; 3]) -> f64 {
        (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
    }

    /// Rotation vector of `R_b * R_a^T` (the rotation from `a` to `b`, in
    /// the base frame).
    fn base_rotation(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [f64; 3] {
        let m: [[f64; 3]; 3] = core::array::from_fn(|i| {
            core::array::from_fn(|j| (0..3).map(|k| b[i][k] * a[j][k]).sum())
        });
        rotation_vector_of(&m)
    }

    fn assert_close(got: [f64; 3], want: [f64; 3], tol: f64) {
        assert!(
            norm(sub(got, want)) <= tol,
            "{got:?} != {want:?} (tol {tol})"
        );
    }

    #[test]
    fn base_translation_moves_the_tool_along_the_requested_axis() {
        let mut task = CuCartesianActionTask::<6, 6>::from_params(params()).unwrap();
        let start = state(NOMINAL);
        // 0.5 * 0.01 m = 5 mm along base +x.
        let (out, report) = run(
            &mut task,
            Some(action(
                [0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
                CuPolicyStatus::Fresh,
            )),
            Some(start.clone()),
        );
        assert_eq!(out.status, CuFollowStatus::Running);
        assert!(matches!(
            report.ik_status,
            Some(CuDiffIkStatus::Nominal { .. })
        ));
        let before = pose(&start.positions);
        let after = pose(&out.positions);
        // Damped least squares executes slightly less than requested.
        assert_close(
            sub(translation(&after), translation(&before)),
            [0.005, 0.0, 0.0],
            0.0005,
        );
        assert!(norm(base_rotation(&rotation(&before), &rotation(&after))) < 0.005);
        // The setpoint velocity executes the change in one period.
        for ((position, velocity), start) in out.positions.iter().zip(&out.velocities).zip(NOMINAL)
        {
            let change = position.si_value() - start;
            assert!((velocity.si_value() * PERIOD - change).abs() < 1e-12);
        }
    }

    #[test]
    fn rpy_and_rotation_vector_agree_on_a_single_axis_rotation() {
        for rotation_kind in [CuRotationEncoding::Rpy, CuRotationEncoding::RotationVector] {
            let mut p = params();
            p.rotation = rotation_kind;
            let mut task = CuCartesianActionTask::<6, 6>::from_params(p).unwrap();
            let start = state(NOMINAL);
            // 1.0 * 0.01 rad of yaw about base z.
            let (out, _) = run(
                &mut task,
                Some(action(
                    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                    CuPolicyStatus::Fresh,
                )),
                Some(start.clone()),
            );
            let before = pose(&start.positions);
            let after = pose(&out.positions);
            assert_close(
                base_rotation(&rotation(&before), &rotation(&after)),
                [0.0, 0.0, 0.01],
                0.001,
            );
            assert!(norm(sub(translation(&after), translation(&before))) < 0.0005);
        }
    }

    #[test]
    fn body_frame_translation_follows_the_tool_axis() {
        let mut p = params();
        p.frame = CuTwistFrame::Body;
        let mut task = CuCartesianActionTask::<6, 6>::from_params(p).unwrap();
        let start = state(NOMINAL);
        // 5 mm along the tool z axis (the approach direction).
        let (out, _) = run(
            &mut task,
            Some(action(
                [0.0, 0.0, 0.5, 0.0, 0.0, 0.0],
                CuPolicyStatus::Fresh,
            )),
            Some(start.clone()),
        );
        let before = pose(&start.positions);
        let after = pose(&out.positions);
        let tool_z = [before[0][2], before[1][2], before[2][2]];
        assert_close(
            sub(translation(&after), translation(&before)),
            tool_z.map(|v| v * 0.005),
            0.0005,
        );
    }

    #[test]
    fn closed_loop_tracking_reaches_the_accumulated_displacement() {
        let mut task = CuCartesianActionTask::<6, 6>::from_params(params()).unwrap();
        let mut q = NOMINAL;
        let start = translation(&pose(&state(q).positions));
        // 40 cycles of 2 mm down: 80 mm, with the arm tracking perfectly.
        for _ in 0..40 {
            let (out, _) = run(
                &mut task,
                Some(action(
                    [0.0, 0.0, -0.2, 0.0, 0.0, 0.0],
                    CuPolicyStatus::Fresh,
                )),
                Some(state(q)),
            );
            assert_eq!(out.status, CuFollowStatus::Running);
            q = out.positions.map(|p| p.si_value());
        }
        let end = translation(&pose(&state(q).positions));
        // DLS shortfall per step does not accumulate into drift sideways.
        let moved = sub(end, start);
        assert!((moved[2] + 0.08).abs() < 0.008, "{moved:?}");
        assert!(
            moved[0].abs() < 0.003 && moved[1].abs() < 0.003,
            "{moved:?}"
        );
    }

    #[test]
    fn cartesian_limits_shorten_the_delta_and_are_reported() {
        let mut p = params();
        p.max_translation = Some(Length::new::<meter>(0.002));
        p.max_rotation = Some(Angle::new::<radian>(0.005));
        let mut task = CuCartesianActionTask::<6, 6>::from_params(p).unwrap();
        let start = state(NOMINAL);
        // 3 * 0.01 = 30 mm requested, 2 mm allowed; 0.03 rad requested, 0.005 allowed.
        let (out, report) = run(
            &mut task,
            Some(action(
                [3.0, 0.0, 0.0, 3.0, 0.0, 0.0],
                CuPolicyStatus::Fresh,
            )),
            Some(start.clone()),
        );
        assert!(report.translation_limited && report.rotation_limited);
        let before = pose(&start.positions);
        let after = pose(&out.positions);
        let moved = norm(sub(translation(&after), translation(&before)));
        assert!(moved <= 0.002 + 1e-4 && moved > 0.0015, "{moved}");
        let turned = norm(base_rotation(&rotation(&before), &rotation(&after)));
        assert!(turned <= 0.005 + 5e-4, "{turned}");
    }

    #[test]
    fn joint_velocity_limits_bound_a_huge_request() {
        let mut p = params();
        p.translation_scale = Length::new::<meter>(1.0);
        let mut task = CuCartesianActionTask::<6, 6>::from_params(p).unwrap();
        // 1 m in one 20 ms cycle: 50 m/s, far beyond the joints.
        let (out, report) = run(
            &mut task,
            Some(action(
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                CuPolicyStatus::Fresh,
            )),
            Some(state(NOMINAL)),
        );
        assert!(matches!(
            report.ik_status,
            Some(CuDiffIkStatus::VelocityScaled { .. })
        ));
        let max_velocity = 2.16;
        for (j, ((position, velocity), start)) in out
            .positions
            .iter()
            .zip(&out.velocities)
            .zip(NOMINAL)
            .enumerate()
        {
            let step = (position.si_value() - start).abs();
            assert!(step <= max_velocity * PERIOD + 1e-9, "joint {j}: {step}");
            assert!(velocity.si_value().abs() <= max_velocity + 1e-9);
        }
    }

    #[test]
    fn a_singular_wrist_is_damped_not_amplified() {
        let mut task = CuCartesianActionTask::<6, 6>::from_params(params()).unwrap();
        // Joint 5 at zero aligns joints 4 and 6: a wrist singularity.
        let singular = [0.3, 0.5, 1.0, 0.2, 0.0, 0.1];
        let (out, report) = run(
            &mut task,
            Some(action(
                [0.0, 0.0, 0.0, 0.5, 0.5, 0.0],
                CuPolicyStatus::Fresh,
            )),
            Some(state(singular)),
        );
        assert_eq!(out.status, CuFollowStatus::Running);
        assert!(
            matches!(
                report.ik_status,
                Some(
                    CuDiffIkStatus::SingularityDamped { .. }
                        | CuDiffIkStatus::VelocityScaled { .. }
                )
            ),
            "{:?}",
            report.ik_status
        );
        for (j, (position, start)) in out.positions.iter().zip(singular).enumerate() {
            let step = (position.si_value() - start).abs();
            assert!(
                step.is_finite() && step <= 2.16 * PERIOD + 1e-9,
                "joint {j}: {step}"
            );
        }
    }

    #[test]
    fn targets_stay_inside_the_joint_limits() {
        let mut task = CuCartesianActionTask::<6, 6>::from_params(params()).unwrap();
        // Joint 5 sits on its upper limit; push the tool every way.
        let mut q = NOMINAL;
        q[4] = 1.65806;
        let chain = igus_rebel_chain();
        for values in [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, -1.0, 0.0],
            [0.0, -1.0, 1.0, 1.0, 0.0, 1.0],
        ] {
            let (out, _) = run(
                &mut task,
                Some(action(values, CuPolicyStatus::Fresh)),
                Some(state(q)),
            );
            assert!(chain.is_within_limits(&out.positions), "{values:?}");
        }
    }

    #[test]
    fn hold_cases_drop_the_target_and_hold_the_measured_position() {
        let mut task = CuCartesianActionTask::<6, 6>::from_params(params()).unwrap();
        let go = action([0.5, 0.0, 0.0, 0.0, 0.0, 0.0], CuPolicyStatus::Fresh);
        let (moved, _) = run(&mut task, Some(go.clone()), Some(state(NOMINAL)));

        // No joint state: the previous target is kept, Idle.
        let (out, report) = run(&mut task, Some(go.clone()), None);
        assert_eq!(out.status, CuFollowStatus::Idle);
        assert_eq!(out.positions, moved.positions);
        assert_eq!(report.ik_status, None);

        let measured = state(NOMINAL).positions;
        let mut not_ready = state(NOMINAL);
        not_ready.status = CuRobotStatus::EmergencyStop;
        let mut nan_values = go.clone();
        nan_values.values[4] = f32::NAN;
        let mut bad_state = state(NOMINAL);
        bad_state.positions[2] = CuJointPosition::Prismatic(Length::new::<meter>(0.0));
        let cases = [
            (None, state(NOMINAL), CuFollowStatus::Idle),
            (
                Some(action([0.5; 6], CuPolicyStatus::Idle)),
                state(NOMINAL),
                CuFollowStatus::Idle,
            ),
            (Some(go.clone()), not_ready, CuFollowStatus::Idle),
            (
                Some(action([0.5; 6], CuPolicyStatus::Fault)),
                state(NOMINAL),
                CuFollowStatus::ErrorEmptyTrajectory,
            ),
            (
                Some(nan_values),
                state(NOMINAL),
                CuFollowStatus::ErrorEmptyTrajectory,
            ),
        ];
        for (act, st, status) in cases {
            let (out, report) = run(&mut task, act.clone(), Some(st));
            assert_eq!(out.status, status, "{act:?}");
            assert_eq!(out.positions, measured, "{act:?}");
            assert!(out.velocities.iter().all(|v| v.si_value() == 0.0));
            assert_eq!(report.ik_status, None);
        }
        let (out, _) = run(&mut task, Some(go), Some(bad_state));
        assert_eq!(out.status, CuFollowStatus::ErrorEmptyTrajectory);
    }

    #[test]
    fn a_held_delta_keeps_its_target() {
        let mut task = CuCartesianActionTask::<6, 6>::from_params(params()).unwrap();
        let values = [0.5, 0.0, 0.0, 0.0, 0.0, 0.0];
        let (first, _) = run(
            &mut task,
            Some(action(values, CuPolicyStatus::Fresh)),
            Some(state(NOMINAL)),
        );
        let (held, report) = run(
            &mut task,
            Some(action(values, CuPolicyStatus::Held)),
            Some(state(NOMINAL)),
        );
        assert_eq!(held.status, CuFollowStatus::Running);
        assert_eq!(held.positions, first.positions);
        assert!(held.velocities.iter().all(|v| v.si_value() == 0.0));
        assert_eq!(report.ik_status, None);
    }

    #[test]
    fn offset_selects_the_pose_slice_and_is_bounded() {
        let mut p = params();
        p.offset = 1;
        let mut task = CuCartesianActionTask::<6, 7>::from_params(p).unwrap();
        // Element 0 is not ours (NaN must not fault); element 1 is dx.
        let (out, _) = run(
            &mut task,
            Some(action(
                [f32::NAN, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
                CuPolicyStatus::Fresh,
            )),
            Some(state(NOMINAL)),
        );
        assert_eq!(out.status, CuFollowStatus::Running);
        assert!(CuCartesianActionTask::<6, 6>::from_params(p).is_err());
        let mut p = params();
        p.offset = usize::MAX;
        assert!(CuCartesianActionTask::<6, 64>::from_params(p).is_err());
    }

    #[test]
    fn freeze_thaw_resumes_identically() {
        let mut original = CuCartesianActionTask::<6, 6>::from_params(params()).unwrap();
        let values = [0.3, -0.2, 0.1, 0.0, 0.2, -0.1];
        run(
            &mut original,
            Some(action(values, CuPolicyStatus::Fresh)),
            Some(state(NOMINAL)),
        );
        let mut bytes = [0u8; 4096];
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
        let mut restored = CuCartesianActionTask::<6, 6>::from_params(params()).unwrap();
        restored.thaw(&mut decoder).unwrap();
        for status in [CuPolicyStatus::Held, CuPolicyStatus::Fresh] {
            let a = run(
                &mut original,
                Some(action(values, status)),
                Some(state(NOMINAL)),
            );
            let b = run(
                &mut restored,
                Some(action(values, status)),
                Some(state(NOMINAL)),
            );
            assert_eq!(a, b, "{status:?}");
        }
    }

    #[test]
    fn rotation_vector_of_handles_small_and_half_turn_rotations() {
        assert_close(
            rotation_vector_of(&rpy_matrix(0.0, 0.0, 0.0)),
            [0.0; 3],
            1e-12,
        );
        assert_close(
            rotation_vector_of(&rpy_matrix(0.0, 0.0, 1e-10)),
            [0.0, 0.0, 1e-10],
            1e-15,
        );
        assert_close(
            rotation_vector_of(&rpy_matrix(0.3, 0.0, 0.0)),
            [0.3, 0.0, 0.0],
            1e-12,
        );
        let half = rotation_vector_of(&rpy_matrix(0.0, core::f64::consts::PI, 0.0));
        assert!(
            (norm(half) - core::f64::consts::PI).abs() < 1e-9,
            "{half:?}"
        );
        assert!(half[0].abs() < 1e-9 && half[2].abs() < 1e-9, "{half:?}");
    }

    #[test]
    fn invalid_params_are_rejected() {
        let mut no_period = params();
        no_period.period = None;
        let mut no_chain = params();
        no_chain.chain = None;
        let mut bad_max = params();
        bad_max.max_translation = Some(Length::new::<meter>(0.0));
        let mut bad_rot = params();
        bad_rot.max_rotation = Some(Angle::new::<radian>(f64::NAN));
        let mut bad_scale = params();
        bad_scale.translation_scale = Length::new::<meter>(f64::INFINITY);
        let mut bad_ik = params();
        bad_ik.diff_ik.base_damping = 0.0;
        for p in [no_period, no_chain, bad_max, bad_rot, bad_scale, bad_ik] {
            assert!(
                CuCartesianActionTask::<6, 6>::from_params(p).is_err(),
                "{p:?}"
            );
        }
    }

    #[cfg(feature = "std")]
    #[test]
    fn params_read_from_ron_and_reject_typos() {
        let cfg: ComponentConfig = ron::from_str(
            r#"#![enable(unwrap_newtypes)] {"chain": "igus_rebel", "period": 0.02,
            "offset": 0, "frame": "body", "rotation": "rpy",
            "translation_scale": 0.01, "rotation_scale": 0.05,
            "max_translation": 0.005, "max_rotation": 0.05,
            "diff_ik": {"base_damping": 0.03}}"#,
        )
        .unwrap();
        let task = CuCartesianActionTask::<6, 7>::new(Some(&cfg), ()).unwrap();
        let p = task.params();
        assert_eq!(p.frame, CuTwistFrame::Body);
        assert_eq!(p.rotation, CuRotationEncoding::Rpy);
        assert_eq!(p.translation_scale.get::<meter>(), 0.01);
        assert_eq!(p.max_rotation.map(|a| a.get::<radian>()), Some(0.05));
        assert_eq!(p.diff_ik.base_damping, 0.03);

        for bad in [
            r#"{"chain": "igus_rebel"}"#,
            r#"{"period": 0.02}"#,
            r#"{"chain": "igus_rebel", "period": 0.02, "frame": "tool"}"#,
            r#"{"chain": "igus_rebel", "period": 0.02, "scale": 0.01}"#,
        ] {
            let cfg: ComponentConfig =
                ron::from_str(&format!("#![enable(unwrap_newtypes)] {bad}")).unwrap();
            assert!(
                CuCartesianActionTask::<6, 7>::new(Some(&cfg), ()).is_err(),
                "{bad}"
            );
        }
    }
}
