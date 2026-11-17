//! [`CuSetpointMuxTask`]: bumpless arbitration between a classical
//! trajectory follower and a learned policy, both emitting
//! [`CuTrajectoryFollowOutput`].
//!
//! The mux owns three sources ([`CuSetpointSource`]): the classical setpoint
//! (for example [`CuFollowTrajectoryTask`](crate::CuFollowTrajectoryTask)
//! running a collision-free transit plan), the policy setpoint (for example
//! `repose_policy_rl::CuJointActionTask` doing contact-rich manipulation),
//! and an internal hold used as a safe fallback. Every switch between them
//! is bumpless: the difference between the last emitted setpoint and the new
//! source (position, velocity, and acceleration) decays to zero along a
//! quintic over `blend_time`, so the emitted position, velocity, and
//! acceleration stay continuous across the switch.

use crate::CuFollowStatus;
use crate::CuTrajectoryFollowOutput;
use bincode::Decode;
use bincode::Encode;
use bincode::de::Decoder;
use bincode::enc::Encoder;
use bincode::error::DecodeError;
use bincode::error::EncodeError;
use cu29::prelude::*;
use cu29::units::si::f64::Time;
use cu29::units::si::time::second;
use repose_kinematics::CuJointAcceleration;
use repose_kinematics::CuJointPosition;
use repose_kinematics::CuJointVelocity;
use serde::Deserialize;
use serde::Serialize;

/// Default [`CuSetpointMuxParams::blend_time`]: 0.2 s.
pub const DEFAULT_MUX_BLEND_TIME_S: f64 = 0.2;

/// Which source a [`CuSetpointMuxTask`] currently follows.
#[derive(
    Default, Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Encode, Decode, Reflect,
)]
#[serde(rename_all = "snake_case")]
pub enum CuSetpointSource {
    /// The classical setpoint (first input). Initial source.
    #[default]
    Classical,
    /// The policy setpoint (second input).
    Policy,
    /// Safe fallback: the position the mux emitted when the policy stopped
    /// (or a source became unusable) is held with zero velocity until the
    /// classical source starts a new trajectory.
    Hold,
}

/// Parameters (node `config:`) of [`CuSetpointMuxTask`].
///
/// ```ron
/// config: {}                    // 0.2 s blend on every switch
/// config: {"blend_time": 0.5}   // seconds; 0.0 switches hard
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Reflect)]
#[serde(default, deny_unknown_fields)]
pub struct CuSetpointMuxParams {
    /// Duration over which the offset to a newly selected source decays to
    /// zero. `0` switches without blending.
    pub blend_time: Time,
}

impl Default for CuSetpointMuxParams {
    fn default() -> Self {
        Self {
            blend_time: Time::new::<second>(DEFAULT_MUX_BLEND_TIME_S),
        }
    }
}

impl CuSetpointMuxParams {
    /// Rejects a negative or non-finite `blend_time`.
    pub fn validate(&self) -> CuResult<()> {
        let s = self.blend_time.get::<second>();
        if !s.is_finite() || s < 0.0 {
            return Err(CuError::from(
                "repose-trajectory: 'blend_time' must be finite and not negative",
            ));
        }
        Ok(())
    }
}

/// Active blend: SI offsets (last emitted minus new source) at `start`.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Blend<const N: usize> {
    start_ns: u64,
    position: [f64; N],
    velocity: [f64; N],
    acceleration: [f64; N],
}

/// Quintic correction from `(p0, v0, a0)` at `t = 0` to rest at `t = duration`:
/// returns `(position, velocity, acceleration)` at `t`.
fn quintic_correction(p0: f64, v0: f64, a0: f64, duration: f64, t: f64) -> (f64, f64, f64) {
    if t >= duration {
        return (0.0, 0.0, 0.0);
    }
    let d = duration;
    let (d2, d3) = (d * d, d * d * d);
    let c3 = -(20.0 * p0 + 12.0 * v0 * d + 3.0 * a0 * d2) / (2.0 * d3);
    let c4 = (30.0 * p0 + 16.0 * v0 * d + 3.0 * a0 * d2) / (2.0 * d3 * d);
    let c5 = -(12.0 * p0 + 6.0 * v0 * d + a0 * d2) / (2.0 * d3 * d2);
    let (t2, t3) = (t * t, t * t * t);
    let t4 = t2 * t2;
    let p = p0 + v0 * t + 0.5 * a0 * t2 + c3 * t3 + c4 * t4 + c5 * t4 * t;
    let v = v0 + a0 * t + 3.0 * c3 * t2 + 4.0 * c4 * t3 + 5.0 * c5 * t4;
    let a = a0 + 6.0 * c3 * t + 12.0 * c4 * t2 + 20.0 * c5 * t3;
    (p, v, a)
}

/// A setpoint whose values are finite and whose joint kinds match `like`.
fn usable<const N: usize>(
    sp: &CuTrajectoryFollowOutput<N>,
    like: Option<&CuTrajectoryFollowOutput<N>>,
) -> bool {
    let finite = sp.positions.iter().all(|q| q.si_value().is_finite())
        && sp.velocities.iter().all(|v| v.si_value().is_finite())
        && sp.accelerations.iter().all(|a| a.si_value().is_finite());
    let kinds = like.is_none_or(|like| {
        sp.positions
            .iter()
            .zip(like.positions.iter())
            .all(|(a, b)| core::mem::discriminant(a) == core::mem::discriminant(b))
    });
    finite && kinds
}

/// Deterministic, bumpless mux between a classical and a policy setpoint.
///
/// Inputs are `(classical, policy)`, both [`CuTrajectoryFollowOutput<N>`];
/// output is one [`CuTrajectoryFollowOutput<N>`] for
/// `repose_controller::CuControllerTask`.
///
/// # Arbitration
///
/// | Source | Switches to | When |
/// |---|---|---|
/// | `Classical` | `Policy` | the classical setpoint was `Running` since the last switch, is now `Finished` (transit reached, e.g. a pre-grasp pose), and the policy setpoint is `Running` |
/// | `Policy` | `Classical` | the classical setpoint is `Running` again (a new plan, e.g. a retract, preempts the policy) |
/// | `Policy` | `Hold` | the policy setpoint is missing, `Idle`, `ErrorEmptyTrajectory` (policy `Fault`), `Finished`, or not usable |
/// | `Classical` | `Hold` | the classical setpoint is not usable |
/// | `Hold` | `Classical` | the classical setpoint is `Running` and usable |
///
/// A setpoint is usable when every position, velocity, and acceleration is
/// finite and its joint kinds match the last emitted setpoint. The classical
/// source must be `Running` again before it can hand over to the policy a
/// second time, so a policy that stopped is never resumed on its own, and a
/// classical `Finished` hold that the arm has since moved away from is never
/// re-entered (that would snap back to the pre-grasp pose): `Hold` waits for
/// a new classical plan, typically one starting from the measured state.
///
/// # Bumpless transfer
///
/// On every switch the mux records the difference between the setpoint it
/// emitted last and the newly selected source, per joint, for position,
/// velocity, and acceleration. It then emits the source plus a quintic
/// correction that starts at that difference and reaches zero difference,
/// with zero rate, at `blend_time`, so position, velocity, and acceleration
/// are continuous across the switch. `Hold` targets the position emitted at
/// the switch with zero velocity; the correction carries the current
/// velocity into a short stop. A switch during a blend starts a new blend
/// from the blended setpoint. The correction can overshoot the source by a
/// fraction of the switching velocity times `blend_time`; the controller's
/// own joint limits still apply.
///
/// # Status
///
/// The source status is passed through, except that a blend into a
/// `Finished` source or into `Hold` reports `Running`, and a settled `Hold`
/// reports `Finished`, so the controller tracks the held target (`Holding`)
/// rather than falling back to the measured position. A missing classical
/// setpoint while `Classical` is active emits the last setpoint with zero
/// velocity and `Idle`.
///
/// # Clock and replay
///
/// The blend clock is the `Tov::Time` of the active source's input (the
/// other input's, then `ctx.now()`, as fallback); that `Tov` is propagated
/// to the output. Source, arming, last setpoint, and blend round-trip
/// through `Freezable`.
#[derive(Reflect)]
#[reflect(no_field_bounds, from_reflect = false)]
pub struct CuSetpointMuxTask<const N: usize> {
    params: CuSetpointMuxParams,
    source: CuSetpointSource,
    /// The classical source ran since the last switch and may hand over.
    armed: bool,
    #[reflect(ignore)]
    last: Option<CuTrajectoryFollowOutput<N>>,
    #[reflect(ignore)]
    hold: [CuJointPosition; N],
    #[reflect(ignore)]
    blend: Option<Blend<N>>,
}

impl<const N: usize> Default for CuSetpointMuxTask<N> {
    fn default() -> Self {
        Self {
            params: CuSetpointMuxParams::default(),
            source: CuSetpointSource::Classical,
            armed: false,
            last: None,
            hold: [CuJointPosition::default(); N],
            blend: None,
        }
    }
}

impl<const N: usize> CuSetpointMuxTask<N> {
    /// Creates a mux following the classical source.
    pub fn from_params(params: CuSetpointMuxParams) -> CuResult<Self> {
        params.validate()?;
        Ok(Self {
            params,
            ..Self::default()
        })
    }

    /// Params this task runs with.
    pub fn params(&self) -> &CuSetpointMuxParams {
        &self.params
    }

    /// Source the mux currently follows.
    pub fn source(&self) -> CuSetpointSource {
        self.source
    }

    /// Whether a bumpless blend is in progress.
    pub fn is_blending(&self) -> bool {
        self.blend.is_some()
    }

    fn hold_setpoint(&self) -> CuTrajectoryFollowOutput<N> {
        CuTrajectoryFollowOutput {
            positions: self.hold,
            velocities: self.hold.map(|q| CuJointVelocity::from_coordinate(q, 0.0)),
            accelerations: self
                .hold
                .map(|q| CuJointAcceleration::from_coordinate(q, 0.0)),
            status: CuFollowStatus::Finished,
        }
    }

    /// Switches to `source` with `reference` as its setpoint this cycle,
    /// starting a blend from the last emitted setpoint.
    fn switch(
        &mut self,
        source: CuSetpointSource,
        reference: &CuTrajectoryFollowOutput<N>,
        now: CuTime,
    ) {
        self.source = source;
        self.armed =
            source == CuSetpointSource::Classical && reference.status == CuFollowStatus::Running;
        self.blend = None;
        let Some(last) = self.last.as_ref() else {
            return;
        };
        if self.params.blend_time.get::<second>() <= 0.0 {
            return;
        }
        let mut blend = Blend {
            start_ns: now.as_nanos(),
            position: [0.0; N],
            velocity: [0.0; N],
            acceleration: [0.0; N],
        };
        for j in 0..N {
            blend.position[j] = last.positions[j].si_value() - reference.positions[j].si_value();
            blend.velocity[j] = last.velocities[j].si_value() - reference.velocities[j].si_value();
            blend.acceleration[j] =
                last.accelerations[j].si_value() - reference.accelerations[j].si_value();
        }
        self.blend = Some(blend);
    }

    /// Enters `Hold` at the last emitted position.
    fn enter_hold(&mut self, now: CuTime) {
        if let Some(last) = self.last.as_ref() {
            self.hold = last.positions;
        }
        let reference = self.hold_setpoint();
        self.switch(CuSetpointSource::Hold, &reference, now);
    }

    /// Applies the active blend to `reference`; ends it once settled.
    fn apply_blend(&mut self, reference: &mut CuTrajectoryFollowOutput<N>, now: CuTime) {
        let Some(blend) = self.blend else {
            return;
        };
        let duration = self.params.blend_time.get::<second>();
        let elapsed = now.as_nanos().saturating_sub(blend.start_ns) as f64 / 1_000_000_000.0;
        if elapsed >= duration {
            self.blend = None;
            return;
        }
        for j in 0..N {
            let (p, v, a) = quintic_correction(
                blend.position[j],
                blend.velocity[j],
                blend.acceleration[j],
                duration,
                elapsed,
            );
            let q = reference.positions[j];
            reference.positions[j] = q.with_si_value(q.si_value() + p);
            reference.velocities[j] =
                CuJointVelocity::from_coordinate(q, reference.velocities[j].si_value() + v);
            reference.accelerations[j] =
                CuJointAcceleration::from_coordinate(q, reference.accelerations[j].si_value() + a);
        }
        if matches!(
            reference.status,
            CuFollowStatus::Finished | CuFollowStatus::Idle
        ) {
            reference.status = CuFollowStatus::Running;
        }
    }

    /// Arbitration and output for one cycle; see the type docs.
    fn step(
        &mut self,
        classical: Option<&CuTrajectoryFollowOutput<N>>,
        policy: Option<&CuTrajectoryFollowOutput<N>>,
        now: CuTime,
    ) -> CuTrajectoryFollowOutput<N> {
        let last = self.last.clone();
        let classical_running =
            classical.filter(|c| c.status == CuFollowStatus::Running && usable(c, last.as_ref()));

        match self.source {
            CuSetpointSource::Classical => match classical {
                Some(c) if !usable(c, last.as_ref()) => self.enter_hold(now),
                Some(c) => {
                    let policy_ready = policy.filter(|p| {
                        p.status == CuFollowStatus::Running && usable(p, last.as_ref())
                    });
                    match (c.status, policy_ready) {
                        (CuFollowStatus::Running, _) => self.armed = true,
                        (CuFollowStatus::Finished, Some(p)) if self.armed => {
                            self.switch(CuSetpointSource::Policy, p, now);
                        }
                        _ => {}
                    }
                }
                None => {}
            },
            CuSetpointSource::Policy => {
                if let Some(c) = classical_running {
                    self.switch(CuSetpointSource::Classical, c, now);
                } else if !policy.is_some_and(|p| {
                    p.status == CuFollowStatus::Running && usable(p, last.as_ref())
                }) {
                    self.enter_hold(now);
                }
            }
            CuSetpointSource::Hold => {
                if let Some(c) = classical_running {
                    self.switch(CuSetpointSource::Classical, c, now);
                }
            }
        }

        let mut out = match self.source {
            CuSetpointSource::Classical => match classical {
                Some(c) => c.clone(),
                None => {
                    self.blend = None;
                    let positions = last
                        .as_ref()
                        .map_or([CuJointPosition::default(); N], |l| l.positions);
                    CuTrajectoryFollowOutput {
                        positions,
                        velocities: positions.map(|q| CuJointVelocity::from_coordinate(q, 0.0)),
                        accelerations: positions
                            .map(|q| CuJointAcceleration::from_coordinate(q, 0.0)),
                        status: CuFollowStatus::Idle,
                    }
                }
            },
            // Usable and `Running`, checked above.
            CuSetpointSource::Policy => policy.cloned().unwrap_or_else(|| self.hold_setpoint()),
            CuSetpointSource::Hold => self.hold_setpoint(),
        };
        // A classical cancel (`Idle` / `ErrorEmptyTrajectory`) passes through
        // unblended: the controller then holds the measured state.
        if matches!(
            out.status,
            CuFollowStatus::Idle | CuFollowStatus::ErrorEmptyTrajectory
        ) && self.source == CuSetpointSource::Classical
        {
            self.blend = None;
        }
        self.apply_blend(&mut out, now);
        self.last = Some(out.clone());
        out
    }
}

/// The frozen state has the same encoded size in every state (no `Option`,
/// no varint counter), because Copper sizes the keyframe buffer once, before
/// the first cycle, from each task's frozen size at that time.
impl<const N: usize> Freezable for CuSetpointMuxTask<N> {
    fn freeze<E: Encoder>(&self, encoder: &mut E) -> Result<(), EncodeError> {
        Encode::encode(&self.source, encoder)?;
        Encode::encode(&self.armed, encoder)?;
        Encode::encode(&self.last.is_some(), encoder)?;
        match self.last.as_ref() {
            Some(last) => Encode::encode(last, encoder)?,
            None => Encode::encode(&CuTrajectoryFollowOutput::<N>::default(), encoder)?,
        }
        Encode::encode(&self.hold, encoder)?;
        let blend = self.blend.unwrap_or(Blend {
            start_ns: 0,
            position: [0.0; N],
            velocity: [0.0; N],
            acceleration: [0.0; N],
        });
        Encode::encode(&self.blend.is_some(), encoder)?;
        Encode::encode(&blend.start_ns.to_le_bytes(), encoder)?;
        Encode::encode(&blend.position, encoder)?;
        Encode::encode(&blend.velocity, encoder)?;
        Encode::encode(&blend.acceleration, encoder)
    }

    fn thaw<D: Decoder>(&mut self, decoder: &mut D) -> Result<(), DecodeError> {
        self.source = Decode::decode(decoder)?;
        self.armed = Decode::decode(decoder)?;
        let has_last: bool = Decode::decode(decoder)?;
        let last: CuTrajectoryFollowOutput<N> = Decode::decode(decoder)?;
        self.last = has_last.then_some(last);
        self.hold = Decode::decode(decoder)?;
        let has_blend: bool = Decode::decode(decoder)?;
        let start: [u8; 8] = Decode::decode(decoder)?;
        let blend = Blend {
            start_ns: u64::from_le_bytes(start),
            position: Decode::decode(decoder)?,
            velocity: Decode::decode(decoder)?,
            acceleration: Decode::decode(decoder)?,
        };
        self.blend = has_blend.then_some(blend);
        Ok(())
    }
}

impl<const N: usize> CuTask for CuSetpointMuxTask<N> {
    type Resources<'r> = ();
    type Input<'m> = input_msg!('m, CuTrajectoryFollowOutput<N>, CuTrajectoryFollowOutput<N>);
    type Output<'m> = output_msg!(CuTrajectoryFollowOutput<N>);

    fn new(cfg: Option<&ComponentConfig>, _res: Self::Resources<'_>) -> CuResult<Self>
    where
        Self: Sized,
    {
        let params = cfg
            .map(ComponentConfig::deserialize_into::<CuSetpointMuxParams>)
            .transpose()?
            .unwrap_or_default();
        Self::from_params(params)
    }

    fn process<'i, 'o>(
        &mut self,
        ctx: &CuContext,
        input: &Self::Input<'i>,
        output: &mut Self::Output<'o>,
    ) -> CuResult<()> {
        let (primary, secondary) = match self.source {
            CuSetpointSource::Policy => (input.1.tov, input.0.tov),
            _ => (input.0.tov, input.1.tov),
        };
        let now = match (primary, secondary) {
            (Tov::Time(t), _) | (_, Tov::Time(t)) => t,
            _ => ctx.now(),
        };
        let out = self.step(input.0.payload(), input.1.payload(), now);
        output.tov = match self.source {
            CuSetpointSource::Policy => input.1.tov,
            _ => input.0.tov,
        };
        output.set_payload(out);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cu29::units::si::angle::radian;
    use cu29::units::si::f64::Angle;

    const MS: u64 = 1_000_000;

    fn sp(q: f64, v: f64, status: CuFollowStatus) -> CuTrajectoryFollowOutput<2> {
        let pos = [CuJointPosition::Revolute(Angle::new::<radian>(q)); 2];
        CuTrajectoryFollowOutput {
            positions: pos,
            velocities: pos.map(|p| CuJointVelocity::from_coordinate(p, v)),
            accelerations: pos.map(|p| CuJointAcceleration::from_coordinate(p, 0.0)),
            status,
        }
    }

    fn at(ns: u64) -> CuTime {
        CuTime::from_nanos(ns)
    }

    fn mux(blend: f64) -> CuSetpointMuxTask<2> {
        CuSetpointMuxTask::from_params(CuSetpointMuxParams {
            blend_time: Time::new::<second>(blend),
        })
        .unwrap()
    }

    /// Classical runs to 1.0 and finishes; returns the time of the next cycle.
    fn transit(m: &mut CuSetpointMuxTask<2>, policy: Option<&CuTrajectoryFollowOutput<2>>) -> u64 {
        m.step(Some(&sp(0.5, 1.0, CuFollowStatus::Running)), policy, at(0));
        m.step(
            Some(&sp(1.0, 0.0, CuFollowStatus::Finished)),
            policy,
            at(MS),
        );
        2 * MS
    }

    fn q(out: &CuTrajectoryFollowOutput<2>) -> f64 {
        out.positions[0].si_value()
    }

    fn v(out: &CuTrajectoryFollowOutput<2>) -> f64 {
        out.velocities[0].si_value()
    }

    #[test]
    fn quintic_correction_starts_at_offset_and_settles() {
        let (p, v, a) = quintic_correction(0.3, -1.2, 4.0, 0.5, 0.0);
        assert!((p - 0.3).abs() < 1e-12 && (v + 1.2).abs() < 1e-12 && (a - 4.0).abs() < 1e-12);
        let (p, v, a) = quintic_correction(0.3, -1.2, 4.0, 0.5, 0.5 - 1e-9);
        assert!(p.abs() < 1e-6 && v.abs() < 1e-6 && a.abs() < 1e-3);
        // Velocity is the derivative of position.
        let h = 1e-6;
        let (p0, v0, a0) = quintic_correction(0.3, -1.2, 4.0, 0.5, 0.2);
        let (p1, v1, _) = quintic_correction(0.3, -1.2, 4.0, 0.5, 0.2 + h);
        assert!(((p1 - p0) / h - v0).abs() < 1e-4);
        assert!(((v1 - v0) / h - a0).abs() < 1e-3);
    }

    #[test]
    fn params_reject_negative_blend_and_parse_ron() {
        assert!(
            CuSetpointMuxParams {
                blend_time: Time::new::<second>(-0.1)
            }
            .validate()
            .is_err()
        );
        let p: CuSetpointMuxParams = ron::from_str("(blend_time: 0.5)").unwrap();
        assert_eq!(p.blend_time.get::<second>(), 0.5);
        let p: CuSetpointMuxParams = ron::from_str("()").unwrap();
        assert_eq!(p.blend_time.get::<second>(), DEFAULT_MUX_BLEND_TIME_S);
    }

    #[test]
    fn policy_is_ignored_until_classical_transit_finishes() {
        let mut m = mux(0.1);
        let policy = sp(5.0, 0.0, CuFollowStatus::Running);
        // Classical idle at boot: no handoff, even with a running policy.
        let out = m.step(
            Some(&sp(0.0, 0.0, CuFollowStatus::Idle)),
            Some(&policy),
            at(0),
        );
        assert_eq!(m.source(), CuSetpointSource::Classical);
        assert_eq!(out.status, CuFollowStatus::Idle);
        // A classical Finished that was never Running does not arm either.
        m.step(
            Some(&sp(1.0, 0.0, CuFollowStatus::Finished)),
            Some(&policy),
            at(MS),
        );
        assert_eq!(m.source(), CuSetpointSource::Classical);
        // Running passes through untouched.
        let out = m.step(
            Some(&sp(0.5, 1.0, CuFollowStatus::Running)),
            Some(&policy),
            at(2 * MS),
        );
        assert_eq!(out, sp(0.5, 1.0, CuFollowStatus::Running));
        // An idle policy blocks the handoff at Finished.
        let idle = sp(0.0, 0.0, CuFollowStatus::Idle);
        m.step(
            Some(&sp(1.0, 0.0, CuFollowStatus::Finished)),
            Some(&idle),
            at(3 * MS),
        );
        assert_eq!(m.source(), CuSetpointSource::Classical);
        // The policy becoming ready later still hands over.
        m.step(
            Some(&sp(1.0, 0.0, CuFollowStatus::Finished)),
            Some(&policy),
            at(4 * MS),
        );
        assert_eq!(m.source(), CuSetpointSource::Policy);
    }

    #[test]
    fn handoff_to_policy_is_bumpless_and_converges() {
        let mut m = mux(0.1);
        let policy = sp(1.2, 0.5, CuFollowStatus::Running);
        let t = transit(&mut m, None);
        assert_eq!(m.source(), CuSetpointSource::Classical);
        let out = m.step(
            Some(&sp(1.0, 0.0, CuFollowStatus::Finished)),
            Some(&policy),
            at(t),
        );
        assert_eq!(m.source(), CuSetpointSource::Policy);
        assert!(m.is_blending());
        // First cycle continues exactly from the last classical setpoint.
        assert!((q(&out) - 1.0).abs() < 1e-12);
        assert!(v(&out).abs() < 1e-12);
        assert_eq!(out.status, CuFollowStatus::Running);
        // 1 kHz cycles: no step larger than a smooth motion allows.
        let mut prev = q(&out);
        for k in 1..=100 {
            let out = m.step(
                Some(&sp(1.0, 0.0, CuFollowStatus::Finished)),
                Some(&policy),
                at(t + k * MS),
            );
            assert!((q(&out) - prev).abs() < 4e-3, "cycle {k}");
            prev = q(&out);
        }
        assert!(!m.is_blending());
        let out = m.step(None, Some(&policy), at(t + 101 * MS));
        assert_eq!(out, policy);
    }

    #[test]
    fn policy_fault_falls_back_to_hold_and_needs_a_new_plan() {
        let mut m = mux(0.0);
        let policy = sp(1.3, 0.0, CuFollowStatus::Running);
        let t = transit(&mut m, Some(&policy));
        m.step(
            Some(&sp(1.0, 0.0, CuFollowStatus::Finished)),
            Some(&policy),
            at(t),
        );
        let fault = sp(0.9, 0.0, CuFollowStatus::ErrorEmptyTrajectory);
        let out = m.step(
            Some(&sp(1.0, 0.0, CuFollowStatus::Finished)),
            Some(&fault),
            at(t + MS),
        );
        assert_eq!(m.source(), CuSetpointSource::Hold);
        // Holds where the policy left the arm, not the stale pre-grasp pose.
        assert!((q(&out) - 1.3).abs() < 1e-12);
        assert_eq!(out.status, CuFollowStatus::Finished);
        // A recovered policy and a still-finished classical do not resume.
        let out = m.step(
            Some(&sp(1.0, 0.0, CuFollowStatus::Finished)),
            Some(&policy),
            at(t + 2 * MS),
        );
        assert_eq!(m.source(), CuSetpointSource::Hold);
        assert!((q(&out) - 1.3).abs() < 1e-12);
        // A new classical plan (retract) takes over.
        let retract = sp(1.29, -0.2, CuFollowStatus::Running);
        let out = m.step(Some(&retract), Some(&policy), at(t + 3 * MS));
        assert_eq!(m.source(), CuSetpointSource::Classical);
        assert_eq!(out, retract);
    }

    #[test]
    fn missing_idle_or_non_finite_policy_holds() {
        for bad in [
            None,
            Some(sp(1.3, 0.0, CuFollowStatus::Idle)),
            Some(sp(f64::NAN, 0.0, CuFollowStatus::Running)),
            Some(sp(1.3, f64::INFINITY, CuFollowStatus::Running)),
        ] {
            let mut m = mux(0.1);
            let policy = sp(1.3, 0.0, CuFollowStatus::Running);
            let t = transit(&mut m, Some(&policy));
            m.step(
                Some(&sp(1.0, 0.0, CuFollowStatus::Finished)),
                Some(&policy),
                at(t),
            );
            let out = m.step(
                Some(&sp(1.0, 0.0, CuFollowStatus::Finished)),
                bad.as_ref(),
                at(t + MS),
            );
            assert_eq!(m.source(), CuSetpointSource::Hold, "{bad:?}");
            assert!(q(&out).is_finite() && v(&out).is_finite());
        }
    }

    #[test]
    fn hold_blend_carries_velocity_into_a_stop() {
        let mut m = mux(0.1);
        let t = transit(&mut m, None);
        // Policy moving at 1 rad/s when it stops.
        m.step(
            Some(&sp(1.0, 0.0, CuFollowStatus::Finished)),
            Some(&sp(1.0, 1.0, CuFollowStatus::Running)),
            at(t),
        );
        for k in 1..=200 {
            m.step(
                None,
                Some(&sp(1.0, 1.0, CuFollowStatus::Running)),
                at(t + k * MS),
            );
        }
        let out = m.step(None, None, at(t + 201 * MS));
        assert_eq!(m.source(), CuSetpointSource::Hold);
        // Velocity continuous at the switch, then a stop.
        assert!((v(&out) - 1.0).abs() < 1e-2);
        assert_eq!(out.status, CuFollowStatus::Running);
        let out = m.step(None, None, at(t + 400 * MS));
        assert_eq!(out.status, CuFollowStatus::Finished);
        assert_eq!(v(&out), 0.0);
    }

    #[test]
    fn new_classical_plan_preempts_policy() {
        let mut m = mux(0.05);
        let policy = sp(1.3, 0.0, CuFollowStatus::Running);
        let t = transit(&mut m, Some(&policy));
        m.step(
            Some(&sp(1.0, 0.0, CuFollowStatus::Finished)),
            Some(&policy),
            at(t),
        );
        assert_eq!(m.source(), CuSetpointSource::Policy);
        m.step(
            Some(&sp(1.25, 0.0, CuFollowStatus::Running)),
            Some(&policy),
            at(t + MS),
        );
        assert_eq!(m.source(), CuSetpointSource::Classical);
        // Re-armed: finishing the new plan hands over again.
        m.step(
            Some(&sp(1.0, 0.0, CuFollowStatus::Finished)),
            Some(&policy),
            at(t + 2 * MS),
        );
        assert_eq!(m.source(), CuSetpointSource::Policy);
    }

    #[test]
    fn mismatched_joint_kinds_hold() {
        let mut m = mux(0.1);
        let t = transit(&mut m, None);
        let mut policy = sp(1.3, 0.0, CuFollowStatus::Running);
        policy.positions[1] = CuJointPosition::Fixed;
        m.step(
            Some(&sp(1.0, 0.0, CuFollowStatus::Finished)),
            Some(&policy),
            at(t),
        );
        assert_eq!(m.source(), CuSetpointSource::Classical);
    }

    #[test]
    fn freeze_thaw_mid_blend_resumes_identically() {
        let mut original = mux(0.1);
        let policy = sp(1.2, 0.5, CuFollowStatus::Running);
        let t = transit(&mut original, Some(&policy));
        original.step(
            Some(&sp(1.0, 0.0, CuFollowStatus::Finished)),
            Some(&policy),
            at(t),
        );
        original.step(None, Some(&policy), at(t + 10 * MS));
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
        let mut restored = mux(0.1);
        restored.thaw(&mut decoder).unwrap();
        assert_eq!(restored.source(), CuSetpointSource::Policy);
        for k in 11..60 {
            let a = original.step(None, Some(&policy), at(t + k * MS));
            let b = restored.step(None, Some(&policy), at(t + k * MS));
            assert_eq!(a, b);
        }
    }

    fn frozen_len(m: &CuSetpointMuxTask<2>) -> usize {
        let mut bytes = [0u8; 4096];
        let mut encoder = bincode::enc::EncoderImpl::new(
            bincode::enc::write::SliceWriter::new(&mut bytes),
            bincode::config::standard(),
        );
        m.freeze(&mut encoder).unwrap();
        encoder.into_writer().bytes_written()
    }

    #[test]
    fn frozen_size_never_grows() {
        // Copper sizes the keyframe buffer from the frozen size before the
        // first cycle; a larger snapshot later fails the keyframe.
        let mut m = mux(0.1);
        let initial = frozen_len(&m);
        let policy = sp(1.2, 0.5, CuFollowStatus::Running);
        let t = transit(&mut m, None);
        assert!(frozen_len(&m) <= initial);
        m.step(
            Some(&sp(1.0, 0.0, CuFollowStatus::Finished)),
            Some(&policy),
            at(t),
        );
        assert!(m.is_blending());
        assert!(frozen_len(&m) <= initial);
        m.step(None, None, at(u64::MAX / 2));
        assert_eq!(m.source(), CuSetpointSource::Hold);
        assert!(frozen_len(&m) <= initial);
    }

    #[test]
    fn process_uses_active_source_tov() {
        let (ctx, _) = CuContext::new_mock_clock();
        let mut m = mux(0.0);
        let mut c = CuMsg::new(Some(sp(0.5, 1.0, CuFollowStatus::Running)));
        c.tov = Tov::Time(at(7 * MS));
        let p = CuMsg::<CuTrajectoryFollowOutput<2>>::new(None);
        let mut out = CuMsg::new(None);
        m.process(&ctx, &(&c, &p), &mut out).unwrap();
        assert_eq!(out.tov, Tov::Time(at(7 * MS)));
        assert_eq!(out.payload().unwrap().status, CuFollowStatus::Running);
    }
}
