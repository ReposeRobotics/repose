//! Fixed-capacity joint-space trajectory payload and allocation-free sampling.
//!
//! [`CuTrajectoryPayload`] stores up to `MAX_POINTS` timestamped joint waypoints
//! and the [`CuTrajectoryInterpolation`] that reconstructs the motion between
//! them. [`CuTrajectoryPayload::sample`] interpolates joint positions and
//! velocities in place into caller-provided buffers without allocating.
//!
//! Joint-space waypoints are timed with [`parameterize`]; Cartesian paths are
//! built with [`blend_cartesian_path`] and timed with
//! [`parameterize_cartesian_path`]. Both timing functions share
//! [`CuTimingParams`] and report a [`CuTimingStatus`], and both set the
//! payload's interpolation so [`CuFollowTrajectoryTask`] executes the motion
//! that was validated.

#![cfg_attr(not(feature = "std"), no_std)]

mod cartesian_blend;
mod parameterize;
mod path_timing;
mod quintic;
mod serde_arrays;
mod task;
#[cfg(test)]
mod test_support;

pub use cartesian_blend::CuCartesianBlendParams;
pub use cartesian_blend::CuCartesianBlendPath;
pub use cartesian_blend::CuCartesianBlendStatus;
pub use cartesian_blend::CuCartesianPathPoint;
pub use cartesian_blend::CuCartesianWaypoint;
pub use cartesian_blend::CuTcpZone;
pub use cartesian_blend::blend_cartesian_path;
pub use parameterize::CuJointTrajectoryLimits;
pub use parameterize::CuTimingOverrides;
pub use parameterize::CuTimingParams;
pub use parameterize::CuTimingStatus;
pub use parameterize::CuTrajectoryProfile;
pub use parameterize::parameterize;
pub use path_timing::CuPathTimingParams;
pub use path_timing::parameterize_cartesian_path;
pub use task::CuFollowStatus;
pub use task::CuFollowTrajectoryParams;
pub use task::CuFollowTrajectoryTask;
pub use task::CuTrajectoryFollowOutput;

use bincode::Decode;
use bincode::Encode;
use cu29::prelude::*;
use cu29::units::si::angular_velocity::radian_per_second;
use cu29::units::si::f64::AngularVelocity;
use cu29::units::si::f64::Time;
use cu29::units::si::f64::Velocity;
use cu29::units::si::time::second;
use cu29::units::si::velocity::meter_per_second;
use repose_kinematics::CuJointAcceleration;
use repose_kinematics::CuJointPosition;
use repose_kinematics::CuJointVelocity;
use serde::Deserialize;
use serde::Serialize;

/// Maximum number of waypoints in a [`CuTrajectoryPayload`].
pub const MAX_TRAJECTORY_POINTS: usize = 64;

/// Single timestamped joint waypoint with positions and velocities.
#[derive(Debug, Clone, PartialEq, Encode, Decode, Serialize, Deserialize, Reflect)]
#[reflect(no_field_bounds, from_reflect = false)]
pub struct CuTrajectoryPoint<const N: usize> {
    /// Elapsed time since trajectory start.
    pub time_from_start: Time,
    #[serde(with = "repose_kinematics::serde_arrays::position_array")]
    pub positions: [CuJointPosition; N],
    #[serde(with = "repose_kinematics::serde_arrays::velocity_array")]
    pub velocities: [CuJointVelocity; N],
    /// Joint accelerations (`rad/s^2` revolute, `m/s^2` prismatic).
    ///
    /// Only quintic sampling ([`CuTrajectoryPayload::sample_quintic`]) reads
    /// them; the other kinds derive acceleration from velocities. Trajectories
    /// that rest at, or have zero acceleration at, every waypoint store zero.
    #[serde(with = "repose_kinematics::serde_arrays::acceleration_array")]
    pub accelerations: [CuJointAcceleration; N],
}

impl<const N: usize> Default for CuTrajectoryPoint<N> {
    fn default() -> Self {
        Self {
            time_from_start: Time::default(),
            positions: [CuJointPosition::default(); N],
            velocities: [CuJointVelocity::default(); N],
            accelerations: [CuJointAcceleration::default(); N],
        }
    }
}

/// Reconstruction of the motion between stored trajectory points.
///
/// Stored on every [`CuTrajectoryPayload`] (set by the parameterizer that
/// produced it) and used by [`CuTrajectoryPayload::sample`].
#[derive(
    Default, Debug, Clone, Copy, PartialEq, Eq, Encode, Decode, Serialize, Deserialize, Reflect,
)]
#[serde(rename_all = "snake_case")]
pub enum CuTrajectoryInterpolation {
    /// Positions and velocities are interpolated linearly
    /// ([`CuTrajectoryPayload::sample_linear`]).
    #[default]
    Linear,
    /// Position and velocity are reconstructed using cubic Hermite interpolation
    /// ([`CuTrajectoryPayload::sample_cubic`]).
    Cubic,
    /// Position and velocity are reconstructed with the quintic Hermite motion
    /// through the stored positions, velocities and accelerations
    /// ([`CuTrajectoryPayload::sample_quintic`]).
    Quintic,
}

/// Fixed-capacity timestamped joint trajectory (`CuMsgPayload`-compliant).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Reflect)]
#[reflect(no_field_bounds, from_reflect = false)]
pub struct CuTrajectoryPayload<const N: usize, const MAX_POINTS: usize = MAX_TRAJECTORY_POINTS> {
    #[serde(with = "crate::serde_arrays::trajectory_point_array")]
    pub points: [CuTrajectoryPoint<N>; MAX_POINTS],
    /// Number of valid waypoints in `points[..len]`.
    pub len: u32,
    /// How [`CuTrajectoryPayload::sample`] reconstructs the motion between
    /// points. [`parameterize`] and [`parameterize_cartesian_path`] set it to
    /// the reconstruction their limits were validated on; hand-built
    /// trajectories default to [`CuTrajectoryInterpolation::Linear`].
    #[serde(default)]
    pub interpolation: CuTrajectoryInterpolation,
}

// The bincode derive currently emits invalid impl generics for const defaults.
impl<const N: usize, const MAX_POINTS: usize> Encode for CuTrajectoryPayload<N, MAX_POINTS> {
    fn encode<E: bincode::enc::Encoder>(
        &self,
        encoder: &mut E,
    ) -> Result<(), bincode::error::EncodeError> {
        Encode::encode(&self.points, encoder)?;
        Encode::encode(&self.len, encoder)?;
        Encode::encode(&self.interpolation, encoder)
    }
}
impl<C, const N: usize, const MAX_POINTS: usize> Decode<C> for CuTrajectoryPayload<N, MAX_POINTS> {
    fn decode<D: bincode::de::Decoder<Context = C>>(
        decoder: &mut D,
    ) -> Result<Self, bincode::error::DecodeError> {
        Ok(Self {
            points: Decode::decode(decoder)?,
            len: Decode::decode(decoder)?,
            interpolation: Decode::decode(decoder)?,
        })
    }
}
impl<'de, C, const N: usize, const MAX_POINTS: usize> bincode::BorrowDecode<'de, C>
    for CuTrajectoryPayload<N, MAX_POINTS>
{
    fn borrow_decode<D: bincode::de::BorrowDecoder<'de, Context = C>>(
        decoder: &mut D,
    ) -> Result<Self, bincode::error::DecodeError> {
        Decode::decode(decoder)
    }
}

impl<const N: usize, const MAX_POINTS: usize> Default for CuTrajectoryPayload<N, MAX_POINTS> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const N: usize, const MAX_POINTS: usize> CuTrajectoryPayload<N, MAX_POINTS> {
    /// Creates an empty trajectory with [`CuTrajectoryInterpolation::Linear`].
    pub fn new() -> Self {
        Self {
            points: core::array::from_fn(|_| CuTrajectoryPoint::default()),
            len: 0,
            interpolation: CuTrajectoryInterpolation::Linear,
        }
    }

    /// Returns `true` when the trajectory holds no waypoints.
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the number of valid waypoints.
    pub const fn len(&self) -> usize {
        self.len as usize
    }

    /// Returns the last waypoint time, or zero when empty.
    pub fn duration(&self) -> Time {
        if self.is_empty() {
            Time::default()
        } else {
            self.points[self.len as usize - 1].time_from_start
        }
    }

    /// Borrows the valid waypoint prefix.
    pub fn points(&self) -> &[CuTrajectoryPoint<N>] {
        &self.points[..self.len as usize]
    }

    /// Appends a waypoint; times must be strictly increasing.
    pub fn push(&mut self, point: CuTrajectoryPoint<N>) -> CuResult<()> {
        if self.len as usize >= MAX_POINTS {
            return Err(CuError::from("Trajectory is full"));
        }
        if !self.is_empty()
            && point.time_from_start <= self.points[self.len as usize - 1].time_from_start
        {
            return Err(CuError::from(
                "Trajectory times must be strictly increasing",
            ));
        }
        self.points[self.len as usize] = point;
        self.len += 1;
        Ok(())
    }

    /// Drops all waypoints; the interpolation is kept.
    pub fn clear(&mut self) {
        self.len = 0;
    }

    /// Samples joint positions and velocities at `t` in place with the
    /// payload's [`interpolation`](Self::interpolation).
    ///
    /// Dispatches to [`sample_linear`](Self::sample_linear),
    /// [`sample_cubic`](Self::sample_cubic) or
    /// [`sample_quintic`](Self::sample_quintic). Sampling before the first or
    /// after the last waypoint holds that endpoint.
    pub fn sample(
        &self,
        t: Time,
        out_positions: &mut [CuJointPosition; N],
        out_velocities: &mut [CuJointVelocity; N],
    ) -> CuResult<()> {
        self.sample_with(self.interpolation, t, out_positions, out_velocities)
    }

    /// Samples at `t` with an explicit `interpolation`, ignoring the payload's.
    pub fn sample_with(
        &self,
        interpolation: CuTrajectoryInterpolation,
        t: Time,
        out_positions: &mut [CuJointPosition; N],
        out_velocities: &mut [CuJointVelocity; N],
    ) -> CuResult<()> {
        match interpolation {
            CuTrajectoryInterpolation::Linear => {
                self.sample_linear(t, out_positions, out_velocities)
            }
            CuTrajectoryInterpolation::Cubic => self.sample_cubic(t, out_positions, out_velocities),
            CuTrajectoryInterpolation::Quintic => {
                self.sample_quintic(t, out_positions, out_velocities)
            }
        }
    }

    /// Samples at `t` like [`sample_with`](Self::sample_with) and also writes
    /// the desired joint accelerations in place.
    ///
    /// Acceleration is the analytic time derivative of the velocity that the
    /// chosen interpolation emits, never a finite difference:
    /// - [`Linear`](CuTrajectoryInterpolation::Linear): the segment's constant
    ///   `(v1 - v0) / dt` (the derivative of the interpolated velocity).
    /// - [`Cubic`](CuTrajectoryInterpolation::Cubic): the Hermite segment's
    ///   second derivative, linear in time within a segment.
    /// - [`Quintic`](CuTrajectoryInterpolation::Quintic): the quintic's
    ///   acceleration, which interpolates the stored knot accelerations and is
    ///   continuous across knots.
    ///
    /// At an interior knot the segment ending there is used (the left-sided
    /// value), so linear and cubic accelerations may jump across a knot. Held
    /// endpoints (`t` at or before the first knot, at or after the last, or a
    /// single-point trajectory) report zero acceleration: the setpoint is
    /// stationary there, whatever the stored endpoint accelerations are. The
    /// units follow the position kind of each joint.
    pub fn sample_with_acceleration(
        &self,
        interpolation: CuTrajectoryInterpolation,
        t: Time,
        out_positions: &mut [CuJointPosition; N],
        out_velocities: &mut [CuJointVelocity; N],
        out_accelerations: &mut [CuJointAcceleration; N],
    ) -> CuResult<()> {
        let acc = Some(out_accelerations);
        match interpolation {
            CuTrajectoryInterpolation::Linear => {
                self.sample_linear_impl(t, out_positions, out_velocities, acc)
            }
            CuTrajectoryInterpolation::Cubic => {
                self.sample_cubic_impl(t, out_positions, out_velocities, acc)
            }
            CuTrajectoryInterpolation::Quintic => {
                self.sample_quintic_impl(t, out_positions, out_velocities, acc)
            }
        }
    }

    /// Samples joint positions and velocities at `t` in place by linear
    /// interpolation between the bracketing waypoints.
    ///
    /// `t` is clamped to `[0, duration]`; sampling before the first or after
    /// the last waypoint holds that endpoint. Errors on an empty trajectory or
    /// on mismatched joint coordinate types.
    pub fn sample_linear(
        &self,
        t: Time,
        out_positions: &mut [CuJointPosition; N],
        out_velocities: &mut [CuJointVelocity; N],
    ) -> CuResult<()> {
        self.sample_linear_impl(t, out_positions, out_velocities, None)
    }

    fn sample_linear_impl(
        &self,
        t: Time,
        out_positions: &mut [CuJointPosition; N],
        out_velocities: &mut [CuJointVelocity; N],
        mut out_accelerations: Option<&mut [CuJointAcceleration; N]>,
    ) -> CuResult<()> {
        if self.is_empty() {
            return Err(CuError::from("Cannot sample an empty trajectory"));
        }
        if self.len() == 1 {
            *out_positions = self.points[0].positions;
            *out_velocities = self.points[0].velocities;
            hold_accelerations(out_positions, out_accelerations.as_deref_mut());
            return Ok(());
        }
        let pts = self.points();
        let duration = self.duration();
        let zero = Time::default();
        let clamped = if t <= zero {
            zero
        } else if t >= duration {
            duration
        } else {
            t
        };
        let t_s = clamped.get::<second>();
        let last = pts.len() - 1;
        if t_s <= pts[0].time_from_start.get::<second>() {
            *out_positions = pts[0].positions;
            *out_velocities = pts[0].velocities;
            hold_accelerations(out_positions, out_accelerations.as_deref_mut());
            return Ok(());
        }
        if t_s >= pts[last].time_from_start.get::<second>() {
            *out_positions = pts[last].positions;
            *out_velocities = pts[last].velocities;
            hold_accelerations(out_positions, out_accelerations.as_deref_mut());
            return Ok(());
        }
        let mut i = 0;
        while i < last && pts[i + 1].time_from_start.get::<second>() < t_s {
            i += 1;
        }
        let t0 = pts[i].time_from_start.get::<second>();
        let t1 = pts[i + 1].time_from_start.get::<second>();
        let fraction = (t_s - t0) / (t1 - t0);
        let mut j = 0;
        while j < N {
            out_positions[j] =
                pts[i].positions[j].interpolate(pts[i + 1].positions[j], fraction)?;
            out_velocities[j] =
                lerp_velocity(pts[i].velocities[j], pts[i + 1].velocities[j], fraction)?;
            if let Some(acc) = out_accelerations.as_deref_mut() {
                // Derivative of the linearly interpolated velocity.
                let dv = pts[i + 1].velocities[j].si_value() - pts[i].velocities[j].si_value();
                acc[j] = CuJointAcceleration::from_coordinate(pts[i].positions[j], dv / (t1 - t0));
            }
            j += 1;
        }
        Ok(())
    }

    /// Samples cubic Hermite interpolation using the stored positions and velocities.
    ///
    /// [`parameterize`] and the non-S-curve profiles of
    /// [`parameterize_cartesian_path`] validate their limits on this
    /// reconstruction; it preserves the continuous position and velocity at
    /// each stored knot.
    pub fn sample_cubic(
        &self,
        t: Time,
        out_positions: &mut [CuJointPosition; N],
        out_velocities: &mut [CuJointVelocity; N],
    ) -> CuResult<()> {
        self.sample_cubic_impl(t, out_positions, out_velocities, None)
    }

    fn sample_cubic_impl(
        &self,
        t: Time,
        out_positions: &mut [CuJointPosition; N],
        out_velocities: &mut [CuJointVelocity; N],
        mut out_accelerations: Option<&mut [CuJointAcceleration; N]>,
    ) -> CuResult<()> {
        if self.is_empty() || self.len() > MAX_POINTS || !t.get::<second>().is_finite() {
            return Err(CuError::from("Invalid cubic trajectory or sample time"));
        }
        let pts = self.points();
        let ts = t.get::<second>();
        if ts <= pts[0].time_from_start.get::<second>() {
            *out_positions = pts[0].positions;
            *out_velocities = pts[0].velocities;
            hold_accelerations(out_positions, out_accelerations.as_deref_mut());
            return Ok(());
        }
        let last = pts.len() - 1;
        if ts >= pts[last].time_from_start.get::<second>() {
            *out_positions = pts[last].positions;
            *out_velocities = pts[last].velocities;
            hold_accelerations(out_positions, out_accelerations.as_deref_mut());
            return Ok(());
        }
        let mut i = 0;
        while i < last && pts[i + 1].time_from_start.get::<second>() < ts {
            i += 1;
        }
        let a = &pts[i];
        let b = &pts[i + 1];
        let t0 = a.time_from_start.get::<second>();
        let duration = b.time_from_start.get::<second>() - t0;
        if !duration.is_finite() || duration <= 0.0 {
            return Err(CuError::from("Invalid trajectory segment duration"));
        }
        let u = (ts - t0) / duration;
        for j in 0..N {
            if core::mem::discriminant(&a.positions[j]) != core::mem::discriminant(&b.positions[j])
                || core::mem::discriminant(&a.velocities[j])
                    != core::mem::discriminant(&b.velocities[j])
                || !matches!(
                    (a.positions[j], a.velocities[j]),
                    (CuJointPosition::Revolute(_), CuJointVelocity::Revolute(_))
                        | (CuJointPosition::Prismatic(_), CuJointVelocity::Prismatic(_))
                        | (CuJointPosition::Fixed, CuJointVelocity::Fixed)
                )
            {
                return Err(CuError::from("Trajectory joint kinds disagree"));
            }
            let q0 = a.positions[j].si_value();
            let q1 = b.positions[j].si_value();
            let v0 = a.velocities[j].si_value();
            let v1 = b.velocities[j].si_value();
            if !q0.is_finite() || !q1.is_finite() || !v0.is_finite() || !v1.is_finite() {
                return Err(CuError::from("Non-finite trajectory knot"));
            }
            let d = q1 - q0;
            let aa = v0 * duration;
            let bb = 3.0 * d - 2.0 * aa - v1 * duration;
            let cc = -2.0 * d + aa + v1 * duration;
            out_positions[j] = a.positions[j].with_si_value(q0 + u * (aa + u * (bb + u * cc)));
            out_velocities[j] = CuJointVelocity::from_coordinate(
                a.positions[j],
                (aa + u * (2.0 * bb + 3.0 * cc * u)) / duration,
            );
            if let Some(acc) = out_accelerations.as_deref_mut() {
                acc[j] = CuJointAcceleration::from_coordinate(
                    a.positions[j],
                    (2.0 * bb + 6.0 * cc * u) / (duration * duration),
                );
            }
        }
        Ok(())
    }

    /// Samples the quintic Hermite motion through the stored knots.
    ///
    /// Each segment is the quintic matching position, velocity and
    /// acceleration at both knots, so all three are continuous across the
    /// trajectory and jerk is finite. [`parameterize_cartesian_path`] with the
    /// S-curve profile verifies its limits on exactly this reconstruction. Sampling before the first
    /// or after the last knot holds that endpoint.
    pub fn sample_quintic(
        &self,
        t: Time,
        out_positions: &mut [CuJointPosition; N],
        out_velocities: &mut [CuJointVelocity; N],
    ) -> CuResult<()> {
        self.sample_quintic_impl(t, out_positions, out_velocities, None)
    }

    fn sample_quintic_impl(
        &self,
        t: Time,
        out_positions: &mut [CuJointPosition; N],
        out_velocities: &mut [CuJointVelocity; N],
        mut out_accelerations: Option<&mut [CuJointAcceleration; N]>,
    ) -> CuResult<()> {
        if self.is_empty() || self.len() > MAX_POINTS || !t.get::<second>().is_finite() {
            return Err(CuError::from("Invalid quintic trajectory or sample time"));
        }
        let pts = self.points();
        let ts = t.get::<second>();
        let last = pts.len() - 1;
        if ts <= pts[0].time_from_start.get::<second>() {
            *out_positions = pts[0].positions;
            *out_velocities = pts[0].velocities;
            hold_accelerations(out_positions, out_accelerations.as_deref_mut());
            return Ok(());
        }
        if ts >= pts[last].time_from_start.get::<second>() {
            *out_positions = pts[last].positions;
            *out_velocities = pts[last].velocities;
            hold_accelerations(out_positions, out_accelerations.as_deref_mut());
            return Ok(());
        }
        let mut i = 0;
        while i < last && pts[i + 1].time_from_start.get::<second>() < ts {
            i += 1;
        }
        let a = &pts[i];
        let b = &pts[i + 1];
        let t0 = a.time_from_start.get::<second>();
        let duration = b.time_from_start.get::<second>() - t0;
        if !duration.is_finite() || duration <= 0.0 {
            return Err(CuError::from("Invalid trajectory segment duration"));
        }
        let u = (ts - t0) / duration;
        for j in 0..N {
            let kinds_agree = matches!(
                (
                    a.positions[j],
                    b.positions[j],
                    a.velocities[j],
                    b.velocities[j]
                ),
                (
                    CuJointPosition::Revolute(_),
                    CuJointPosition::Revolute(_),
                    CuJointVelocity::Revolute(_),
                    CuJointVelocity::Revolute(_)
                ) | (
                    CuJointPosition::Prismatic(_),
                    CuJointPosition::Prismatic(_),
                    CuJointVelocity::Prismatic(_),
                    CuJointVelocity::Prismatic(_)
                ) | (
                    CuJointPosition::Fixed,
                    CuJointPosition::Fixed,
                    CuJointVelocity::Fixed,
                    CuJointVelocity::Fixed
                )
            );
            if !kinds_agree {
                return Err(CuError::from("Trajectory joint kinds disagree"));
            }
            let quintic = quintic::Quintic::new(
                [a.positions[j].si_value(), b.positions[j].si_value()],
                [a.velocities[j].si_value(), b.velocities[j].si_value()],
                [a.accelerations[j].si_value(), b.accelerations[j].si_value()],
                duration,
            );
            let (q, v, acceleration) = quintic.eval(u);
            if !q.is_finite() || !v.is_finite() || !acceleration.is_finite() {
                return Err(CuError::from("Non-finite trajectory knot"));
            }
            out_positions[j] = a.positions[j].with_si_value(q);
            out_velocities[j] = CuJointVelocity::from_coordinate(a.positions[j], v);
            if let Some(acc) = out_accelerations.as_deref_mut() {
                acc[j] = CuJointAcceleration::from_coordinate(a.positions[j], acceleration);
            }
        }
        Ok(())
    }
}

/// Writes zero acceleration (of each joint's unit kind) for a held setpoint.
fn hold_accelerations<const N: usize>(
    positions: &[CuJointPosition; N],
    accelerations: Option<&mut [CuJointAcceleration; N]>,
) {
    if let Some(acc) = accelerations {
        for j in 0..N {
            acc[j] = CuJointAcceleration::from_coordinate(positions[j], 0.0);
        }
    }
}

/// Linearly interpolates joint velocities of the same joint type.
fn lerp_velocity(
    a: CuJointVelocity,
    b: CuJointVelocity,
    fraction: f64,
) -> CuResult<CuJointVelocity> {
    if !fraction.is_finite() {
        return Err(CuError::from("Non-finite interpolation fraction"));
    }
    match (a, b) {
        (CuJointVelocity::Revolute(x), CuJointVelocity::Revolute(y)) => {
            let xv = x.get::<radian_per_second>();
            let yv = y.get::<radian_per_second>();
            Ok(CuJointVelocity::Revolute(AngularVelocity::new::<
                radian_per_second,
            >(
                xv + fraction * (yv - xv)
            )))
        }
        (CuJointVelocity::Prismatic(x), CuJointVelocity::Prismatic(y)) => {
            let xv = x.get::<meter_per_second>();
            let yv = y.get::<meter_per_second>();
            Ok(CuJointVelocity::Prismatic(
                Velocity::new::<meter_per_second>(xv + fraction * (yv - xv)),
            ))
        }
        (CuJointVelocity::Fixed, CuJointVelocity::Fixed) => Ok(CuJointVelocity::Fixed),
        _ => Err(CuError::from("Mismatched joint velocity types")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cu29::units::si::angle::radian;

    fn point(t: f64, q: f64, v: f64) -> CuTrajectoryPoint<1> {
        CuTrajectoryPoint {
            time_from_start: Time::new::<second>(t),
            positions: [CuJointPosition::Revolute(
                cu29::units::si::f64::Angle::new::<radian>(q),
            )],
            velocities: [CuJointVelocity::Revolute(AngularVelocity::new::<
                radian_per_second,
            >(v))],
            ..Default::default()
        }
    }

    fn two_point_traj() -> CuTrajectoryPayload<1> {
        let mut traj = CuTrajectoryPayload::new();
        traj.push(point(0.0, 0.0, 0.0)).unwrap();
        traj.push(point(2.0, 2.0, 4.0)).unwrap();
        traj
    }

    #[test]
    fn test_push_duration_and_points() {
        let traj = two_point_traj();
        assert_eq!(traj.len(), 2);
        assert!(!traj.is_empty());
        assert_eq!(traj.duration().get::<second>(), 2.0);
        assert_eq!(traj.points().len(), 2);
    }

    #[test]
    fn test_push_rejects_full_and_non_monotonic() {
        let mut traj = CuTrajectoryPayload::<1, 2>::new();
        traj.push(point(0.0, 0.0, 0.0)).unwrap();
        traj.push(point(1.0, 1.0, 1.0)).unwrap();
        assert!(traj.push(point(2.0, 2.0, 2.0)).is_err());
        let mut traj = CuTrajectoryPayload::<1>::new();
        traj.push(point(1.0, 0.0, 0.0)).unwrap();
        assert!(traj.push(point(1.0, 1.0, 1.0)).is_err());
        assert!(traj.push(point(0.5, 1.0, 1.0)).is_err());
        assert_eq!(traj.len(), 1);
        traj.clear();
        assert!(traj.is_empty());
        assert_eq!(traj.duration().get::<second>(), 0.0);
    }

    #[test]
    fn test_sample_endpoints_midpoint_and_clamping() {
        let traj = two_point_traj();
        let mut q = [CuJointPosition::default(); 1];
        let mut qd = [CuJointVelocity::default(); 1];
        traj.sample(Time::new::<second>(0.0), &mut q, &mut qd)
            .unwrap();
        assert_eq!(q[0].si_value(), 0.0);
        traj.sample(Time::new::<second>(1.0), &mut q, &mut qd)
            .unwrap();
        assert!((q[0].si_value() - 1.0).abs() < 1e-12);
        assert!((qd[0].si_value() - 2.0).abs() < 1e-12);
        traj.sample(Time::new::<second>(2.0), &mut q, &mut qd)
            .unwrap();
        assert_eq!(q[0].si_value(), 2.0);
        traj.sample(Time::new::<second>(-5.0), &mut q, &mut qd)
            .unwrap();
        assert_eq!(q[0].si_value(), 0.0);
        traj.sample(Time::new::<second>(99.0), &mut q, &mut qd)
            .unwrap();
        assert_eq!(q[0].si_value(), 2.0);
        assert_eq!(qd[0].si_value(), 4.0);
    }

    #[test]
    fn test_sample_empty_errors() {
        let traj = CuTrajectoryPayload::<1>::new();
        let mut q = [CuJointPosition::default(); 1];
        let mut qd = [CuJointVelocity::default(); 1];
        assert!(
            traj.sample(Time::new::<second>(0.0), &mut q, &mut qd)
                .is_err()
        );
    }

    #[test]
    fn test_sample_dispatches_on_payload_interpolation() {
        let mut traj = two_point_traj();
        let mut q = [CuJointPosition::default(); 1];
        let mut qd = [CuJointVelocity::default(); 1];
        let mut q2 = q;
        let mut qd2 = qd;
        let t = Time::new::<second>(1.0);
        for interpolation in [
            CuTrajectoryInterpolation::Linear,
            CuTrajectoryInterpolation::Cubic,
            CuTrajectoryInterpolation::Quintic,
        ] {
            traj.interpolation = interpolation;
            traj.sample(t, &mut q, &mut qd).unwrap();
            match interpolation {
                CuTrajectoryInterpolation::Linear => traj.sample_linear(t, &mut q2, &mut qd2),
                CuTrajectoryInterpolation::Cubic => traj.sample_cubic(t, &mut q2, &mut qd2),
                CuTrajectoryInterpolation::Quintic => traj.sample_quintic(t, &mut q2, &mut qd2),
            }
            .unwrap();
            assert_eq!((q, qd), (q2, qd2), "{interpolation:?}");
        }
    }

    #[test]
    fn test_bincode_round_trip_keeps_interpolation() {
        let mut traj = CuTrajectoryPayload::<1, 4>::new();
        traj.push(point(0.0, 0.0, 0.0)).unwrap();
        traj.push(point(1.0, 1.0, 0.0)).unwrap();
        traj.interpolation = CuTrajectoryInterpolation::Quintic;
        let mut bytes = [0u8; 1024];
        let len =
            bincode::encode_into_slice(&traj, &mut bytes, bincode::config::standard()).unwrap();
        let (decoded, _): (CuTrajectoryPayload<1, 4>, usize) =
            bincode::decode_from_slice(&bytes[..len], bincode::config::standard()).unwrap();
        assert_eq!(decoded, traj);
    }

    #[test]
    fn test_stretch_time_scales_and_limit_ratio_drops() {
        let mut traj = CuTrajectoryPayload::<1, 4>::new();
        traj.push(point(0.0, 0.0, 0.0)).unwrap();
        traj.push(point(1.0, 1.0, 0.0)).unwrap();
        let limits = CuJointTrajectoryLimits::new(
            [CuJointVelocity::Revolute(AngularVelocity::new::<
                radian_per_second,
            >(1.0))],
            [CuJointAcceleration::from_coordinate(
                traj.points()[0].positions[0],
                10.0,
            )],
        );
        let before = traj.quintic_limit_ratio(&limits).unwrap();
        traj.stretch_time(2.0).unwrap();
        assert_eq!(traj.duration().get::<second>(), 2.0);
        let after = traj.quintic_limit_ratio(&limits).unwrap();
        assert!(after < before);
        assert!(traj.stretch_time(0.0).is_err());
    }

    fn mixed_point(t: f64, q: [f64; 2], v: [f64; 2], a: [f64; 2]) -> CuTrajectoryPoint<2> {
        use cu29::units::si::acceleration::meter_per_second_squared;
        use cu29::units::si::angular_acceleration::radian_per_second_squared;
        use cu29::units::si::f64::{Acceleration, AngularAcceleration, Length, Velocity};
        use cu29::units::si::length::meter;
        CuTrajectoryPoint {
            time_from_start: Time::new::<second>(t),
            positions: [
                CuJointPosition::Revolute(cu29::units::si::f64::Angle::new::<radian>(q[0])),
                CuJointPosition::Prismatic(Length::new::<meter>(q[1])),
            ],
            velocities: [
                CuJointVelocity::Revolute(AngularVelocity::new::<radian_per_second>(v[0])),
                CuJointVelocity::Prismatic(Velocity::new::<meter_per_second>(v[1])),
            ],
            accelerations: [
                CuJointAcceleration::Revolute(
                    AngularAcceleration::new::<radian_per_second_squared>(a[0]),
                ),
                CuJointAcceleration::Prismatic(Acceleration::new::<meter_per_second_squared>(a[1])),
            ],
        }
    }

    fn three_knot_traj() -> CuTrajectoryPayload<2> {
        let mut traj = CuTrajectoryPayload::new();
        traj.push(mixed_point(0.0, [0.0, 0.1], [0.0, 0.0], [0.0, 0.0]))
            .unwrap();
        traj.push(mixed_point(1.0, [0.5, 0.3], [1.2, -0.4], [2.0, 0.7]))
            .unwrap();
        traj.push(mixed_point(2.5, [0.9, -0.2], [0.0, 0.0], [-1.0, 0.0]))
            .unwrap();
        traj
    }

    fn sample_all(
        traj: &CuTrajectoryPayload<2>,
        kind: CuTrajectoryInterpolation,
        t: f64,
    ) -> (
        [CuJointPosition; 2],
        [CuJointVelocity; 2],
        [CuJointAcceleration; 2],
    ) {
        let mut q = [CuJointPosition::default(); 2];
        let mut v = [CuJointVelocity::default(); 2];
        let mut a = [CuJointAcceleration::default(); 2];
        traj.sample_with_acceleration(kind, Time::new::<second>(t), &mut q, &mut v, &mut a)
            .unwrap();
        (q, v, a)
    }

    /// The reported acceleration is the derivative of the reported velocity
    /// (and, for the smooth kinds, the second derivative of position),
    /// checked here against high-resolution central differences.
    #[test]
    fn acceleration_matches_finite_difference_of_velocity_for_each_kind() {
        let traj = three_knot_traj();
        let h = 1e-6;
        for kind in [
            CuTrajectoryInterpolation::Linear,
            CuTrajectoryInterpolation::Cubic,
            CuTrajectoryInterpolation::Quintic,
        ] {
            let mut checked = 0;
            let mut t: f64 = 0.013;
            while t < 2.5 {
                // Stay clear of the knots, where one-sided values differ.
                if (t - 1.0).abs() > 1e-3 && (t - 2.5).abs() > 1e-3 {
                    let (_, _, acc) = sample_all(&traj, kind, t);
                    let (q_hi, v_hi, _) = sample_all(&traj, kind, t + h);
                    let (q_lo, v_lo, _) = sample_all(&traj, kind, t - h);
                    let (q_mid, _, _) = sample_all(&traj, kind, t);
                    for j in 0..2 {
                        let fd = (v_hi[j].si_value() - v_lo[j].si_value()) / (2.0 * h);
                        assert!(
                            (acc[j].si_value() - fd).abs() < 1e-5 * (1.0 + fd.abs()),
                            "{kind:?} t={t} j={j}: {} vs {fd}",
                            acc[j].si_value()
                        );
                        if kind != CuTrajectoryInterpolation::Linear {
                            // Smooth kinds: velocity is also d(position)/dt and
                            // acceleration the second difference of position.
                            let second_diff = (q_hi[j].si_value() - 2.0 * q_mid[j].si_value()
                                + q_lo[j].si_value())
                                / (h * h);
                            assert!(
                                (acc[j].si_value() - second_diff).abs() < 5e-2 * (1.0 + fd.abs()),
                                "{kind:?} t={t} j={j}: {} vs second diff {second_diff}",
                                acc[j].si_value()
                            );
                        }
                    }
                    checked += 1;
                }
                t += 0.037;
            }
            assert!(checked > 50);
        }
    }

    #[test]
    fn acceleration_units_follow_the_joint_kind() {
        let traj = three_knot_traj();
        for kind in [
            CuTrajectoryInterpolation::Linear,
            CuTrajectoryInterpolation::Cubic,
            CuTrajectoryInterpolation::Quintic,
        ] {
            let (_, _, acc) = sample_all(&traj, kind, 0.5);
            assert!(matches!(acc[0], CuJointAcceleration::Revolute(_)));
            assert!(matches!(acc[1], CuJointAcceleration::Prismatic(_)));
        }
    }

    #[test]
    fn quintic_acceleration_is_continuous_and_hits_the_knot_value() {
        let traj = three_knot_traj();
        let kind = CuTrajectoryInterpolation::Quintic;
        let (_, _, before) = sample_all(&traj, kind, 1.0 - 1e-9);
        let (_, _, after) = sample_all(&traj, kind, 1.0 + 1e-9);
        let (_, _, at) = sample_all(&traj, kind, 1.0);
        for j in 0..2 {
            let knot = traj.points()[1].accelerations[j].si_value();
            for acc in [before, after, at] {
                assert!((acc[j].si_value() - knot).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn linear_and_cubic_knot_acceleration_is_left_sided() {
        let traj = three_knot_traj();
        // Linear: at the knot the ending segment's constant (1.2 - 0) / 1.
        let (_, _, at) = sample_all(&traj, CuTrajectoryInterpolation::Linear, 1.0);
        let (_, _, after) = sample_all(&traj, CuTrajectoryInterpolation::Linear, 1.0 + 1e-9);
        assert!((at[0].si_value() - 1.2).abs() < 1e-12);
        assert!((after[0].si_value() - (0.0 - 1.2) / 1.5).abs() < 1e-12);
        // Cubic: equals the value just before the knot.
        let (_, _, at) = sample_all(&traj, CuTrajectoryInterpolation::Cubic, 1.0);
        let (_, _, before) = sample_all(&traj, CuTrajectoryInterpolation::Cubic, 1.0 - 1e-9);
        assert!((at[0].si_value() - before[0].si_value()).abs() < 1e-6);
    }

    #[test]
    fn held_endpoints_report_zero_acceleration() {
        let traj = three_knot_traj();
        for kind in [
            CuTrajectoryInterpolation::Linear,
            CuTrajectoryInterpolation::Cubic,
            CuTrajectoryInterpolation::Quintic,
        ] {
            for t in [-1.0, 0.0, 2.5, 9.0] {
                let (_, _, acc) = sample_all(&traj, kind, t);
                assert_eq!(acc[0].si_value(), 0.0, "{kind:?} t={t}");
                assert_eq!(acc[1].si_value(), 0.0, "{kind:?} t={t}");
                assert!(matches!(acc[1], CuJointAcceleration::Prismatic(_)));
            }
        }
        // A single-point trajectory holds its point.
        let mut one = CuTrajectoryPayload::<2>::new();
        one.push(mixed_point(0.0, [0.1, 0.2], [0.3, 0.4], [5.0, 6.0]))
            .unwrap();
        for kind in [
            CuTrajectoryInterpolation::Linear,
            CuTrajectoryInterpolation::Cubic,
            CuTrajectoryInterpolation::Quintic,
        ] {
            let (_, _, acc) = sample_all(&one, kind, 0.4);
            assert_eq!(acc[0].si_value(), 0.0);
        }
    }

    #[test]
    fn sample_with_acceleration_matches_sample_with_for_positions_and_velocities() {
        let traj = three_knot_traj();
        for kind in [
            CuTrajectoryInterpolation::Linear,
            CuTrajectoryInterpolation::Cubic,
            CuTrajectoryInterpolation::Quintic,
        ] {
            for t in [0.0, 0.3, 1.0, 1.7, 2.5, 3.0] {
                let (q, v, _) = sample_all(&traj, kind, t);
                let mut q2 = [CuJointPosition::default(); 2];
                let mut v2 = [CuJointVelocity::default(); 2];
                traj.sample_with(kind, Time::new::<second>(t), &mut q2, &mut v2)
                    .unwrap();
                assert_eq!(q, q2);
                assert_eq!(v, v2);
            }
        }
    }
}
