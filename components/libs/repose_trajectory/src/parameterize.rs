//! Synchronized time-parameterization for joint-space paths.
//!
//! [`parameterize`] converts an untimed waypoint sequence into a timestamped
//! [`CuTrajectoryPayload`] with a selectable [`CuTrajectoryProfile`] family
//! (trapezoidal, jerk-limited S-curve, minimum-jerk quintic, cubic Hermite)
//! under the per-joint [`CuJointTrajectoryLimits`] of a [`CuTimingParams`].
//! Per-call [`CuTimingOverrides`] fix segment durations and waypoint
//! velocities; the default overrides give synchronized rest-to-rest timing.

use bincode::Decode;
use bincode::Encode;
use cu29::prelude::*;
use cu29::units::si::f64::Time;
use cu29::units::si::time::second;
use repose_kinematics::CuJointAcceleration;
use repose_kinematics::CuJointJerk;
use repose_kinematics::CuJointPosition;
use repose_kinematics::CuJointVelocity;
use serde::Deserialize;
use serde::Serialize;

use crate::CuTrajectoryInterpolation;
use crate::CuTrajectoryPayload;
use crate::CuTrajectoryPoint;
use crate::quintic::polynomial_extrema;

/// Per-joint speed, acceleration, and jerk limits for time-parameterization.
///
/// Every entry uses the typed joint representation, so revolute (`rad/s`,
/// `rad/s^2`, `rad/s^3`), prismatic (`m/s`, `m/s^2`, `m/s^3`), and fixed joints
/// each carry their own unit; the entry kind must match the joint kind of the
/// parameterized waypoints. `max_jerks` is only used by
/// [`CuTrajectoryProfile::SCurve`]; `None` (the default, and an omitted RON
/// key) means "no jerk bound" and lets the S-curve degrade gracefully to
/// trapezoidal timing and shape. A single `f64::INFINITY` entry leaves that
/// joint unbounded.
///
/// ```ron
/// {
///     "max_velocities": [{"revolute": 0.8}, {"prismatic": 0.2}],
///     "max_accelerations": [{"revolute": 2.0}, {"prismatic": 1.0}],
///     "max_jerks": [{"revolute": 10.0}, {"prismatic": 5.0}], // optional
/// }
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Encode, Decode, Reflect)]
#[reflect(no_field_bounds, from_reflect = false)]
#[serde(deny_unknown_fields)]
pub struct CuJointTrajectoryLimits<const N: usize> {
    #[serde(with = "repose_kinematics::serde_arrays::velocity_array")]
    pub max_velocities: [CuJointVelocity; N],
    #[serde(with = "repose_kinematics::serde_arrays::acceleration_array")]
    pub max_accelerations: [CuJointAcceleration; N],
    /// Per-joint jerk bounds; `None` means unbounded.
    #[serde(default, with = "crate::serde_arrays::option_jerk_array")]
    pub max_jerks: Option<[CuJointJerk; N]>,
}

impl<const N: usize> Default for CuJointTrajectoryLimits<N> {
    /// Zero limits: a placeholder that [`CuJointTrajectoryLimits::validate`]
    /// rejects, since limits have no sensible default.
    fn default() -> Self {
        Self::new(
            [CuJointVelocity::default(); N],
            [CuJointAcceleration::default(); N],
        )
    }
}

impl<const N: usize> CuJointTrajectoryLimits<N> {
    /// Creates limits from per-joint typed velocities and accelerations with
    /// no jerk bound; add one with [`CuJointTrajectoryLimits::with_jerks`].
    pub const fn new(
        max_velocities: [CuJointVelocity; N],
        max_accelerations: [CuJointAcceleration; N],
    ) -> Self {
        Self {
            max_velocities,
            max_accelerations,
            max_jerks: None,
        }
    }

    /// Sets per-joint jerk bounds. Each entry must be positive (finite or
    /// `f64::INFINITY`) for revolute and prismatic joints; fixed-joint entries
    /// are ignored.
    pub const fn with_jerks(mut self, max_jerks: [CuJointJerk; N]) -> Self {
        self.max_jerks = Some(max_jerks);
        self
    }

    /// SI acceleration bound of joint `j`.
    pub(crate) fn acc(&self, j: usize) -> f64 {
        self.max_accelerations[j].si_value()
    }

    /// SI jerk bound of joint `j`; `f64::INFINITY` when unbounded.
    pub(crate) fn jerk(&self, j: usize) -> f64 {
        match self.max_jerks {
            Some(jerks) => match jerks[j] {
                CuJointJerk::Fixed => f64::INFINITY,
                jerk => jerk.si_value(),
            },
            None => f64::INFINITY,
        }
    }

    /// Checks entry kinds against each other and that every non-fixed joint
    /// has a positive finite velocity and acceleration bound and, when given,
    /// a positive (possibly infinite) jerk bound. The joint kind is taken from
    /// `max_velocities`.
    pub fn validate(&self) -> CuResult<()> {
        for j in 0..N {
            let vel = self.max_velocities[j];
            let acc = self.max_accelerations[j];
            let jerk = self.max_jerks.map(|jerks| jerks[j]);
            let kinds_agree = match vel {
                CuJointVelocity::Revolute(_) => {
                    matches!(acc, CuJointAcceleration::Revolute(_))
                        && jerk.is_none_or(|k| matches!(k, CuJointJerk::Revolute(_)))
                }
                CuJointVelocity::Prismatic(_) => {
                    matches!(acc, CuJointAcceleration::Prismatic(_))
                        && jerk.is_none_or(|k| matches!(k, CuJointJerk::Prismatic(_)))
                }
                CuJointVelocity::Fixed => true,
            };
            if !kinds_agree {
                return Err(CuError::from(
                    "Trajectory limit kinds disagree across velocity, acceleration and jerk",
                ));
            }
            if matches!(vel, CuJointVelocity::Fixed) {
                continue;
            }
            let (v, a) = (vel.si_value(), acc.si_value());
            if !v.is_finite() || v <= 0.0 || !a.is_finite() || a <= 0.0 {
                return Err(CuError::from(
                    "Trajectory velocity and acceleration limits must be positive and finite",
                ));
            }
            let k = self.jerk(j);
            if k.is_nan() || k <= 0.0 {
                return Err(CuError::from("Trajectory jerk limits must be positive"));
            }
        }
        Ok(())
    }
}

/// Closed set of time-parameterization profile families.
///
/// Deserialized from snake-case RON (`trapezoidal`, `s_curve`,
/// `minimum_jerk`, `cubic`) with no stringly-typed dispatch.
#[derive(
    Default, Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Encode, Decode, Reflect,
)]
#[serde(rename_all = "snake_case")]
pub enum CuTrajectoryProfile {
    /// Synchronized trapezoidal (or triangular) rest-to-rest profile per
    /// segment.
    #[default]
    Trapezoidal,
    /// Jerk-limited symmetric 7-segment S-curve rest-to-rest per segment,
    /// synchronized across joints by time-scaling (velocity scales by
    /// `s <= 1`, acceleration by `s^2`, jerk by `s^3`, so the scaled profile
    /// still honors every bound). Infinite jerk entries reproduce the
    /// trapezoidal shape and timing exactly.
    SCurve,
    /// Quintic minimum-jerk polynomial per segment per joint with zero
    /// endpoint accelerations, honoring the specified waypoint velocities.
    MinimumJerk,
    /// Cubic Hermite polynomial per segment per joint honoring the specified
    /// waypoint velocities exactly.
    Cubic,
}

fn default_sample_period() -> Time {
    Time::new::<second>(0.02)
}

/// Timing parameters shared by [`parameterize`] and
/// [`parameterize_cartesian_path`](crate::parameterize_cartesian_path).
///
/// `limits` is required (it has no sensible default); `profile` defaults to
/// trapezoidal and `sample_period` to 0.02 s.
///
/// ```ron
/// {
///     "limits": {
///         "max_velocities": [{"revolute": 0.8}],
///         "max_accelerations": [{"revolute": 2.0}],
///     },
///     "profile": "s_curve",
///     "sample_period": 0.01,
/// }
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Reflect)]
#[reflect(no_field_bounds, from_reflect = false)]
// Field-level defaults: serde does not support container `default` on const-generic structs.
#[serde(deny_unknown_fields)]
pub struct CuTimingParams<const N: usize> {
    /// Per-joint velocity, acceleration and optional jerk limits.
    pub limits: CuJointTrajectoryLimits<N>,
    /// Profile family of every segment (joint space) or move (Cartesian path).
    #[serde(default)]
    pub profile: CuTrajectoryProfile,
    /// Period at which the timed motion is resampled into the payload.
    #[serde(default = "default_sample_period")]
    pub sample_period: Time,
}

impl<const N: usize> Default for CuTimingParams<N> {
    fn default() -> Self {
        Self {
            limits: CuJointTrajectoryLimits::default(),
            profile: CuTrajectoryProfile::default(),
            sample_period: default_sample_period(),
        }
    }
}

impl<const N: usize> CuTimingParams<N> {
    /// Checks the limits and that `sample_period` is positive and finite.
    pub fn validate(&self) -> CuResult<()> {
        self.limits.validate()?;
        let dt = self.sample_period.get::<second>();
        if !dt.is_finite() || dt <= 0.0 {
            return Err(CuError::from(
                "Trajectory sample period must be positive and finite",
            ));
        }
        Ok(())
    }
}

/// Per-call segment timing and waypoint velocities for [`parameterize`].
///
/// `segment_durations`, when present, must hold exactly
/// `waypoints.len() - 1` positive finite durations; the trajectory total is
/// their exact sum. `waypoint_velocities`, when present, must hold exactly
/// `waypoints.len()` entries aligned with the waypoints (endpoints
/// included); `None` means zero velocity at every waypoint. The default is
/// synchronized rest-to-rest timing.
#[derive(Default, Debug, Clone, Copy)]
pub struct CuTimingOverrides<'a, const N: usize> {
    pub segment_durations: Option<&'a [Time]>,
    pub waypoint_velocities: Option<&'a [[CuJointVelocity; N]]>,
}

/// Outcome of [`parameterize`] and
/// [`parameterize_cartesian_path`](crate::parameterize_cartesian_path).
///
/// The Cartesian path leaves its output empty unless `Success`.
#[derive(
    Default, Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Encode, Decode, Reflect,
)]
#[serde(rename_all = "snake_case")]
pub enum CuTimingStatus {
    /// Trajectory written to `out`, ending at the final waypoint.
    Success,
    /// No waypoints were provided; `out` is left empty.
    #[default]
    EmptyPath,
    /// A limit, a fixed segment duration, a waypoint velocity, the sample
    /// period, or the path is non-positive, non-finite, mismatched to the
    /// joint kinds, or inconsistent with the profile (non-zero waypoint
    /// velocity under a rest-to-rest profile, fixed duration below the
    /// synchronized minimum, segment velocity above a limit, wrong override
    /// slice length, path outside joint limits, or unreachable speeds).
    InvalidInput,
    /// `out` filled before the resampled trajectory completed.
    CapacityExceeded,
    /// The interpolated Cartesian path collides between knots `sample` and
    /// `sample + 1`.
    Collision { sample: usize },
    /// The interpolated Cartesian path leaves this waypoint's TCP zone.
    ZoneExceeded { waypoint: usize },
    /// The Cartesian path cannot keep moving through this waypoint's zone.
    NoContinuousMotion { waypoint: usize },
    /// Every joint reverses at this non-stop Cartesian path knot outside any
    /// zone.
    UnexpectedStop { sample: usize },
}

impl CuTimingStatus {
    /// Returns `true` when the trajectory was fully parameterized.
    pub const fn is_success(&self) -> bool {
        matches!(self, Self::Success)
    }
}

/// Times a joint-space waypoint path, writing timestamped points into `out`
/// in place without allocating.
///
/// Only fixed-size stack buffers are used; nothing is heap-allocated. `out`
/// gets [`CuTrajectoryInterpolation::Cubic`]: the stored points carry exact
/// positions and velocities (accelerations are stored as zero), and the
/// velocity limits of the polynomial profiles are certified on the cubic
/// Hermite segments.
///
/// # Profiles
///
/// - `Trapezoidal`: each segment duration is the maximum over joints of the
///   trapezoidal (triangular when the displacement is too short to reach full
///   speed) time under the velocity/acceleration limits, so every joint has
///   zero velocity at each waypoint. Requires zero waypoint velocities (any
///   non-zero entry is [`CuTimingStatus::InvalidInput`]). A fixed segment
///   duration must be at least the synchronized minimum for that segment
///   (else `InvalidInput`); the profile is stretched onto it exactly, lowering
///   peak velocity accordingly.
/// - `SCurve`: jerk-limited symmetric 7-segment rest-to-rest S-curve per
///   segment, synchronized across joints. Requires zero waypoint velocities.
///   The segment duration derives from the velocity, acceleration, and jerk
///   limits (infinite jerk reproduces trapezoidal timing); a fixed duration
///   at least the minimum is honored exactly (else `InvalidInput`).
/// - `MinimumJerk`: quintic minimum-jerk polynomial per segment per joint
///   with zero endpoint accelerations, honoring the specified waypoint
///   velocities. Duration: fixed when given, otherwise derived so zero-via
///   peak velocity (`|v| <= 1.875 * distance / duration`) and acceleration
///   (`|a| <= (10 / sqrt(3)) * distance / duration^2`) satisfy the limits;
///   with any non-zero waypoint velocity the derived duration falls back to
///   the trapezoidal minimum. The exact velocity extrema of every segment,
///   waypoint velocities included, are validated against the limits
///   (exceeding is `InvalidInput`).
/// - `Cubic`: cubic Hermite polynomial per segment per joint honoring the
///   specified waypoint velocities exactly. Duration: fixed when given,
///   otherwise the trapezoidal minimum, stretched for rest-to-rest segments
///   so the cubic peak `1.5 * distance / duration` stays within the velocity
///   limit (with non-zero waypoint velocities the trapezoidal minimum is
///   used as is); the exact velocity extrema of every
///   segment, waypoint velocities included, are validated against the limits
///   (exceeding is `InvalidInput`).
///
/// # Guarantees (all profiles)
///
/// The first point is at `t = 0` at `waypoints[0]` with its specified
/// velocity, the last point is exactly at the final waypoint with its
/// specified velocity, times are strictly increasing, and the total duration
/// equals the sum of the segment durations. Points are resampled at
/// `params.sample_period`. An empty path yields `EmptyPath`, a single
/// waypoint yields one point (`Success`), and a full buffer yields
/// `CapacityExceeded`.
///
/// # Example
///
/// ```rust
/// use repose_kinematics::CuJointAcceleration;
/// use repose_kinematics::CuJointPosition;
/// use repose_kinematics::CuJointVelocity;
/// use repose_trajectory::CuJointTrajectoryLimits;
/// use repose_trajectory::CuTimingOverrides;
/// use repose_trajectory::CuTimingParams;
/// use repose_trajectory::CuTimingStatus;
/// use repose_trajectory::CuTrajectoryPayload;
/// use repose_trajectory::parameterize;
/// use cu29::units::si::angle::radian;
/// use cu29::units::si::angular_acceleration::radian_per_second_squared;
/// use cu29::units::si::angular_velocity::radian_per_second;
/// use cu29::units::si::f64::Angle;
/// use cu29::units::si::f64::AngularAcceleration;
/// use cu29::units::si::f64::AngularVelocity;
/// use cu29::units::si::f64::Time;
/// use cu29::units::si::time::second;
///
/// let waypoints = [
///     [CuJointPosition::Revolute(Angle::new::<radian>(0.0))],
///     [CuJointPosition::Revolute(Angle::new::<radian>(1.0))],
/// ];
/// let params = CuTimingParams {
///     limits: CuJointTrajectoryLimits::new(
///         [CuJointVelocity::Revolute(AngularVelocity::new::<radian_per_second>(0.5))],
///         [CuJointAcceleration::Revolute(
///             AngularAcceleration::new::<radian_per_second_squared>(1.0),
///         )],
///     ),
///     sample_period: Time::new::<second>(0.02),
///     ..Default::default()
/// };
/// let mut out = CuTrajectoryPayload::<1, 512>::new();
/// let status = parameterize(&waypoints, &params, CuTimingOverrides::default(), &mut out).unwrap();
/// assert_eq!(status, CuTimingStatus::Success);
/// ```
pub fn parameterize<const N: usize, const MAX_POINTS: usize>(
    waypoints: &[[CuJointPosition; N]],
    params: &CuTimingParams<N>,
    overrides: CuTimingOverrides<'_, N>,
    out: &mut CuTrajectoryPayload<N, MAX_POINTS>,
) -> CuResult<CuTimingStatus> {
    out.clear();
    out.interpolation = CuTrajectoryInterpolation::Cubic;
    let limits = &params.limits;
    let profile = params.profile;
    let timing = &overrides;
    if waypoints.is_empty() {
        return Ok(CuTimingStatus::EmptyPath);
    }
    let dt = params.sample_period.get::<second>();
    if !dt.is_finite() || dt <= 0.0 {
        return Ok(CuTimingStatus::InvalidInput);
    }
    validate_waypoints(waypoints)?;
    if !limits_valid(waypoints[0], limits) {
        return Ok(CuTimingStatus::InvalidInput);
    }
    if profile == CuTrajectoryProfile::SCurve && !jerks_valid(waypoints[0], limits) {
        return Ok(CuTimingStatus::InvalidInput);
    }
    let last = waypoints.len() - 1;
    if let Some(durations) = timing.segment_durations {
        if durations.len() != last {
            return Ok(CuTimingStatus::InvalidInput);
        }
        let mut s = 0;
        while s < last {
            let span = durations[s].get::<second>();
            if !span.is_finite() || span <= 0.0 {
                return Ok(CuTimingStatus::InvalidInput);
            }
            s += 1;
        }
    }
    if timing.waypoint_velocities.is_some_and(|vias| {
        vias.len() != waypoints.len()
            || !vias_valid(waypoints, vias)
            || !vias_within_limits(waypoints, vias, limits)
    }) {
        return Ok(CuTimingStatus::InvalidInput);
    }
    let rest_only = matches!(
        profile,
        CuTrajectoryProfile::Trapezoidal | CuTrajectoryProfile::SCurve
    );
    let all_zero_via = !vias_nonzero(timing);
    if rest_only && !all_zero_via {
        return Ok(CuTimingStatus::InvalidInput);
    }
    if waypoints.len() == 1 {
        let solo = waypoint_velocity(timing, 0, waypoints[0]);
        return push_single_point(out, Time::default(), waypoints[0], solo);
    }
    let quintic = profile == CuTrajectoryProfile::MinimumJerk;
    let check_poly_limits = matches!(
        profile,
        CuTrajectoryProfile::MinimumJerk | CuTrajectoryProfile::Cubic
    );
    let mut total = 0.0;
    let mut first_duration = 0.0;
    let mut s = 0;
    while s < last {
        let duration = match timing.segment_durations {
            Some(fixed) => {
                let exact = fixed[s].get::<second>();
                if rest_only {
                    let minimum = segment_minimum(
                        &waypoints[s],
                        &waypoints[s + 1],
                        limits,
                        profile,
                        all_zero_via,
                    );
                    if exact < minimum && minimum - exact > 1e-9 * minimum.max(1.0) {
                        return Ok(CuTimingStatus::InvalidInput);
                    }
                }
                exact
            }
            None => segment_minimum(
                &waypoints[s],
                &waypoints[s + 1],
                limits,
                profile,
                all_zero_via,
            ),
        };
        if check_poly_limits
            && duration > 0.0
            && !segment_within_limits(
                &waypoints[s],
                &waypoints[s + 1],
                s,
                duration,
                limits,
                quintic,
                timing,
            )
        {
            return Ok(CuTimingStatus::InvalidInput);
        }
        if s == 0 {
            first_duration = duration;
        }
        total += duration;
        s += 1;
    }
    if total <= 0.0 {
        let solo = waypoint_velocity(timing, 0, waypoints[0]);
        return push_single_point(out, Time::default(), waypoints[0], solo);
    }
    let zero_acc = zero_accelerations(waypoints[0]);
    if !try_push(
        out,
        CuTrajectoryPoint {
            time_from_start: Time::default(),
            positions: waypoints[0],
            velocities: waypoint_velocity(timing, 0, waypoints[0]),
            accelerations: zero_acc,
        },
    ) {
        return Ok(CuTimingStatus::CapacityExceeded);
    }
    let mut cursor = SegmentCursor::new(
        waypoints,
        limits,
        profile,
        timing,
        all_zero_via,
        first_duration,
    );
    let mut k = 1u64;
    loop {
        let t = k as f64 * dt;
        if t >= total {
            break;
        }
        let (positions, velocities) = cursor.eval_at(t);
        if !try_push(
            out,
            CuTrajectoryPoint {
                time_from_start: Time::new::<second>(t),
                positions,
                velocities,
                accelerations: zero_acc,
            },
        ) {
            return Ok(CuTimingStatus::CapacityExceeded);
        }
        k += 1;
    }
    if !try_push(
        out,
        CuTrajectoryPoint {
            time_from_start: Time::new::<second>(total),
            positions: waypoints[last],
            velocities: waypoint_velocity(timing, last, waypoints[last]),
            accelerations: zero_acc,
        },
    ) {
        return Ok(CuTimingStatus::CapacityExceeded);
    }
    Ok(CuTimingStatus::Success)
}

/// Checks that all waypoints share joint kinds and hold finite coordinates.
fn validate_waypoints<const N: usize>(waypoints: &[[CuJointPosition; N]]) -> CuResult<()> {
    let mut i = 0;
    while i < waypoints.len() {
        let mut j = 0;
        while j < N {
            if core::mem::discriminant(&waypoints[i][j])
                != core::mem::discriminant(&waypoints[0][j])
                || !waypoints[i][j].si_value().is_finite()
            {
                return Err(CuError::from(
                    "Waypoint joint kinds must match and be finite",
                ));
            }
            j += 1;
        }
        i += 1;
    }
    Ok(())
}

/// Checks limit positivity, finiteness, and kind agreement with the joints.
fn limits_valid<const N: usize>(
    reference: [CuJointPosition; N],
    limits: &CuJointTrajectoryLimits<N>,
) -> bool {
    let mut j = 0;
    while j < N {
        let acc = limits.acc(j);
        if !acc.is_finite() {
            return false;
        }
        let vel = limits.max_velocities[j];
        if core::mem::discriminant(&vel)
            != core::mem::discriminant(&coordinate_velocity_kind(reference[j]))
            || core::mem::discriminant(&limits.max_accelerations[j])
                != core::mem::discriminant(&CuJointAcceleration::from_coordinate(reference[j], 0.0))
        {
            return false;
        }
        match reference[j] {
            CuJointPosition::Fixed => {
                if acc < 0.0 {
                    return false;
                }
            }
            _ => {
                if !vel.si_value().is_finite() || vel.si_value() <= 0.0 || acc <= 0.0 {
                    return false;
                }
            }
        }
        j += 1;
    }
    true
}

/// Checks jerk kind and positivity for joints that move; fixed-joint entries
/// are ignored and `None` limits are always valid.
///
/// Each entry must be strictly positive; `f64::INFINITY` is explicitly
/// allowed and means "no jerk bound" (the S-curve then coincides with the
/// trapezoid). NaN, non-positive, and negative-infinite entries fail.
fn jerks_valid<const N: usize>(
    reference: [CuJointPosition; N],
    limits: &CuJointTrajectoryLimits<N>,
) -> bool {
    let mut j = 0;
    while j < N {
        if matches!(reference[j], CuJointPosition::Fixed) {
            j += 1;
            continue;
        }
        if let Some(jerks) = limits.max_jerks
            && core::mem::discriminant(&jerks[j])
                != core::mem::discriminant(&CuJointJerk::from_coordinate(reference[j], 0.0))
        {
            return false;
        }
        let jerk = limits.jerk(j);
        if jerk <= 0.0 || jerk.is_nan() {
            return false;
        }
        j += 1;
    }
    true
}

/// Checks waypoint-velocity kind agreement and finiteness against the joints.
fn vias_valid<const N: usize>(
    waypoints: &[[CuJointPosition; N]],
    vias: &[[CuJointVelocity; N]],
) -> bool {
    let mut i = 0;
    while i < waypoints.len() {
        let mut j = 0;
        while j < N {
            if core::mem::discriminant(&vias[i][j])
                != core::mem::discriminant(&coordinate_velocity_kind(waypoints[i][j]))
                || !vias[i][j].si_value().is_finite()
            {
                return false;
            }
            j += 1;
        }
        i += 1;
    }
    true
}

/// Returns `true` when any specified waypoint velocity is non-zero.
fn vias_nonzero<const N: usize>(timing: &CuTimingOverrides<'_, N>) -> bool {
    match timing.waypoint_velocities {
        Some(vias) => {
            let mut i = 0;
            while i < vias.len() {
                let mut j = 0;
                while j < N {
                    if vias[i][j].si_value() != 0.0 {
                        return true;
                    }
                    j += 1;
                }
                i += 1;
            }
            false
        }
        None => false,
    }
}

/// Resolves the specified velocity at waypoint `index`, defaulting to rest.
fn waypoint_velocity<const N: usize>(
    timing: &CuTimingOverrides<'_, N>,
    index: usize,
    reference: [CuJointPosition; N],
) -> [CuJointVelocity; N] {
    match timing.waypoint_velocities {
        Some(vias) => vias[index],
        None => core::array::from_fn(|j| CuJointVelocity::from_coordinate(reference[j], 0.0)),
    }
}

/// Zero accelerations of the joint kinds of `reference`.
fn zero_accelerations<const N: usize>(reference: [CuJointPosition; N]) -> [CuJointAcceleration; N] {
    reference.map(|q| CuJointAcceleration::from_coordinate(q, 0.0))
}

/// Maps a joint kind to the matching zero-magnitude velocity variant.
fn coordinate_velocity_kind(q: CuJointPosition) -> CuJointVelocity {
    CuJointVelocity::from_coordinate(q, 0.0)
}

/// Trapezoidal (or triangular) traversal time for one joint displacement.
fn joint_profile_duration(distance: f64, max_vel: f64, max_acc: f64) -> f64 {
    if distance <= 0.0 {
        return 0.0;
    }
    let accel_time = max_vel / max_acc;
    if distance >= max_vel * accel_time {
        distance / max_vel + accel_time
    } else {
        2.0 * libm::sqrt(distance / max_acc)
    }
}

/// Rest-to-rest cubic traversal time for one joint displacement: the
/// trapezoidal time, stretched to `1.5 * distance / max_vel` when the cubic
/// peak would otherwise exceed the velocity limit (`distance > 2 v^2 / a`).
fn cubic_joint_duration(distance: f64, max_vel: f64, max_acc: f64) -> f64 {
    if distance <= 0.0 {
        return 0.0;
    }
    let accel_time = max_vel / max_acc;
    if distance >= max_vel * accel_time {
        let cruise = distance / max_vel;
        (cruise + accel_time).max(1.5 * cruise)
    } else {
        // Triangular: 2 * sqrt(d / a) <= d / v + v / a, and the cubic peak
        // 1.5 * d / T stays below v whenever d < 2 v^2 / a.
        2.0 * libm::sqrt(distance / max_acc)
    }
}

/// Jerk-limited S-curve traversal time for one joint displacement.
///
/// With infinite jerk this is exactly the trapezoidal time; with finite jerk
/// it is the symmetric 7-segment duration from [`scurve_shape`].
fn scurve_joint_duration(distance: f64, max_vel: f64, max_acc: f64, max_jerk: f64) -> f64 {
    if distance <= 0.0 {
        return 0.0;
    }
    if !max_jerk.is_finite() {
        return joint_profile_duration(distance, max_vel, max_acc);
    }
    let (jerk_time, accel_time, cruise_time) = scurve_shape(distance, max_vel, max_acc, max_jerk);
    4.0 * jerk_time + 2.0 * accel_time + cruise_time
}

/// Solves the symmetric rest-to-rest 7-segment S-curve phase durations.
///
/// Returns `(jerk_time, accel_time, cruise_time)` for the phase sequence
/// `+J, 0, -J, cruise, -J, 0, +J`. Assumes `distance > 0` with finite
/// positive limits. Peak velocity, acceleration, and jerk never exceed the
/// given bounds: full limits are reached only when the displacement admits
/// them, otherwise the cruise, the constant-acceleration phase, and finally
/// the jerk time shrink in that order.
fn scurve_shape(distance: f64, max_vel: f64, max_acc: f64, max_jerk: f64) -> (f64, f64, f64) {
    let jerk_full = max_acc / max_jerk;
    let accel_full = max_vel / max_acc - jerk_full;
    if accel_full >= 0.0 {
        let reach = max_vel * (accel_full + 2.0 * jerk_full);
        if distance >= reach {
            return (jerk_full, accel_full, (distance - reach) / max_vel);
        }
        let discriminant = jerk_full * jerk_full + 4.0 * distance / max_acc;
        let accel = 0.5 * (libm::sqrt(discriminant) - 3.0 * jerk_full);
        if accel >= 0.0 {
            return (jerk_full, accel, 0.0);
        }
        return (libm::cbrt(distance / (2.0 * max_jerk)), 0.0, 0.0);
    }
    let jerk = libm::sqrt(max_vel / max_jerk);
    if distance >= 2.0 * max_vel * jerk {
        return (jerk, 0.0, distance / max_vel - 2.0 * jerk);
    }
    (libm::cbrt(distance / (2.0 * max_jerk)), 0.0, 0.0)
}

/// Evaluates a quintic minimum-jerk segment at normalized time `u`.
///
/// Boundary conditions are `q(0) = q0`, `q(1) = q1`, `dq(0) = v0`,
/// `dq(1) = v1` (SI units per second), with zero endpoint accelerations.
/// Positions and velocities are absolute SI values, not offsets.
///
/// Shared with the Cartesian path timing.
pub(crate) fn quintic_eval(
    u: f64,
    q0: f64,
    q1: f64,
    v0: f64,
    v1: f64,
    duration: f64,
) -> (f64, f64) {
    if duration <= 0.0 {
        return (q0, v0);
    }
    let distance = q1 - q0;
    let start_term = distance - v0 * duration;
    let end_term = (v1 - v0) * duration;
    let a3 = 10.0 * start_term - 4.0 * end_term;
    let a4 = -15.0 * start_term + 7.0 * end_term;
    let a5 = 6.0 * start_term - 3.0 * end_term;
    let u2 = u * u;
    let u3 = u2 * u;
    let u4 = u3 * u;
    let position = q0 + v0 * duration * u + a3 * u3 + a4 * u4 + a5 * u4 * u;
    let velocity = v0 + (3.0 * a3 * u2 + 4.0 * a4 * u3 + 5.0 * a5 * u4) / duration;
    (position, velocity)
}

/// Evaluates a cubic Hermite segment at normalized time `u`.
///
/// Interpolates `q(0) = q0`, `q(1) = q1` with endpoint velocities `v0`, `v1`
/// (SI units per second). Positions and velocities are absolute SI values.
///
/// Shared with the Cartesian path timing.
pub(crate) fn cubic_eval(u: f64, q0: f64, q1: f64, v0: f64, v1: f64, duration: f64) -> (f64, f64) {
    if duration <= 0.0 {
        return (q0, v0);
    }
    let u2 = u * u;
    let u3 = u2 * u;
    let h00 = 2.0 * u3 - 3.0 * u2 + 1.0;
    let h10 = u3 - 2.0 * u2 + u;
    let h01 = -2.0 * u3 + 3.0 * u2;
    let h11 = u3 - u2;
    let position = h00 * q0 + h10 * duration * v0 + h01 * q1 + h11 * duration * v1;
    let dh00 = 6.0 * u2 - 6.0 * u;
    let dh10 = 3.0 * u2 - 4.0 * u + 1.0;
    let dh01 = -6.0 * u2 + 6.0 * u;
    let dh11 = 3.0 * u2 - 2.0 * u;
    let velocity = (dh00 * q0 + dh10 * duration * v0 + dh01 * q1 + dh11 * duration * v1) / duration;
    (position, velocity)
}

/// Synchronized minimum duration for one segment under the profile.
///
/// Trapezoidal and cubic use the slowest joint's trapezoidal time. S-curve
/// uses the slowest joint's jerk-limited time. Minimum-jerk uses the slowest
/// joint's quintic rest-to-rest time when every waypoint velocity is zero,
/// and the trapezoidal minimum otherwise.
fn segment_minimum<const N: usize>(
    from: &[CuJointPosition; N],
    to: &[CuJointPosition; N],
    limits: &CuJointTrajectoryLimits<N>,
    profile: CuTrajectoryProfile,
    all_zero_via: bool,
) -> f64 {
    match profile {
        CuTrajectoryProfile::MinimumJerk if all_zero_via => {
            minjerk_segment_duration(from, to, limits)
        }
        CuTrajectoryProfile::Cubic if all_zero_via => cubic_segment_duration(from, to, limits),
        CuTrajectoryProfile::MinimumJerk | CuTrajectoryProfile::Cubic => {
            segment_duration(from, to, limits)
        }
        CuTrajectoryProfile::Trapezoidal => segment_duration(from, to, limits),
        CuTrajectoryProfile::SCurve => scurve_segment_duration(from, to, limits),
    }
}

/// Segment duration synchronized across joints (slowest joint dominates).
fn segment_duration<const N: usize>(
    from: &[CuJointPosition; N],
    to: &[CuJointPosition; N],
    limits: &CuJointTrajectoryLimits<N>,
) -> f64 {
    let mut duration = 0.0;
    let mut j = 0;
    while j < N {
        let distance = (to[j].si_value() - from[j].si_value()).abs();
        let leg =
            joint_profile_duration(distance, limits.max_velocities[j].si_value(), limits.acc(j));
        if leg > duration {
            duration = leg;
        }
        j += 1;
    }
    duration
}

/// Rest-to-rest cubic Hermite segment duration synchronized across joints.
///
/// A rest-to-rest cubic peaks at `1.5 * distance / duration` mid-segment, so
/// the trapezoidal minimum alone exceeds the velocity limit once a joint
/// travels more than `2 * v^2 / a`. The duration is the trapezoidal minimum,
/// stretched just enough to keep that peak at the velocity limit.
fn cubic_segment_duration<const N: usize>(
    from: &[CuJointPosition; N],
    to: &[CuJointPosition; N],
    limits: &CuJointTrajectoryLimits<N>,
) -> f64 {
    let mut duration = 0.0;
    let mut j = 0;
    while j < N {
        let distance = (to[j].si_value() - from[j].si_value()).abs();
        let leg =
            cubic_joint_duration(distance, limits.max_velocities[j].si_value(), limits.acc(j));
        if leg > duration {
            duration = leg;
        }
        j += 1;
    }
    duration
}

/// S-curve segment duration synchronized across joints.
fn scurve_segment_duration<const N: usize>(
    from: &[CuJointPosition; N],
    to: &[CuJointPosition; N],
    limits: &CuJointTrajectoryLimits<N>,
) -> f64 {
    let mut duration = 0.0;
    let mut j = 0;
    while j < N {
        let distance = (to[j].si_value() - from[j].si_value()).abs();
        let leg = scurve_joint_duration(
            distance,
            limits.max_velocities[j].si_value(),
            limits.acc(j),
            limits.jerk(j),
        );
        if leg > duration {
            duration = leg;
        }
        j += 1;
    }
    duration
}

/// Quintic rest-to-rest segment duration synchronized across joints.
///
/// A rest-to-rest quintic peaks at `|v| = 1.875 * distance / duration` and
/// `|a| = (10 / sqrt(3)) * distance / duration^2`, so the smallest duration
/// honoring the velocity and acceleration limits is their maximum.
fn minjerk_segment_duration<const N: usize>(
    from: &[CuJointPosition; N],
    to: &[CuJointPosition; N],
    limits: &CuJointTrajectoryLimits<N>,
) -> f64 {
    let acc_factor = 10.0 / libm::sqrt(3.0);
    let mut duration = 0.0;
    let mut j = 0;
    while j < N {
        let distance = (to[j].si_value() - from[j].si_value()).abs();
        if distance > 0.0 {
            let vel = limits.max_velocities[j].si_value();
            let acc = limits.acc(j);
            let vel_time = 1.875 * distance / vel;
            let acc_time = libm::sqrt(acc_factor * distance / acc);
            if vel_time > duration {
                duration = vel_time;
            }
            if acc_time > duration {
                duration = acc_time;
            }
        }
        j += 1;
    }
    duration
}

/// Precomputed per-joint parameters for a trapezoidal (or triangular) segment.
///
/// Triangular profiles set `accel_time = cruise_end = half` so the cruise
/// branch is never entered while keeping the evaluation branchless between
/// trapezoidal and triangular shapes.
#[derive(Clone, Copy)]
struct TrapezoidJointCache {
    start: f64,
    sign: f64,
    distance: f64,
    scale: f64,
    max_vel: f64,
    max_acc: f64,
    half_acc: f64,
    accel_time: f64,
    cruise_end: f64,
    total: f64,
    accel_distance: f64,
}

impl TrapezoidJointCache {
    const IDLE: Self = Self {
        start: 0.0,
        sign: 1.0,
        distance: 0.0,
        scale: 0.0,
        max_vel: 0.0,
        max_acc: 0.0,
        half_acc: 0.0,
        accel_time: 0.0,
        cruise_end: 0.0,
        total: 0.0,
        accel_distance: 0.0,
    };

    /// Computes unscaled trapezoidal/triangular parameters (`scale` is set
    /// once the synchronized or fixed segment duration is known).
    fn unscaled(start: f64, displacement: f64, max_vel: f64, max_acc: f64) -> Self {
        let distance = displacement.abs();
        let sign = if displacement >= 0.0 { 1.0 } else { -1.0 };
        if distance <= 0.0 {
            return Self {
                start,
                sign,
                ..Self::IDLE
            };
        }
        let accel_time = max_vel / max_acc;
        let half_acc = 0.5 * max_acc;
        if distance >= max_vel * accel_time {
            let cruise = distance / max_vel - accel_time;
            let total = distance / max_vel + accel_time;
            let accel_distance = 0.5 * max_vel * accel_time;
            Self {
                start,
                sign,
                distance,
                scale: 0.0,
                max_vel,
                max_acc,
                half_acc,
                accel_time,
                cruise_end: accel_time + cruise,
                total,
                accel_distance,
            }
        } else {
            let half = libm::sqrt(distance / max_acc);
            let total = 2.0 * half;
            Self {
                start,
                sign,
                distance,
                scale: 0.0,
                max_vel: 0.0,
                max_acc,
                half_acc,
                accel_time: half,
                cruise_end: half,
                total,
                accel_distance: 0.0,
            }
        }
    }

    #[inline]
    fn eval(&self, tau: f64) -> (f64, f64) {
        if self.distance <= 0.0 || self.scale <= 0.0 {
            return (self.start, 0.0);
        }
        let offset = tau * self.scale;
        let (pos_off, vel) = if offset <= 0.0 {
            (0.0, 0.0)
        } else if offset < self.accel_time {
            (self.half_acc * offset * offset, self.max_acc * offset)
        } else if offset < self.cruise_end {
            (
                self.accel_distance + self.max_vel * (offset - self.accel_time),
                self.max_vel,
            )
        } else if offset < self.total {
            let remaining = self.total - offset;
            (
                self.distance - self.half_acc * remaining * remaining,
                self.max_acc * remaining,
            )
        } else {
            (self.distance, 0.0)
        };
        (
            self.start + self.sign * pos_off,
            self.sign * (vel * self.scale),
        )
    }
}

/// Precomputed per-joint parameters for a finite-jerk S-curve segment.
///
/// Exploits exact symmetry around `total / 2`: the decelerating phases mirror
/// the accelerating half (`p(T - r) = D - p(r)`, `v(T - r) = v(r)`), allowing
/// O(1) division-free evaluation at every sample.
#[derive(Clone, Copy)]
struct FiniteSCurveCache {
    start: f64,
    sign: f64,
    distance: f64,
    scale: f64,
    max_jerk: f64,
    jerk_sixth: f64,
    a_peak: f64,
    t1: f64,
    t2: f64,
    t3: f64,
    t4: f64,
    total: f64,
    p1: f64,
    v1: f64,
    p2: f64,
    v2: f64,
    p3: f64,
    v_peak: f64,
}

impl FiniteSCurveCache {
    fn unscaled(start: f64, displacement: f64, max_vel: f64, max_acc: f64, max_jerk: f64) -> Self {
        let distance = displacement.abs();
        let sign = if displacement >= 0.0 { 1.0 } else { -1.0 };
        let (jerk_time, accel_time, cruise_time) =
            scurve_shape(distance, max_vel, max_acc, max_jerk);
        let jerk_sixth = max_jerk / 6.0;
        let t1 = jerk_time;
        let t2 = jerk_time + accel_time;
        let t3 = 2.0 * jerk_time + accel_time;
        let t4 = t3 + cruise_time;
        let total = 4.0 * jerk_time + 2.0 * accel_time + cruise_time;
        let a_peak = max_jerk * jerk_time;
        let tj2 = jerk_time * jerk_time;
        let v1 = 0.5 * max_jerk * tj2;
        let p1 = jerk_sixth * tj2 * jerk_time;
        let v2 = v1 + a_peak * accel_time;
        let p2 = p1 + v1 * accel_time + 0.5 * a_peak * accel_time * accel_time;
        let v_peak = v2 + a_peak * jerk_time - 0.5 * max_jerk * tj2;
        let p3 = p2 + v2 * jerk_time + 0.5 * a_peak * tj2 - jerk_sixth * tj2 * jerk_time;
        Self {
            start,
            sign,
            distance,
            scale: 0.0,
            max_jerk,
            jerk_sixth,
            a_peak,
            t1,
            t2,
            t3,
            t4,
            total,
            p1,
            v1,
            p2,
            v2,
            p3,
            v_peak,
        }
    }

    /// Evaluates the accelerating half (`0 <= r <= t3`) from precomputed phase
    /// boundaries.
    #[inline]
    fn eval_half(&self, r: f64) -> (f64, f64) {
        if r < self.t1 {
            let r2 = r * r;
            (self.jerk_sixth * r2 * r, 0.5 * self.max_jerk * r2)
        } else if r < self.t2 {
            let s = r - self.t1;
            (
                self.p1 + self.v1 * s + 0.5 * self.a_peak * s * s,
                self.v1 + self.a_peak * s,
            )
        } else {
            let s = r - self.t2;
            let s2 = s * s;
            (
                self.p2 + self.v2 * s + 0.5 * self.a_peak * s2 - self.jerk_sixth * s2 * s,
                self.v2 + self.a_peak * s - 0.5 * self.max_jerk * s2,
            )
        }
    }

    #[inline]
    fn eval(&self, tau: f64) -> (f64, f64) {
        if self.distance <= 0.0 || self.scale <= 0.0 {
            return (self.start, 0.0);
        }
        let offset = tau * self.scale;
        if offset <= 0.0 {
            return (self.start, 0.0);
        }
        if offset >= self.total {
            return (self.start + self.sign * self.distance, 0.0);
        }
        let (pos_off, vel) = if offset < self.t3 {
            self.eval_half(offset)
        } else if offset < self.t4 {
            (self.p3 + self.v_peak * (offset - self.t3), self.v_peak)
        } else {
            let (pos_r, vel_r) = self.eval_half(self.total - offset);
            (self.distance - pos_r, vel_r)
        };
        (
            self.start + self.sign * pos_off,
            self.sign * (vel * self.scale),
        )
    }
}

/// Precomputed per-joint parameters for an S-curve segment.
///
/// Joints with an infinite jerk limit reuse [`TrapezoidJointCache`] directly.
#[derive(Clone, Copy)]
enum SCurveJointCache {
    Idle { start: f64 },
    Finite(FiniteSCurveCache),
    Trapezoid(TrapezoidJointCache),
}

impl SCurveJointCache {
    const IDLE: Self = Self::Idle { start: 0.0 };

    #[inline]
    fn eval(&self, tau: f64) -> (f64, f64) {
        match *self {
            Self::Idle { start } => (start, 0.0),
            Self::Trapezoid(ref trap) => trap.eval(tau),
            Self::Finite(ref finite) => finite.eval(tau),
        }
    }
}

/// Precomputed per-joint polynomial coefficients for a minimum-jerk segment.
#[derive(Clone, Copy)]
struct QuinticJointCache {
    q0: f64,
    v0: f64,
    v0_dur: f64,
    a3: f64,
    a4: f64,
    a5: f64,
    d3: f64,
    d4: f64,
    d5: f64,
}

impl QuinticJointCache {
    const ZERO: Self = Self {
        q0: 0.0,
        v0: 0.0,
        v0_dur: 0.0,
        a3: 0.0,
        a4: 0.0,
        a5: 0.0,
        d3: 0.0,
        d4: 0.0,
        d5: 0.0,
    };

    fn from_joint(q0: f64, q1: f64, v0: f64, v1: f64, duration: f64, inv_duration: f64) -> Self {
        if duration <= 0.0 {
            return Self {
                q0,
                v0,
                ..Self::ZERO
            };
        }
        let distance = q1 - q0;
        let v0_dur = v0 * duration;
        let start_term = distance - v0_dur;
        let end_term = (v1 - v0) * duration;
        let a3 = 10.0 * start_term - 4.0 * end_term;
        let a4 = -15.0 * start_term + 7.0 * end_term;
        let a5 = 6.0 * start_term - 3.0 * end_term;
        Self {
            q0,
            v0,
            v0_dur,
            a3,
            a4,
            a5,
            d3: 3.0 * a3 * inv_duration,
            d4: 4.0 * a4 * inv_duration,
            d5: 5.0 * a5 * inv_duration,
        }
    }

    #[inline]
    fn eval(&self, u: f64, u2: f64, u3: f64, u4: f64, u5: f64) -> (f64, f64) {
        let position = self.q0 + self.v0_dur * u + self.a3 * u3 + self.a4 * u4 + self.a5 * u5;
        let velocity = self.v0 + (self.d3 * u2 + self.d4 * u3 + self.d5 * u4);
        (position, velocity)
    }
}

/// Precomputed per-joint polynomial coefficients for a cubic Hermite segment.
#[derive(Clone, Copy)]
struct CubicJointCache {
    q0: f64,
    v0: f64,
    v0_dur: f64,
    a2: f64,
    a3: f64,
    d2: f64,
    d3: f64,
}

impl CubicJointCache {
    const ZERO: Self = Self {
        q0: 0.0,
        v0: 0.0,
        v0_dur: 0.0,
        a2: 0.0,
        a3: 0.0,
        d2: 0.0,
        d3: 0.0,
    };

    fn from_joint(q0: f64, q1: f64, v0: f64, v1: f64, duration: f64, inv_duration: f64) -> Self {
        if duration <= 0.0 {
            return Self {
                q0,
                v0,
                ..Self::ZERO
            };
        }
        let distance = q1 - q0;
        let v0_dur = v0 * duration;
        let v1_dur = v1 * duration;
        let a2 = 3.0 * distance - 2.0 * v0_dur - v1_dur;
        let a3 = -2.0 * distance + v0_dur + v1_dur;
        Self {
            q0,
            v0,
            v0_dur,
            a2,
            a3,
            d2: 2.0 * a2 * inv_duration,
            d3: 3.0 * a3 * inv_duration,
        }
    }

    #[inline]
    fn eval(&self, u: f64, u2: f64, u3: f64) -> (f64, f64) {
        let position = self.q0 + self.v0_dur * u + self.a2 * u2 + self.a3 * u3;
        let velocity = self.v0 + (self.d2 * u + self.d3 * u2);
        (position, velocity)
    }
}

/// Stack-allocated per-segment parameter cache for all `N` joints.
enum SegmentCache<const N: usize> {
    Trapezoidal([TrapezoidJointCache; N]),
    SCurve([SCurveJointCache; N]),
    MinimumJerk([QuinticJointCache; N]),
    Cubic([CubicJointCache; N]),
}

impl<const N: usize> SegmentCache<N> {
    fn empty(profile: CuTrajectoryProfile) -> Self {
        match profile {
            CuTrajectoryProfile::Trapezoidal => Self::Trapezoidal([TrapezoidJointCache::IDLE; N]),
            CuTrajectoryProfile::SCurve => Self::SCurve([SCurveJointCache::IDLE; N]),
            CuTrajectoryProfile::MinimumJerk => Self::MinimumJerk([QuinticJointCache::ZERO; N]),
            CuTrajectoryProfile::Cubic => Self::Cubic([CubicJointCache::ZERO; N]),
        }
    }
}

/// Builds the trapezoidal per-joint cache and returns `(effective_duration, cache)`.
///
/// When `fixed_duration` is `None`, the synchronized segment duration is
/// derived from the same pass without recomputing per-joint shapes.
fn build_trapezoidal_cache<const N: usize>(
    from: &[CuJointPosition; N],
    to: &[CuJointPosition; N],
    limits: &CuJointTrajectoryLimits<N>,
    fixed_duration: Option<f64>,
) -> (f64, [TrapezoidJointCache; N]) {
    let mut cache = [TrapezoidJointCache::IDLE; N];
    let mut derived = 0.0;
    let mut j = 0;
    while j < N {
        let start = from[j].si_value();
        let displacement = to[j].si_value() - start;
        let entry = TrapezoidJointCache::unscaled(
            start,
            displacement,
            limits.max_velocities[j].si_value(),
            limits.acc(j),
        );
        if entry.total > derived {
            derived = entry.total;
        }
        cache[j] = entry;
        j += 1;
    }
    let effective = fixed_duration.unwrap_or(derived);
    if effective > 0.0 {
        let mut j = 0;
        while j < N {
            cache[j].scale = cache[j].total / effective;
            j += 1;
        }
    }
    (effective, cache)
}

/// Builds the S-curve per-joint cache and returns `(effective_duration, cache)`.
///
/// When `fixed_duration` is `None`, the synchronized segment duration is
/// derived from the same pass without recomputing `scurve_shape`.
fn build_scurve_cache<const N: usize>(
    from: &[CuJointPosition; N],
    to: &[CuJointPosition; N],
    limits: &CuJointTrajectoryLimits<N>,
    fixed_duration: Option<f64>,
) -> (f64, [SCurveJointCache; N]) {
    let mut cache = [SCurveJointCache::IDLE; N];
    let mut derived = 0.0;
    let mut j = 0;
    while j < N {
        let start = from[j].si_value();
        let displacement = to[j].si_value() - start;
        let own = if displacement.abs() <= 0.0 {
            cache[j] = SCurveJointCache::Idle { start };
            0.0
        } else {
            let max_vel = limits.max_velocities[j].si_value();
            let max_acc = limits.acc(j);
            let max_jerk = limits.jerk(j);
            if !max_jerk.is_finite() {
                let trap = TrapezoidJointCache::unscaled(start, displacement, max_vel, max_acc);
                let total = trap.total;
                cache[j] = SCurveJointCache::Trapezoid(trap);
                total
            } else {
                let finite =
                    FiniteSCurveCache::unscaled(start, displacement, max_vel, max_acc, max_jerk);
                let total = finite.total;
                cache[j] = SCurveJointCache::Finite(finite);
                total
            }
        };
        if own > derived {
            derived = own;
        }
        j += 1;
    }
    let effective = fixed_duration.unwrap_or(derived);
    if effective > 0.0 {
        let mut j = 0;
        while j < N {
            match &mut cache[j] {
                SCurveJointCache::Idle { .. } => {}
                SCurveJointCache::Finite(finite) => {
                    finite.scale = finite.total / effective;
                }
                SCurveJointCache::Trapezoid(trap) => {
                    trap.scale = trap.total / effective;
                }
            }
            j += 1;
        }
    }
    (effective, cache)
}

/// Builds the minimum-jerk quintic polynomial cache for segment `seg`.
fn build_quintic_cache<const N: usize>(
    from: &[CuJointPosition; N],
    to: &[CuJointPosition; N],
    seg: usize,
    duration: f64,
    inv_duration: f64,
    timing: &CuTimingOverrides<'_, N>,
) -> [QuinticJointCache; N] {
    let mut cache = [QuinticJointCache::ZERO; N];
    let mut j = 0;
    while j < N {
        let q0 = from[j].si_value();
        let q1 = to[j].si_value();
        let (v0, v1) = match timing.waypoint_velocities {
            Some(vias) => (vias[seg][j].si_value(), vias[seg + 1][j].si_value()),
            None => (0.0, 0.0),
        };
        cache[j] = QuinticJointCache::from_joint(q0, q1, v0, v1, duration, inv_duration);
        j += 1;
    }
    cache
}

/// Builds the cubic Hermite polynomial cache for segment `seg`.
fn build_cubic_cache<const N: usize>(
    from: &[CuJointPosition; N],
    to: &[CuJointPosition; N],
    seg: usize,
    duration: f64,
    inv_duration: f64,
    timing: &CuTimingOverrides<'_, N>,
) -> [CubicJointCache; N] {
    let mut cache = [CubicJointCache::ZERO; N];
    let mut j = 0;
    while j < N {
        let q0 = from[j].si_value();
        let q1 = to[j].si_value();
        let (v0, v1) = match timing.waypoint_velocities {
            Some(vias) => (vias[seg][j].si_value(), vias[seg + 1][j].si_value()),
            None => (0.0, 0.0),
        };
        cache[j] = CubicJointCache::from_joint(q0, q1, v0, v1, duration, inv_duration);
        j += 1;
    }
    cache
}

/// Monotonic segment cursor with a stack-allocated per-segment evaluation cache.
///
/// Advances forward as resampling time `t` increases, matching the exact
/// boundary and zero-duration semantics of scanning from segment `0` while
/// computing each segment's parameters only once on entry.
struct SegmentCursor<'a, const N: usize> {
    waypoints: &'a [[CuJointPosition; N]],
    limits: &'a CuJointTrajectoryLimits<N>,
    profile: CuTrajectoryProfile,
    timing: &'a CuTimingOverrides<'a, N>,
    all_zero_via: bool,
    last: usize,
    seg: usize,
    start: f64,
    duration: f64,
    end: f64,
    inv_duration: f64,
    cached_seg: Option<usize>,
    cache: SegmentCache<N>,
}

impl<'a, const N: usize> SegmentCursor<'a, N> {
    fn new(
        waypoints: &'a [[CuJointPosition; N]],
        limits: &'a CuJointTrajectoryLimits<N>,
        profile: CuTrajectoryProfile,
        timing: &'a CuTimingOverrides<'a, N>,
        all_zero_via: bool,
        first_duration: f64,
    ) -> Self {
        let inv_duration = if first_duration > 0.0 {
            1.0 / first_duration
        } else {
            0.0
        };
        Self {
            waypoints,
            limits,
            profile,
            timing,
            all_zero_via,
            last: waypoints.len() - 1,
            seg: 0,
            start: 0.0,
            duration: first_duration,
            end: first_duration,
            inv_duration,
            cached_seg: None,
            cache: SegmentCache::empty(profile),
        }
    }

    fn step_to(&mut self, seg: usize) {
        self.seg = seg;
        match self.timing.segment_durations {
            Some(fixed) => {
                self.duration = fixed[seg].get::<second>();
            }
            None => match self.profile {
                CuTrajectoryProfile::Trapezoidal => {
                    let (duration, cache) = build_trapezoidal_cache(
                        &self.waypoints[seg],
                        &self.waypoints[seg + 1],
                        self.limits,
                        None,
                    );
                    self.duration = duration;
                    self.cache = SegmentCache::Trapezoidal(cache);
                    self.cached_seg = Some(seg);
                }
                CuTrajectoryProfile::SCurve => {
                    let (duration, cache) = build_scurve_cache(
                        &self.waypoints[seg],
                        &self.waypoints[seg + 1],
                        self.limits,
                        None,
                    );
                    self.duration = duration;
                    self.cache = SegmentCache::SCurve(cache);
                    self.cached_seg = Some(seg);
                }
                CuTrajectoryProfile::MinimumJerk | CuTrajectoryProfile::Cubic => {
                    self.duration = segment_minimum(
                        &self.waypoints[seg],
                        &self.waypoints[seg + 1],
                        self.limits,
                        self.profile,
                        self.all_zero_via,
                    );
                }
            },
        }
        self.end = self.start + self.duration;
        self.inv_duration = if self.duration > 0.0 {
            1.0 / self.duration
        } else {
            0.0
        };
    }

    fn populate_cache(&mut self) {
        let seg = self.seg;
        let from = &self.waypoints[seg];
        let to = &self.waypoints[seg + 1];
        self.cache = match self.profile {
            CuTrajectoryProfile::Trapezoidal => {
                let (_, cache) =
                    build_trapezoidal_cache(from, to, self.limits, Some(self.duration));
                SegmentCache::Trapezoidal(cache)
            }
            CuTrajectoryProfile::SCurve => {
                let (_, cache) = build_scurve_cache(from, to, self.limits, Some(self.duration));
                SegmentCache::SCurve(cache)
            }
            CuTrajectoryProfile::MinimumJerk => SegmentCache::MinimumJerk(build_quintic_cache(
                from,
                to,
                seg,
                self.duration,
                self.inv_duration,
                self.timing,
            )),
            CuTrajectoryProfile::Cubic => SegmentCache::Cubic(build_cubic_cache(
                from,
                to,
                seg,
                self.duration,
                self.inv_duration,
                self.timing,
            )),
        };
        self.cached_seg = Some(seg);
    }

    /// Advances the cursor to the segment containing monotonic time `t` and
    /// evaluates every joint from the per-segment cache.
    fn eval_at(&mut self, t: f64) -> ([CuJointPosition; N], [CuJointVelocity; N]) {
        while self.seg + 1 < self.last && t >= self.end {
            self.start += self.duration;
            self.step_to(self.seg + 1);
        }
        if self.cached_seg != Some(self.seg) {
            self.populate_cache();
        }
        let reference = &self.waypoints[0];
        let tau = (t - self.start).clamp(0.0, self.duration);
        let mut positions = [CuJointPosition::Fixed; N];
        let mut velocities = [CuJointVelocity::Fixed; N];
        match &self.cache {
            SegmentCache::Trapezoidal(joints) => {
                let mut j = 0;
                while j < N {
                    let (pos, vel) = joints[j].eval(tau);
                    positions[j] = reference[j].with_si_value(pos);
                    velocities[j] = CuJointVelocity::from_coordinate(reference[j], vel);
                    j += 1;
                }
            }
            SegmentCache::SCurve(joints) => {
                let mut j = 0;
                while j < N {
                    let (pos, vel) = joints[j].eval(tau);
                    positions[j] = reference[j].with_si_value(pos);
                    velocities[j] = CuJointVelocity::from_coordinate(reference[j], vel);
                    j += 1;
                }
            }
            SegmentCache::MinimumJerk(joints) => {
                let u = if self.duration > 0.0 {
                    (tau * self.inv_duration).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                let u2 = u * u;
                let u3 = u2 * u;
                let u4 = u3 * u;
                let u5 = u4 * u;
                let mut j = 0;
                while j < N {
                    let (pos, vel) = joints[j].eval(u, u2, u3, u4, u5);
                    positions[j] = reference[j].with_si_value(pos);
                    velocities[j] = CuJointVelocity::from_coordinate(reference[j], vel);
                    j += 1;
                }
            }
            SegmentCache::Cubic(joints) => {
                let u = if self.duration > 0.0 {
                    (tau * self.inv_duration).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                let u2 = u * u;
                let u3 = u2 * u;
                let mut j = 0;
                while j < N {
                    let (pos, vel) = joints[j].eval(u, u2, u3);
                    positions[j] = reference[j].with_si_value(pos);
                    velocities[j] = CuJointVelocity::from_coordinate(reference[j], vel);
                    j += 1;
                }
            }
        }
        (positions, velocities)
    }
}

/// Checks waypoint-velocity magnitudes against the per-joint velocity limits.
fn vias_within_limits<const N: usize>(
    waypoints: &[[CuJointPosition; N]],
    vias: &[[CuJointVelocity; N]],
    limits: &CuJointTrajectoryLimits<N>,
) -> bool {
    let mut i = 0;
    while i < vias.len() {
        let mut j = 0;
        while j < N {
            if matches!(waypoints[i][j], CuJointPosition::Fixed) {
                j += 1;
                continue;
            }
            let bound = limits.max_velocities[j].si_value() * (1.0 + 1e-9) + 1e-12;
            if vias[i][j].si_value().abs() > bound {
                return false;
            }
            j += 1;
        }
        i += 1;
    }
    true
}

/// Checks the exact velocity extrema of one Hermite segment — waypoint
/// velocities at the ends included — against the velocity limits, with a
/// small float tolerance.
///
/// Sampling alone cannot certify this: a segment shorter than the sample
/// period receives no interior samples, and a peak between two samples stays
/// unseen, so the analytic extrema are checked instead.
fn segment_within_limits<const N: usize>(
    from: &[CuJointPosition; N],
    to: &[CuJointPosition; N],
    seg: usize,
    duration: f64,
    limits: &CuJointTrajectoryLimits<N>,
    quintic: bool,
    timing: &CuTimingOverrides<'_, N>,
) -> bool {
    let mut j = 0;
    while j < N {
        if matches!(from[j], CuJointPosition::Fixed) {
            j += 1;
            continue;
        }
        let distance = to[j].si_value() - from[j].si_value();
        let (v0, v1) = match timing.waypoint_velocities {
            Some(vias) => (vias[seg][j].si_value(), vias[seg + 1][j].si_value()),
            None => (0.0, 0.0),
        };
        let (high, low, _) = polynomial_extrema(quintic, distance, v0, v1, duration);
        let bound = limits.max_velocities[j].si_value() * (1.0 + 1e-9) + 1e-12;
        if !high.is_finite() || !low.is_finite() || high > bound || low < -bound {
            return false;
        }
        j += 1;
    }
    true
}

/// Pushes one fully-specified point; a full buffer maps to `CapacityExceeded`.
fn push_single_point<const N: usize, const MAX_POINTS: usize>(
    out: &mut CuTrajectoryPayload<N, MAX_POINTS>,
    time_from_start: Time,
    q: [CuJointPosition; N],
    velocities: [CuJointVelocity; N],
) -> CuResult<CuTimingStatus> {
    match out.push(CuTrajectoryPoint {
        time_from_start,
        positions: q,
        velocities,
        accelerations: zero_accelerations(q),
    }) {
        Ok(()) => Ok(CuTimingStatus::Success),
        Err(_) => Ok(CuTimingStatus::CapacityExceeded),
    }
}

/// Pushes a point, returning `false` when `out` is full.
fn try_push<const N: usize, const MAX_POINTS: usize>(
    out: &mut CuTrajectoryPayload<N, MAX_POINTS>,
    point: CuTrajectoryPoint<N>,
) -> bool {
    out.push(point).is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use cu29::units::si::angle::radian;
    use cu29::units::si::angular_velocity::radian_per_second;
    use cu29::units::si::f64::Angle;
    use cu29::units::si::f64::AngularVelocity;

    fn rev(q: f64) -> CuJointPosition {
        CuJointPosition::Revolute(Angle::new::<radian>(q))
    }

    fn rev_acc(a: f64) -> CuJointAcceleration {
        CuJointAcceleration::from_coordinate(rev(0.0), a)
    }

    fn rev_limits(v: f64, a: f64) -> CuJointTrajectoryLimits<1> {
        CuJointTrajectoryLimits::new(
            [CuJointVelocity::Revolute(AngularVelocity::new::<
                radian_per_second,
            >(v))],
            [rev_acc(a)],
        )
    }

    fn rev_limits_jerk(v: f64, a: f64, j: f64) -> CuJointTrajectoryLimits<1> {
        rev_limits(v, a).with_jerks([CuJointJerk::from_coordinate(rev(0.0), j)])
    }

    /// Default params and overrides: the synchronized trapezoidal timing.
    fn trapezoid<const N: usize, const M: usize>(
        waypoints: &[[CuJointPosition; N]],
        limits: &CuJointTrajectoryLimits<N>,
        sample_period: Time,
        out: &mut CuTrajectoryPayload<N, M>,
    ) -> CuResult<CuTimingStatus> {
        let params = CuTimingParams {
            limits: *limits,
            sample_period,
            ..Default::default()
        };
        parameterize(waypoints, &params, CuTimingOverrides::default(), out)
    }

    fn with_profile<const N: usize, const M: usize>(
        waypoints: &[[CuJointPosition; N]],
        limits: &CuJointTrajectoryLimits<N>,
        profile: CuTrajectoryProfile,
        overrides: &CuTimingOverrides<'_, N>,
        sample_period: Time,
        out: &mut CuTrajectoryPayload<N, M>,
    ) -> CuResult<CuTimingStatus> {
        let params = CuTimingParams {
            limits: *limits,
            profile,
            sample_period,
        };
        parameterize(waypoints, &params, *overrides, out)
    }

    fn zero_vel() -> CuJointVelocity {
        CuJointVelocity::Revolute(AngularVelocity::new::<radian_per_second>(0.0))
    }

    fn rev_vel(v: f64) -> CuJointVelocity {
        CuJointVelocity::Revolute(AngularVelocity::new::<radian_per_second>(v))
    }

    fn max_sampled_velocity<const M: usize>(traj: &CuTrajectoryPayload<1, M>) -> (f64, f64) {
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        let mut i = 0;
        while i < traj.len() {
            let v = traj.points()[i].velocities[0].si_value();
            lo = lo.min(v);
            hi = hi.max(v);
            i += 1;
        }
        (lo, hi)
    }

    fn assert_strictly_increasing<const M: usize>(traj: &CuTrajectoryPayload<1, M>) {
        let mut i = 1;
        while i < traj.len() {
            assert!(
                traj.points()[i].time_from_start > traj.points()[i - 1].time_from_start,
                "times must be strictly increasing"
            );
            i += 1;
        }
    }

    #[test]
    fn test_empty_path_yields_empty_status() {
        let mut out = CuTrajectoryPayload::<1>::new();
        let status = trapezoid(
            &[],
            &rev_limits(1.0, 1.0),
            Time::new::<second>(0.01),
            &mut out,
        )
        .unwrap();
        assert_eq!(status, CuTimingStatus::EmptyPath);
        assert!(out.is_empty());
    }

    #[test]
    fn test_single_waypoint_is_zero_velocity_point() {
        let mut out = CuTrajectoryPayload::<1>::new();
        let status = trapezoid(
            &[[rev(0.5)]],
            &rev_limits(1.0, 1.0),
            Time::new::<second>(0.01),
            &mut out,
        )
        .unwrap();
        assert_eq!(status, CuTimingStatus::Success);
        assert_eq!(out.len(), 1);
        assert_eq!(out.points()[0].time_from_start.get::<second>(), 0.0);
        assert_eq!(out.points()[0].positions[0].si_value(), 0.5);
        assert_eq!(out.points()[0].velocities[0].si_value(), 0.0);
    }

    #[test]
    fn test_trapezoidal_duration_and_velocity_limits() {
        let mut out = CuTrajectoryPayload::<1, 512>::new();
        let status = trapezoid(
            &[[rev(0.0)], [rev(1.0)]],
            &rev_limits(0.5, 1.0),
            Time::new::<second>(0.02),
            &mut out,
        )
        .unwrap();
        assert_eq!(status, CuTimingStatus::Success);
        // Trapezoidal: T = D/v + v/a = 2.0 + 0.5.
        assert!((out.duration().get::<second>() - 2.5).abs() < 1e-9);
        let (lo, hi) = max_sampled_velocity(&out);
        assert!(hi <= 0.5 + 1e-9, "velocity limit exceeded: {hi}");
        assert!(lo >= -1e-9, "unexpected reverse motion: {lo}");
        assert_eq!(out.points()[0].velocities[0].si_value(), 0.0);
        let last = out.len() - 1;
        assert_eq!(out.points()[last].positions[0].si_value(), 1.0);
        assert_eq!(out.points()[last].velocities[0].si_value(), 0.0);
    }

    #[test]
    fn test_triangular_duration_matches_formula() {
        let mut out = CuTrajectoryPayload::<1, 256>::new();
        let status = trapezoid(
            &[[rev(0.0)], [rev(0.1)]],
            &rev_limits(1.0, 1.0),
            Time::new::<second>(0.005),
            &mut out,
        )
        .unwrap();
        assert_eq!(status, CuTimingStatus::Success);
        // Triangular (D < v^2/a): T = 2*sqrt(D/a).
        let expected = 2.0 * (0.1f64).sqrt();
        assert!((out.duration().get::<second>() - expected).abs() < 1e-9);
        let (_, peak) = max_sampled_velocity(&out);
        assert!((peak - (0.1f64).sqrt()).abs() < 0.02);
    }

    #[test]
    fn test_slow_joint_dominates_synchronized_duration() {
        let slow = CuJointTrajectoryLimits::new(
            [
                CuJointVelocity::Revolute(AngularVelocity::new::<radian_per_second>(2.0)),
                CuJointVelocity::Revolute(AngularVelocity::new::<radian_per_second>(0.25)),
            ],
            [rev_acc(2.0), rev_acc(0.5)],
        );
        let mut out = CuTrajectoryPayload::<2, 512>::new();
        let status = trapezoid(
            &[[rev(0.0), rev(0.0)], [rev(0.1), rev(1.0)]],
            &slow,
            Time::new::<second>(0.05),
            &mut out,
        )
        .unwrap();
        assert_eq!(status, CuTimingStatus::Success);
        // Slow joint trapezoidal: T = 1/0.25 + 0.25/0.5 = 4.5; fast joint ~0.45.
        assert!((out.duration().get::<second>() - 4.5).abs() < 1e-9);
        let last = out.len() - 1;
        assert!((out.points()[last].positions[0].si_value() - 0.1).abs() < 1e-9);
        assert!((out.points()[last].positions[1].si_value() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_capacity_exceeded_status() {
        let mut out = CuTrajectoryPayload::<1, 3>::new();
        let status = trapezoid(
            &[[rev(0.0)], [rev(1.0)]],
            &rev_limits(0.5, 1.0),
            Time::new::<second>(0.01),
            &mut out,
        )
        .unwrap();
        assert_eq!(status, CuTimingStatus::CapacityExceeded);
        assert_eq!(out.len(), 3);
    }

    #[test]
    fn test_invalid_limits_rejected() {
        let wps = [[rev(0.0)], [rev(1.0)]];
        let mut out = CuTrajectoryPayload::<1>::new();
        let zero_vel = CuJointTrajectoryLimits::new(
            [CuJointVelocity::Revolute(AngularVelocity::new::<
                radian_per_second,
            >(0.0))],
            [rev_acc(1.0)],
        );
        let status = trapezoid(&wps, &zero_vel, Time::new::<second>(0.01), &mut out).unwrap();
        assert_eq!(status, CuTimingStatus::InvalidInput);
        let neg_acc = rev_limits(1.0, -2.0);
        let status = trapezoid(&wps, &neg_acc, Time::new::<second>(0.01), &mut out).unwrap();
        assert_eq!(status, CuTimingStatus::InvalidInput);
        let status = trapezoid(
            &wps,
            &rev_limits(1.0, 1.0),
            Time::new::<second>(0.0),
            &mut out,
        )
        .unwrap();
        assert_eq!(status, CuTimingStatus::InvalidInput);
    }

    #[test]
    fn test_profile_default_is_trapezoidal() {
        assert_eq!(
            CuTrajectoryProfile::default(),
            CuTrajectoryProfile::Trapezoidal
        );
    }

    #[test]
    fn test_default_profile_matches_legacy_entry_point() {
        let wps = [[rev(0.0)], [rev(0.5)], [rev(1.0)]];
        let limits = rev_limits(0.5, 1.0);
        let dt = Time::new::<second>(0.01);
        let mut legacy = CuTrajectoryPayload::<1, 512>::new();
        let mut profiled = CuTrajectoryPayload::<1, 512>::new();
        let legacy_status = trapezoid(&wps, &limits, dt, &mut legacy).unwrap();
        let profiled_status = with_profile(
            &wps,
            &limits,
            CuTrajectoryProfile::Trapezoidal,
            &CuTimingOverrides::default(),
            dt,
            &mut profiled,
        )
        .unwrap();
        assert_eq!(legacy_status, profiled_status);
        assert_eq!(legacy.points(), profiled.points());
    }

    #[test]
    fn test_cubic_honors_via_velocity() {
        let wps = [[rev(0.0)], [rev(1.0)], [rev(2.0)]];
        let limits = rev_limits(2.0, 5.0);
        let vias = [[zero_vel()], [rev_vel(0.8)], [zero_vel()]];
        let fixed = [Time::new::<second>(1.0), Time::new::<second>(1.0)];
        let timing = CuTimingOverrides {
            segment_durations: Some(fixed.as_slice()),
            waypoint_velocities: Some(vias.as_slice()),
        };
        let mut out = CuTrajectoryPayload::<1, 512>::new();
        let status = with_profile(
            &wps,
            &limits,
            CuTrajectoryProfile::Cubic,
            &timing,
            Time::new::<second>(0.01),
            &mut out,
        )
        .unwrap();
        assert_eq!(status, CuTimingStatus::Success);
        assert!((out.duration().get::<second>() - 2.0).abs() < 1e-9);
        let mid = out
            .points()
            .iter()
            .find(|p| (p.time_from_start.get::<second>() - 1.0).abs() < 1e-9)
            .expect("resampled interior waypoint");
        assert!((mid.positions[0].si_value() - 1.0).abs() < 1e-9);
        assert!((mid.velocities[0].si_value() - 0.8).abs() < 1e-9);
        assert_eq!(out.points()[0].velocities[0].si_value(), 0.0);
        let last = out.len() - 1;
        assert_eq!(out.points()[last].positions[0].si_value(), 2.0);
        assert_eq!(out.points()[last].velocities[0].si_value(), 0.0);
        assert_strictly_increasing(&out);
    }

    #[test]
    fn test_minimum_jerk_honors_via_and_zero_end_acceleration() {
        let wps = [[rev(0.0)], [rev(1.0)], [rev(2.0)]];
        let limits = rev_limits(2.0, 5.0);
        let vias = [[zero_vel()], [rev_vel(0.8)], [zero_vel()]];
        let fixed = [Time::new::<second>(1.0), Time::new::<second>(1.0)];
        let timing = CuTimingOverrides {
            segment_durations: Some(fixed.as_slice()),
            waypoint_velocities: Some(vias.as_slice()),
        };
        let mut out = CuTrajectoryPayload::<1, 2048>::new();
        let status = with_profile(
            &wps,
            &limits,
            CuTrajectoryProfile::MinimumJerk,
            &timing,
            Time::new::<second>(0.002),
            &mut out,
        )
        .unwrap();
        assert_eq!(status, CuTimingStatus::Success);
        assert!((out.duration().get::<second>() - 2.0).abs() < 1e-9);
        let mid = out
            .points()
            .iter()
            .find(|p| (p.time_from_start.get::<second>() - 1.0).abs() < 1e-9)
            .expect("resampled interior waypoint");
        assert!((mid.velocities[0].si_value() - 0.8).abs() < 1e-9);
        // Quintic endpoint accelerations are zero; finite differences of the
        // 2 ms resampling only see jerk-bounded growth (~0.04 rad/s^2 here).
        let dt = 0.002;
        let accel_start = (out.points()[1].velocities[0].si_value()
            - out.points()[0].velocities[0].si_value())
            / dt;
        assert!(accel_start.abs() < 0.5, "start accel: {accel_start}");
        let last = out.len() - 1;
        let accel_end = (out.points()[last].velocities[0].si_value()
            - out.points()[last - 1].velocities[0].si_value())
            / dt;
        assert!(accel_end.abs() < 0.5, "end accel: {accel_end}");
        assert_strictly_increasing(&out);
    }

    #[test]
    fn test_minimum_jerk_derived_duration_matches_quintic_factors() {
        let wps = [[rev(0.0)], [rev(1.0)]];
        let limits = rev_limits(1.0, 1.0);
        let mut out = CuTrajectoryPayload::<1, 512>::new();
        let status = with_profile(
            &wps,
            &limits,
            CuTrajectoryProfile::MinimumJerk,
            &CuTimingOverrides::default(),
            Time::new::<second>(0.005),
            &mut out,
        )
        .unwrap();
        assert_eq!(status, CuTimingStatus::Success);
        // Rest-to-rest quintic peaks: |v| = 1.875 D/T, |a| = (10/sqrt(3)) D/T^2.
        let expected = 1.875f64.max((10.0f64 / 3.0f64.sqrt()).sqrt());
        assert!((out.duration().get::<second>() - expected).abs() < 1e-9);
        let (lo, hi) = max_sampled_velocity(&out);
        assert!(hi <= 1.0 + 1e-9, "velocity limit exceeded: {hi}");
        assert!(lo >= -1e-9, "unexpected reverse motion: {lo}");
        assert_strictly_increasing(&out);
    }

    #[test]
    fn test_scurve_respects_velocity_acceleration_and_rest() {
        let wps = [[rev(0.0)], [rev(1.0)]];
        let limits = rev_limits_jerk(0.5, 1.0, 2.0);
        let mut out = CuTrajectoryPayload::<1, 1024>::new();
        let status = with_profile(
            &wps,
            &limits,
            CuTrajectoryProfile::SCurve,
            &CuTimingOverrides::default(),
            Time::new::<second>(0.005),
            &mut out,
        )
        .unwrap();
        assert_eq!(status, CuTimingStatus::Success);
        // Jerk time 0.5 s, no constant-accel phase, 1 s cruise: T = 3.0.
        assert!((out.duration().get::<second>() - 3.0).abs() < 1e-9);
        let (lo, hi) = max_sampled_velocity(&out);
        assert!(hi <= 0.5 + 1e-9, "velocity limit exceeded: {hi}");
        assert!(lo >= -1e-9, "unexpected reverse motion: {lo}");
        assert_eq!(out.points()[0].velocities[0].si_value(), 0.0);
        let last = out.len() - 1;
        assert_eq!(out.points()[last].positions[0].si_value(), 1.0);
        assert_eq!(out.points()[last].velocities[0].si_value(), 0.0);
        let mut peak_accel: f64 = 0.0;
        let mut i = 1;
        while i + 2 < out.len() {
            let gap = out.points()[i + 1].time_from_start.get::<second>()
                - out.points()[i - 1].time_from_start.get::<second>();
            if (gap - 0.01).abs() < 1e-12 {
                let accel = (out.points()[i + 1].velocities[0].si_value()
                    - out.points()[i - 1].velocities[0].si_value())
                    / gap;
                peak_accel = peak_accel.max(accel.abs());
            }
            i += 1;
        }
        assert!(
            peak_accel <= 1.0 + 0.05,
            "accel limit exceeded: {peak_accel}"
        );
        assert_strictly_increasing(&out);
    }

    #[test]
    fn test_scurve_infinite_jerk_matches_trapezoidal_timing() {
        let wps = [[rev(0.0)], [rev(1.0)]];
        let limits = rev_limits(0.5, 1.0);
        assert_eq!(limits.max_jerks, None);
        let mut out = CuTrajectoryPayload::<1, 256>::new();
        let status = with_profile(
            &wps,
            &limits,
            CuTrajectoryProfile::SCurve,
            &CuTimingOverrides::default(),
            Time::new::<second>(0.02),
            &mut out,
        )
        .unwrap();
        assert_eq!(status, CuTimingStatus::Success);
        // Trapezoidal: T = D/v + v/a = 2.0 + 0.5.
        assert!((out.duration().get::<second>() - 2.5).abs() < 1e-9);
        let (_, peak) = max_sampled_velocity(&out);
        assert!(peak <= 0.5 + 1e-9, "velocity limit exceeded: {peak}");
    }

    #[test]
    fn test_scurve_rejects_non_positive_jerk() {
        let wps = [[rev(0.0)], [rev(1.0)]];
        for bad in [0.0, -1.0, f64::NAN, f64::NEG_INFINITY] {
            let limits = rev_limits_jerk(0.5, 1.0, bad);
            let mut out = CuTrajectoryPayload::<1>::new();
            let status = with_profile(
                &wps,
                &limits,
                CuTrajectoryProfile::SCurve,
                &CuTimingOverrides::default(),
                Time::new::<second>(0.01),
                &mut out,
            )
            .unwrap();
            assert_eq!(status, CuTimingStatus::InvalidInput, "jerk: {bad}");
        }
    }

    #[test]
    fn test_rest_only_profiles_reject_nonzero_via() {
        let wps = [[rev(0.0)], [rev(1.0)]];
        let limits = rev_limits(1.0, 1.0);
        let vias = [[zero_vel()], [rev_vel(0.5)]];
        let timing = CuTimingOverrides {
            segment_durations: None,
            waypoint_velocities: Some(vias.as_slice()),
        };
        for profile in [
            CuTrajectoryProfile::Trapezoidal,
            CuTrajectoryProfile::SCurve,
        ] {
            let mut out = CuTrajectoryPayload::<1>::new();
            let status = with_profile(
                &wps,
                &limits,
                profile,
                &timing,
                Time::new::<second>(0.01),
                &mut out,
            )
            .unwrap();
            assert_eq!(status, CuTimingStatus::InvalidInput);
        }
    }

    #[test]
    fn test_timing_shape_violations_rejected() {
        let wps = [[rev(0.0)], [rev(1.0)], [rev(2.0)]];
        let limits = rev_limits(1.0, 1.0);
        let dt = Time::new::<second>(0.01);
        // One duration for two segments.
        let one = [Time::new::<second>(1.0)];
        let short_timing = CuTimingOverrides {
            segment_durations: Some(one.as_slice()),
            waypoint_velocities: None,
        };
        // Via count must match waypoint count.
        let vias = [[zero_vel()], [zero_vel()]];
        let via_timing = CuTimingOverrides {
            segment_durations: None,
            waypoint_velocities: Some(vias.as_slice()),
        };
        // Fixed durations below the trapezoidal minimum (2.0 s here) ...
        let short = [Time::new::<second>(0.5), Time::new::<second>(0.5)];
        let fast_timing = CuTimingOverrides {
            segment_durations: Some(short.as_slice()),
            waypoint_velocities: None,
        };
        // ... and non-positive fixed durations.
        let zero = [Time::new::<second>(0.0), Time::new::<second>(1.0)];
        let zero_timing = CuTimingOverrides {
            segment_durations: Some(zero.as_slice()),
            waypoint_velocities: None,
        };
        for timing in [short_timing, via_timing, fast_timing, zero_timing] {
            let mut out = CuTrajectoryPayload::<1>::new();
            let status = with_profile(
                &wps,
                &limits,
                CuTrajectoryProfile::Trapezoidal,
                &timing,
                dt,
                &mut out,
            )
            .unwrap();
            assert_eq!(status, CuTimingStatus::InvalidInput);
        }
    }

    #[test]
    fn test_fixed_durations_produce_exact_total() {
        let wps = [[rev(0.0)], [rev(1.0)], [rev(1.5)]];
        let limits = rev_limits(3.0, 5.0);
        let fixed = [Time::new::<second>(0.7), Time::new::<second>(1.3)];
        let timing = CuTimingOverrides {
            segment_durations: Some(fixed.as_slice()),
            waypoint_velocities: None,
        };
        for profile in [CuTrajectoryProfile::Cubic, CuTrajectoryProfile::MinimumJerk] {
            let mut out = CuTrajectoryPayload::<1, 512>::new();
            let status = with_profile(
                &wps,
                &limits,
                profile,
                &timing,
                Time::new::<second>(0.01),
                &mut out,
            )
            .unwrap();
            assert_eq!(status, CuTimingStatus::Success);
            assert!((out.duration().get::<second>() - 2.0).abs() < 1e-9);
            let last = out.len() - 1;
            assert_eq!(out.points()[last].positions[0].si_value(), 1.5);
            assert_eq!(out.points()[last].velocities[0].si_value(), 0.0);
            assert_strictly_increasing(&out);
        }
    }

    #[test]
    fn test_cubic_long_segment_stretched_to_velocity_limit() {
        let wps = [[rev(0.0)], [rev(1.0)]];
        let limits = rev_limits(0.5, 5.0);
        let mut out = CuTrajectoryPayload::<1, 512>::new();
        // The trapezoidal minimum (2.1 s) would peak at 1.5 / 2.1 = 0.71
        // rad/s; the cubic duration is stretched to 1.5 * 1.0 / 0.5 = 3 s.
        let status = with_profile(
            &wps,
            &limits,
            CuTrajectoryProfile::Cubic,
            &CuTimingOverrides::default(),
            Time::new::<second>(0.01),
            &mut out,
        )
        .unwrap();
        assert_eq!(status, CuTimingStatus::Success);
        let last = out.points().last().unwrap();
        assert!((last.time_from_start.get::<second>() - 3.0).abs() < 1e-12);
        assert!((last.positions[0].si_value() - 1.0).abs() < 1e-12);
        for p in out.points() {
            assert!(p.velocities[0].si_value().abs() <= 0.5 + 1e-9);
        }
    }

    #[test]
    fn test_cubic_velocity_violation_rejected() {
        let wps = [[rev(0.0)], [rev(1.0)]];
        let limits = rev_limits(0.5, 5.0);
        let mut out = CuTrajectoryPayload::<1, 512>::new();
        // A fixed 2.1 s segment peaks at 1.5 / 2.1 = 0.71 rad/s, above the
        // 0.5 rad/s limit.
        let durations = [Time::new::<second>(2.1)];
        let status = with_profile(
            &wps,
            &limits,
            CuTrajectoryProfile::Cubic,
            &CuTimingOverrides {
                segment_durations: Some(&durations),
                ..Default::default()
            },
            Time::new::<second>(0.01),
            &mut out,
        )
        .unwrap();
        assert_eq!(status, CuTimingStatus::InvalidInput);
        assert!(out.is_empty());
    }

    #[test]
    fn test_parameterize_tags_cubic_interpolation() {
        let mut out = CuTrajectoryPayload::<1, 512>::new();
        trapezoid(
            &[[rev(0.0)], [rev(1.0)]],
            &rev_limits(0.5, 1.0),
            Time::new::<second>(0.02),
            &mut out,
        )
        .unwrap();
        assert_eq!(out.interpolation, CuTrajectoryInterpolation::Cubic);
    }

    #[test]
    fn test_params_validate() {
        assert!(CuTimingParams::<1>::default().validate().is_err());
        let mut params = CuTimingParams {
            limits: rev_limits(0.5, 1.0),
            ..Default::default()
        };
        assert!(params.validate().is_ok());
        assert_eq!(params.sample_period.get::<second>(), 0.02);
        params.sample_period = Time::new::<second>(0.0);
        assert!(params.validate().is_err());
        params.sample_period = Time::new::<second>(0.01);
        params.limits = rev_limits_jerk(0.5, 1.0, -1.0);
        assert!(params.validate().is_err());
        params.limits = rev_limits(0.5, 1.0);
        params.limits.max_accelerations = [CuJointAcceleration::Fixed];
        assert!(params.validate().is_err());
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_params_from_ron() {
        let parse = |text: &str| {
            let cfg: ComponentConfig =
                ron::from_str(&format!("#![enable(unwrap_newtypes)] {text}")).unwrap();
            cfg.deserialize_into::<CuTimingParams<1>>()
        };
        let params = parse(
            r#"{"limits": {"max_velocities": [{"revolute": 0.5}], "max_accelerations": [{"revolute": 1.0}]}}"#,
        )
        .unwrap();
        assert_eq!(params.limits, rev_limits(0.5, 1.0));
        assert_eq!(params.profile, CuTrajectoryProfile::Trapezoidal);
        assert_eq!(params.sample_period.get::<second>(), 0.02);
        let params = parse(
            r#"{"limits": {"max_velocities": [{"revolute": 0.5}], "max_accelerations": [{"revolute": 1.0}],
                "max_jerks": [{"revolute": 2.0}]}, "profile": "s_curve", "sample_period": 0.01}"#,
        )
        .unwrap();
        assert_eq!(params.limits, rev_limits_jerk(0.5, 1.0, 2.0));
        assert_eq!(params.profile, CuTrajectoryProfile::SCurve);
        assert_eq!(params.sample_period.get::<second>(), 0.01);
        assert!(
            parse(
                r#"{"limits": {"max_velocities": [{"revolute": 0.5}], "max_accelerations": [{"revolute": 1.0}]},
                "sample_period_s": 0.01}"#
            )
            .is_err()
        );
        assert!(parse(r#"{"profile": "cubic"}"#).is_err());
    }

    fn pris(q: f64) -> CuJointPosition {
        CuJointPosition::Prismatic(cu29::units::si::f64::Length::new::<
            cu29::units::si::length::meter,
        >(q))
    }

    fn pris_vel(v: f64) -> CuJointVelocity {
        CuJointVelocity::from_coordinate(pris(0.0), v)
    }

    fn pris_acc(a: f64) -> CuJointAcceleration {
        CuJointAcceleration::from_coordinate(pris(0.0), a)
    }

    fn pris_jerk(j: f64) -> CuJointJerk {
        CuJointJerk::from_coordinate(pris(0.0), j)
    }

    fn rev_jerk(j: f64) -> CuJointJerk {
        CuJointJerk::from_coordinate(rev(0.0), j)
    }

    fn reference_trapezoid_eval(
        offset: f64,
        distance: f64,
        max_vel: f64,
        max_acc: f64,
    ) -> (f64, f64) {
        if distance <= 0.0 || offset <= 0.0 {
            return (0.0, 0.0);
        }
        let accel_time = max_vel / max_acc;
        if distance >= max_vel * accel_time {
            let cruise = distance / max_vel - accel_time;
            let total = distance / max_vel + accel_time;
            let accel_distance = 0.5 * max_vel * accel_time;
            if offset < accel_time {
                (0.5 * max_acc * offset * offset, max_acc * offset)
            } else if offset < accel_time + cruise {
                (accel_distance + max_vel * (offset - accel_time), max_vel)
            } else if offset < total {
                let remaining = total - offset;
                (
                    distance - 0.5 * max_acc * remaining * remaining,
                    max_acc * remaining,
                )
            } else {
                (distance, 0.0)
            }
        } else {
            let half = libm::sqrt(distance / max_acc);
            let total = 2.0 * half;
            if offset < half {
                (0.5 * max_acc * offset * offset, max_acc * offset)
            } else if offset < total {
                let remaining = total - offset;
                (
                    distance - 0.5 * max_acc * remaining * remaining,
                    max_acc * remaining,
                )
            } else {
                (distance, 0.0)
            }
        }
    }

    fn reference_scurve_eval(
        offset: f64,
        distance: f64,
        max_vel: f64,
        max_acc: f64,
        max_jerk: f64,
    ) -> (f64, f64) {
        if distance <= 0.0 || offset <= 0.0 {
            return (0.0, 0.0);
        }
        let (jerk_time, accel_time, cruise_time) =
            scurve_shape(distance, max_vel, max_acc, max_jerk);
        let total = 4.0 * jerk_time + 2.0 * accel_time + cruise_time;
        if offset >= total {
            return (distance, 0.0);
        }
        let mut remaining = offset;
        let mut position = 0.0;
        let mut velocity = 0.0;
        let mut acceleration = 0.0;
        let mut phase = 0;
        while phase < 7 {
            let (jerk, duration) = match phase {
                0 => (max_jerk, jerk_time),
                1 => (0.0, accel_time),
                2 => (-max_jerk, jerk_time),
                3 => (0.0, cruise_time),
                4 => (-max_jerk, jerk_time),
                5 => (0.0, accel_time),
                _ => (max_jerk, jerk_time),
            };
            let step = if remaining < duration {
                remaining
            } else {
                duration
            };
            position += velocity * step
                + 0.5 * acceleration * step * step
                + jerk * step * step * step / 6.0;
            velocity += acceleration * step + 0.5 * jerk * step * step;
            acceleration += jerk * step;
            remaining -= step;
            if remaining <= 0.0 {
                break;
            }
            phase += 1;
        }
        (position, velocity)
    }

    fn reference_eval_at<const N: usize>(
        t: f64,
        waypoints: &[[CuJointPosition; N]],
        limits: &CuJointTrajectoryLimits<N>,
        profile: CuTrajectoryProfile,
        timing: &CuTimingOverrides<'_, N>,
        all_zero_via: bool,
    ) -> ([CuJointPosition; N], [CuJointVelocity; N]) {
        let mut seg = 0;
        let mut seg_start = 0.0;
        while seg + 2 < waypoints.len() {
            let duration = match timing.segment_durations {
                Some(fixed) => fixed[seg].get::<second>(),
                None => segment_minimum(
                    &waypoints[seg],
                    &waypoints[seg + 1],
                    limits,
                    profile,
                    all_zero_via,
                ),
            };
            if t < seg_start + duration {
                break;
            }
            seg_start += duration;
            seg += 1;
        }
        let seg_duration = match timing.segment_durations {
            Some(fixed) => fixed[seg].get::<second>(),
            None => segment_minimum(
                &waypoints[seg],
                &waypoints[seg + 1],
                limits,
                profile,
                all_zero_via,
            ),
        };
        let from = waypoints[seg];
        let to = waypoints[seg + 1];
        let start_vel = waypoint_velocity(timing, seg, from);
        let end_vel = waypoint_velocity(timing, seg + 1, to);
        let tau = (t - seg_start).clamp(0.0, seg_duration);
        let mut positions = [CuJointPosition::Fixed; N];
        let mut velocities = [CuJointVelocity::Fixed; N];
        let mut j = 0;
        while j < N {
            let start = from[j].si_value();
            let displacement = to[j].si_value() - start;
            let distance = displacement.abs();
            let max_vel = limits.max_velocities[j].si_value();
            let max_acc = limits.acc(j);
            let max_jerk = limits.jerk(j);
            match profile {
                CuTrajectoryProfile::Trapezoidal | CuTrajectoryProfile::SCurve => {
                    let sign = if displacement >= 0.0 { 1.0 } else { -1.0 };
                    let jerk_limited = profile == CuTrajectoryProfile::SCurve;
                    let own = if jerk_limited {
                        scurve_joint_duration(distance, max_vel, max_acc, max_jerk)
                    } else {
                        joint_profile_duration(distance, max_vel, max_acc)
                    };
                    let (offset, vel) = if seg_duration <= 0.0 || own <= 0.0 {
                        (0.0, 0.0)
                    } else {
                        let scale = own / seg_duration;
                        let (off, v) = if jerk_limited && max_jerk.is_finite() {
                            reference_scurve_eval(tau * scale, distance, max_vel, max_acc, max_jerk)
                        } else {
                            reference_trapezoid_eval(tau * scale, distance, max_vel, max_acc)
                        };
                        (off, v * scale)
                    };
                    positions[j] = from[j].with_si_value(start + sign * offset);
                    velocities[j] = CuJointVelocity::from_coordinate(from[j], sign * vel);
                }
                CuTrajectoryProfile::MinimumJerk | CuTrajectoryProfile::Cubic => {
                    let u = if seg_duration > 0.0 {
                        (tau / seg_duration).clamp(0.0, 1.0)
                    } else {
                        0.0
                    };
                    let (position, vel) = if profile == CuTrajectoryProfile::MinimumJerk {
                        quintic_eval(
                            u,
                            start,
                            start + displacement,
                            start_vel[j].si_value(),
                            end_vel[j].si_value(),
                            seg_duration,
                        )
                    } else {
                        cubic_eval(
                            u,
                            start,
                            start + displacement,
                            start_vel[j].si_value(),
                            end_vel[j].si_value(),
                            seg_duration,
                        )
                    };
                    positions[j] = from[j].with_si_value(position);
                    velocities[j] = CuJointVelocity::from_coordinate(from[j], vel);
                }
            }
            j += 1;
        }
        (positions, velocities)
    }

    fn assert_matches_reference<const N: usize, const M: usize>(
        waypoints: &[[CuJointPosition; N]],
        limits: &CuJointTrajectoryLimits<N>,
        profile: CuTrajectoryProfile,
        overrides: &CuTimingOverrides<'_, N>,
        sample_period: Time,
        out: &mut CuTrajectoryPayload<N, M>,
    ) {
        let status =
            with_profile(waypoints, limits, profile, overrides, sample_period, out).unwrap();
        assert_eq!(status, CuTimingStatus::Success, "{profile:?}");
        assert!(out.len() >= 2, "{profile:?}");
        let all_zero_via = !vias_nonzero(overrides);
        let pts = out.points();
        let mut i = 1;
        while i < pts.len() {
            assert!(
                pts[i].time_from_start > pts[i - 1].time_from_start,
                "{profile:?}: non-increasing time at {i}"
            );
            i += 1;
        }
        let mut k = 1;
        while k + 1 < pts.len() {
            let t = pts[k].time_from_start.get::<second>();
            let (exp_pos, exp_vel) =
                reference_eval_at(t, waypoints, limits, profile, overrides, all_zero_via);
            let mut j = 0;
            while j < N {
                assert_eq!(
                    core::mem::discriminant(&pts[k].positions[j]),
                    core::mem::discriminant(&exp_pos[j]),
                    "{profile:?}: position kind mismatch at knot {k} joint {j}"
                );
                assert_eq!(
                    core::mem::discriminant(&pts[k].velocities[j]),
                    core::mem::discriminant(&exp_vel[j]),
                    "{profile:?}: velocity kind mismatch at knot {k} joint {j}"
                );
                let dp = (pts[k].positions[j].si_value() - exp_pos[j].si_value()).abs();
                let dv = (pts[k].velocities[j].si_value() - exp_vel[j].si_value()).abs();
                assert!(
                    dp <= 1e-12,
                    "{profile:?}: position drift {dp} at knot {k} joint {j} (t={t})"
                );
                assert!(
                    dv <= 1e-12,
                    "{profile:?}: velocity drift {dv} at knot {k} joint {j} (t={t})"
                );
                j += 1;
            }
            k += 1;
        }
    }

    #[test]
    fn test_cursor_matches_reference_all_profiles_multi_segment() {
        // Exercises all 4 S-curve regimes (cruise + const-accel, accel-limited
        // no cruise, jerk-only short move, vel-limited no const-accel) as well
        // as trapezoidal/triangular legs, reversals, and stationary joints.
        let waypoints = [
            [rev(0.0), rev(0.0), rev(0.5), rev(0.0)],
            [rev(3.0), rev(-0.4), rev(0.5), rev(0.02)],
            [rev(2.95), rev(0.8), rev(-0.2), rev(1.5)],
            [rev(0.2), rev(0.8), rev(0.1), rev(1.48)],
            [rev(-1.0), rev(-0.5), rev(0.9), rev(0.0)],
        ];
        let limits = CuJointTrajectoryLimits::new(
            [rev_vel(1.0), rev_vel(0.8), rev_vel(1.2), rev_vel(0.6)],
            [rev_acc(2.0), rev_acc(3.0), rev_acc(2.5), rev_acc(4.0)],
        )
        .with_jerks([
            rev_jerk(10.0),
            rev_jerk(2.0), // accel_full < 0: vel-limited without const-accel
            rev_jerk(8.0),
            rev_jerk(12.0),
        ]);
        let dt = Time::new::<second>(0.05);
        let mut out = CuTrajectoryPayload::<4, 512>::new();
        for profile in [
            CuTrajectoryProfile::Trapezoidal,
            CuTrajectoryProfile::SCurve,
            CuTrajectoryProfile::MinimumJerk,
            CuTrajectoryProfile::Cubic,
        ] {
            assert_matches_reference(
                &waypoints,
                &limits,
                profile,
                &CuTimingOverrides::default(),
                dt,
                &mut out,
            );
            let last = out.len() - 1;
            assert_eq!(out.points()[0].positions, waypoints[0]);
            assert_eq!(out.points()[last].positions, waypoints[4]);
        }
    }

    #[test]
    fn test_zero_displacement_and_short_segments_all_profiles() {
        // Leading zero segment, consecutive middle zero segments, a tiny
        // segment shorter than sample_period, and a trailing zero segment.
        let waypoints = [
            [rev(0.0), rev(1.0)],
            [rev(0.0), rev(1.0)],    // leading zero-duration
            [rev(1.0), rev(0.2)],    // normal move
            [rev(1.0), rev(0.2)],    // middle zero-duration #1
            [rev(1.0), rev(0.2)],    // middle zero-duration #2
            [rev(1.0001), rev(0.2)], // shorter than dt = 0.05 s
            [rev(0.3), rev(-0.6)],   // normal move
            [rev(0.3), rev(-0.6)],   // trailing zero-duration
        ];
        let limits = CuJointTrajectoryLimits::new(
            [rev_vel(1.0), rev_vel(1.0)],
            [rev_acc(2.0), rev_acc(2.0)],
        )
        .with_jerks([rev_jerk(10.0), rev_jerk(10.0)]);
        let dt = Time::new::<second>(0.05);
        let mut out = CuTrajectoryPayload::<2, 256>::new();
        for profile in [
            CuTrajectoryProfile::Trapezoidal,
            CuTrajectoryProfile::SCurve,
            CuTrajectoryProfile::MinimumJerk,
            CuTrajectoryProfile::Cubic,
        ] {
            assert_matches_reference(
                &waypoints,
                &limits,
                profile,
                &CuTimingOverrides::default(),
                dt,
                &mut out,
            );
        }

        // All-zero-displacement path collapses to a single point for all profiles.
        let stationary = [
            [rev(0.5), rev(-0.2)],
            [rev(0.5), rev(-0.2)],
            [rev(0.5), rev(-0.2)],
        ];
        let mut single = CuTrajectoryPayload::<2, 16>::new();
        for profile in [
            CuTrajectoryProfile::Trapezoidal,
            CuTrajectoryProfile::SCurve,
            CuTrajectoryProfile::MinimumJerk,
            CuTrajectoryProfile::Cubic,
        ] {
            let status = with_profile(
                &stationary,
                &limits,
                profile,
                &CuTimingOverrides::default(),
                dt,
                &mut single,
            )
            .unwrap();
            assert_eq!(status, CuTimingStatus::Success);
            assert_eq!(single.len(), 1);
            assert_eq!(single.points()[0].positions, stationary[0]);
        }
    }

    #[test]
    fn test_fixed_durations_all_profiles_match_reference() {
        let waypoints = [
            [rev(0.0), rev(0.0)],
            [rev(0.5), rev(-0.3)],
            [rev(0.5), rev(-0.3)], // zero displacement with positive fixed duration
            [rev(1.2), rev(0.4)],
        ];
        let limits = CuJointTrajectoryLimits::new(
            [rev_vel(1.0), rev_vel(1.0)],
            [rev_acc(2.0), rev_acc(2.0)],
        )
        .with_jerks([rev_jerk(10.0), rev_jerk(10.0)]);
        // Include an exact sample-boundary alignment (dt = 0.05 divides 2.0, 0.25, 2.5).
        let fixed = [
            Time::new::<second>(2.0),
            Time::new::<second>(0.25),
            Time::new::<second>(2.5),
        ];
        let overrides = CuTimingOverrides {
            segment_durations: Some(&fixed),
            waypoint_velocities: None,
        };
        let dt = Time::new::<second>(0.05);
        let mut out = CuTrajectoryPayload::<2, 256>::new();
        for profile in [
            CuTrajectoryProfile::Trapezoidal,
            CuTrajectoryProfile::SCurve,
            CuTrajectoryProfile::MinimumJerk,
            CuTrajectoryProfile::Cubic,
        ] {
            assert_matches_reference(&waypoints, &limits, profile, &overrides, dt, &mut out);
            assert!((out.duration().get::<second>() - 4.75).abs() < 1e-12);
        }
    }

    #[test]
    fn test_nonzero_via_velocities_match_reference() {
        let waypoints = [
            [rev(0.0), rev(0.0), rev(1.0)],
            [rev(1.0), rev(-0.6), rev(0.5)],
            [rev(2.2), rev(-0.2), rev(-0.3)],
            [rev(1.8), rev(0.4), rev(0.0)],
        ];
        let vias = [
            [rev_vel(0.1), rev_vel(-0.1), rev_vel(0.0)],
            [rev_vel(0.6), rev_vel(-0.2), rev_vel(-0.4)],
            [rev_vel(0.2), rev_vel(0.3), rev_vel(-0.1)],
            [zero_vel(), zero_vel(), zero_vel()],
        ];
        let limits = CuJointTrajectoryLimits::new(
            [rev_vel(2.0), rev_vel(2.0), rev_vel(2.0)],
            [rev_acc(4.0), rev_acc(4.0), rev_acc(4.0)],
        );
        let fixed = [
            Time::new::<second>(1.5),
            Time::new::<second>(1.8),
            Time::new::<second>(1.2),
        ];
        let dt = Time::new::<second>(0.02);
        let mut out = CuTrajectoryPayload::<3, 512>::new();
        for durations in [None, Some(fixed.as_slice())] {
            let overrides = CuTimingOverrides {
                segment_durations: durations,
                waypoint_velocities: Some(&vias),
            };
            for profile in [CuTrajectoryProfile::MinimumJerk, CuTrajectoryProfile::Cubic] {
                assert_matches_reference(&waypoints, &limits, profile, &overrides, dt, &mut out);
            }
        }
    }

    #[test]
    fn test_mixed_joint_kinds_all_profiles_match_reference() {
        let waypoints = [
            [rev(0.0), pris(0.0), CuJointPosition::Fixed],
            [rev(0.8), pris(0.3), CuJointPosition::Fixed],
            [rev(0.8), pris(0.3), CuJointPosition::Fixed],
            [rev(-0.4), pris(0.05), CuJointPosition::Fixed],
        ];
        let limits = CuJointTrajectoryLimits::new(
            [rev_vel(0.8), pris_vel(0.25), CuJointVelocity::Fixed],
            [rev_acc(1.5), pris_acc(0.6), CuJointAcceleration::Fixed],
        )
        .with_jerks([rev_jerk(6.0), pris_jerk(3.0), CuJointJerk::Fixed]);
        let dt = Time::new::<second>(0.02);
        let mut out = CuTrajectoryPayload::<3, 512>::new();
        for profile in [
            CuTrajectoryProfile::Trapezoidal,
            CuTrajectoryProfile::SCurve,
            CuTrajectoryProfile::MinimumJerk,
            CuTrajectoryProfile::Cubic,
        ] {
            assert_matches_reference(
                &waypoints,
                &limits,
                profile,
                &CuTimingOverrides::default(),
                dt,
                &mut out,
            );
            for p in out.points() {
                assert_eq!(p.positions[2], CuJointPosition::Fixed);
                assert_eq!(p.velocities[2], CuJointVelocity::Fixed);
                assert_eq!(p.accelerations[2], CuJointAcceleration::Fixed);
            }
        }
    }

    #[test]
    fn test_scurve_infinite_and_mixed_jerks_match_reference() {
        let waypoints = [
            [rev(0.0), pris(0.0), CuJointPosition::Fixed],
            [rev(1.2), pris(0.4), CuJointPosition::Fixed],
            [rev(0.2), pris(0.1), CuJointPosition::Fixed],
        ];
        let dt = Time::new::<second>(0.02);

        // Mixed finite + infinite jerk + fixed joint.
        let mixed_limits = CuJointTrajectoryLimits::new(
            [rev_vel(0.8), pris_vel(0.3), CuJointVelocity::Fixed],
            [rev_acc(1.6), pris_acc(0.8), CuJointAcceleration::Fixed],
        )
        .with_jerks([rev_jerk(5.0), pris_jerk(f64::INFINITY), CuJointJerk::Fixed]);
        let fixed = [Time::new::<second>(3.0), Time::new::<second>(2.5)];
        let mut out = CuTrajectoryPayload::<3, 512>::new();
        for durations in [None, Some(fixed.as_slice())] {
            let overrides = CuTimingOverrides {
                segment_durations: durations,
                waypoint_velocities: None,
            };
            assert_matches_reference(
                &waypoints,
                &mixed_limits,
                CuTrajectoryProfile::SCurve,
                &overrides,
                dt,
                &mut out,
            );
        }

        // `max_jerks: None` on SCurve must match Trapezoidal point-for-point.
        let unbounded_limits = CuJointTrajectoryLimits::new(
            [rev_vel(0.8), pris_vel(0.3), CuJointVelocity::Fixed],
            [rev_acc(1.6), pris_acc(0.8), CuJointAcceleration::Fixed],
        );
        let mut scurve_out = CuTrajectoryPayload::<3, 512>::new();
        assert_matches_reference(
            &waypoints,
            &unbounded_limits,
            CuTrajectoryProfile::SCurve,
            &CuTimingOverrides::default(),
            dt,
            &mut scurve_out,
        );
        let mut trap_out = CuTrajectoryPayload::<3, 512>::new();
        assert_matches_reference(
            &waypoints,
            &unbounded_limits,
            CuTrajectoryProfile::Trapezoidal,
            &CuTimingOverrides::default(),
            dt,
            &mut trap_out,
        );
        assert_eq!(scurve_out.points(), trap_out.points());
    }
}
