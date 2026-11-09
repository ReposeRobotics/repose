# Continuous motion and TCP zones

`repose_trajectory::cartesian_blend::blend_cartesian_path` takes a serial chain,
collision model, joint waypoints, and one typed `Length` zone radius per
waypoint. A zero interior radius is an exact stop; a positive radius replaces
that TCP corner with a sampled quadratic fly-by path. The first and last radii
must be zero. Tool orientation follows shortest-arc interpolation, and IK is
seeded from the previous solution. Input/output capacity is fixed at compile
time; invalid geometry, IK failure, collision, or capacity exhaustion clears the
output and returns a status.

Geometry and timing are separate: the zone is a property of each waypoint,
the profile a property of the motion. `path_timing::parameterize_cartesian_path`
times the path with any `CuTrajectoryProfile` from `CuPathTimingConfig::profile`.
Every profile flies through every positive zone and stops exactly at zero
zones. The knots are joined by a C² natural cubic spline over a speed-weighted
joint-space arc length; exact stops split the path into rest-to-rest pieces,
each zone is crossed at a constant corner speed bounded by its curvature, and
the moves between stops and corners follow the profile from one boundary speed
to the next. Joint velocity and acceleration stay within the configured limits;
S-curve joint jerk is checked on the quintic reconstruction, including curvature
contributions. Every profile is certified on the reconstruction its follower
executes — quintic Hermite for the S-curve, cubic Hermite for the others — and
uniform time stretching brings any excess within the limits. The geometric spline is
rechecked for joint limits, collisions, and zone containment, and a path that
reverses inside a zone is rejected with `NoContinuousMotion` naming the
waypoint. Read the result with
`CuTrajectoryPayload::sample_quintic` for S-curve motion, or configure a
`CuFollowTrajectoryTask` with `config: {"interpolation": "quintic"}`. Other
profiles use `sample_cubic` and the `cubic` follower mode. Match the follower's `MAX_POINTS`
type parameter to the generated payload capacity. The default follower
interpolation remains linear. Interpolation mode is included in newly frozen follower state;
earlier frozen snapshots need migration.

Timing is conservative and automatic. Finite joint jerk bounds apply to the
S-curve profile with quintic execution. Collision and zone checks apply to the geometric spline; the reconstructed
execution curve still needs a separate collision and zone recheck. Checks are discrete and depend on their configured
resolution; they are not swept-volume collision guarantees. The zone radius is a
TCP-position tolerance, not a separate orientation tolerance or an exact
reproduction of a vendor controller's zone algorithm.

