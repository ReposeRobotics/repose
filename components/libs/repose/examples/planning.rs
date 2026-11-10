//! Joint-space planning around an obstacle, then trajectory timing.
//!
//! A sphere blocks the straight joint-space line from start to goal, so the
//! planner has to route around it; the resulting waypoints are timed into a
//! trajectory. No simulator, no Copper task graph: plain library calls with a
//! fixed seed, so every run prints the same numbers. Run with:
//! `cargo run -p repose --example planning`

use cu_spatial_payloads::Transform3D;
use cu29::prelude::CuError;
use cu29::prelude::CuResult;
use cu29::units::si::angle::degree;
use cu29::units::si::angular_acceleration::radian_per_second_squared;
use cu29::units::si::angular_velocity::radian_per_second;
use cu29::units::si::f64::Angle;
use cu29::units::si::f64::AngularAcceleration;
use cu29::units::si::f64::AngularVelocity;
use cu29::units::si::f64::Length;
use cu29::units::si::f64::Time;
use cu29::units::si::length::meter;
use cu29::units::si::time::second;
use repose::collision::CuCollisionShape;
use repose::collision::CuSphere;
use repose::collision::igus_rebel_collision_model;
use repose::kinematics::CuJointAcceleration;
use repose::kinematics::CuJointPosition;
use repose::kinematics::CuJointVelocity;
use repose::kinematics::igus_rebel_chain;
use repose::motion_planning::CuJointPlanPath;
use repose::motion_planning::CuJointPlanRequest;
use repose::motion_planning::CuJointPlanner;
use repose::motion_planning::MAX_WAYPOINTS;
use repose::planners::CuAnyPlanner;
use repose::planners::CuPlannerChoice;
use repose::trajectory::CuJointTrajectoryLimits;
use repose::trajectory::CuTimingOverrides;
use repose::trajectory::CuTimingParams;
use repose::trajectory::CuTimingStatus;
use repose::trajectory::CuTrajectoryPayload;
use repose::trajectory::CuTrajectoryProfile;
use repose::trajectory::parameterize;

/// Joint angles in degrees, as typed revolute positions.
fn deg(values: [f64; 6]) -> [CuJointPosition; 6] {
    values.map(|d| CuJointPosition::Revolute(Angle::new::<degree>(d)))
}

/// Start and goal in degrees; the goal is far enough from the start that the
/// mid-point obstacle blocks the straight joint-space line.
const START: [f64; 6] = [-1.1, 45.5, 60.0, -3.3, -80.9, 28.7];
const GOAL: [f64; 6] = [90.0, 30.0, 60.0, 0.0, 60.0, 0.0];

fn main() -> CuResult<()> {
    let chain = igus_rebel_chain();
    let model = igus_rebel_collision_model();
    let start = deg(START);
    let goal = deg(GOAL);
    // Half-way along the straight joint line from start to goal.
    let mid = deg(core::array::from_fn(|j| 0.5 * (START[j] + GOAL[j])));

    // The obstacle: a 10 cm sphere centered on the tool point of `mid`, so
    // the straight joint-space line runs through it.
    let mut request = CuJointPlanRequest {
        start,
        goal,
        ..Default::default()
    };
    let mut tcp = Transform3D::<f64>::identity();
    chain.forward_kinematics(&mid, &mut tcp)?;
    let sphere = CuSphere::new(tcp.translation(), Length::new::<meter>(0.1))?;
    request
        .scene
        .add_obstacle(CuCollisionShape::Sphere(sphere), None)?;
    let env = request.scene.collision_env();

    // The example only shows something if the endpoints are free but the
    // straight line between them is blocked.
    for (name, q) in [("start", &start), ("goal", &goal)] {
        if !model.is_configuration_collision_free(&chain, q, &env)? {
            return Err(CuError::from(format!("{name} is in collision")));
        }
    }
    if model.is_segment_collision_free(&chain, &start, &goal, Angle::new::<degree>(2.0), &env)? {
        return Err(CuError::from(
            "straight-line path is not blocked; the example shows nothing",
        ));
    }
    println!("straight line: blocked");

    // `6` is the joint count of the igus ReBeL. The seed makes the sampling
    // planner deterministic: the same seed replays the same job bit for bit,
    // which is what Copper's record-and-replay relies on.
    let mut planner = CuAnyPlanner::<6>::new(chain, model, CuPlannerChoice::RrtConnect)?;
    let mut status = planner.begin(&request, 1)?;
    let mut steps = 0;
    while !status.is_final() && steps < 400 {
        status = planner.step(64)?;
        steps += 1;
    }
    let mut path = CuJointPlanPath::<6>::default();
    status = planner.write_best(&mut path);
    if !status.has_path() {
        return Err(CuError::from(format!("planning failed: {status:?}")));
    }
    println!(
        "plan: {} waypoints, cost {:.3}, status {status:?}",
        path.len, path.cost
    );

    // Time the waypoints into a trajectory: 1 rad/s and 2 rad/s² per joint,
    // sampled every 10 ms into at most 1024 points (`<6, 1024>`).
    let mut waypoints = [[CuJointPosition::default(); 6]; MAX_WAYPOINTS];
    for (i, waypoint) in path.waypoints().iter().enumerate() {
        waypoints[i] = waypoint.positions;
    }
    let params = CuTimingParams {
        limits: CuJointTrajectoryLimits::new(
            [CuJointVelocity::Revolute(AngularVelocity::new::<radian_per_second>(1.0)); 6],
            [CuJointAcceleration::Revolute(AngularAcceleration::new::<radian_per_second_squared>(
                2.0,
            )); 6],
        ),
        profile: CuTrajectoryProfile::Trapezoidal,
        sample_period: Time::new::<second>(0.01),
    };
    let mut trajectory = Box::new(CuTrajectoryPayload::<6, 1024>::new());
    let timing = parameterize(
        &waypoints[..path.len as usize],
        &params,
        CuTimingOverrides::default(),
        &mut trajectory,
    )?;
    if timing != CuTimingStatus::Success {
        return Err(CuError::from(format!("timing failed: {timing:?}")));
    }
    println!(
        "trajectory: {} points, duration {:.3} s",
        trajectory.points().len(),
        trajectory.duration().get::<second>()
    );
    Ok(())
}
