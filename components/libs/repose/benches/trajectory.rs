//! Joint-space trajectory timing and sampling for a 6-DOF path.

use cu29::units::si::angle::degree;
use cu29::units::si::angular_acceleration::radian_per_second_squared;
use cu29::units::si::angular_jerk::radian_per_second_cubed;
use cu29::units::si::angular_velocity::radian_per_second;
use cu29::units::si::f64::Angle;
use cu29::units::si::f64::AngularAcceleration;
use cu29::units::si::f64::AngularJerk;
use cu29::units::si::f64::AngularVelocity;
use cu29::units::si::f64::Time;
use cu29::units::si::time::second;
use divan::Bencher;
use divan::black_box;
use repose::kinematics::CuJointAcceleration;
use repose::kinematics::CuJointJerk;
use repose::kinematics::CuJointPosition;
use repose::kinematics::CuJointVelocity;
use repose::trajectory::CuJointTrajectoryLimits;
use repose::trajectory::CuTimingOverrides;
use repose::trajectory::CuTimingParams;
use repose::trajectory::CuTimingStatus;
use repose::trajectory::CuTrajectoryPayload;
use repose::trajectory::CuTrajectoryProfile;
use repose::trajectory::parameterize;

fn main() {
    divan::main();
}

const CAPACITY: usize = 1024;

fn deg(values: [f64; 6]) -> [CuJointPosition; 6] {
    values.map(|d| CuJointPosition::Revolute(Angle::new::<degree>(d)))
}

fn waypoints() -> [[CuJointPosition; 6]; 5] {
    [
        deg([-1.1, 45.5, 60.0, -3.3, -80.9, 28.7]),
        deg([10.0, 40.0, 55.0, 5.0, -40.0, 20.0]),
        deg([20.0, 35.0, 50.0, 10.0, 0.0, 10.0]),
        deg([10.0, 30.0, 55.0, 5.0, 30.0, 5.0]),
        deg([0.0, 30.0, 60.0, 0.0, 60.0, 0.0]),
    ]
}

fn timing_params(profile: CuTrajectoryProfile) -> CuTimingParams<6> {
    CuTimingParams {
        limits: CuJointTrajectoryLimits::new(
            [CuJointVelocity::Revolute(AngularVelocity::new::<radian_per_second>(1.0)); 6],
            [CuJointAcceleration::Revolute(AngularAcceleration::new::<radian_per_second_squared>(
                2.0,
            )); 6],
        )
        .with_jerks([CuJointJerk::Revolute(AngularJerk::new::<radian_per_second_cubed>(10.0)); 6]),
        profile,
        sample_period: Time::new::<second>(0.01),
    }
}

const PROFILES: [CuTrajectoryProfile; 4] = [
    CuTrajectoryProfile::Trapezoidal,
    CuTrajectoryProfile::SCurve,
    CuTrajectoryProfile::MinimumJerk,
    CuTrajectoryProfile::Cubic,
];

#[divan::bench(args = PROFILES)]
fn parameterize_profile(bencher: Bencher, profile: CuTrajectoryProfile) {
    let waypoints = waypoints();
    let params = timing_params(profile);
    let mut out = Box::new(CuTrajectoryPayload::<6, CAPACITY>::new());
    let status = parameterize(&waypoints, &params, CuTimingOverrides::default(), &mut out).unwrap();
    assert_eq!(status, CuTimingStatus::Success, "{profile:?}");
    bencher.bench_local(|| {
        parameterize(
            black_box(&waypoints),
            black_box(&params),
            CuTimingOverrides::default(),
            black_box(&mut out),
        )
        .unwrap()
    });
}

/// Samples the whole timed trajectory at 1 kHz, as a follower task would.
#[divan::bench]
fn sample_quintic_1khz(bencher: Bencher) {
    let waypoints = waypoints();
    let params = timing_params(CuTrajectoryProfile::SCurve);
    let mut traj = Box::new(CuTrajectoryPayload::<6, CAPACITY>::new());
    parameterize(&waypoints, &params, CuTimingOverrides::default(), &mut traj).unwrap();
    let samples = (traj.duration().get::<second>() * 1000.0) as usize;
    let mut positions = [CuJointPosition::default(); 6];
    let mut velocities = [CuJointVelocity::default(); 6];
    bencher.bench_local(|| {
        for i in 0..samples {
            let t = Time::new::<second>(i as f64 * 1e-3);
            traj.sample_quintic(black_box(t), &mut positions, &mut velocities)
                .unwrap();
        }
    });
}
