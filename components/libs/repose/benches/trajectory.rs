//! Joint-space and Cartesian trajectory timing, sampling, following, and
//! bumpless setpoint multiplexing for a 6-DOF path.

use cu29::prelude::*;
use cu29::units::si::angle::degree;
use cu29::units::si::angular_acceleration::radian_per_second_squared;
use cu29::units::si::angular_jerk::radian_per_second_cubed;
use cu29::units::si::angular_velocity::radian_per_second;
use cu29::units::si::f64::Angle;
use cu29::units::si::f64::AngularAcceleration;
use cu29::units::si::f64::AngularJerk;
use cu29::units::si::f64::AngularVelocity;
use cu29::units::si::f64::Length;
use cu29::units::si::f64::Time;
use cu29::units::si::length::meter;
use cu29::units::si::time::second;
use divan::Bencher;
use divan::black_box;
use repose::collision::CuCollisionEnv;
use repose::collision::igus_rebel_collision_model;
use repose::kinematics::CuJointAcceleration;
use repose::kinematics::CuJointJerk;
use repose::kinematics::CuJointPosition;
use repose::kinematics::CuJointVelocity;
use repose::kinematics::igus_rebel_chain;
use repose::trajectory::CuCartesianBlendParams;
use repose::trajectory::CuCartesianBlendPath;
use repose::trajectory::CuCartesianBlendStatus;
use repose::trajectory::CuCartesianWaypoint;
use repose::trajectory::CuFollowStatus;
use repose::trajectory::CuFollowTrajectoryTask;
use repose::trajectory::CuJointTrajectoryLimits;
use repose::trajectory::CuPathTimingParams;
use repose::trajectory::CuSetpointMuxParams;
use repose::trajectory::CuSetpointMuxTask;
use repose::trajectory::CuTimingOverrides;
use repose::trajectory::CuTimingParams;
use repose::trajectory::CuTimingStatus;
use repose::trajectory::CuTrajectoryFollowOutput;
use repose::trajectory::CuTrajectoryPayload;
use repose::trajectory::CuTrajectoryProfile;
use repose::trajectory::blend_cartesian_path;
use repose::trajectory::parameterize;
use repose::trajectory::parameterize_cartesian_path;

fn main() {
    divan::main();
}

const CAPACITY: usize = 1024;
const CARTESIAN_PATH_CAPACITY: usize = 128;

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

/// Steps [`CuFollowTrajectoryTask`] across an adopted S-curve trajectory at 1 kHz,
/// producing positions, velocities, and accelerations on every control cycle.
#[divan::bench]
fn follow_trajectory_1khz(bencher: Bencher) {
    let waypoints = waypoints();
    let params = timing_params(CuTrajectoryProfile::SCurve);
    let mut traj = Box::new(CuTrajectoryPayload::<6, CAPACITY>::new());
    parameterize(&waypoints, &params, CuTimingOverrides::default(), &mut traj).unwrap();
    let samples = (traj.duration().get::<second>() * 1000.0) as u64;
    let (ctx, _clock) = CuContext::new_mock_clock();
    let mut follower = Box::new(CuFollowTrajectoryTask::<6, CAPACITY>::default());
    let mut adopt = Box::new(CuMsg::new(Some((*traj).clone())));
    adopt.tov = Tov::Time(CuTime::from_nanos(0));
    let mut idle_in = Box::new(CuMsg::<CuTrajectoryPayload<6, CAPACITY>>::new(None));
    let mut out = CuMsg::<CuTrajectoryFollowOutput<6>>::default();
    follower.process(&ctx, &adopt, &mut out).unwrap();
    bencher.bench_local(|| {
        adopt.tov = Tov::Time(CuTime::from_nanos(0));
        follower.process(&ctx, &adopt, black_box(&mut out)).unwrap();
        for i in 1..samples {
            idle_in.tov = Tov::Time(CuTime::from_nanos(i * 1_000_000));
            follower
                .process(&ctx, &idle_in, black_box(&mut out))
                .unwrap();
        }
    });
}

/// Steady-state [`CuSetpointMuxTask`] cycle passing through a running setpoint.
#[divan::bench]
fn setpoint_mux_passthrough(bencher: Bencher) {
    let (ctx, _clock) = CuContext::new_mock_clock();
    let mut mux = CuSetpointMuxTask::<6>::default();
    let mut classical = CuMsg::new(Some(CuTrajectoryFollowOutput::<6> {
        positions: deg([10.0, 35.0, 50.0, 2.0, -30.0, 15.0]),
        velocities: [CuJointVelocity::Revolute(AngularVelocity::new::<radian_per_second>(0.2)); 6],
        accelerations: [CuJointAcceleration::Revolute(AngularAcceleration::new::<
            radian_per_second_squared,
        >(0.5)); 6],
        status: CuFollowStatus::Running,
    }));
    classical.tov = Tov::Time(CuTime::from_nanos(1_000_000));
    let policy = CuMsg::<CuTrajectoryFollowOutput<6>>::new(None);
    let mut out = CuMsg::<CuTrajectoryFollowOutput<6>>::default();
    mux.process(&ctx, &(&classical, &policy), &mut out).unwrap();
    bencher.bench_local(|| {
        mux.process(&ctx, black_box(&(&classical, &policy)), black_box(&mut out))
            .unwrap();
    });
}

/// Full 200 ms bumpless quintic handoff (plus completion at 201 ms) from
/// classical to policy inside [`CuSetpointMuxTask`].
#[divan::bench]
fn setpoint_mux_blend_1khz(bencher: Bencher) {
    let (ctx, _clock) = CuContext::new_mock_clock();
    let mut arm_classical = CuMsg::new(Some(CuTrajectoryFollowOutput::<6> {
        positions: deg([0.0, 30.0, 60.0, 0.0, 45.0, 0.0]),
        velocities: [CuJointVelocity::Revolute(AngularVelocity::new::<radian_per_second>(0.1)); 6],
        accelerations: [CuJointAcceleration::Revolute(AngularAcceleration::new::<
            radian_per_second_squared,
        >(0.0)); 6],
        status: CuFollowStatus::Running,
    }));
    arm_classical.tov = Tov::Time(CuTime::from_nanos(0));
    let mut done_classical = CuMsg::new(Some(CuTrajectoryFollowOutput::<6> {
        positions: deg([0.0, 30.0, 60.0, 0.0, 45.0, 0.0]),
        velocities: [CuJointVelocity::Revolute(AngularVelocity::new::<radian_per_second>(0.0)); 6],
        accelerations: [CuJointAcceleration::Revolute(AngularAcceleration::new::<
            radian_per_second_squared,
        >(0.0)); 6],
        status: CuFollowStatus::Finished,
    }));
    let mut policy = CuMsg::new(Some(CuTrajectoryFollowOutput::<6> {
        positions: deg([2.0, 32.0, 58.0, 1.0, 43.0, -2.0]),
        velocities: [CuJointVelocity::Revolute(AngularVelocity::new::<radian_per_second>(0.15)); 6],
        accelerations: [CuJointAcceleration::Revolute(AngularAcceleration::new::<
            radian_per_second_squared,
        >(0.3)); 6],
        status: CuFollowStatus::Running,
    }));
    let empty_policy = CuMsg::<CuTrajectoryFollowOutput<6>>::new(None);
    let mut out = CuMsg::<CuTrajectoryFollowOutput<6>>::default();
    bencher.bench_local(|| {
        let mut mux = CuSetpointMuxTask::<6>::from_params(CuSetpointMuxParams::default()).unwrap();
        mux.process(&ctx, &(&arm_classical, &empty_policy), &mut out)
            .unwrap();
        for cycle in 1..=201u64 {
            let t = Tov::Time(CuTime::from_nanos(cycle * 1_000_000));
            done_classical.tov = t;
            policy.tov = t;
            mux.process(
                &ctx,
                black_box(&(&done_classical, &policy)),
                black_box(&mut out),
            )
            .unwrap();
        }
    });
}

/// Cartesian TCP corner blending followed by S-curve path parameterization on
/// the 6-DOF igus ReBeL.
#[divan::bench]
fn blend_and_time_cartesian_path(bencher: Bencher) {
    let chain = igus_rebel_chain();
    let model = igus_rebel_collision_model();
    let env = CuCollisionEnv::EMPTY;
    let waypoints = [
        CuCartesianWaypoint::new(
            deg([0.0, 35.0, 55.0, 0.0, 40.0, 0.0]),
            Length::new::<meter>(0.0),
        ),
        CuCartesianWaypoint::new(
            deg([12.0, 40.0, 50.0, 4.0, 35.0, 6.0]),
            Length::new::<meter>(0.015),
        ),
        CuCartesianWaypoint::new(
            deg([24.0, 36.0, 54.0, 0.0, 38.0, 12.0]),
            Length::new::<meter>(0.0),
        ),
    ];
    let blend_params = CuCartesianBlendParams::default();
    let path_timing = CuPathTimingParams {
        timing: timing_params(CuTrajectoryProfile::SCurve),
        ..Default::default()
    };
    let mut blended = Box::new(CuCartesianBlendPath::<6, CARTESIAN_PATH_CAPACITY>::default());
    let mut timed = Box::new(CuTrajectoryPayload::<6, CAPACITY>::new());
    assert_eq!(
        blend_cartesian_path(
            &chain,
            &model,
            &env,
            &waypoints,
            &blend_params,
            &mut blended
        )
        .unwrap(),
        CuCartesianBlendStatus::Success,
    );
    assert_eq!(
        parameterize_cartesian_path(&chain, &model, &env, &blended, &path_timing, &mut timed)
            .unwrap(),
        CuTimingStatus::Success,
    );
    bencher.bench_local(|| {
        let blend_status = blend_cartesian_path(
            black_box(&chain),
            black_box(&model),
            black_box(&env),
            black_box(&waypoints),
            black_box(&blend_params),
            black_box(&mut blended),
        )
        .unwrap();
        assert_eq!(blend_status, CuCartesianBlendStatus::Success);
        let timing_status = parameterize_cartesian_path(
            black_box(&chain),
            black_box(&model),
            black_box(&env),
            black_box(&blended),
            black_box(&path_timing),
            black_box(&mut timed),
        )
        .unwrap();
        assert_eq!(timing_status, CuTimingStatus::Success);
    });
}
