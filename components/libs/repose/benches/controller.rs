//! Unified joint controller hot paths on the 6-DOF igus ReBeL: rate-limited
//! position tracking, pure-PD effort mode, model-based inverse-dynamics effort
//! mode, and the gravity-compensated hold brake.

use cu29::prelude::*;
use cu29::units::si::angle::degree;
use cu29::units::si::angular_acceleration::radian_per_second_squared;
use cu29::units::si::angular_velocity::radian_per_second;
use cu29::units::si::f64::Angle;
use cu29::units::si::f64::AngularAcceleration;
use cu29::units::si::f64::AngularVelocity;
use cu29::units::si::f64::Length;
use cu29::units::si::f64::Mass;
use cu29::units::si::f64::MomentOfInertia;
use cu29::units::si::f64::Time;
use cu29::units::si::length::meter;
use cu29::units::si::mass::kilogram;
use cu29::units::si::moment_of_inertia::kilogram_square_meter;
use cu29::units::si::time::second;
use divan::Bencher;
use divan::black_box;
use repose::controller::CuControllerMode;
use repose::controller::CuControllerParams;
use repose::controller::CuControllerStatus;
use repose::controller::CuControllerTask;
use repose::controller::CuEffortGains;
use repose::controller::CuEffortModel;
use repose::controller::CuJointCommandPayload;
use repose::controller::CuRateLimit;
use repose::kinematics::CuChainInertia;
use repose::kinematics::CuJointAcceleration;
use repose::kinematics::CuJointPosition;
use repose::kinematics::CuJointStatePayload;
use repose::kinematics::CuJointVelocity;
use repose::kinematics::CuLinkInertia;
use repose::kinematics::CuRobotPreset;
use repose::kinematics::CuRobotStatus;
use repose::trajectory::CuFollowStatus;
use repose::trajectory::CuTrajectoryFollowOutput;

fn main() {
    divan::main();
}

fn deg(values: [f64; 6]) -> [CuJointPosition; 6] {
    values.map(|d| CuJointPosition::Revolute(Angle::new::<degree>(d)))
}

fn vel(values: [f64; 6]) -> [CuJointVelocity; 6] {
    values.map(|v| CuJointVelocity::Revolute(AngularVelocity::new::<radian_per_second>(v)))
}

fn acc(values: [f64; 6]) -> [CuJointAcceleration; 6] {
    values.map(|a| {
        CuJointAcceleration::Revolute(AngularAcceleration::new::<radian_per_second_squared>(a))
    })
}

fn cylinder_link(mass: f64, radius: f64, length: f64) -> CuLinkInertia {
    let transverse = mass * (3.0 * radius * radius + length * length) / 12.0;
    let axial = mass * radius * radius / 2.0;
    CuLinkInertia::new(
        Mass::new::<kilogram>(mass),
        [0.0, 0.0, length / 2.0].map(Length::new::<meter>),
        [transverse, transverse, axial, 0.0, 0.0, 0.0]
            .map(MomentOfInertia::new::<kilogram_square_meter>),
    )
    .unwrap()
}

fn rebel_effort_model() -> CuEffortModel<6> {
    let inertia = CuChainInertia::new([
        cylinder_link(1.6, 0.05, 0.106),
        cylinder_link(1.9, 0.045, 0.2415),
        cylinder_link(1.3, 0.045, 0.1575),
        cylinder_link(1.0, 0.04, 0.142),
        cylinder_link(0.7, 0.04, 0.0768),
        cylinder_link(0.3, 0.035, 0.04),
    ])
    .unwrap();
    CuEffortModel::new(CuRobotPreset::IgusRebel, inertia)
}

fn rebel_gains() -> CuEffortGains<6> {
    CuEffortGains {
        kp: [60.0, 60.0, 40.0, 12.0, 8.0, 4.0],
        kv: [4.0, 4.0, 2.5, 1.0, 0.8, 0.4],
    }
}

fn fixtures(
    status: CuFollowStatus,
) -> (
    CuMsg<CuTrajectoryFollowOutput<6>>,
    CuMsg<CuJointStatePayload<6>>,
) {
    let mut setpoint = CuMsg::new(Some(CuTrajectoryFollowOutput::<6> {
        positions: deg([10.5, 36.0, 54.0, 2.0, -39.0, 20.5]),
        velocities: vel([0.3, -0.2, 0.4, -0.1, 0.5, -0.2]),
        accelerations: acc([0.8, -0.5, 1.0, -0.3, 1.2, -0.6]),
        status,
    }));
    let mut state = CuMsg::new(Some(CuJointStatePayload::<6> {
        status: CuRobotStatus::Ready,
        positions: deg([10.0, 35.5, 54.5, 1.8, -39.5, 20.0]),
        velocities: vel([0.25, -0.15, 0.35, -0.08, 0.45, -0.15]),
    }));
    let stamp = Tov::Time(CuTime::from_nanos(1_000_000));
    setpoint.tov = stamp;
    state.tov = stamp;
    (setpoint, state)
}

#[divan::bench]
fn controller_position_rate_limited(bencher: Bencher) {
    let (ctx, clock) = CuContext::new_mock_clock();
    clock.set_value(1_000_000);
    let mut task = CuControllerTask::<6>::from_params(CuControllerParams {
        mode: CuControllerMode::Position,
        state_timeout: Some(Time::new::<second>(0.2)),
        rate_limit: Some(CuRateLimit {
            max_speed: 1.5,
            max_acceleration: 6.0,
        }),
        ..CuControllerParams::default()
    })
    .unwrap();
    let (mut setpoint, mut state) = fixtures(CuFollowStatus::Running);
    let mut out = CuMsg::<CuJointCommandPayload<6>>::default();
    let mut ns = 1_000_000u64;
    task.process(&ctx, &(&setpoint, &state), &mut out).unwrap();
    assert_eq!(out.payload().unwrap().status, CuControllerStatus::Running);
    bencher.bench_local(|| {
        ns = ns.wrapping_add(1_000_000);
        clock.set_value(ns);
        let stamp = Tov::Time(CuTime::from_nanos(ns));
        setpoint.tov = stamp;
        state.tov = stamp;
        task.process(&ctx, black_box(&(&setpoint, &state)), black_box(&mut out))
            .unwrap();
    });
}

#[divan::bench]
fn controller_effort_pd(bencher: Bencher) {
    let (ctx, _clock) = CuContext::new_mock_clock();
    let mut task = CuControllerTask::<6>::from_params(CuControllerParams {
        mode: CuControllerMode::Effort,
        effort_gains: rebel_gains(),
        ..CuControllerParams::default()
    })
    .unwrap();
    let (setpoint, state) = fixtures(CuFollowStatus::Running);
    let mut out = CuMsg::<CuJointCommandPayload<6>>::default();
    task.process(&ctx, &(&setpoint, &state), &mut out).unwrap();
    assert_eq!(out.payload().unwrap().status, CuControllerStatus::Running);
    bencher.bench_local(|| {
        task.process(&ctx, black_box(&(&setpoint, &state)), black_box(&mut out))
            .unwrap();
    });
}

#[divan::bench]
fn controller_effort_model_based(bencher: Bencher) {
    let (ctx, _clock) = CuContext::new_mock_clock();
    let mut task = CuControllerTask::<6>::from_params(CuControllerParams {
        mode: CuControllerMode::Effort,
        effort_gains: rebel_gains(),
        effort_model: Some(rebel_effort_model()),
        ..CuControllerParams::default()
    })
    .unwrap();
    let (setpoint, state) = fixtures(CuFollowStatus::Running);
    let mut out = CuMsg::<CuJointCommandPayload<6>>::default();
    task.process(&ctx, &(&setpoint, &state), &mut out).unwrap();
    assert_eq!(out.payload().unwrap().status, CuControllerStatus::Running);
    bencher.bench_local(|| {
        task.process(&ctx, black_box(&(&setpoint, &state)), black_box(&mut out))
            .unwrap();
    });
}

#[divan::bench]
fn controller_effort_hold_brake(bencher: Bencher) {
    let (ctx, _clock) = CuContext::new_mock_clock();
    let mut task = CuControllerTask::<6>::from_params(CuControllerParams {
        mode: CuControllerMode::Effort,
        effort_gains: rebel_gains(),
        effort_model: Some(rebel_effort_model()),
        ..CuControllerParams::default()
    })
    .unwrap();
    let (_setpoint, state) = fixtures(CuFollowStatus::Idle);
    let empty_setpoint = CuMsg::<CuTrajectoryFollowOutput<6>>::new(None);
    let mut out = CuMsg::<CuJointCommandPayload<6>>::default();
    task.process(&ctx, &(&empty_setpoint, &state), &mut out)
        .unwrap();
    assert_eq!(out.payload().unwrap().status, CuControllerStatus::Idle);
    bencher.bench_local(|| {
        task.process(
            &ctx,
            black_box(&(&empty_setpoint, &state)),
            black_box(&mut out),
        )
        .unwrap();
    });
}
