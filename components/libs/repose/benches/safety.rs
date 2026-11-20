//! Safety shield overhead on the 6-DOF igus ReBeL with an 11-obstacle scene,
//! workspace fence, TCP speed check, and optional braking-horizon lookahead.

use cu29::units::si::angle::degree;
use cu29::units::si::angular_velocity::radian_per_second;
use cu29::units::si::f64::Angle;
use cu29::units::si::f64::AngularVelocity;
use cu29::units::si::f64::Length;
use cu29::units::si::f64::Time;
use cu29::units::si::f64::Velocity;
use cu29::units::si::length::meter;
use cu29::units::si::time::second;
use cu29::units::si::velocity::meter_per_second;
use divan::Bencher;
use divan::black_box;
use repose::collision::CuCapsule;
use repose::collision::CuCollisionShape;
use repose::collision::CuPlanningScene;
use repose::collision::CuSphere;
use repose::kinematics::CuJointAcceleration;
use repose::kinematics::CuJointPosition;
use repose::kinematics::CuJointStatePayload;
use repose::kinematics::CuJointVelocity;
use repose::kinematics::CuRobotPreset;
use repose::kinematics::CuRobotStatus;
use repose::safety::CuSafetyReport;
use repose::safety::CuSafetyShieldParams;
use repose::safety::CuSafetyShieldTask;
use repose::safety::CuSafetyVerdict;
use repose::safety::CuWorkspaceBounds;
use repose::trajectory::CuFollowStatus;
use repose::trajectory::CuTrajectoryFollowOutput;

fn main() {
    divan::main();
}

fn deg(values: [f64; 6]) -> [CuJointPosition; 6] {
    values.map(|d| CuJointPosition::Revolute(Angle::new::<degree>(d)))
}

fn m(v: [f64; 3]) -> [Length; 3] {
    v.map(Length::new::<meter>)
}

const START: [f64; 6] = [-1.1, 45.5, 60.0, -3.3, -80.9, 28.7];

fn cluttered_scene() -> CuPlanningScene {
    let mut scene = CuPlanningScene::new();
    let mut add = |shape| {
        scene.add_obstacle(shape, None).unwrap();
    };
    add(CuCollisionShape::Capsule(
        CuCapsule::new(
            m([-1.5, -1.5, -0.2]),
            m([1.5, -1.5, -0.2]),
            Length::new::<meter>(0.05),
        )
        .unwrap(),
    ));
    add(CuCollisionShape::Capsule(
        CuCapsule::new(
            m([-1.2, 1.2, 0.0]),
            m([-1.2, 1.2, 1.5]),
            Length::new::<meter>(0.05),
        )
        .unwrap(),
    ));
    add(CuCollisionShape::Capsule(
        CuCapsule::new(
            m([1.2, 1.2, 0.0]),
            m([1.2, 1.2, 1.5]),
            Length::new::<meter>(0.05),
        )
        .unwrap(),
    ));
    for i in 0..8 {
        let a = core::f64::consts::TAU * f64::from(i) / 8.0;
        add(CuCollisionShape::Sphere(
            CuSphere::new(
                m([1.6 * a.cos(), 1.6 * a.sin(), 1.8]),
                Length::new::<meter>(0.08),
            )
            .unwrap(),
        ));
    }
    scene
}

fn shield_params(braking_horizon: Option<Time>) -> CuSafetyShieldParams<6> {
    CuSafetyShieldParams {
        chain: Some(CuRobotPreset::IgusRebel.into()),
        collision_model: Some(CuRobotPreset::IgusRebel.into()),
        workspace: Some(CuWorkspaceBounds::new(m([-1.0, -1.0, -0.1]), m([1.0, 1.0, 1.5])).unwrap()),
        braking_horizon,
        max_tcp_speed: Some(Velocity::new::<meter_per_second>(2.0)),
        ..Default::default()
    }
}

#[divan::bench]
fn safety_shield_clear(bencher: Bencher) {
    let mut task = CuSafetyShieldTask::<6>::from_params(shield_params(None)).unwrap();
    let scene = cluttered_scene();
    let positions = deg(START);
    let setpoint = CuTrajectoryFollowOutput {
        positions,
        velocities: [CuJointVelocity::Revolute(AngularVelocity::new::<radian_per_second>(0.0)); 6],
        accelerations: positions.map(|p| CuJointAcceleration::from_coordinate(p, 0.0)),
        status: CuFollowStatus::Running,
    };
    let state = CuJointStatePayload {
        status: CuRobotStatus::Ready,
        positions,
        velocities: setpoint.velocities,
    };
    let mut out_sp = CuTrajectoryFollowOutput::default();
    let mut out_rep = CuSafetyReport::default();

    bencher.bench_local(|| {
        task.evaluate(
            black_box(Some(&setpoint)),
            black_box(Some(&state)),
            black_box(Some(&scene)),
            black_box(&mut out_sp),
            black_box(&mut out_rep),
        )
        .unwrap();
        assert_eq!(out_rep.verdict, CuSafetyVerdict::Clear);
    });
}

#[divan::bench]
fn safety_shield_braking_lookahead(bencher: Bencher) {
    let mut task =
        CuSafetyShieldTask::<6>::from_params(shield_params(Some(Time::new::<second>(0.1))))
            .unwrap();
    let scene = cluttered_scene();
    let positions = deg(START);
    let setpoint = CuTrajectoryFollowOutput {
        positions,
        velocities: [CuJointVelocity::Revolute(AngularVelocity::new::<radian_per_second>(0.1)); 6],
        accelerations: positions.map(|p| CuJointAcceleration::from_coordinate(p, 0.0)),
        status: CuFollowStatus::Running,
    };
    let state = CuJointStatePayload {
        status: CuRobotStatus::Ready,
        positions,
        velocities: setpoint.velocities,
    };
    let mut out_sp = CuTrajectoryFollowOutput::default();
    let mut out_rep = CuSafetyReport::default();

    bencher.bench_local(|| {
        task.evaluate(
            black_box(Some(&setpoint)),
            black_box(Some(&state)),
            black_box(Some(&scene)),
            black_box(&mut out_sp),
            black_box(&mut out_rep),
        )
        .unwrap();
        assert_eq!(out_rep.verdict, CuSafetyVerdict::Clear);
    });
}
