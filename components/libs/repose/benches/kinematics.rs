//! Kinematics and rigid-body dynamics hot paths on the 6-DOF igus ReBeL preset:
//! forward kinematics, Jacobians, inverse kinematics, inverse dynamics, and
//! gravity compensation.

use cu_spatial_payloads::Transform3D;
use cu29::units::si::angle::degree;
use cu29::units::si::angular_acceleration::radian_per_second_squared;
use cu29::units::si::angular_velocity::radian_per_second;
use cu29::units::si::f64::Angle;
use cu29::units::si::f64::AngularAcceleration;
use cu29::units::si::f64::AngularVelocity;
use cu29::units::si::f64::Length;
use cu29::units::si::f64::Mass;
use cu29::units::si::f64::MomentOfInertia;
use cu29::units::si::f64::Velocity;
use cu29::units::si::length::meter;
use cu29::units::si::mass::kilogram;
use cu29::units::si::moment_of_inertia::kilogram_square_meter;
use cu29::units::si::velocity::meter_per_second;
use divan::Bencher;
use divan::black_box;
use repose::kinematics::CuChainInertia;
use repose::kinematics::CuDiffIkParams;
use repose::kinematics::CuIkParams;
use repose::kinematics::CuInverseDynamicsInput;
use repose::kinematics::CuJointAcceleration;
use repose::kinematics::CuJointEffort;
use repose::kinematics::CuJointPosition;
use repose::kinematics::CuJointVelocity;
use repose::kinematics::CuLinkInertia;
use repose::kinematics::CuTwist6D;
use repose::kinematics::CuTwistFrame;
use repose::kinematics::igus_rebel_chain;
use repose::kinematics::standard_gravity;

fn main() {
    divan::main();
}

fn deg(values: [f64; 6]) -> [CuJointPosition; 6] {
    values.map(|d| CuJointPosition::Revolute(Angle::new::<degree>(d)))
}

const START: [f64; 6] = [-1.1, 45.5, 60.0, -3.3, -80.9, 28.7];
const GOAL: [f64; 6] = [0.0, 30.0, 60.0, 0.0, 60.0, 0.0];

#[divan::bench]
fn forward_kinematics(bencher: Bencher) {
    let chain = igus_rebel_chain();
    let q = deg(START);
    let mut out = Transform3D::<f64>::identity();
    bencher.bench_local(|| {
        chain
            .forward_kinematics(black_box(&q), black_box(&mut out))
            .unwrap();
    });
}

#[divan::bench]
fn forward_frames(bencher: Bencher) {
    let chain = igus_rebel_chain();
    let q = deg(START);
    let mut frames = [Transform3D::<f64>::identity(); 6];
    let mut ee = Transform3D::<f64>::identity();
    bencher.bench_local(|| {
        chain
            .forward_frames(black_box(&q), black_box(&mut frames), black_box(&mut ee))
            .unwrap();
    });
}

#[divan::bench]
fn geometric_jacobian(bencher: Bencher) {
    let chain = igus_rebel_chain();
    let q = deg(START);
    let mut jac = [[0.0_f64; 6]; 6];
    bencher.bench_local(|| {
        chain
            .geometric_jacobian(black_box(&q), black_box(&mut jac))
            .unwrap();
    });
}

#[divan::bench]
fn body_jacobian(bencher: Bencher) {
    let chain = igus_rebel_chain();
    let q = deg(START);
    let mut jac = [[0.0_f64; 6]; 6];
    bencher.bench_local(|| {
        chain
            .body_jacobian(black_box(&q), black_box(&mut jac))
            .unwrap();
    });
}

#[divan::bench]
fn inverse_kinematics_dls(bencher: Bencher) {
    let chain = igus_rebel_chain();
    let mut target = Transform3D::<f64>::identity();
    chain.forward_kinematics(&deg(GOAL), &mut target).unwrap();
    let seed = deg(START);
    let params = CuIkParams::default();
    let mut solution = seed;
    bencher.bench_local(|| {
        chain
            .inverse_kinematics_dls(
                black_box(&target),
                black_box(&seed),
                black_box(&params),
                black_box(&mut solution),
            )
            .unwrap()
    });
}

#[divan::bench]
fn differential_ik(bencher: Bencher) {
    let chain = igus_rebel_chain();
    let q = deg(START);
    let twist = CuTwist6D::new(
        [
            Velocity::new::<meter_per_second>(0.05),
            Velocity::new::<meter_per_second>(0.0),
            Velocity::new::<meter_per_second>(-0.02),
        ],
        [
            AngularVelocity::new::<radian_per_second>(0.0),
            AngularVelocity::new::<radian_per_second>(0.1),
            AngularVelocity::new::<radian_per_second>(0.0),
        ],
        CuTwistFrame::Base,
    );
    let params = CuDiffIkParams::default();
    let mut qd = [CuJointVelocity::default(); 6];
    bencher.bench_local(|| {
        chain
            .differential_ik(
                black_box(&q),
                black_box(&twist),
                black_box(&params),
                black_box(&mut qd),
            )
            .unwrap()
    });
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

fn rebel_inertia() -> CuChainInertia<6> {
    CuChainInertia::new([
        cylinder_link(1.6, 0.05, 0.106),
        cylinder_link(1.9, 0.045, 0.2415),
        cylinder_link(1.3, 0.045, 0.1575),
        cylinder_link(1.0, 0.04, 0.142),
        cylinder_link(0.7, 0.04, 0.0768),
        cylinder_link(0.3, 0.035, 0.04),
    ])
    .unwrap()
}

#[divan::bench]
fn inverse_dynamics(bencher: Bencher) {
    let chain = igus_rebel_chain();
    let inertia = rebel_inertia();
    let q = deg(START);
    let qd = [0.4, -0.3, 0.5, -0.2, 0.6, -0.4]
        .map(|v| CuJointVelocity::Revolute(AngularVelocity::new::<radian_per_second>(v)));
    let qdd = [1.2, -0.8, 1.5, -0.5, 2.0, -1.0].map(|a| {
        CuJointAcceleration::Revolute(AngularAcceleration::new::<radian_per_second_squared>(a))
    });
    let gravity = standard_gravity();
    let mut efforts = [CuJointEffort::default(); 6];
    bencher.bench_local(|| {
        let input = CuInverseDynamicsInput::new(
            black_box(&q),
            black_box(&qd),
            black_box(&qdd),
            black_box(gravity),
        );
        chain
            .inverse_dynamics(black_box(&inertia), &input, black_box(&mut efforts))
            .unwrap();
    });
}

#[divan::bench]
fn gravity_compensation(bencher: Bencher) {
    let chain = igus_rebel_chain();
    let inertia = rebel_inertia();
    let q = deg(START);
    let gravity = standard_gravity();
    let mut efforts = [CuJointEffort::default(); 6];
    bencher.bench_local(|| {
        chain
            .gravity_compensation(
                black_box(&inertia),
                black_box(&q),
                black_box(gravity),
                black_box(&mut efforts),
            )
            .unwrap();
    });
}
