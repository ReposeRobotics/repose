//! Kinematics hot paths on the 6-DOF igus ReBeL preset: forward kinematics,
//! Jacobians, and inverse kinematics.

use cu_spatial_payloads::Transform3D;
use cu29::units::si::angle::degree;
use cu29::units::si::angular_velocity::radian_per_second;
use cu29::units::si::f64::Angle;
use cu29::units::si::f64::AngularVelocity;
use cu29::units::si::f64::Velocity;
use cu29::units::si::velocity::meter_per_second;
use divan::Bencher;
use divan::black_box;
use repose::kinematics::CuDiffIkParams;
use repose::kinematics::CuIkParams;
use repose::kinematics::CuJointPosition;
use repose::kinematics::CuJointVelocity;
use repose::kinematics::CuTwist6D;
use repose::kinematics::CuTwistFrame;
use repose::kinematics::igus_rebel_chain;

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
