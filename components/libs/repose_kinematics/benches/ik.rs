//! CodSpeed benchmarks for inverse kinematics on the 6-DOF igus ReBeL chain.
//!
//! Run locally with `cargo bench -p repose-kinematics --bench ik`, or under
//! CodSpeed with `cargo codspeed build -p repose-kinematics` followed by
//! `cargo codspeed run -p repose-kinematics`.

use cu_spatial_payloads::Transform3D;
use cu29::units::si::angle::radian;
use cu29::units::si::angular_velocity::radian_per_second;
use cu29::units::si::f64::Angle;
use cu29::units::si::f64::AngularVelocity;
use cu29::units::si::f64::Velocity;
use cu29::units::si::velocity::meter_per_second;
use repose_kinematics::CuDiffIkParams;
use repose_kinematics::CuIkParams;
use repose_kinematics::CuIkSolver;
use repose_kinematics::CuJointPosition;
use repose_kinematics::CuJointVelocity;
use repose_kinematics::CuTwist6D;
use repose_kinematics::CuTwistFrame;
use repose_kinematics::igus_rebel_chain;

fn main() {
    divan::main();
}

fn positions(values: [f64; 6]) -> [CuJointPosition; 6] {
    values.map(|v| Angle::new::<radian>(v).into())
}

/// A reachable target pose with a nearby seed, mirroring the round-trip test.
fn ik_fixture() -> (Transform3D<f64>, [CuJointPosition; 6]) {
    let chain = igus_rebel_chain();
    let q_goal = positions([0.3, 0.45, -0.35, 0.25, -0.4, 0.5]);
    let mut target = Transform3D::<f64>::identity();
    chain
        .forward_kinematics(&q_goal, &mut target)
        .expect("fixture FK must succeed");
    let seed = positions([0.1, 0.2, -0.1, 0.0, -0.1, 0.1]);
    (target, seed)
}

#[divan::bench]
fn inverse_kinematics_dls(bencher: divan::Bencher) {
    let chain = igus_rebel_chain();
    let (target, seed) = ik_fixture();
    let cfg = CuIkParams::default();
    let mut out_q = [CuJointPosition::default(); 6];
    bencher.bench_local(|| {
        let status = chain
            .inverse_kinematics_dls(
                divan::black_box(&target),
                divan::black_box(&seed),
                &cfg,
                &mut out_q,
            )
            .expect("IK fixture must solve");
        assert!(status.is_converged(), "IK fixture did not converge");
        divan::black_box((status, out_q))
    });
}

#[divan::bench]
fn inverse_kinematics_pseudoinverse(bencher: divan::Bencher) {
    let chain = igus_rebel_chain();
    let (target, seed) = ik_fixture();
    let cfg = CuIkParams {
        solver: CuIkSolver::Pseudoinverse,
        ..CuIkParams::default()
    };
    let mut out_q = [CuJointPosition::default(); 6];
    bencher.bench_local(|| {
        let status = chain
            .inverse_kinematics_dls(
                divan::black_box(&target),
                divan::black_box(&seed),
                &cfg,
                &mut out_q,
            )
            .expect("IK fixture must solve");
        assert!(status.is_converged(), "IK fixture did not converge");
        divan::black_box((status, out_q))
    });
}

#[divan::bench]
fn differential_ik(bencher: divan::Bencher) {
    let chain = igus_rebel_chain();
    let q = positions([0.2, 0.45, -0.6, 0.3, -0.5, 0.2]);
    let twist = CuTwist6D::new(
        [0.05, -0.03, 0.04].map(Velocity::new::<meter_per_second>),
        [0.1, -0.05, 0.08].map(AngularVelocity::new::<radian_per_second>),
        CuTwistFrame::Base,
    );
    let cfg = CuDiffIkParams::default();
    let mut out_qd = [CuJointVelocity::default(); 6];
    bencher.bench_local(|| {
        let status = chain
            .differential_ik(divan::black_box(&q), &twist, &cfg, &mut out_qd)
            .expect("differential IK fixture must succeed");
        divan::black_box((status, out_qd))
    });
}
