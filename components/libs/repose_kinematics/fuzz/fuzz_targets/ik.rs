//! Fuzz target for the igus ReBeL kinematics surface: FK, Jacobians,
//! manipulability, differential IK, and DLS/pseudoinverse IK.
//!
//! Input layout (19+ bytes): 6 seed joint bytes, 6 goal joint bytes,
//! 6 twist bytes, 1 solver selector byte. Bytes are mapped into bounded,
//! finite ranges, so every input exercises the solvers rather than the
//! input validation. The oracles are the library's own contracts: no
//! panics, finite outputs on success, and joint limits respected when
//! IK reports convergence.

#![no_main]

use cu_spatial_payloads::Transform3D;
use cu29::units::si::angle::radian;
use cu29::units::si::angular_velocity::radian_per_second;
use cu29::units::si::f64::Angle;
use cu29::units::si::f64::AngularVelocity;
use cu29::units::si::f64::Velocity;
use cu29::units::si::velocity::meter_per_second;
use libfuzzer_sys::fuzz_target;
use repose_kinematics::CuDiffIkParams;
use repose_kinematics::CuIkParams;
use repose_kinematics::CuIkSolver;
use repose_kinematics::CuJointPosition;
use repose_kinematics::CuJointVelocity;
use repose_kinematics::CuTwist6D;
use repose_kinematics::CuTwistFrame;
use repose_kinematics::igus_rebel_chain;

/// Maps a byte to a joint angle in `[-2.8, 2.8]` rad, inside ReBeL limits.
fn angle(byte: u8) -> CuJointPosition {
    let radians = -2.8 + (f64::from(byte) / 255.0) * 5.6;
    Angle::new::<radian>(radians).into()
}

fn positions(bytes: &[u8]) -> [CuJointPosition; 6] {
    [
        angle(bytes[0]),
        angle(bytes[1]),
        angle(bytes[2]),
        angle(bytes[3]),
        angle(bytes[4]),
        angle(bytes[5]),
    ]
}

/// Maps a byte to a centred, bounded twist component.
fn centred(byte: u8, scale: f64) -> f64 {
    (f64::from(byte) - 127.5) / 127.5 * scale
}

fuzz_target!(|data: &[u8]| {
    if data.len() < 19 {
        return;
    }
    let chain = igus_rebel_chain();
    let seed = positions(&data[0..6]);
    let goal = positions(&data[6..12]);

    let mut target = Transform3D::<f64>::identity();
    if chain.forward_kinematics(&goal, &mut target).is_err() {
        return;
    }
    let mut seed_pose = Transform3D::<f64>::identity();
    if chain.forward_kinematics(&seed, &mut seed_pose).is_err() {
        return;
    }

    let mut jac = [[0.0_f64; 6]; 6];
    let _ = chain.geometric_jacobian(&seed, &mut jac);
    let _ = chain.body_jacobian(&seed, &mut jac);
    if let Ok(w_sq) = chain.manipulability_squared(&seed) {
        assert!(w_sq.is_finite() && w_sq >= 0.0);
    }

    // Pose IK from the seed to a reachable target.
    let cfg = CuIkParams {
        solver: if data[18] & 1 == 0 {
            CuIkSolver::DampedLeastSquares
        } else {
            CuIkSolver::Pseudoinverse
        },
        ..CuIkParams::default()
    };
    let mut out_q = [CuJointPosition::default(); 6];
    if let Ok(status) = chain.inverse_kinematics_dls(&target, &seed, &cfg, &mut out_q) {
        assert!(out_q.iter().all(|q| q.si_value().is_finite()));
        if status.is_converged() {
            assert!(chain.is_within_limits(&out_q));
        }
    }

    // Differential IK with a bounded base-frame twist.
    let twist = CuTwist6D::new(
        [
            Velocity::new::<meter_per_second>(centred(data[12], 0.5)),
            Velocity::new::<meter_per_second>(centred(data[13], 0.5)),
            Velocity::new::<meter_per_second>(centred(data[14], 0.5)),
        ],
        [
            AngularVelocity::new::<radian_per_second>(centred(data[15], 1.0)),
            AngularVelocity::new::<radian_per_second>(centred(data[16], 1.0)),
            AngularVelocity::new::<radian_per_second>(centred(data[17], 1.0)),
        ],
        if data[18] & 2 == 0 {
            CuTwistFrame::Base
        } else {
            CuTwistFrame::Body
        },
    );
    let mut out_qd = [CuJointVelocity::default(); 6];
    if chain
        .differential_ik(&seed, &twist, &CuDiffIkParams::default(), &mut out_qd)
        .is_ok()
    {
        assert!(out_qd.iter().all(|qd| qd.si_value().is_finite()));
    }
});
