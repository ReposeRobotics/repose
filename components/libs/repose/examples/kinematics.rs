//! Forward then inverse kinematics on the built-in 6-DOF igus ReBeL arm.
//!
//! No simulator, no Copper task graph: plain library calls. Run with:
//! `cargo run -p repose --example kinematics`

use cu_spatial_payloads::Transform3D;
use cu29::prelude::CuError;
use cu29::prelude::CuResult;
use cu29::units::si::angle::degree;
use cu29::units::si::f64::Angle;
use cu29::units::si::length::meter;
use cu29::units::si::length::millimeter;
use repose::kinematics::CuIkParams;
use repose::kinematics::CuIkStatus;
use repose::kinematics::CuJointPosition;
use repose::kinematics::igus_rebel_chain;

/// Joint angles in degrees, as typed revolute positions.
fn deg(values: [f64; 6]) -> [CuJointPosition; 6] {
    values.map(|d| CuJointPosition::Revolute(Angle::new::<degree>(d)))
}

fn main() -> CuResult<()> {
    let chain = igus_rebel_chain();
    let home = deg([0.0, 30.0, 60.0, 0.0, 60.0, 0.0]);
    let seed = deg([-1.1, 45.5, 60.0, -3.3, -80.9, 28.7]);
    let mut tcp = Transform3D::<f64>::identity();
    chain.forward_kinematics(&home, &mut tcp)?;
    let mut solution = seed;
    let status =
        chain.inverse_kinematics_dls(&tcp, &seed, &CuIkParams::default(), &mut solution)?;
    let CuIkStatus::Converged {
        iterations,
        position_error,
        orientation_error,
    } = status
    else {
        return Err(CuError::from(format!("ik did not converge: {status:?}")));
    };

    let t = tcp.translation();
    println!(
        "tcp: x={:.4} m y={:.4} m z={:.4} m",
        t[0].get::<meter>(),
        t[1].get::<meter>(),
        t[2].get::<meter>()
    );
    let mut solved = [0.0; 6];
    for (i, q) in solution.iter().enumerate() {
        solved[i] = q.angle()?.get::<degree>();
    }
    println!(
        "ik: converged in {iterations} iterations, position error {:.4} mm, orientation error {:.4}°",
        position_error.get::<millimeter>(),
        orientation_error.get::<degree>()
    );
    println!(
        "solution: [{:.2} {:.2} {:.2} {:.2} {:.2} {:.2}]°",
        solved[0], solved[1], solved[2], solved[3], solved[4], solved[5]
    );
    Ok(())
}
