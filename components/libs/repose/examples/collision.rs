//! Collision checking on the built-in 6-DOF igus ReBeL arm.
//!
//! One sphere obstacle sits on the tool path of a mid configuration, so one
//! configuration is free and one collides. No simulator, no Copper task
//! graph: plain library calls. Run with:
//! `cargo run -p repose --example collision`

use cu_spatial_payloads::Transform3D;
use cu29::prelude::CuError;
use cu29::prelude::CuResult;
use cu29::units::si::angle::degree;
use cu29::units::si::f64::Angle;
use cu29::units::si::f64::Length;
use cu29::units::si::length::meter;
use cu29::units::si::length::millimeter;
use repose::collision::CuCollisionReport;
use repose::collision::CuCollisionShape;
use repose::collision::CuPlanningScene;
use repose::collision::CuSphere;
use repose::collision::igus_rebel_collision_model;
use repose::kinematics::CuJointPosition;
use repose::kinematics::igus_rebel_chain;

/// Joint angles in degrees, as typed revolute positions.
fn deg(values: [f64; 6]) -> [CuJointPosition; 6] {
    values.map(|d| CuJointPosition::Revolute(Angle::new::<degree>(d)))
}

fn main() -> CuResult<()> {
    let chain = igus_rebel_chain();
    let model = igus_rebel_collision_model();
    let free = deg([-1.1, 45.5, 60.0, -3.3, -80.9, 28.7]);
    let blocked = deg([45.0, 30.0, 60.0, 0.0, 60.0, 0.0]);

    // The obstacle: a 10 cm sphere centered on the tool point of `blocked`,
    // so that configuration is guaranteed to collide with it.
    let mut tcp = Transform3D::<f64>::identity();
    chain.forward_kinematics(&blocked, &mut tcp)?;
    // The capacities default (32 obstacles, 8 attached shapes); the annotation
    // pins them because nothing else constrains the type.
    let mut scene: CuPlanningScene = CuPlanningScene::new();
    let sphere = CuSphere::new(tcp.translation(), Length::new::<meter>(0.1))?;
    scene.add_obstacle(CuCollisionShape::Sphere(sphere), None)?;
    let env = scene.collision_env();

    // `16` is the maximum number of contacts stored in the report; the hot
    // path writes into this caller-owned buffer instead of allocating.
    let mut report = CuCollisionReport::<16>::default();
    model.check_configuration(&chain, &free, &env, &mut report)?;
    if !report.is_collision_free() {
        return Err(CuError::from(
            "expected the free configuration to be collision-free",
        ));
    }
    println!(
        "free: collision-free, clearance {:.1} mm",
        report.min_signed_distance().get::<millimeter>()
    );

    model.check_configuration(&chain, &blocked, &env, &mut report)?;
    if report.is_collision_free() {
        return Err(CuError::from(
            "expected the blocked configuration to collide",
        ));
    }
    println!(
        "blocked: in collision, {} contact(s), min distance {:.1} mm",
        report.total_contacts(),
        report.min_signed_distance().get::<millimeter>()
    );
    Ok(())
}
