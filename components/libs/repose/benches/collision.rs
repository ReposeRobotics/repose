//! Collision checking on the igus ReBeL model: self-collision, a cluttered
//! planning scene, and discretized segment checks.

use cu29::units::si::angle::degree;
use cu29::units::si::f64::Angle;
use cu29::units::si::f64::Length;
use cu29::units::si::length::meter;
use divan::Bencher;
use divan::black_box;
use repose::collision::CuCapsule;
use repose::collision::CuCollisionEnv;
use repose::collision::CuCollisionReport;
use repose::collision::CuCollisionShape;
use repose::collision::CuPlanningScene;
use repose::collision::CuSphere;
use repose::collision::igus_rebel_collision_model;
use repose::kinematics::CuJointPosition;
use repose::kinematics::igus_rebel_chain;

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
const GOAL: [f64; 6] = [0.0, 30.0, 60.0, 0.0, 60.0, 0.0];

/// A scene with a table, two posts and a ring of spheres around the robot,
/// all clear of the benchmark configurations.
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

#[divan::bench]
fn check_configuration_self(bencher: Bencher) {
    let chain = igus_rebel_chain();
    let model = igus_rebel_collision_model();
    let q = deg(START);
    let env = CuCollisionEnv::EMPTY;
    let mut report = CuCollisionReport::<16>::default();
    bencher.bench_local(|| {
        model
            .check_configuration(&chain, black_box(&q), &env, black_box(&mut report))
            .unwrap();
    });
}

#[divan::bench]
fn check_configuration_scene(bencher: Bencher) {
    let chain = igus_rebel_chain();
    let model = igus_rebel_collision_model();
    let scene = cluttered_scene();
    let env = scene.collision_env();
    let q = deg(START);
    let mut report = CuCollisionReport::<16>::default();
    bencher.bench_local(|| {
        model
            .check_configuration(&chain, black_box(&q), &env, black_box(&mut report))
            .unwrap();
    });
}

#[divan::bench]
fn is_configuration_collision_free_scene(bencher: Bencher) {
    let chain = igus_rebel_chain();
    let model = igus_rebel_collision_model();
    let scene = cluttered_scene();
    let env = scene.collision_env();
    let q = deg(GOAL);
    bencher.bench_local(|| {
        model
            .is_configuration_collision_free(&chain, black_box(&q), &env)
            .unwrap()
    });
}

#[divan::bench]
fn is_segment_collision_free_scene(bencher: Bencher) {
    let chain = igus_rebel_chain();
    let model = igus_rebel_collision_model();
    let scene = cluttered_scene();
    let env = scene.collision_env();
    let start = deg(START);
    let goal = deg(GOAL);
    let step = Angle::new::<degree>(2.0);
    bencher.bench_local(|| {
        model
            .is_segment_collision_free(&chain, black_box(&start), black_box(&goal), step, &env)
            .unwrap()
    });
}
