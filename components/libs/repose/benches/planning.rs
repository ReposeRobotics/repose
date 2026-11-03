//! Every built-in joint planner, through the `repose::planners` registry, on
//! the igus ReBeL: one full anytime job from `begin` to a final status.

use cu_spatial_payloads::Transform3D;
use cu29::units::si::angle::degree;
use cu29::units::si::f64::Angle;
use cu29::units::si::f64::Length;
use cu29::units::si::length::meter;
use divan::Bencher;
use divan::black_box;
use repose::collision::CuCollisionShape;
use repose::collision::CuSphere;
use repose::collision::igus_rebel_collision_model;
use repose::kinematics::CuJointPosition;
use repose::kinematics::igus_rebel_chain;
use repose::motion_planning::CuJointPlanPath;
use repose::motion_planning::CuJointPlanRequest;
use repose::motion_planning::CuJointPlanner;
use repose::planners::CuAnyPlanner;
use repose::planners::CuPlannerChoice;

fn main() {
    divan::main();
}

fn deg(values: [f64; 6]) -> [CuJointPosition; 6] {
    values.map(|d| CuJointPosition::Revolute(Angle::new::<degree>(d)))
}

/// Upper bound on `step` calls, so a planner that never reaches a final status
/// still runs a fixed, deterministic amount of work.
const MAX_STEPS: u32 = 64;
const STEP_BUDGET: u32 = 64;
const SEED: u64 = 1;
const OBSTACLE_RADIUS: f64 = 0.1;
const START: [f64; 6] = [-1.1, 45.5, 60.0, -3.3, -80.9, 28.7];
const GOAL: [f64; 6] = [90.0, 30.0, 60.0, 0.0, 60.0, 0.0];

/// Start and goal of the igus ReBeL, with a sphere sitting on the tool path of
/// the straight joint-space line so that planners have to route around it.
fn request() -> CuJointPlanRequest<6> {
    let chain = igus_rebel_chain();
    let model = igus_rebel_collision_model();
    let mut request = CuJointPlanRequest {
        start: deg(START),
        goal: deg(GOAL),
        ..Default::default()
    };
    let mid = deg(core::array::from_fn(|j| 0.5 * (START[j] + GOAL[j])));
    let mut tcp = Transform3D::<f64>::identity();
    chain.forward_kinematics(&mid, &mut tcp).unwrap();
    let center = tcp.translation();
    request
        .scene
        .add_obstacle(
            CuCollisionShape::Sphere(
                CuSphere::new(center, Length::new::<meter>(OBSTACLE_RADIUS)).unwrap(),
            ),
            None,
        )
        .unwrap();
    let env = request.scene.collision_env();
    for q in [&request.start, &request.goal] {
        assert!(
            model
                .is_configuration_collision_free(&chain, q, &env)
                .unwrap()
        );
    }
    assert!(
        !model
            .is_configuration_collision_free(&chain, &mid, &env)
            .unwrap()
    );
    request
}

#[divan::bench(args = CuPlannerChoice::ALL)]
fn plan(bencher: Bencher, choice: &CuPlannerChoice) {
    let request = request();
    bencher
        .with_inputs(|| {
            (
                CuAnyPlanner::<6>::new(igus_rebel_chain(), igus_rebel_collision_model(), *choice)
                    .unwrap(),
                Box::new(CuJointPlanPath::<6>::default()),
            )
        })
        .bench_local_values(|(mut planner, mut path)| {
            let mut status = planner.begin(black_box(&request), SEED).unwrap();
            let mut steps = 0;
            while !status.is_final() && steps < MAX_STEPS {
                status = planner.step(STEP_BUDGET).unwrap();
                steps += 1;
            }
            planner.write_best(&mut path);
            (planner, path)
        });
}
