//! Every built-in joint planner (through `repose::planners`), plus pose-goal,
//! Cartesian, and path-constrained planning on the 6-DOF igus ReBeL.

use cu_spatial_payloads::Transform3D;
use cu29::cutask_anytime::AnytimeStatus;
use cu29::cutask_anytime::CuAnytimeTask;
use cu29::prelude::*;
use cu29::units::si::angle::degree;
use cu29::units::si::angle::radian;
use cu29::units::si::angular_acceleration::radian_per_second_squared;
use cu29::units::si::angular_jerk::radian_per_second_cubed;
use cu29::units::si::angular_velocity::radian_per_second;
use cu29::units::si::f64::Angle;
use cu29::units::si::f64::AngularAcceleration;
use cu29::units::si::f64::AngularJerk;
use cu29::units::si::f64::AngularVelocity;
use cu29::units::si::f64::Length;
use cu29::units::si::f64::Time;
use cu29::units::si::length::meter;
use cu29::units::si::time::second;
use divan::Bencher;
use divan::black_box;
use repose::collision::CuCollisionModelSpec;
use repose::collision::CuCollisionShape;
use repose::collision::CuSphere;
use repose::collision::igus_rebel_collision_model;
use repose::kinematics::CuChainSpec;
use repose::kinematics::CuJointAcceleration;
use repose::kinematics::CuJointJerk;
use repose::kinematics::CuJointPosition;
use repose::kinematics::CuJointVelocity;
use repose::kinematics::CuRobotPreset;
use repose::kinematics::igus_rebel_chain;
use repose::motion_planning::CuCartesianPlanParams;
use repose::motion_planning::CuCartesianPlanResult;
use repose::motion_planning::CuCartesianPlanTask;
use repose::motion_planning::CuJointPlanPath;
use repose::motion_planning::CuJointPlanRequest;
use repose::motion_planning::CuJointPlanner;
use repose::motion_planning::CuJointPlannerParams;
use repose::motion_planning::CuJointRrtConnect;
use repose::motion_planning::CuJointRrtConnectParams;
use repose::motion_planning::CuOrientationConstraint;
use repose::motion_planning::CuPathConstraint;
use repose::motion_planning::CuPathConstraintSet;
use repose::motion_planning::CuPlanStatus;
use repose::motion_planning::CuPoseIkParams;
use repose::motion_planning::CuPosePlanRequest;
use repose::motion_planning::CuPosePlannerParams;
use repose::motion_planning::CuPosePlannerTask;
use repose::motion_planning::CuTcpPose;
use repose::motion_planning::CuToolAxis;
use repose::planners::CuAnyPlanner;
use repose::planners::CuPlannerChoice;
use repose::trajectory::CuJointTrajectoryLimits;
use repose::trajectory::CuPathTimingParams;
use repose::trajectory::CuTimingParams;
use repose::trajectory::CuTrajectoryProfile;

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

/// Extracts Euler XYZ `(roll, pitch, yaw)` matching `Transform3D::from_translation_euler_xyz`.
fn pose_from_joint_target(q: &[CuJointPosition; 6]) -> CuTcpPose {
    let chain = igus_rebel_chain();
    let mut tf = Transform3D::<f64>::identity();
    chain.forward_kinematics(q, &mut tf).unwrap();
    let m = tf.to_matrix();
    let sy = (-m[2][0]).clamp(-1.0, 1.0);
    let pitch = sy.asin();
    let (roll, yaw) = if sy.abs() < 1.0 - 1e-9 {
        (m[2][1].atan2(m[2][2]), m[1][0].atan2(m[0][0]))
    } else {
        (m[1][2].atan2(m[1][1]), 0.0)
    };
    CuTcpPose {
        translation: tf.translation(),
        rotation: [roll, pitch, yaw].map(Angle::new::<radian>),
    }
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

/// Resolves a TCP pose goal with bounded IK against the obstacle scene and runs
/// RRT-Connect through [`CuPosePlannerTask`].
#[divan::bench]
fn plan_pose_goal(bencher: Bencher) {
    let joint_req = request();
    let pose_goal = pose_from_joint_target(&deg([25.0, 35.0, 55.0, 5.0, -45.0, 15.0]));
    let pose_msg = CuMsg::new(Some(CuPosePlanRequest::<6> {
        scene: joint_req.scene,
        exclusions: joint_req.exclusions,
        start: joint_req.start,
        goal: pose_goal,
    }));
    let (ctx, _clock) = CuContext::new_mock_clock();
    let params = CuPosePlannerParams {
        planner: CuJointPlannerParams {
            chain: Some(CuChainSpec::Preset(CuRobotPreset::IgusRebel)),
            collision_model: Some(CuCollisionModelSpec::Preset(CuRobotPreset::IgusRebel)),
            params: CuJointRrtConnectParams::default(),
            base_budget: Some(STEP_BUDGET),
            refine_budget: Some(STEP_BUDGET),
        },
        goal: CuPoseIkParams::default(),
    };
    bencher
        .with_inputs(|| {
            (
                Box::new(
                    CuPosePlannerTask::<CuJointRrtConnect<6, 6>, 6, 6>::from_params(
                        params.clone(),
                        SEED,
                    )
                    .unwrap(),
                ),
                Box::new(CuMsg::<CuJointPlanPath<6>>::default()),
            )
        })
        .bench_local_values(|(mut task, mut out)| {
            let mut status = task.base(&ctx, black_box(&pose_msg), &mut out).unwrap();
            let mut steps = 0;
            while matches!(status, AnytimeStatus::Improved(_))
                && steps < MAX_STEPS
                && out.payload().is_none_or(|p| !p.status.is_final())
            {
                status = task.refine(&ctx, &mut out).unwrap();
                steps += 1;
            }
            assert!(
                out.payload().is_some_and(|p| p.status.has_path()),
                "pose goal benchmark must produce a valid path"
            );
            (task, out)
        });
}

/// Plans and S-curve-times a straight TCP Cartesian move on the 6-DOF igus ReBeL.
#[divan::bench]
fn plan_cartesian(bencher: Bencher) {
    let start = deg([0.0, 35.0, 55.0, 0.0, 40.0, 0.0]);
    let goal = pose_from_joint_target(&deg([15.0, 38.0, 52.0, 4.0, 36.0, 8.0]));
    let req = CuPosePlanRequest::<6> {
        start,
        goal,
        ..Default::default()
    };
    let params = CuCartesianPlanParams::<6, 6> {
        chain: CuChainSpec::Preset(CuRobotPreset::IgusRebel),
        collision_model: CuCollisionModelSpec::Preset(CuRobotPreset::IgusRebel),
        goal: CuPoseIkParams::default(),
        path: Default::default(),
        timing: CuPathTimingParams {
            timing: CuTimingParams {
                limits: CuJointTrajectoryLimits::new(
                    [CuJointVelocity::Revolute(AngularVelocity::new::<radian_per_second>(1.0)); 6],
                    [CuJointAcceleration::Revolute(AngularAcceleration::new::<
                        radian_per_second_squared,
                    >(2.0)); 6],
                )
                .with_jerks(
                    [CuJointJerk::Revolute(AngularJerk::new::<radian_per_second_cubed>(10.0)); 6],
                ),
                profile: CuTrajectoryProfile::SCurve,
                sample_period: Time::new::<second>(0.02),
            },
            ..Default::default()
        },
    };
    let mut task =
        Box::new(CuCartesianPlanTask::<6, 128, 1024, 6>::from_params(params, SEED).unwrap());
    let mut out = Box::new(CuCartesianPlanResult::<6, 1024>::default());
    task.plan(&req, &mut out).unwrap();
    assert_eq!(out.status, CuPlanStatus::Converged);
    bencher.bench_local(|| {
        task.plan(black_box(&req), black_box(&mut out)).unwrap();
        assert_eq!(out.status, CuPlanStatus::Converged);
    });
}

/// Runs RRT-Connect under an active orientation path constraint so every sample
/// evaluates both collision geometry and tool-frame pose bounds.
#[divan::bench]
fn plan_constrained_rrt_connect(bencher: Bencher) {
    let chain = igus_rebel_chain();
    let start = deg([0.0, 35.0, 55.0, 0.0, 40.0, 0.0]);
    let goal = deg([40.0, 32.0, 58.0, 0.0, 38.0, 10.0]);
    let mut start_tf = Transform3D::<f64>::identity();
    chain.forward_kinematics(&start, &mut start_tf).unwrap();
    let m = start_tf.to_matrix();
    let ref_z = [m[0][2], m[1][2], m[2][2]];
    let mut constraints = CuPathConstraintSet::new();
    constraints
        .push(CuPathConstraint::Orientation(
            CuOrientationConstraint::new(CuToolAxis::Z, ref_z, Angle::new::<degree>(45.0)).unwrap(),
        ))
        .unwrap();

    let mut req = CuJointPlanRequest {
        start,
        goal,
        ..Default::default()
    };
    let mid = deg(core::array::from_fn(|j| {
        0.5 * (start[j].si_value().to_degrees() + goal[j].si_value().to_degrees())
    }));
    let mut mid_tf = Transform3D::<f64>::identity();
    chain.forward_kinematics(&mid, &mut mid_tf).unwrap();
    req.scene
        .add_obstacle(
            CuCollisionShape::Sphere(
                CuSphere::new(mid_tf.translation(), Length::new::<meter>(0.06)).unwrap(),
            ),
            None,
        )
        .unwrap();

    bencher
        .with_inputs(|| {
            let mut planner = CuJointRrtConnect::new(
                igus_rebel_chain(),
                igus_rebel_collision_model(),
                CuJointRrtConnectParams::default(),
            )
            .unwrap();
            planner.set_path_constraints(&constraints).unwrap();
            (planner, Box::new(CuJointPlanPath::<6>::default()))
        })
        .bench_local_values(|(mut planner, mut path)| {
            let mut status = planner.begin(black_box(&req), SEED).unwrap();
            let mut steps = 0;
            while !status.is_final() && steps < MAX_STEPS {
                status = planner.step(STEP_BUDGET).unwrap();
                steps += 1;
            }
            assert!(status.has_path(), "constrained benchmark must find a path");
            planner.write_best(&mut path);
            (planner, path)
        });
}
