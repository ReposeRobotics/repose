//! Pose planning integration tests.
use super::*;
use cu29::cutask_anytime::CuAnytimeTask;
use cu29::units::si::acceleration::meter_per_second_squared;
use cu29::units::si::angle::radian;
use cu29::units::si::f64::Acceleration;
use cu29::units::si::f64::Angle;
use cu29::units::si::f64::Length;
use cu29::units::si::f64::Velocity;
use cu29::units::si::length::meter;
use cu29::units::si::velocity::meter_per_second;
use repose_collision::CuCollisionModelParams;
use repose_collision::CuCollisionShape;
use repose_collision::CuLinkShape;
use repose_collision::CuRobotCollisionModel;
use repose_collision::CuSphere;
use repose_kinematics::CuJointAcceleration;
use repose_kinematics::CuJointAxis;
use repose_kinematics::CuJointLimits;
use repose_kinematics::CuJointSpec;
use repose_kinematics::CuJointVelocity;
use repose_kinematics::CuKinematicChain;
use repose_trajectory::CuJointTrajectoryLimits;
use repose_trajectory::CuPathTimingParams;
use repose_trajectory::CuTimingParams;

fn m(v: f64) -> Length {
    Length::new::<meter>(v)
}

/// XYZ gantry whose TCP position equals its joint positions.
pub(crate) fn robot() -> (CuKinematicChain<3>, CuRobotCollisionModel<3, 1>) {
    let limits = CuJointLimits::prismatic(m(-1.0), m(1.0), Velocity::new::<meter_per_second>(1.0));
    let origin = [m(0.0); 3];
    let rpy = [Angle::new::<radian>(0.0); 3];
    let chain = CuKinematicChain::new([
        CuJointSpec::prismatic(origin, rpy, CuJointAxis::PosX, limits),
        CuJointSpec::prismatic(origin, rpy, CuJointAxis::PosY, limits),
        CuJointSpec::prismatic(origin, rpy, CuJointAxis::PosZ, limits),
    ]);
    let sphere = CuCollisionShape::Sphere(CuSphere::new([m(0.0); 3], m(0.01)).unwrap());
    let model = CuRobotCollisionModel::new(
        &[CuLinkShape::new(sphere, Some(2))],
        CuCollisionModelParams {
            safety_margin: m(0.0),
            skip_adjacent_links: false,
        },
    )
    .unwrap();
    (chain, model)
}

pub(crate) fn limits() -> CuJointTrajectoryLimits<3> {
    CuJointTrajectoryLimits::new(
        [CuJointVelocity::Prismatic(Velocity::new::<meter_per_second>(0.4)); 3],
        [CuJointAcceleration::Prismatic(Acceleration::new::<meter_per_second_squared>(1.0)); 3],
    )
}

fn request() -> CuPosePlanRequest<3> {
    CuPosePlanRequest {
        start: [m(0.0).into(); 3],
        goal: CuTcpPose {
            translation: [m(0.3), m(0.1), m(0.0)],
            rotation: [Angle::new::<radian>(0.0); 3],
        },
        ..Default::default()
    }
}
fn cartesian<const PATH: usize, const CAP: usize>() -> CuCartesianPlanTask<3, PATH, CAP, 1> {
    let (chain, _) = robot();
    let shape = CuCollisionShape::Sphere(CuSphere::new([m(0.0); 3], m(0.01)).unwrap());
    CuCartesianPlanTask::from_params(
        CuCartesianPlanParams {
            chain: chain.into(),
            collision_model: repose_collision::CuCollisionModelSpec::Inline {
                shapes: [CuLinkShape::new(shape, Some(2))],
                params: CuCollisionModelParams {
                    safety_margin: m(0.0),
                    skip_adjacent_links: false,
                },
            },
            goal: Default::default(),
            path: Default::default(),
            timing: CuPathTimingParams {
                timing: CuTimingParams {
                    limits: limits(),
                    ..Default::default()
                },
                ..Default::default()
            },
        },
        42,
    )
    .unwrap()
}
#[test]
fn straight_pose_move_is_timed_and_preserves_scene_revision() {
    let mut task = cartesian::<128, 1024>();
    let req = request();
    let mut out = CuCartesianPlanResult::default();
    task.plan(&req, &mut out).unwrap();
    assert_eq!(out.status, CuPlanStatus::Converged);
    assert_eq!(out.scene_revision, req.scene.revision());
    assert!(out.trajectory.len > 2);
    for point in out.trajectory.points() {
        let x = point.positions[0].si_value();
        let y = point.positions[1].si_value();
        assert!((y - x / 3.0).abs() < 0.001);
    }
    let last = out.trajectory.points().last().unwrap();
    assert!((last.positions[0].si_value() - 0.3).abs() < 0.001);
}
#[test]
fn unreachable_and_invalid_requests_clear_previous_trajectory() {
    let mut task = cartesian::<128, 1024>();
    let mut req = request();
    let mut out = CuCartesianPlanResult::default();
    task.plan(&req, &mut out).unwrap();
    assert!(out.trajectory.len > 0);
    req.goal.translation[0] = m(3.0);
    task.plan(&req, &mut out).unwrap();
    assert_eq!(out.status, CuPlanStatus::NoIkSolution);
    assert_eq!(out.trajectory.len, 0);
    req.goal.translation[0] = m(f64::NAN);
    task.plan(&req, &mut out).unwrap();
    assert_eq!(out.status, CuPlanStatus::InvalidRequest);
}
#[test]
fn scene_obstacle_blocks_straight_line() {
    let mut task = cartesian::<128, 1024>();
    let mut req = request();
    let id = repose_collision::CuObjectId::new(9);
    req.scene
        .add_obstacle(
            CuCollisionShape::Sphere(CuSphere::new([m(0.15), m(0.05), m(0.0)], m(0.04)).unwrap()),
            Some(id),
        )
        .unwrap();
    let mut out = CuCartesianPlanResult::default();
    task.plan(&req, &mut out).unwrap();
    assert_eq!(out.status, CuPlanStatus::Exhausted);
    assert_eq!(out.trajectory.len, 0);
    assert_eq!(out.scene_revision, req.scene.revision());
}
#[test]
fn capacity_failure_is_explicit() {
    let mut task = cartesian::<2, 2>();
    let mut out = CuCartesianPlanResult::default();
    task.plan(&request(), &mut out).unwrap();
    assert_eq!(out.status, CuPlanStatus::CapacityExceeded);
    assert_eq!(out.trajectory.len, 0);
}
#[test]
fn pose_task_runs_joint_planner_and_publishes_ik_failure() {
    let (chain, _model) = robot();
    let params = CuPosePlannerParams {
        planner: CuJointPlannerParams {
            chain: Some(chain.into()),
            collision_model: Some(repose_collision::CuCollisionModelSpec::Inline {
                shapes: [CuLinkShape::new(
                    CuCollisionShape::Sphere(CuSphere::new([m(0.0); 3], m(0.01)).unwrap()),
                    Some(2),
                )],
                params: CuCollisionModelParams {
                    safety_margin: m(0.0),
                    skip_adjacent_links: false,
                },
            }),
            params: CuJointDirectParams::default(),
            ..Default::default()
        },
        goal: Default::default(),
    };
    let mut task = CuPosePlannerTask::<CuJointDirect<3, 1>, 3, 1>::from_params(params, 42).unwrap();
    let (ctx, _) = CuContext::new_mock_clock();
    let mut output = CuMsg::default();
    task.base(&ctx, &CuMsg::new(Some(request())), &mut output)
        .unwrap();
    assert_eq!(output.payload().unwrap().status, CuPlanStatus::Converged);
    let mut req = request();
    req.goal.translation[0] = m(3.0);
    task.base(&ctx, &CuMsg::new(Some(req)), &mut output)
        .unwrap();
    assert_eq!(output.payload().unwrap().status, CuPlanStatus::NoIkSolution);
    assert_eq!(output.payload().unwrap().len, 0);
}

#[test]
fn attached_tool_and_exclusions_are_used_during_the_entire_move() {
    let mut task = cartesian::<128, 1024>();
    let mut req = request();
    let tool = req
        .scene
        .add_attached(
            Some(2),
            CuCollisionShape::Sphere(CuSphere::new([m(0.0), m(0.08), m(0.0)], m(0.025)).unwrap()),
            None,
        )
        .unwrap();
    let obstacle = req
        .scene
        .add_obstacle(
            CuCollisionShape::Sphere(CuSphere::new([m(0.15), m(0.13), m(0.0)], m(0.025)).unwrap()),
            None,
        )
        .unwrap();
    let mut out = CuCartesianPlanResult::default();
    task.plan(&req, &mut out).unwrap();
    assert_eq!(out.status, CuPlanStatus::Exhausted);
    req.exclusions
        .add(repose_collision::CuSceneExclusion::obstacle(tool, obstacle))
        .unwrap();
    task.plan(&req, &mut out).unwrap();
    assert_eq!(out.status, CuPlanStatus::Converged);
    assert_eq!(out.exclusions, req.exclusions);
    assert_eq!(out.scene_revision, req.scene.revision());
}

#[test]
fn replay_restores_seed_progress_and_payload_roundtrips() {
    let mut task = cartesian::<64, 128>();
    let req = request();
    let mut out = CuCartesianPlanResult::default();
    task.plan(&req, &mut out).unwrap();
    let mut bytes = [0u8; 64];
    task.freeze(&mut bincode::enc::EncoderImpl::new(
        bincode::enc::write::SliceWriter::new(&mut bytes),
        bincode::config::standard(),
    ))
    .unwrap();
    let mut restored = cartesian::<64, 128>();
    restored
        .thaw(&mut bincode::de::DecoderImpl::new(
            bincode::de::read::SliceReader::new(&bytes),
            bincode::config::standard(),
            (),
        ))
        .unwrap();
    let mut expected = CuCartesianPlanResult::default();
    let mut actual = CuCartesianPlanResult::default();
    task.plan(&req, &mut expected).unwrap();
    restored.plan(&req, &mut actual).unwrap();
    assert_eq!(expected, actual);
    let encoded = bincode::encode_to_vec(&actual, bincode::config::standard()).unwrap();
    let (decoded, _): (CuCartesianPlanResult<3, 128>, usize) =
        bincode::decode_from_slice(&encoded, bincode::config::standard()).unwrap();
    assert_eq!(decoded, actual);
}

#[test]
fn pose_goal_task_reads_ron_and_emits_once_when_ready() {
    let cfg: ComponentConfig =
        ron::from_str(r#"#![enable(unwrap_newtypes)] {"goal":{"translation":[0.3,0.1,0.0],"rotation":[0.0,0.0,0.0]}}"#)
            .unwrap();
    let mut task = CuPoseGoalTask::<3>::new(Some(&cfg), ()).unwrap();
    let (ctx, _) = CuContext::new_mock_clock();
    let mut out = CuMsg::default();
    let mut state = repose_kinematics::CuJointStatePayload {
        positions: [m(0.0).into(); 3],
        ..Default::default()
    };
    state.status = repose_kinematics::CuRobotStatus::NotReady;
    task.process(&ctx, &CuMsg::new(Some(state.clone())), &mut out)
        .unwrap();
    assert!(out.payload().is_none());
    state.status = repose_kinematics::CuRobotStatus::Ready;
    let input = CuMsg::new(Some(state));
    task.process(&ctx, &input, &mut out).unwrap();
    assert_eq!(out.payload().unwrap().goal, request().goal);
    task.process(&ctx, &input, &mut out).unwrap();
    assert!(out.payload().is_none());
}

#[test]
fn endpoint_collisions_and_invalid_attachment_frames_have_distinct_statuses() {
    let mut task = cartesian::<64, 128>();
    let mut req = request();
    let mut out = CuCartesianPlanResult::default();
    req.scene
        .add_obstacle(
            CuCollisionShape::Sphere(CuSphere::new(req.goal.translation, m(0.04)).unwrap()),
            None,
        )
        .unwrap();
    task.plan(&req, &mut out).unwrap();
    assert_eq!(out.status, CuPlanStatus::InvalidGoal);
    req.start = req.goal.translation.map(Into::into);
    task.plan(&req, &mut out).unwrap();
    assert_eq!(out.status, CuPlanStatus::InvalidStart);
    let mut req = request();
    req.scene
        .add_attached(
            Some(99),
            CuCollisionShape::Sphere(CuSphere::new([m(0.0); 3], m(0.01)).unwrap()),
            None,
        )
        .unwrap();
    task.plan(&req, &mut out).unwrap();
    assert_eq!(out.status, CuPlanStatus::InvalidRequest);
    assert_eq!(out.trajectory.len, 0);
}

#[test]
fn follower_adapter_clears_failures_and_propagates_time() {
    let mut task = cartesian::<64, 128>();
    let mut result = CuCartesianPlanResult::default();
    task.plan(&request(), &mut result).unwrap();
    let mut input = CuMsg::new(Some(result));
    input.tov = cu29::clock::Tov::Time(cu29::clock::CuTime::from_nanos(42));
    let mut adapter = CuCartesianTrajectoryTask::<3, 128>::new(None, ()).unwrap();
    let (ctx, _) = CuContext::new_mock_clock();
    let mut output = CuMsg::default();
    adapter.process(&ctx, &input, &mut output).unwrap();
    assert!(output.payload().unwrap().len > 0);
    assert_eq!(output.tov, input.tov);
    input.payload_mut().as_mut().unwrap().status = CuPlanStatus::NoIkSolution;
    adapter.process(&ctx, &input, &mut output).unwrap();
    assert!(output.payload().is_none());
}
