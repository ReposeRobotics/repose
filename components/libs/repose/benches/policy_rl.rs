//! Reinforcement-learning and VLA policy execution hot paths: observation
//! building, single-step and chunked ONNX inference, ACT temporal ensembling,
//! receding-horizon chunk stepping, and joint / Cartesian / gripper action
//! decoding on the 6-DOF igus ReBeL.

use cu29::prelude::*;
use cu29::units::si::angle::radian;
use cu29::units::si::angular_velocity::radian_per_second;
use cu29::units::si::f64::Angle;
use cu29::units::si::f64::AngularVelocity;
use cu29::units::si::f64::Length;
use cu29::units::si::f64::Time;
use cu29::units::si::length::meter;
use cu29::units::si::time::second;
use divan::Bencher;
use divan::black_box;
use repose::io::CuGripperCommandPayload;
use repose::kinematics::CuChainSpec;
use repose::kinematics::CuJointPosition;
use repose::kinematics::CuJointPositionsPayload;
use repose::kinematics::CuJointStatePayload;
use repose::kinematics::CuJointVelocity;
use repose::kinematics::CuRobotPreset;
use repose::kinematics::CuRobotStatus;
use repose::trajectory::CuFollowStatus;
use repose::trajectory::CuTrajectoryFollowOutput;
use repose_policy_rl::CuActionChunkMode;
use repose_policy_rl::CuActionChunkParams;
use repose_policy_rl::CuActionChunkTask;
use repose_policy_rl::CuCartesianActionParams;
use repose_policy_rl::CuCartesianActionReport;
use repose_policy_rl::CuCartesianActionTask;
use repose_policy_rl::CuChunkPolicyParams;
use repose_policy_rl::CuChunkPolicyTask;
use repose_policy_rl::CuGripperActionParams;
use repose_policy_rl::CuGripperActionTask;
use repose_policy_rl::CuGripperCloseWhen;
use repose_policy_rl::CuJointActionKind;
use repose_policy_rl::CuJointActionParams;
use repose_policy_rl::CuJointActionTask;
use repose_policy_rl::CuJointReachObservationParams;
use repose_policy_rl::CuJointReachObservationTask;
use repose_policy_rl::CuObservationTerm;
use repose_policy_rl::CuPolicyAction;
use repose_policy_rl::CuPolicyActionChunk;
use repose_policy_rl::CuPolicyObservation;
use repose_policy_rl::CuPolicyParams;
use repose_policy_rl::CuPolicyStatus;
use repose_policy_rl::CuPolicyTask;
use repose_policy_rl::CuRotationEncoding;

fn main() {
    divan::main();
}

fn fixture_path(name: &str) -> String {
    format!(
        "{}/../../tasks/repose_policy_rl/tests/fixtures/{name}",
        env!("CARGO_MANIFEST_DIR")
    )
}

fn pos(rad: f64) -> CuJointPosition {
    CuJointPosition::Revolute(Angle::new::<radian>(rad))
}

fn vel(rad_s: f64) -> CuJointVelocity {
    CuJointVelocity::Revolute(AngularVelocity::new::<radian_per_second>(rad_s))
}

fn state_msg() -> CuMsg<CuJointStatePayload<6>> {
    CuMsg::new(Some(CuJointStatePayload::<6> {
        status: CuRobotStatus::Ready,
        positions: [0.1, 0.45, 0.9, -0.1, 0.7, 0.2].map(pos),
        velocities: [0.05, -0.04, 0.08, 0.02, -0.06, 0.03].map(vel),
    }))
}

#[divan::bench]
fn joint_reach_observation(bencher: Bencher) {
    let (ctx, _clock) = CuContext::new_mock_clock();
    let mut task =
        CuJointReachObservationTask::from_params(CuJointReachObservationParams::<6, 18> {
            observation: vec![
                CuObservationTerm::JointPositions { scale: 1.0 },
                CuObservationTerm::JointVelocities { scale: 0.1 },
                CuObservationTerm::GoalError { scale: 1.0 },
            ],
            ..CuJointReachObservationParams::default()
        })
        .unwrap();
    let state = state_msg();
    let goal = CuMsg::new(Some(CuJointPositionsPayload::<6> {
        positions: [0.5, 0.6, 0.8, 0.0, 0.5, -0.1].map(pos),
    }));
    let mut out = CuMsg::<CuPolicyObservation<18>>::default();
    task.process(&ctx, &(&state, &goal), &mut out).unwrap();
    bencher.bench_local(|| {
        task.process(&ctx, black_box(&(&state, &goal)), black_box(&mut out))
            .unwrap();
    });
}

#[divan::bench]
fn policy_infer_linear_24_6(bencher: Bencher) {
    let (ctx, _clock) = CuContext::new_mock_clock();
    let mut task = CuPolicyTask::from_params(CuPolicyParams::<18, 24, 6> {
        model: fixture_path("linear_24_6.onnx"),
        previous_action: true,
        ..CuPolicyParams::default()
    })
    .unwrap();
    let obs = CuMsg::new(Some(CuPolicyObservation::<18> {
        values: core::array::from_fn(|i| (i as f32 - 8.0) * 0.05),
    }));
    let mut out = (
        CuMsg::<CuPolicyObservation<24>>::default(),
        CuMsg::<CuPolicyAction<6>>::default(),
    );
    task.process(&ctx, &obs, &mut out).unwrap();
    assert_eq!(out.1.payload().unwrap().status, CuPolicyStatus::Fresh);
    bencher.bench_local(|| {
        task.process(&ctx, black_box(&obs), black_box(&mut out))
            .unwrap();
    });
}

#[divan::bench]
fn chunk_policy_infer_4_3_2(bencher: Bencher) {
    let (ctx, _clock) = CuContext::new_mock_clock();
    let mut task = CuChunkPolicyTask::<4, 3, 2>::from_params(CuChunkPolicyParams {
        model: fixture_path("chunk_4_3_2.onnx"),
        ..CuChunkPolicyParams::default()
    })
    .unwrap();
    let obs = CuMsg::new(Some(CuPolicyObservation::<4> {
        values: [0.2, -0.1, 0.4, 0.05],
    }));
    let mut out = CuMsg::<CuPolicyActionChunk<3, 2>>::default();
    task.process(&ctx, &obs, &mut out).unwrap();
    assert_eq!(out.payload().unwrap().status, CuPolicyStatus::Fresh);
    bencher.bench_local(|| {
        task.process(&ctx, black_box(&obs), black_box(&mut out))
            .unwrap();
    });
}

#[divan::bench]
fn action_chunk_temporal_ensemble(bencher: Bencher) {
    let (ctx, _clock) = CuContext::new_mock_clock();
    let mut task = CuActionChunkTask::<50, 7>::from_params(CuActionChunkParams {
        mode: CuActionChunkMode::TemporalEnsemble { decay: 0.01 },
        step_decimation: 1,
        execution_horizon: Some(32),
    })
    .unwrap();
    let fresh = CuMsg::new(Some(CuPolicyActionChunk::<50, 7> {
        values: core::array::from_fn(|s| {
            let phase = s as f32 * 0.02;
            [0.2 + phase, -0.1, 0.15, 0.05, -0.05, 0.1, 1.0 - phase]
        }),
        status: CuPolicyStatus::Fresh,
    }));
    let mut out = CuMsg::<CuPolicyAction<7>>::default();
    for _ in 0..8 {
        task.process(&ctx, &fresh, &mut out).unwrap();
    }
    bencher.bench_local(|| {
        task.process(&ctx, black_box(&fresh), black_box(&mut out))
            .unwrap();
    });
}

#[divan::bench]
fn action_chunk_replace(bencher: Bencher) {
    let (ctx, _clock) = CuContext::new_mock_clock();
    let mut task = CuActionChunkTask::<50, 7>::from_params(CuActionChunkParams {
        mode: CuActionChunkMode::Replace,
        step_decimation: 1,
        execution_horizon: Some(32),
    })
    .unwrap();
    let fresh = CuMsg::new(Some(CuPolicyActionChunk::<50, 7> {
        values: core::array::from_fn(|s| [s as f32 * 0.01; 7]),
        status: CuPolicyStatus::Fresh,
    }));
    let empty = CuMsg::<CuPolicyActionChunk<50, 7>>::new(None);
    let mut out = CuMsg::<CuPolicyAction<7>>::default();
    let mut cycle = 0usize;
    bencher.bench_local(|| {
        let msg = if cycle.is_multiple_of(8) {
            &fresh
        } else {
            &empty
        };
        cycle = cycle.wrapping_add(1);
        task.process(&ctx, black_box(msg), black_box(&mut out))
            .unwrap();
    });
}

#[divan::bench]
fn joint_action_delta(bencher: Bencher) {
    let (ctx, _clock) = CuContext::new_mock_clock();
    let mut task = CuJointActionTask::<6, 6>::from_params(CuJointActionParams {
        chain: Some(CuChainSpec::Preset(CuRobotPreset::IgusRebel)),
        offset: 0,
        kind: CuJointActionKind::PositionDelta,
        scale: [0.05; 6],
        default_positions: [CuJointPosition::default(); 6],
        max_step: Some(0.03),
        period: Some(Time::new::<second>(0.02)),
    })
    .unwrap();
    let action = CuMsg::new(Some(CuPolicyAction::<6> {
        values: [0.4, -0.3, 0.25, -0.2, 0.5, -0.15],
        status: CuPolicyStatus::Fresh,
    }));
    let state = state_msg();
    let mut out = CuMsg::<CuTrajectoryFollowOutput<6>>::default();
    task.process(&ctx, &(&action, &state), &mut out).unwrap();
    assert_eq!(out.payload().unwrap().status, CuFollowStatus::Running);
    bencher.bench_local(|| {
        task.process(&ctx, black_box(&(&action, &state)), black_box(&mut out))
            .unwrap();
    });
}

#[divan::bench]
fn cartesian_action_vla(bencher: Bencher) {
    let (ctx, _clock) = CuContext::new_mock_clock();
    let mut arm = CuCartesianActionTask::<6, 7>::from_params(CuCartesianActionParams {
        chain: Some(CuChainSpec::Preset(CuRobotPreset::IgusRebel)),
        period: Some(Time::new::<second>(0.02)),
        rotation: CuRotationEncoding::Rpy,
        translation_scale: Length::new::<meter>(0.005),
        rotation_scale: Angle::new::<radian>(0.02),
        ..CuCartesianActionParams::default()
    })
    .unwrap();
    let mut gripper = CuGripperActionTask::<7>::from_params(CuGripperActionParams {
        offset: 6,
        threshold: 0.5,
        close_when: CuGripperCloseWhen::Below,
        hysteresis: 0.1,
    })
    .unwrap();
    let action = CuMsg::new(Some(CuPolicyAction::<7> {
        values: [0.2, -0.1, 0.15, 0.05, -0.05, 0.1, 0.2],
        status: CuPolicyStatus::Fresh,
    }));
    let state = state_msg();
    let mut arm_out = (
        CuMsg::<CuTrajectoryFollowOutput<6>>::default(),
        CuMsg::<CuCartesianActionReport>::default(),
    );
    let mut grip_out = CuMsg::<CuGripperCommandPayload>::default();
    arm.process(&ctx, &(&action, &state), &mut arm_out).unwrap();
    gripper.process(&ctx, &action, &mut grip_out).unwrap();
    bencher.bench_local(|| {
        arm.process(&ctx, black_box(&(&action, &state)), black_box(&mut arm_out))
            .unwrap();
        gripper
            .process(&ctx, black_box(&action), black_box(&mut grip_out))
            .unwrap();
    });
}
