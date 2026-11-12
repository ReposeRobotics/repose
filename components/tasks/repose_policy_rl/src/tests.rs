//! Crate-level tests of [`CuPolicyTask`] against the committed ONNX fixtures.

use crate::CuObservationTerm;
use crate::CuPolicyAction;
use crate::CuPolicyObservation;
use crate::CuPolicyParams;
use crate::CuPolicyStatus;
use crate::CuPolicyTask;
use cu29::prelude::*;
use cu29::units::si::angle::radian;
use cu29::units::si::angular_velocity::radian_per_second;
use cu29::units::si::f64::Angle;
use cu29::units::si::f64::AngularVelocity;
use repose_kinematics::CuJointPosition;
use repose_kinematics::CuJointPositionsPayload;
use repose_kinematics::CuJointStatePayload;
use repose_kinematics::CuJointVelocity;
use repose_kinematics::CuRobotStatus;

fn fixture(name: &str) -> String {
    format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"))
}

fn pos(rad: f64) -> CuJointPosition {
    CuJointPosition::Revolute(Angle::new::<radian>(rad))
}

fn vel(rad_s: f64) -> CuJointVelocity {
    CuJointVelocity::Revolute(AngularVelocity::new::<radian_per_second>(rad_s))
}

/// `N = 2`, `OBS = 4`, `ACT = 4` params over `joint_positions + goal_error`.
fn identity_params() -> CuPolicyParams<2, 4, 4> {
    CuPolicyParams {
        model: fixture("identity_4.onnx"),
        observation: vec![
            CuObservationTerm::JointPositions { scale: 1.0 },
            CuObservationTerm::GoalError { scale: 1.0 },
        ],
        ..CuPolicyParams::default()
    }
}

fn state2(positions: [f64; 2]) -> CuJointStatePayload<2> {
    CuJointStatePayload {
        status: CuRobotStatus::Ready,
        positions: positions.map(pos),
        velocities: [vel(0.0); 2],
    }
}

fn goal2(positions: [f64; 2]) -> CuJointPositionsPayload<2> {
    CuJointPositionsPayload {
        positions: positions.map(pos),
    }
}

fn run2<const OBS: usize, const ACT: usize>(
    task: &mut CuPolicyTask<2, OBS, ACT>,
    state: Option<CuJointStatePayload<2>>,
    goal: Option<CuJointPositionsPayload<2>>,
) -> (CuPolicyObservation<OBS>, CuPolicyAction<ACT>) {
    let state_msg = CuMsg::new(state);
    let goal_msg = CuMsg::new(goal);
    let input = (&state_msg, &goal_msg);
    let mut output = (
        CuMsg::<CuPolicyObservation<OBS>>::default(),
        CuMsg::<CuPolicyAction<ACT>>::default(),
    );
    let (ctx, _clock) = CuContext::new_mock_clock();
    task.process(&ctx, &input, &mut output).unwrap();
    (
        output.0.payload().unwrap().clone(),
        output.1.payload().unwrap().clone(),
    )
}

#[test]
fn params_round_trip_and_typo_rejection() {
    let ron = format!(
        r#"#![enable(unwrap_newtypes)]
{{
    "model": "{}",
    "decimation": 2,
    "default_positions": [{{"revolute": 0.1}}, {{"revolute": -0.2}}],
    "observation": [
        {{"joint_positions": {{"scale": 1.0}}}},
        {{"goal_error": {{"scale": 2.0}}}},
    ],
}}"#,
        fixture("identity_4.onnx")
    );
    let cfg: ComponentConfig = ron::from_str(&ron).unwrap();
    let task = CuPolicyTask::<2, 4, 4>::new(Some(&cfg), ()).unwrap();
    assert_eq!(task.params().decimation, 2);
    assert_eq!(task.params().default_positions[0].si_value(), 0.1);
    assert_eq!(task.params().default_positions[1].si_value(), -0.2);

    // Serializing the params and reading them back is an identity.
    let text = ron::to_string(task.params()).unwrap();
    let back: CuPolicyParams<2, 4, 4> = ron::from_str(&text).unwrap();
    assert_eq!(&back, task.params());

    let typo = format!(
        r#"#![enable(unwrap_newtypes)]
{{"model": "{}", "decimaton": 2}}"#,
        fixture("identity_4.onnx")
    );
    let cfg: ComponentConfig = ron::from_str(&typo).unwrap();
    let err = CuPolicyTask::<2, 4, 4>::new(Some(&cfg), ()).err().unwrap();
    assert!(format!("{err}").contains("decimaton"), "{err}");
}

#[test]
fn layout_width_mismatch_is_rejected() {
    let params = CuPolicyParams::<2, 5, 4> {
        model: fixture("identity_4.onnx"),
        observation: vec![
            CuObservationTerm::JointPositions { scale: 1.0 },
            CuObservationTerm::GoalError { scale: 1.0 },
        ],
        ..CuPolicyParams::default()
    };
    assert!(params.validate().is_err());
    assert!(CuPolicyTask::from_params(params).is_err());
}

#[test]
fn invalid_params_are_rejected() {
    let base = identity_params();
    assert!(CuPolicyParams::<2, 4, 4>::default().validate().is_err());
    assert!(
        CuPolicyParams {
            decimation: 0,
            ..base.clone()
        }
        .validate()
        .is_err()
    );
    assert!(
        CuPolicyParams {
            model: String::new(),
            ..base
        }
        .validate()
        .is_err()
    );
}

#[test]
fn model_shape_mismatch_is_rejected_at_startup() {
    // bad_shape needs a 5-wide input; pinning OBS = 4 must fail.
    let params = CuPolicyParams::<2, 4, 4> {
        model: fixture("bad_shape.onnx"),
        ..identity_params()
    };
    assert!(CuPolicyTask::from_params(params).is_err());

    // identity_4 produces 4 outputs; ACT = 6 must fail.
    let params = CuPolicyParams::<2, 4, 6> {
        model: fixture("identity_4.onnx"),
        observation: vec![
            CuObservationTerm::JointPositions { scale: 1.0 },
            CuObservationTerm::GoalError { scale: 1.0 },
        ],
        ..CuPolicyParams::default()
    };
    assert!(CuPolicyTask::from_params(params).is_err());
}

#[test]
fn hash_mismatch_is_rejected_and_correct_hash_accepted() {
    let bytes = std::fs::read(fixture("identity_4.onnx")).unwrap();
    let hash = crate::backend::hash_model_bytes(&bytes);
    let wrong = CuPolicyParams::<2, 4, 4> {
        model_hash: Some(hash ^ 1),
        ..identity_params()
    };
    assert!(CuPolicyTask::from_params(wrong).is_err());
    let right = CuPolicyParams::<2, 4, 4> {
        model_hash: Some(hash),
        ..identity_params()
    };
    assert!(CuPolicyTask::from_params(right).is_ok());
}

#[test]
fn identity_model_gives_exact_output() {
    let mut task = CuPolicyTask::from_params(identity_params()).unwrap();
    let (observation, action) = run2(
        &mut task,
        Some(state2([0.5, -0.25])),
        Some(goal2([1.5, 0.75])),
    );
    assert_eq!(observation.values, [0.5, -0.25, 1.0, 1.0]);
    assert_eq!(action.values, observation.values);
    assert_eq!(action.status, CuPolicyStatus::Fresh);
}

#[test]
fn linear_model_selects_goal_error_slice() {
    let params = CuPolicyParams::<6, 24, 6> {
        model: fixture("linear_24_6.onnx"),
        observation: vec![
            CuObservationTerm::JointPositions { scale: 1.0 },
            CuObservationTerm::JointVelocities { scale: 0.1 },
            CuObservationTerm::GoalError { scale: 1.0 },
            CuObservationTerm::PreviousAction,
        ],
        ..CuPolicyParams::default()
    };
    let mut task = CuPolicyTask::from_params(params).unwrap();
    let state = CuJointStatePayload::<6> {
        status: CuRobotStatus::Ready,
        positions: [0.0, 1.0, 2.0, 3.0, 4.0, 5.0].map(pos),
        velocities: [vel(10.0); 6],
    };
    let goal = CuJointPositionsPayload::<6> {
        positions: [2.0; 6].map(pos),
    };
    let state_msg = CuMsg::new(Some(state));
    let goal_msg = CuMsg::new(Some(goal));
    let input = (&state_msg, &goal_msg);
    let mut output = (
        CuMsg::<CuPolicyObservation<24>>::default(),
        CuMsg::<CuPolicyAction<6>>::default(),
    );
    let (ctx, _clock) = CuContext::new_mock_clock();
    task.process(&ctx, &input, &mut output).unwrap();
    let observation = output.0.payload().unwrap();
    assert_eq!(&observation.values[0..6], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    assert_eq!(&observation.values[6..12], [1.0; 6]);
    assert_eq!(
        &observation.values[12..18],
        [2.0, 1.0, 0.0, -1.0, -2.0, -3.0]
    );
    assert_eq!(&observation.values[18..24], [0.0; 6]);
    let action = output.1.payload().unwrap();
    // The fixture gain is 0.5 (see make_fixtures.py).
    assert_eq!(action.values, [1.0, 0.5, 0.0, -0.5, -1.0, -1.5]);
    assert_eq!(action.status, CuPolicyStatus::Fresh);

    // The next cycle's previous_action term carries this raw action.
    let state_msg = CuMsg::new(Some(CuJointStatePayload::<6> {
        status: CuRobotStatus::Ready,
        positions: [pos(0.0); 6],
        velocities: [vel(0.0); 6],
    }));
    let input = (&state_msg, &goal_msg);
    task.process(&ctx, &input, &mut output).unwrap();
    let observation = output.0.payload().unwrap();
    assert_eq!(
        &observation.values[18..24],
        [1.0, 0.5, 0.0, -0.5, -1.0, -1.5]
    );
}

#[test]
fn decimation_holds_between_inferences() {
    let params = CuPolicyParams::<2, 4, 4> {
        decimation: 3,
        ..identity_params()
    };
    let mut task = CuPolicyTask::from_params(params).unwrap();
    let goal = goal2([1.5, 0.75]);

    let (_, action) = run2(&mut task, Some(state2([0.5, -0.25])), Some(goal.clone()));
    assert_eq!(action.status, CuPolicyStatus::Fresh);
    assert_eq!(action.values, [0.5, -0.25, 1.0, 1.0]);

    // The state moved, but cycles 1 and 2 repeat the inferred action while
    // the observation still shows the new state.
    for _ in 0..2 {
        let (observation, action) = run2(&mut task, Some(state2([0.25, 0.0])), Some(goal.clone()));
        assert_eq!(observation.values, [0.25, 0.0, 1.25, 0.75]);
        assert_eq!(action.status, CuPolicyStatus::Held);
        assert_eq!(action.values, [0.5, -0.25, 1.0, 1.0]);
    }

    let (_, action) = run2(&mut task, Some(state2([0.25, 0.0])), Some(goal));
    assert_eq!(action.status, CuPolicyStatus::Fresh);
    assert_eq!(action.values, [0.25, 0.0, 1.25, 0.75]);
}

#[test]
fn nan_model_output_is_fault_with_zero_values() {
    let params = CuPolicyParams::<2, 4, 4> {
        model: fixture("nan.onnx"),
        ..identity_params()
    };
    let mut task = CuPolicyTask::from_params(params).unwrap();
    let (observation, action) = run2(
        &mut task,
        Some(state2([0.5, -0.25])),
        Some(goal2([1.5, 0.75])),
    );
    assert_eq!(observation.values, [0.5, -0.25, 1.0, 1.0]);
    assert_eq!(action.status, CuPolicyStatus::Fault);
    assert_eq!(action.values, [0.0; 4]);
}

#[test]
fn missing_or_unready_inputs_are_idle() {
    let mut task = CuPolicyTask::from_params(identity_params()).unwrap();
    let (observation, action) = run2(&mut task, Some(state2([0.5, -0.25])), None);
    assert_eq!(observation.values, [0.0; 4]);
    assert_eq!(action.status, CuPolicyStatus::Idle);
    assert_eq!(action.values, [0.0; 4]);

    let mut not_ready = state2([0.5, -0.25]);
    not_ready.status = CuRobotStatus::NotReady;
    let (_, action) = run2(&mut task, Some(not_ready), Some(goal2([1.5, 0.75])));
    assert_eq!(action.status, CuPolicyStatus::Idle);
}

#[test]
fn freeze_thaw_resumes_identically() {
    let params = CuPolicyParams::<2, 4, 4> {
        decimation: 2,
        ..identity_params()
    };
    let mut task = CuPolicyTask::from_params(params.clone()).unwrap();
    let goal = goal2([1.5, 0.75]);
    let (_, action) = run2(&mut task, Some(state2([0.5, -0.25])), Some(goal.clone()));
    assert_eq!(action.status, CuPolicyStatus::Fresh);

    let mut bytes = [0u8; 256];
    let mut encoder = bincode::enc::EncoderImpl::new(
        bincode::enc::write::SliceWriter::new(&mut bytes),
        bincode::config::standard(),
    );
    task.freeze(&mut encoder).unwrap();
    let len = encoder.into_writer().bytes_written();

    let mut restored = CuPolicyTask::from_params(params).unwrap();
    restored
        .thaw(&mut bincode::de::DecoderImpl::new(
            bincode::de::read::SliceReader::new(&bytes[..len]),
            bincode::config::standard(),
            (),
        ))
        .unwrap();

    // Both tasks are mid-decimation: the next cycle holds, the one after
    // infers, and the outputs agree exactly.
    for expected in [CuPolicyStatus::Held, CuPolicyStatus::Fresh] {
        let (obs_a, act_a) = run2(&mut task, Some(state2([0.25, 0.0])), Some(goal.clone()));
        let (obs_b, act_b) = run2(&mut restored, Some(state2([0.25, 0.0])), Some(goal.clone()));
        assert_eq!(act_a.status, expected);
        assert_eq!(obs_a, obs_b);
        assert_eq!(act_a, act_b);
    }
}
