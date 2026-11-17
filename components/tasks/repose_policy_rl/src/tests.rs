//! Crate-level tests of [`CuPolicyTask`] against the committed ONNX fixtures,
//! of [`CuJointReachObservationTask`], and of the two chained.

use crate::CuJointReachObservationParams;
use crate::CuJointReachObservationTask;
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

/// `IN = OBS = ACT = 4` params over the identity model, no previous action.
fn identity_params() -> CuPolicyParams<4, 4, 4> {
    CuPolicyParams {
        model: fixture("identity_4.onnx"),
        ..CuPolicyParams::default()
    }
}

/// `N = 2`, `IN = 4` params over `joint_positions + goal_error`.
fn reach_params() -> CuJointReachObservationParams<2, 4> {
    CuJointReachObservationParams {
        observation: vec![
            CuObservationTerm::JointPositions { scale: 1.0 },
            CuObservationTerm::GoalError { scale: 1.0 },
        ],
        ..CuJointReachObservationParams::default()
    }
}

/// `N = 6`, `IN = 18` params over positions, velocities (scale `velocity`),
/// and goal error.
fn reach6_params(velocity: f64) -> CuJointReachObservationParams<6, 18> {
    CuJointReachObservationParams {
        observation: vec![
            CuObservationTerm::JointPositions { scale: 1.0 },
            CuObservationTerm::JointVelocities { scale: velocity },
            CuObservationTerm::GoalError { scale: 1.0 },
        ],
        ..CuJointReachObservationParams::default()
    }
}

/// `IN = 18`, `OBS = 24`, `ACT = 6` over `linear_24_6`, previous action on.
fn linear_params() -> CuPolicyParams<18, 24, 6> {
    CuPolicyParams {
        model: fixture("linear_24_6.onnx"),
        previous_action: true,
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

fn features<const IN: usize>(values: [f32; IN]) -> Option<CuPolicyObservation<IN>> {
    Some(CuPolicyObservation { values })
}

/// One policy cycle on `input`; returns `(model input, action)`.
fn run<const IN: usize, const OBS: usize, const ACT: usize>(
    task: &mut CuPolicyTask<IN, OBS, ACT>,
    input: Option<CuPolicyObservation<IN>>,
) -> (CuPolicyObservation<OBS>, CuPolicyAction<ACT>) {
    let input_msg = CuMsg::new(input);
    let mut output = (
        CuMsg::<CuPolicyObservation<OBS>>::default(),
        CuMsg::<CuPolicyAction<ACT>>::default(),
    );
    let (ctx, _clock) = CuContext::new_mock_clock();
    task.process(&ctx, &input_msg, &mut output).unwrap();
    (
        output.0.payload().unwrap().clone(),
        output.1.payload().unwrap().clone(),
    )
}

/// One observation cycle; returns the payload, if any.
fn observe<const N: usize, const IN: usize>(
    task: &mut CuJointReachObservationTask<N, IN>,
    state: Option<CuJointStatePayload<N>>,
    goal: Option<CuJointPositionsPayload<N>>,
) -> Option<CuPolicyObservation<IN>> {
    let state_msg = CuMsg::new(state);
    let goal_msg = CuMsg::new(goal);
    let mut output = CuMsg::<CuPolicyObservation<IN>>::default();
    // A stale payload must not survive a cycle without usable inputs.
    output.set_payload(CuPolicyObservation { values: [7.0; IN] });
    let (ctx, _clock) = CuContext::new_mock_clock();
    task.process(&ctx, &(&state_msg, &goal_msg), &mut output)
        .unwrap();
    output.payload().cloned()
}

// --- Policy task: params and model loading ---------------------------------

#[test]
fn params_round_trip_and_typo_rejection() {
    let ron = format!(
        r#"#![enable(unwrap_newtypes)]
{{
    "model": "{}",
    "decimation": 2,
    "previous_action": true,
}}"#,
        fixture("identity_4.onnx")
    );
    let cfg: ComponentConfig = ron::from_str(&ron).unwrap();
    // identity_4 is [1, 4] -> [1, 4], so with the previous action appended
    // the input carries no features: IN = 0, OBS = IN + ACT = 4.
    let params: CuPolicyParams<0, 4, 4> = cfg.deserialize_into().unwrap();
    assert_eq!(params.decimation, 2);
    assert!(params.previous_action);
    let task = CuPolicyTask::<0, 4, 4>::new(Some(&cfg), ()).unwrap();
    assert_eq!(task.params(), &params);

    // Serializing the params and reading them back is an identity.
    let text = ron::to_string(task.params()).unwrap();
    let back: CuPolicyParams<0, 4, 4> = ron::from_str(&text).unwrap();
    assert_eq!(&back, task.params());

    let typo = format!(
        r#"#![enable(unwrap_newtypes)]
{{"model": "{}", "decimaton": 2}}"#,
        fixture("identity_4.onnx")
    );
    let cfg: ComponentConfig = ron::from_str(&typo).unwrap();
    let err = CuPolicyTask::<4, 4, 4>::new(Some(&cfg), ()).err().unwrap();
    assert!(format!("{err}").contains("decimaton"), "{err}");
}

#[test]
fn invalid_params_are_rejected() {
    let base = identity_params();
    assert!(CuPolicyParams::<4, 4, 4>::default().validate().is_err());
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
fn previous_action_sets_the_required_model_input_width() {
    let with = |previous_action| CuPolicyParams::<18, 24, 6> {
        previous_action,
        ..linear_params()
    };
    assert!(with(true).validate().is_ok());
    let err = with(false).validate().err().unwrap();
    assert!(format!("{err}").contains("OBS must equal IN"), "{err}");

    let without = |previous_action| CuPolicyParams::<4, 4, 4> {
        previous_action,
        ..identity_params()
    };
    assert!(without(false).validate().is_ok());
    let err = without(true).validate().err().unwrap();
    assert!(
        format!("{err}").contains("OBS must equal IN + ACT"),
        "{err}"
    );
    assert!(CuPolicyTask::from_params(without(true)).is_err());
}

#[test]
fn model_shape_mismatch_is_rejected_at_startup() {
    // bad_shape needs a 5-wide input; pinning OBS = 4 must fail.
    let params = CuPolicyParams::<4, 4, 4> {
        model: fixture("bad_shape.onnx"),
        ..identity_params()
    };
    assert!(CuPolicyTask::from_params(params).is_err());

    // identity_4 produces 4 outputs; ACT = 6 must fail.
    let params = CuPolicyParams::<4, 4, 6> {
        model: fixture("identity_4.onnx"),
        ..CuPolicyParams::default()
    };
    assert!(CuPolicyTask::from_params(params).is_err());
}

#[test]
fn hash_mismatch_is_rejected_and_correct_hash_accepted() {
    let bytes = std::fs::read(fixture("identity_4.onnx")).unwrap();
    let hash = crate::backend::hash_model_bytes(&bytes);
    let wrong = CuPolicyParams::<4, 4, 4> {
        model_hash: Some(hash ^ 1),
        ..identity_params()
    };
    assert!(CuPolicyTask::from_params(wrong).is_err());
    let right = CuPolicyParams::<4, 4, 4> {
        model_hash: Some(hash),
        ..identity_params()
    };
    assert!(CuPolicyTask::from_params(right).is_ok());
}

#[test]
fn double_model_is_rejected_at_startup() {
    let params = CuPolicyParams::<4, 4, 4> {
        model: fixture("double_4.onnx"),
        ..identity_params()
    };
    let err = CuPolicyTask::from_params(params).err().unwrap();
    assert!(format!("{err}").contains("must be f32"), "{err}");
}

#[test]
fn declared_input_width_mismatch_is_rejected_at_startup() {
    // The graph (an Identity) would run at any width; the declared [1, 5]
    // input must still be rejected for OBS = 4.
    let params = CuPolicyParams::<4, 4, 4> {
        model: fixture("identity_5.onnx"),
        ..identity_params()
    };
    let err = CuPolicyTask::from_params(params).err().unwrap();
    assert!(format!("{err}").contains("shape [1, OBS]"), "{err}");
}

#[test]
fn dynamic_batch_model_loads_and_runs() {
    let params = CuPolicyParams::<4, 4, 4> {
        model: fixture("batch_4.onnx"),
        ..identity_params()
    };
    let mut task = CuPolicyTask::from_params(params).unwrap();
    let (_, action) = run(&mut task, features([0.5, -0.25, 1.0, 1.0]));
    assert_eq!(action.status, CuPolicyStatus::Fresh);
    assert_eq!(action.values, [0.5, -0.25, 1.0, 1.0]);
}

#[test]
fn wrong_width_behind_an_unknown_batch_dimension_is_rejected() {
    let params = CuPolicyParams::<4, 4, 4> {
        model: fixture("open_batch_5.onnx"),
        ..identity_params()
    };
    let err = CuPolicyTask::from_params(params).err().unwrap();
    assert!(format!("{err}").contains("shape [1, OBS]"), "{err}");
}

// --- Policy task: cycles ----------------------------------------------------

#[test]
fn identity_model_gives_exact_output() {
    let mut task = CuPolicyTask::from_params(identity_params()).unwrap();
    let (observation, action) = run(&mut task, features([0.5, -0.25, 1.0, 1.0]));
    assert_eq!(observation.values, [0.5, -0.25, 1.0, 1.0]);
    assert_eq!(action.values, observation.values);
    assert_eq!(action.status, CuPolicyStatus::Fresh);
}

#[test]
fn previous_action_is_appended_after_the_features() {
    let mut task = CuPolicyTask::from_params(linear_params()).unwrap();
    let mut input = [0.0f32; 18];
    input[12..18].copy_from_slice(&[2.0, 1.0, 0.0, -1.0, -2.0, -3.0]);
    let (observation, action) = run(&mut task, features(input));
    assert_eq!(observation.values[..18], input);
    assert_eq!(observation.values[18..24], [0.0; 6]);
    // The fixture gain is 0.5 on the goal-error slice (see make_fixtures.py).
    assert_eq!(action.values, [1.0, 0.5, 0.0, -0.5, -1.0, -1.5]);

    let (observation, _) = run(&mut task, features([0.25; 18]));
    assert_eq!(observation.values[..18], [0.25; 18]);
    assert_eq!(
        observation.values[18..24],
        [1.0, 0.5, 0.0, -0.5, -1.0, -1.5]
    );
}

#[test]
fn decimation_holds_between_inferences() {
    let params = CuPolicyParams::<4, 4, 4> {
        decimation: 3,
        ..identity_params()
    };
    let mut task = CuPolicyTask::from_params(params).unwrap();

    let (_, action) = run(&mut task, features([0.5, -0.25, 1.0, 1.0]));
    assert_eq!(action.status, CuPolicyStatus::Fresh);
    assert_eq!(action.values, [0.5, -0.25, 1.0, 1.0]);

    // The input moved, but cycles 1 and 2 repeat the inferred action while
    // the observation still shows the new input.
    for _ in 0..2 {
        let (observation, action) = run(&mut task, features([0.25, 0.0, 1.25, 0.75]));
        assert_eq!(observation.values, [0.25, 0.0, 1.25, 0.75]);
        assert_eq!(action.status, CuPolicyStatus::Held);
        assert_eq!(action.values, [0.5, -0.25, 1.0, 1.0]);
    }

    let (_, action) = run(&mut task, features([0.25, 0.0, 1.25, 0.75]));
    assert_eq!(action.status, CuPolicyStatus::Fresh);
    assert_eq!(action.values, [0.25, 0.0, 1.25, 0.75]);
}

#[test]
fn nan_model_output_is_fault_with_zero_values() {
    let params = CuPolicyParams::<4, 4, 4> {
        model: fixture("nan.onnx"),
        ..identity_params()
    };
    let mut task = CuPolicyTask::from_params(params).unwrap();
    let (observation, action) = run(&mut task, features([0.5, -0.25, 1.0, 1.0]));
    assert_eq!(observation.values, [0.5, -0.25, 1.0, 1.0]);
    assert_eq!(action.status, CuPolicyStatus::Fault);
    assert_eq!(action.values, [0.0; 4]);
}

#[test]
fn missing_input_is_idle() {
    let mut task = CuPolicyTask::from_params(identity_params()).unwrap();
    let (observation, action) = run(&mut task, None);
    assert_eq!(observation.values, [0.0; 4]);
    assert_eq!(action.status, CuPolicyStatus::Idle);
    assert_eq!(action.values, [0.0; 4]);
}

#[test]
fn missing_input_resets_the_previous_action() {
    let mut task = CuPolicyTask::from_params(linear_params()).unwrap();
    let mut input = [0.0f32; 18];
    input[12..18].copy_from_slice(&[1.0; 6]);
    let (_, action) = run(&mut task, features(input));
    assert_eq!(action.values, [0.5; 6]);
    let (_, action) = run(&mut task, None);
    assert_eq!(action.status, CuPolicyStatus::Idle);
    let (observation, _) = run(&mut task, features(input));
    assert_eq!(observation.values[18..24], [0.0; 6]);
}

#[test]
fn freeze_thaw_resumes_identically() {
    let params = CuPolicyParams::<4, 4, 4> {
        decimation: 2,
        ..identity_params()
    };
    let mut task = CuPolicyTask::from_params(params.clone()).unwrap();
    let (_, action) = run(&mut task, features([0.5, -0.25, 1.0, 1.0]));
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
        let (obs_a, act_a) = run(&mut task, features([0.25, 0.0, 1.25, 0.75]));
        let (obs_b, act_b) = run(&mut restored, features([0.25, 0.0, 1.25, 0.75]));
        assert_eq!(act_a.status, expected);
        assert_eq!(obs_a, obs_b);
        assert_eq!(act_a, act_b);
    }
}

#[test]
fn fault_lasts_until_the_next_inference() {
    let params = CuPolicyParams::<4, 4, 4> {
        model: fixture("nan.onnx"),
        decimation: 2,
        ..identity_params()
    };
    let mut task = CuPolicyTask::from_params(params).unwrap();
    for _ in 0..4 {
        let (_, action) = run(&mut task, features([0.5, -0.25, 1.0, 1.0]));
        assert_eq!(action.status, CuPolicyStatus::Fault);
        assert_eq!(action.values, [0.0; 4]);
    }
}

#[test]
fn idle_restarts_the_decimation_count() {
    let params = CuPolicyParams::<4, 4, 4> {
        decimation: 3,
        ..identity_params()
    };
    let mut task = CuPolicyTask::from_params(params).unwrap();
    let input = features([0.5, -0.25, 1.0, 1.0]);
    let statuses =
        [input.clone(), None, input.clone(), input].map(|input| run(&mut task, input).1.status);
    assert_eq!(
        statuses,
        [
            CuPolicyStatus::Fresh,
            CuPolicyStatus::Idle,
            CuPolicyStatus::Fresh,
            CuPolicyStatus::Held,
        ]
    );
}

#[test]
fn nan_input_is_fault_and_recovers() {
    let mut task = CuPolicyTask::from_params(identity_params()).unwrap();
    let (_, action) = run(&mut task, features([f32::NAN, -0.25, 1.0, 1.0]));
    assert_eq!(action.status, CuPolicyStatus::Fault);
    assert_eq!(action.values, [0.0; 4]);
    let (_, action) = run(&mut task, features([0.5, -0.25, 1.0, 1.0]));
    assert_eq!(action.status, CuPolicyStatus::Fresh);
}

#[test]
fn bad_inputs_fault_a_held_cycle() {
    let params = CuPolicyParams::<4, 4, 4> {
        decimation: 3,
        ..identity_params()
    };
    let mut task = CuPolicyTask::from_params(params).unwrap();
    let good = [0.5, -0.25, 1.0, 1.0];
    let actions = [good, [f32::NAN, -0.25, 1.0, 1.0], good, good]
        .map(|input| run(&mut task, features(input)).1);
    assert_eq!(actions[0].status, CuPolicyStatus::Fresh);
    // A non-finite input on a held cycle faults, and the fault lasts until
    // the next inference.
    assert_eq!(actions[1].status, CuPolicyStatus::Fault);
    assert_eq!(actions[1].values, [0.0; 4]);
    assert_eq!(actions[2].status, CuPolicyStatus::Fault);
    assert_eq!(actions[3].status, CuPolicyStatus::Fresh);
}

#[test]
fn held_cycle_fault_clears_the_previous_action() {
    let params = CuPolicyParams::<18, 24, 6> {
        decimation: 3,
        ..linear_params()
    };
    let mut task = CuPolicyTask::from_params(params).unwrap();
    let mut good = [0.0f32; 18];
    good[12..18].copy_from_slice(&[1.0; 6]);
    // Inference: a nonzero action (0.5 * goal error).
    let (_, action) = run(&mut task, features(good));
    assert_eq!(action.status, CuPolicyStatus::Fresh);
    assert_eq!(action.values, [0.5; 6]);
    // A held cycle with a non-finite input faults and publishes zeros.
    let (_, action) = run(&mut task, features([f32::NAN; 18]));
    assert_eq!(action.status, CuPolicyStatus::Fault);
    // The next model input's previous action is what was published (zeros),
    // not the action from before the fault.
    let (observation, _) = run(&mut task, features(good));
    assert_eq!(observation.values[18..24], [0.0; 6]);
}

#[test]
fn policy_tov_is_propagated_from_the_input() {
    let mut task = CuPolicyTask::from_params(identity_params()).unwrap();
    let mut input = CuMsg::new(features([0.5, -0.25, 1.0, 1.0]));
    input.tov = Tov::Time(CuTime::from_nanos(424_242));
    let mut output = (
        CuMsg::<CuPolicyObservation<4>>::default(),
        CuMsg::<CuPolicyAction<4>>::default(),
    );
    let (ctx, _clock) = CuContext::new_mock_clock();
    task.process(&ctx, &input, &mut output).unwrap();
    assert_eq!(output.0.tov, input.tov);
    assert_eq!(output.1.tov, input.tov);
}

// --- Observation task -------------------------------------------------------

#[test]
fn observation_params_round_trip_and_typo_rejection() {
    let ron = r#"#![enable(unwrap_newtypes)]
{
    "default_positions": [{"revolute": 0.1}, {"revolute": -0.2}],
    "observation": [
        {"joint_positions": {"scale": 1.0}},
        {"goal_error": {"scale": 2.0}},
    ],
}"#;
    let cfg: ComponentConfig = ron::from_str(ron).unwrap();
    let task = CuJointReachObservationTask::<2, 4>::new(Some(&cfg), ()).unwrap();
    assert_eq!(task.params().default_positions[0].si_value(), 0.1);
    assert_eq!(task.params().default_positions[1].si_value(), -0.2);
    assert_eq!(
        task.params().observation[1],
        CuObservationTerm::GoalError { scale: 2.0 }
    );

    let text = ron::to_string(task.params()).unwrap();
    let back: CuJointReachObservationParams<2, 4> = ron::from_str(&text).unwrap();
    assert_eq!(&back, task.params());

    let typo = r#"#![enable(unwrap_newtypes)]
{"observaton": []}"#;
    let cfg: ComponentConfig = ron::from_str(typo).unwrap();
    let err = CuJointReachObservationTask::<2, 4>::new(Some(&cfg), ())
        .err()
        .unwrap();
    assert!(format!("{err}").contains("observaton"), "{err}");
}

#[test]
fn layout_width_mismatch_is_rejected() {
    let params = CuJointReachObservationParams::<2, 5> {
        observation: reach_params().observation,
        ..CuJointReachObservationParams::default()
    };
    assert!(params.validate().is_err());
    assert!(CuJointReachObservationTask::from_params(params).is_err());
}

#[test]
fn non_finite_default_positions_are_rejected() {
    let mut params = reach_params();
    params.default_positions[1] = pos(f64::NAN);
    assert!(params.validate().is_err());
}

#[test]
fn observation_builds_the_layout() {
    let mut params = reach_params();
    params.default_positions = [pos(0.25), pos(-0.25)];
    let mut task = CuJointReachObservationTask::from_params(params).unwrap();
    let observation = observe(
        &mut task,
        Some(state2([0.5, -0.25])),
        Some(goal2([1.5, 0.75])),
    )
    .unwrap();
    assert_eq!(observation.values, [0.25, 0.0, 1.0, 1.0]);
}

#[test]
fn missing_or_unready_inputs_emit_no_payload() {
    let mut task = CuJointReachObservationTask::from_params(reach_params()).unwrap();
    assert!(observe(&mut task, Some(state2([0.5, -0.25])), None).is_none());
    assert!(observe(&mut task, None, Some(goal2([1.5, 0.75]))).is_none());

    let mut not_ready = state2([0.5, -0.25]);
    not_ready.status = CuRobotStatus::NotReady;
    assert!(observe(&mut task, Some(not_ready), Some(goal2([1.5, 0.75]))).is_none());
}

#[test]
fn mismatched_joint_kinds_emit_nan() {
    use cu29::units::si::f64::Length;
    use cu29::units::si::length::meter;
    let mut task = CuJointReachObservationTask::from_params(reach_params()).unwrap();
    // A prismatic goal against a revolute state: meters minus radians.
    let mut goal = goal2([1.5, 0.75]);
    goal.positions[1] = CuJointPosition::Prismatic(Length::new::<meter>(0.75));
    let observation = observe(&mut task, Some(state2([0.5, -0.25])), Some(goal)).unwrap();
    assert!(observation.values.iter().all(|value| value.is_nan()));
}

#[test]
fn velocity_kind_mismatch_emits_nan() {
    use cu29::units::si::f64::Velocity;
    use cu29::units::si::velocity::meter_per_second;
    let params = CuJointReachObservationParams::<2, 4> {
        observation: vec![
            CuObservationTerm::JointPositions { scale: 1.0 },
            CuObservationTerm::JointVelocities { scale: 1.0 },
        ],
        ..CuJointReachObservationParams::default()
    };
    let mut task = CuJointReachObservationTask::from_params(params).unwrap();
    let mut state = state2([0.5, -0.25]);
    state.velocities[1] = CuJointVelocity::Prismatic(Velocity::new::<meter_per_second>(0.1));
    let observation = observe(&mut task, Some(state), Some(goal2([1.5, 0.75]))).unwrap();
    assert!(observation.values.iter().all(|value| value.is_nan()));
}

#[test]
fn observation_tov_is_propagated_from_the_joint_state() {
    let mut task = CuJointReachObservationTask::from_params(reach_params()).unwrap();
    let mut state_msg = CuMsg::new(Some(state2([0.5, -0.25])));
    state_msg.tov = Tov::Time(CuTime::from_nanos(424_242));
    let goal_msg = CuMsg::new(Some(goal2([1.5, 0.75])));
    let mut output = CuMsg::<CuPolicyObservation<4>>::default();
    let (ctx, _clock) = CuContext::new_mock_clock();
    task.process(&ctx, &(&state_msg, &goal_msg), &mut output)
        .unwrap();
    assert_eq!(output.tov, state_msg.tov);
}

// --- Observation task feeding the policy task -------------------------------

/// One cycle of `observation -> policy`, wired as a Copper graph would.
fn chain<const N: usize, const IN: usize, const OBS: usize, const ACT: usize>(
    observation: &mut CuJointReachObservationTask<N, IN>,
    policy: &mut CuPolicyTask<IN, OBS, ACT>,
    state: CuJointStatePayload<N>,
    goal: CuJointPositionsPayload<N>,
) -> (CuPolicyObservation<OBS>, CuPolicyAction<ACT>) {
    let state_msg = CuMsg::new(Some(state));
    let goal_msg = CuMsg::new(Some(goal));
    let mut features = CuMsg::<CuPolicyObservation<IN>>::default();
    let mut output = (
        CuMsg::<CuPolicyObservation<OBS>>::default(),
        CuMsg::<CuPolicyAction<ACT>>::default(),
    );
    let (ctx, _clock) = CuContext::new_mock_clock();
    observation
        .process(&ctx, &(&state_msg, &goal_msg), &mut features)
        .unwrap();
    policy.process(&ctx, &features, &mut output).unwrap();
    (
        output.0.payload().unwrap().clone(),
        output.1.payload().unwrap().clone(),
    )
}

#[test]
fn chained_linear_model_selects_goal_error_slice() {
    let mut observation = CuJointReachObservationTask::from_params(reach6_params(0.1)).unwrap();
    let mut policy = CuPolicyTask::from_params(linear_params()).unwrap();
    let state = CuJointStatePayload::<6> {
        status: CuRobotStatus::Ready,
        positions: [0.0, 1.0, 2.0, 3.0, 4.0, 5.0].map(pos),
        velocities: [vel(10.0); 6],
    };
    let goal = CuJointPositionsPayload::<6> {
        positions: [2.0; 6].map(pos),
    };
    let (model_input, action) = chain(&mut observation, &mut policy, state, goal.clone());
    assert_eq!(&model_input.values[0..6], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    assert_eq!(&model_input.values[6..12], [1.0; 6]);
    assert_eq!(
        &model_input.values[12..18],
        [2.0, 1.0, 0.0, -1.0, -2.0, -3.0]
    );
    assert_eq!(&model_input.values[18..24], [0.0; 6]);
    // The fixture gain is 0.5 (see make_fixtures.py).
    assert_eq!(action.values, [1.0, 0.5, 0.0, -0.5, -1.0, -1.5]);
    assert_eq!(action.status, CuPolicyStatus::Fresh);

    // The next cycle's appended previous action carries this raw action.
    let state = CuJointStatePayload::<6> {
        status: CuRobotStatus::Ready,
        positions: [pos(0.0); 6],
        velocities: [vel(0.0); 6],
    };
    let (model_input, _) = chain(&mut observation, &mut policy, state, goal);
    assert_eq!(
        &model_input.values[18..24],
        [1.0, 0.5, 0.0, -0.5, -1.0, -1.5]
    );
}

#[test]
fn chained_unready_robot_is_idle_and_kind_mismatch_is_fault() {
    use cu29::units::si::f64::Length;
    use cu29::units::si::length::meter;
    let mut observation = CuJointReachObservationTask::from_params(reach_params()).unwrap();
    let mut policy = CuPolicyTask::from_params(identity_params()).unwrap();

    let mut not_ready = state2([0.5, -0.25]);
    not_ready.status = CuRobotStatus::NotReady;
    let (_, action) = chain(&mut observation, &mut policy, not_ready, goal2([1.5, 0.75]));
    assert_eq!(action.status, CuPolicyStatus::Idle);

    let mut goal = goal2([1.5, 0.75]);
    goal.positions[1] = CuJointPosition::Prismatic(Length::new::<meter>(0.75));
    let (_, action) = chain(&mut observation, &mut policy, state2([0.5, -0.25]), goal);
    assert_eq!(action.status, CuPolicyStatus::Fault);
    assert_eq!(action.values, [0.0; 4]);

    let (_, action) = chain(
        &mut observation,
        &mut policy,
        state2([0.5, -0.25]),
        goal2([1.5, 0.75]),
    );
    assert_eq!(action.status, CuPolicyStatus::Fresh);
    assert_eq!(action.values, [0.5, -0.25, 1.0, 1.0]);
}

// --- Chunked ONNX policy + temporal ensembling ------------------------------

#[test]
fn chunk_policy_loads_fixed_and_dynamic_batch_and_rejects_wrong_shapes() {
    use crate::CuChunkPolicyParams;
    use crate::CuChunkPolicyTask;

    for model_name in ["chunk_4_3_2.onnx", "chunk_batch_4_3_2.onnx"] {
        let mut task = CuChunkPolicyTask::<4, 3, 2>::from_params(CuChunkPolicyParams {
            model: fixture(model_name),
            ..CuChunkPolicyParams::default()
        })
        .unwrap();
        let (ctx, _clock) = CuContext::new_mock_clock();
        let obs = CuMsg::new(features([1.0, 2.0, 3.0, 6.0]));
        let mut out = CuMsg::default();
        task.process(&ctx, &obs, &mut out).unwrap();
        let chunk = out.payload().unwrap();
        assert_eq!(chunk.status, CuPolicyStatus::Fresh);
        // obs[0]+obs[2] = 4.0, obs[1]+obs[3] = 8.0; scales = [0.25, 0.5, 0.75]
        assert_eq!(chunk.values, [[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]]);
    }

    // Wrong H, wrong ACT, wrong OBS, or a rank-2 [1, 4] single-step model must fail at startup.
    assert!(
        CuChunkPolicyTask::<4, 4, 2>::from_params(CuChunkPolicyParams {
            model: fixture("chunk_4_3_2.onnx"),
            ..CuChunkPolicyParams::default()
        })
        .is_err()
    );
    assert!(
        CuChunkPolicyTask::<4, 3, 3>::from_params(CuChunkPolicyParams {
            model: fixture("chunk_4_3_2.onnx"),
            ..CuChunkPolicyParams::default()
        })
        .is_err()
    );
    assert!(
        CuChunkPolicyTask::<5, 3, 2>::from_params(CuChunkPolicyParams {
            model: fixture("chunk_4_3_2.onnx"),
            ..CuChunkPolicyParams::default()
        })
        .is_err()
    );
    assert!(
        CuChunkPolicyTask::<4, 1, 4>::from_params(CuChunkPolicyParams {
            model: fixture("identity_4.onnx"),
            ..CuChunkPolicyParams::default()
        })
        .is_err()
    );
    // Wrong hash pin fails at startup.
    assert!(
        CuChunkPolicyTask::<4, 3, 2>::from_params(CuChunkPolicyParams {
            model: fixture("chunk_4_3_2.onnx"),
            model_hash: Some(42),
            ..CuChunkPolicyParams::default()
        })
        .is_err()
    );
}

#[test]
fn chained_observation_chunk_policy_and_temporal_ensembler() {
    use crate::CuActionChunkMode;
    use crate::CuActionChunkParams;
    use crate::CuActionChunkTask;
    use crate::CuChunkPolicyParams;
    use crate::CuChunkPolicyTask;

    let mut obs_task = CuJointReachObservationTask::from_params(reach_params()).unwrap();
    let mut chunk_policy = CuChunkPolicyTask::<4, 3, 2>::from_params(CuChunkPolicyParams {
        model: fixture("chunk_4_3_2.onnx"),
        decimation: 2,
        model_hash: None,
    })
    .unwrap();
    let mut ensembler = CuActionChunkTask::<3, 2>::from_params(CuActionChunkParams {
        mode: CuActionChunkMode::TemporalEnsemble { decay: 0.0 },
        step_decimation: 1,
        execution_horizon: None,
    })
    .unwrap();

    let (ctx, _clock) = CuContext::new_mock_clock();
    let mut run_cycle = |state: Option<CuJointStatePayload<2>>| -> CuPolicyAction<2> {
        let obs = observe(&mut obs_task, state, Some(goal2([3.0, 6.0])));
        let obs_msg = CuMsg::new(obs);
        let mut chunk_msg = CuMsg::default();
        chunk_policy
            .process(&ctx, &obs_msg, &mut chunk_msg)
            .unwrap();
        let mut act_msg = CuMsg::default();
        ensembler.process(&ctx, &chunk_msg, &mut act_msg).unwrap();
        act_msg.payload().unwrap().clone()
    };

    // Cycle 0 (Fresh inference): pos + goal_error = goal = [3.0, 6.0] ->
    // chunk = [[0.75, 1.5], [1.5, 3.0], [2.25, 4.5]]. Emits step 0.
    let a0 = run_cycle(Some(state2([1.0, 2.0])));
    assert_eq!(a0.status, CuPolicyStatus::Fresh);
    assert_eq!(a0.values, [0.75, 1.5]);

    // Cycle 1 (decimated Held from chunk_policy): ensembler steps to step 1!
    let a1 = run_cycle(Some(state2([1.2, 2.4])));
    assert_eq!(a1.status, CuPolicyStatus::Fresh);
    assert_eq!(a1.values, [1.5, 3.0]);

    // Cycle 2 (Fresh inference): overlaps chunk0[2] = [2.25, 4.5] with
    // chunk2[0] = [0.75, 1.5] -> uniform average [1.5, 3.0]!
    let a2 = run_cycle(Some(state2([1.5, 3.0])));
    assert_eq!(a2.status, CuPolicyStatus::Fresh);
    assert_eq!(a2.values, [1.5, 3.0]);

    // Robot goes NotReady mid-chunk -> observation emits None -> chunk_policy
    // emits Idle -> ensembler flushes buffered tail immediately to Idle!
    let mut unready = state2([1.5, 3.0]);
    unready.status = CuRobotStatus::NotReady;
    let stopped = run_cycle(Some(unready));
    assert_eq!(stopped.status, CuPolicyStatus::Idle);
    assert_eq!(stopped.values, [0.0, 0.0]);
}
