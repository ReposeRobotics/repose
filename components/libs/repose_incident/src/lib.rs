//! Black-box incident detector, timeline summarizer, and counterfactual
//! checkpoint scorecard engine for `copper-rs`.
//!
//! Turns any sequence of recorded robot cycles ([`CuCycleSample`]) — from a
//! `.copper` log reader, simulation run, or live ring buffer — into windowed
//! [`CuIncidentRecord`] manifests with deterministic [`CuIncidentBlame`]
//! attribution, run-wide [`CuIncidentSummary`] aggregates, and
//! [`CuCheckpointDiff`] counterfactual regression gates.
//!
//! This crate is `std`-only: JSON manifest export and incident summarization
//! run on the host (`logreader`, `resim`, Flightdeck, and CI gates). There is no
//! `no_std` surface and no `#![no_std]` attribute; the `std` feature exists
//! and forwards to dependencies so workspace-wide `--features std` and
//! `--no-default-features` invocations keep working.
//!
//! # Example
//! ```rust
//! use cu29::units::si::f64::Length;
//! use cu29::units::si::length::meter;
//! use repose_controller::CuControllerStatus;
//! use repose_incident::{
//!     CuCheckpointDiff, CuCycleSample, CuIncidentBlame, CuIncidentDetector,
//!     CuIncidentDetectorParams,
//! };
//! use repose_kinematics::CuRobotStatus;
//!
//! let mut detector = CuIncidentDetector::<6>::from_params(CuIncidentDetectorParams {
//!     model_hash: Some(0xdead_beef),
//!     ..CuIncidentDetectorParams::default()
//! })?;
//!
//! let mut baseline_sample = CuCycleSample::new(0, 10_000_000);
//! baseline_sample.robot_status = Some(CuRobotStatus::Ready);
//! baseline_sample.controller_status = Some(CuControllerStatus::Running);
//! baseline_sample.in_collision = true;
//! baseline_sample.min_clearance = Some(Length::new::<meter>(-0.012));
//! detector.observe(&baseline_sample)?;
//!
//! let baseline = detector.finish();
//! assert_eq!(baseline.collision_episodes, 1);
//! assert_eq!(baseline.incidents[0].blame, CuIncidentBlame::TrajectoryOrCollision);
//!
//! let mut candidate_detector = CuIncidentDetector::<6>::from_params(CuIncidentDetectorParams {
//!     model_hash: Some(0xcafe_babe),
//!     ..CuIncidentDetectorParams::default()
//! })?;
//! let mut candidate_sample = CuCycleSample::new(0, 10_000_000);
//! candidate_sample.robot_status = Some(CuRobotStatus::Ready);
//! candidate_sample.controller_status = Some(CuControllerStatus::Running);
//! candidate_sample.min_clearance = Some(Length::new::<meter>(0.034));
//! candidate_detector.observe(&candidate_sample)?;
//!
//! let candidate = candidate_detector.finish();
//! let diff = CuCheckpointDiff::compare(&baseline, &candidate);
//! assert!(diff.passed_gate());
//! # Ok::<(), cu29::CuError>(())
//! ```

mod detector;
mod scorecard;
mod serde_arrays;

pub use detector::CuCycleSample;
pub use detector::CuIncidentBlame;
pub use detector::CuIncidentDetector;
pub use detector::CuIncidentDetectorParams;
pub use detector::CuIncidentRecord;
pub use detector::CuIncidentSummary;
pub use detector::CuIncidentTrigger;
pub use detector::CuIncidentWindow;
pub use scorecard::CuCheckpointDiff;

#[cfg(test)]
mod tests {
    use crate::CuCheckpointDiff;
    use crate::CuCycleSample;
    use crate::CuIncidentBlame;
    use crate::CuIncidentDetector;
    use crate::CuIncidentDetectorParams;
    use crate::CuIncidentSummary;
    use crate::CuIncidentTrigger;
    use cu29::units::si::angle::radian;
    use cu29::units::si::f64::Angle;
    use cu29::units::si::f64::Length;
    use cu29::units::si::f64::Time;
    use cu29::units::si::length::meter;
    use cu29::units::si::time::second;
    use repose_collision::CuSceneRevision;
    use repose_controller::CuControllerStatus;
    use repose_kinematics::CuJointPosition;
    use repose_kinematics::CuJointVelocity;
    use repose_kinematics::CuRobotStatus;
    use repose_policy_rl::CuPolicyStatus;
    use repose_trajectory::CuFollowStatus;

    fn q(vals: [f64; 6]) -> [CuJointPosition; 6] {
        vals.map(|v| CuJointPosition::Revolute(Angle::new::<radian>(v)))
    }

    fn nominal_sample(cycle: u64, time_ns: u64) -> CuCycleSample<6> {
        CuCycleSample {
            cycle,
            time_ns,
            robot_status: Some(CuRobotStatus::Ready),
            policy_status: Some(CuPolicyStatus::Fresh),
            follow_status: Some(CuFollowStatus::Running),
            controller_status: Some(CuControllerStatus::Running),
            min_clearance: Some(Length::new::<meter>(0.15)),
            in_collision: false,
            action_clamped: false,
            scene_revision: Some(CuSceneRevision::new(1)),
            positions: Some(q([0.0; 6])),
            velocities: Some([CuJointVelocity::default(); 6]),
        }
    }

    #[test]
    fn test_clean_run_produces_zero_incidents_and_accurate_aggregates() {
        let mut detector =
            CuIncidentDetector::<6>::from_params(CuIncidentDetectorParams::default()).unwrap();

        let clearances = [0.18, 0.12, 0.085, 0.14];
        let step_values = [0.0, 0.05, 0.175, 0.20];
        for (i, (&clr, &pos)) in clearances.iter().zip(&step_values).enumerate() {
            let mut sample = nominal_sample(i as u64, (i as u64) * 20_000_000);
            sample.min_clearance = Some(Length::new::<meter>(clr));
            sample.positions = Some(q([pos, 0.0, 0.0, 0.0, 0.0, 0.0]));
            detector.observe(&sample).unwrap();
        }

        let summary = detector.finish();
        assert_eq!(summary.total_cycles, 4);
        assert!((summary.duration_s - 0.06).abs() < 1e-12);
        assert_eq!(summary.collision_episodes, 0);
        assert_eq!(summary.near_miss_episodes, 0);
        assert_eq!(summary.policy_faults, 0);
        assert_eq!(summary.controller_faults, 0);
        assert_eq!(summary.clamped_cycles, 0);
        assert_eq!(summary.min_clearance_m, Some(0.085));
        assert!((summary.peak_position_step_si - 0.125).abs() < 1e-12);
        assert!(summary.incidents.is_empty());

        let json = summary.to_json_pretty().unwrap();
        let parsed: CuIncidentSummary = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.total_cycles, summary.total_cycles);
        assert_eq!(parsed.min_clearance_m, summary.min_clearance_m);
        assert!((parsed.duration_s - summary.duration_s).abs() < 1e-15);
        assert!((parsed.peak_position_step_si - summary.peak_position_step_si).abs() < 1e-15);
        assert_eq!(parsed.incidents, summary.incidents);
    }

    #[test]
    fn test_policy_fault_then_controller_hold_coalesces_with_learned_policy_blame() {
        let pre_s = 3.0;
        let post_s = 2.0;
        let mut detector = CuIncidentDetector::<6>::from_params(CuIncidentDetectorParams {
            pre_window: Time::new::<second>(pre_s),
            post_window: Time::new::<second>(post_s),
            cooldown: Time::new::<second>(1.0),
            model_hash: Some(0xabc1_2345),
            ..CuIncidentDetectorParams::default()
        })
        .unwrap();

        // Clean lead-in up to 4.0 s.
        for c in 0..5 {
            detector
                .observe(&nominal_sample(c, c * 1_000_000_000))
                .unwrap();
        }

        // Cycle 5 at T = 5.0 s: policy faults.
        let trigger_t = 5_000_000_000_u64;
        let mut s5 = nominal_sample(5, trigger_t);
        s5.policy_status = Some(CuPolicyStatus::Fault);
        detector.observe(&s5).unwrap();

        // Cycle 6 at T + 20 ms: controller enters fault/hold in reaction.
        let mut s6 = nominal_sample(6, trigger_t + 20_000_000);
        s6.policy_status = Some(CuPolicyStatus::Fault);
        s6.controller_status = Some(CuControllerStatus::Fault);
        s6.follow_status = Some(CuFollowStatus::ErrorEmptyTrajectory);
        detector.observe(&s6).unwrap();

        let summary = detector.finish();
        assert_eq!(summary.policy_faults, 1);
        assert_eq!(summary.controller_faults, 1);
        assert_eq!(summary.incidents.len(), 1);

        let rec = &summary.incidents[0];
        assert_eq!(rec.id, 1);
        assert_eq!(rec.primary_trigger, CuIncidentTrigger::PolicyFault);
        assert_eq!(rec.blame, CuIncidentBlame::LearnedPolicy);
        assert_eq!(rec.trigger_cycle, 5);
        assert_eq!(rec.trigger_time_ns, trigger_t);
        assert_eq!(rec.window_start_ns, trigger_t - 3_000_000_000);
        assert_eq!(rec.window_end_ns, trigger_t + 2_000_000_000);
        assert_eq!(rec.window().start_ns, rec.window_start_ns);
        assert_eq!(rec.window().end_ns, rec.window_end_ns);
        assert_eq!(rec.model_hash, Some(0xabc1_2345));
        assert_eq!(rec.event_count, 2);
    }

    #[test]
    fn test_scene_revision_increment_before_near_miss_and_collision_attributes_environment() {
        let mut detector =
            CuIncidentDetector::<6>::from_params(CuIncidentDetectorParams::default()).unwrap();

        let mut s0 = nominal_sample(0, 100_000_000);
        s0.scene_revision = Some(CuSceneRevision::new(1));
        detector.observe(&s0).unwrap();

        // Obstacle appears in scene -> revision increments to 2.
        let mut s1 = nominal_sample(1, 120_000_000);
        s1.scene_revision = Some(CuSceneRevision::new(2));
        s1.min_clearance = Some(Length::new::<meter>(0.015));
        detector.observe(&s1).unwrap();

        // Collision occurs 40 ms later on the updated scene.
        let mut s2 = nominal_sample(2, 160_000_000);
        s2.scene_revision = Some(CuSceneRevision::new(2));
        s2.min_clearance = Some(Length::new::<meter>(-0.008));
        s2.in_collision = true;
        detector.observe(&s2).unwrap();

        let summary = detector.finish();
        assert_eq!(summary.near_miss_episodes, 1);
        assert_eq!(summary.collision_episodes, 1);
        assert_eq!(summary.incidents.len(), 1);
        let rec = &summary.incidents[0];
        assert_eq!(rec.primary_trigger, CuIncidentTrigger::CollisionEntered);
        assert_eq!(rec.blame, CuIncidentBlame::EnvironmentChange);
        assert_eq!(rec.scene_revision, Some(2));
        assert_eq!(rec.min_clearance_m, Some(-0.008));
        assert_eq!(rec.event_count, 2);
    }

    #[test]
    fn test_hardware_robot_fault_attributes_hardware_or_bridge_even_with_controller_fault() {
        let mut detector =
            CuIncidentDetector::<6>::from_params(CuIncidentDetectorParams::default()).unwrap();

        detector.observe(&nominal_sample(0, 0)).unwrap();

        let mut fault_sample = nominal_sample(1, 20_000_000);
        fault_sample.robot_status = Some(CuRobotStatus::Fault(42));
        fault_sample.controller_status = Some(CuControllerStatus::Fault);
        detector.observe(&fault_sample).unwrap();

        let summary = detector.finish();
        assert_eq!(summary.incidents.len(), 1);
        assert_eq!(
            summary.incidents[0].primary_trigger,
            CuIncidentTrigger::RobotFault
        );
        assert_eq!(
            summary.incidents[0].blame,
            CuIncidentBlame::HardwareOrBridge
        );
    }

    #[test]
    fn test_controller_fault_while_robot_ready_attributes_controller_or_tracking() {
        let mut detector =
            CuIncidentDetector::<6>::from_params(CuIncidentDetectorParams::default()).unwrap();

        detector.observe(&nominal_sample(0, 0)).unwrap();

        let mut s1 = nominal_sample(1, 20_000_000);
        s1.controller_status = Some(CuControllerStatus::Fault);
        detector.observe(&s1).unwrap();

        let summary = detector.finish();
        assert_eq!(summary.incidents.len(), 1);
        assert_eq!(
            summary.incidents[0].primary_trigger,
            CuIncidentTrigger::ControllerFault
        );
        assert_eq!(
            summary.incidents[0].blame,
            CuIncidentBlame::ControllerOrTracking
        );
    }

    #[test]
    fn test_clamp_burst_and_setpoint_discontinuity_trigger_and_cooldown_splits_episodes() {
        let mut detector = CuIncidentDetector::<6>::from_params(CuIncidentDetectorParams {
            clamp_burst_cycles: 3,
            max_position_step: Some(0.25),
            cooldown: Time::new::<second>(0.5),
            ..CuIncidentDetectorParams::default()
        })
        .unwrap();

        // Episode 1: 3 consecutive clamped cycles at t = 0, 20ms, 40ms.
        for c in 0..3 {
            let mut s = nominal_sample(c, c * 20_000_000);
            s.action_clamped = true;
            detector.observe(&s).unwrap();
        }
        // 4th clamped cycle does not re-trigger the same burst.
        let mut s3 = nominal_sample(3, 60_000_000);
        s3.action_clamped = true;
        detector.observe(&s3).unwrap();

        // Episode 2: after cooldown (> 0.5 s later), a setpoint jump on joint 2.
        let mut s4 = nominal_sample(4, 1_000_000_000);
        s4.positions = Some(q([0.0, 0.0, 0.4, 0.0, 0.0, 0.0]));
        detector.observe(&s4).unwrap();

        let summary = detector.finish();
        assert_eq!(summary.clamped_cycles, 4);
        assert_eq!(summary.incidents.len(), 2);
        assert_eq!(
            summary.incidents[0].primary_trigger,
            CuIncidentTrigger::ActionClampedBurst
        );
        assert_eq!(summary.incidents[0].blame, CuIncidentBlame::LearnedPolicy);
        assert_eq!(
            summary.incidents[1].primary_trigger,
            CuIncidentTrigger::SetpointDiscontinuity {
                joint: 2,
                step_si: 0.4,
            }
        );
        assert_eq!(summary.incidents[1].blame, CuIncidentBlame::LearnedPolicy);
    }

    #[test]
    fn test_checkpoint_diff_catches_regression_and_approves_fixed_candidate() {
        let baseline = CuIncidentSummary {
            total_cycles: 250,
            duration_s: 5.0,
            model_hash: Some(0x1111_2222),
            collision_episodes: 1,
            near_miss_episodes: 1,
            policy_faults: 0,
            controller_faults: 0,
            clamped_cycles: 12,
            min_clearance_m: Some(-0.012),
            peak_position_step_si: 0.04,
            incidents: Vec::new(),
        };

        let fixed_candidate = CuIncidentSummary {
            total_cycles: 250,
            duration_s: 5.0,
            model_hash: Some(0x3333_4444),
            collision_episodes: 0,
            near_miss_episodes: 0,
            policy_faults: 0,
            controller_faults: 0,
            clamped_cycles: 2,
            min_clearance_m: Some(0.034),
            peak_position_step_si: 0.03,
            incidents: Vec::new(),
        };

        let pass_diff = CuCheckpointDiff::compare(&baseline, &fixed_candidate);
        assert_eq!(pass_diff.baseline_model_hash, Some(0x1111_2222));
        assert_eq!(pass_diff.candidate_model_hash, Some(0x3333_4444));
        assert_eq!(pass_diff.collision_delta, -1);
        assert_eq!(pass_diff.near_miss_delta, -1);
        assert_eq!(pass_diff.policy_fault_delta, 0);
        assert_eq!(pass_diff.clamped_cycles_delta, -10);
        assert!((pass_diff.min_clearance_delta_m.unwrap() - 0.046).abs() < 1e-12);
        assert!(!pass_diff.regression);
        assert!(pass_diff.passed_gate());

        let json = pass_diff.to_json_pretty().unwrap();
        let parsed: CuCheckpointDiff = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, pass_diff);

        // Regression 1: more collisions.
        let more_collisions = CuIncidentSummary {
            collision_episodes: 2,
            min_clearance_m: Some(-0.025),
            ..baseline.clone()
        };
        let fail_collision = CuCheckpointDiff::compare(&baseline, &more_collisions);
        assert!(fail_collision.regression);
        assert!(!fail_collision.passed_gate());

        // Regression 2: worse clearance below threshold even with 0 collisions.
        let clean_baseline = CuIncidentSummary {
            collision_episodes: 0,
            near_miss_episodes: 0,
            min_clearance_m: Some(0.05),
            ..baseline.clone()
        };
        let tight_candidate = CuIncidentSummary {
            collision_episodes: 0,
            near_miss_episodes: 1,
            min_clearance_m: Some(0.011),
            ..fixed_candidate.clone()
        };
        let fail_clearance = CuCheckpointDiff::compare(&clean_baseline, &tight_candidate);
        assert!(fail_clearance.regression);
        assert!(!fail_clearance.passed_gate());

        // Regression 3: policy fault introduced.
        let faulty_candidate = CuIncidentSummary {
            policy_faults: 1,
            ..fixed_candidate
        };
        let fail_fault = CuCheckpointDiff::compare(&clean_baseline, &faulty_candidate);
        assert!(fail_fault.regression);
        assert!(!fail_fault.passed_gate());
    }

    #[test]
    fn test_validation_and_monotonic_time_reject_invalid_inputs() {
        let bad_params = CuIncidentDetectorParams {
            pre_window: Time::new::<second>(-1.0),
            ..CuIncidentDetectorParams::default()
        };
        assert!(CuIncidentDetector::<6>::from_params(bad_params).is_err());

        let bad_step = CuIncidentDetectorParams {
            max_position_step: Some(0.0),
            ..CuIncidentDetectorParams::default()
        };
        assert!(CuIncidentDetector::<6>::from_params(bad_step).is_err());

        let bad_threshold = CuIncidentDetectorParams {
            near_miss_threshold: Some(Length::new::<meter>(-0.01)),
            ..CuIncidentDetectorParams::default()
        };
        assert!(CuIncidentDetector::<6>::from_params(bad_threshold).is_err());

        let mut detector = CuIncidentDetector::<6>::default();
        detector.observe(&nominal_sample(0, 1_000_000)).unwrap();
        assert!(detector.observe(&nominal_sample(1, 500_000)).is_err());

        // Non-finite clearance (+inf when no collision pairs checked) is ignored in JSON.
        let mut inf_sample = nominal_sample(1, 2_000_000);
        inf_sample.min_clearance = Some(Length::new::<meter>(f64::INFINITY));
        detector.observe(&inf_sample).unwrap();
        let summary = detector.finish();
        assert_eq!(summary.min_clearance_m, Some(0.15));
        assert!(summary.to_json_pretty().is_ok());

        // CuCycleSample round-trips through JSON with Some and None positions.
        let finite_sample = nominal_sample(1, 2_000_000);
        let encoded = serde_json::to_string(&finite_sample).unwrap();
        let decoded: CuCycleSample<6> = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, finite_sample);

        let empty_sample = CuCycleSample::<6>::new(7, 42);
        let encoded_empty = serde_json::to_string(&empty_sample).unwrap();
        let decoded_empty: CuCycleSample<6> = serde_json::from_str(&encoded_empty).unwrap();
        assert_eq!(decoded_empty, empty_sample);
    }

    #[test]
    fn ongoing_robot_fault_attributes_delayed_controller_fault_to_hardware() {
        let params = CuIncidentDetectorParams {
            cooldown: Time::new::<second>(0.05),
            ..CuIncidentDetectorParams::default()
        };
        let mut detector = CuIncidentDetector::<6>::from_params(params).unwrap();

        // Cycle 0: robot enters fault alone -> opens episode 1.
        let mut s0 = nominal_sample(0, 0);
        s0.robot_status = Some(CuRobotStatus::NotReady);
        detector.observe(&s0).unwrap();

        // Cycle 1 after cooldown (200 ms): robot is still non-Ready and controller faults.
        let mut s1 = nominal_sample(1, 200_000_000);
        s1.robot_status = Some(CuRobotStatus::NotReady);
        s1.controller_status = Some(CuControllerStatus::Fault);
        detector.observe(&s1).unwrap();

        let summary = detector.finish();
        assert_eq!(summary.incidents.len(), 2);
        assert_eq!(
            summary.incidents[1].primary_trigger,
            CuIncidentTrigger::ControllerFault
        );
        assert_eq!(
            summary.incidents[1].blame,
            CuIncidentBlame::HardwareOrBridge
        );
    }
}
