//! Real-time Physical AI and trajectory execution safety shield for `copper-rs`.
//!
//! [`CuSafetyShieldTask`] sits between a setpoint source
//! (`CuFollowTrajectoryTask`, `CuSetpointMuxTask`, or `repose-policy-rl` action
//! tasks) and `CuControllerTask`. Every cycle it checks the desired setpoint
//! against:
//!
//! - kinematic chain joint limits and coordinate validity (`HeldJointLimits`),
//! - robot self-collision, environment obstacles, and attached objects
//!   (`HeldCollision`),
//! - an optional Cartesian TCP keep-in box [`CuWorkspaceBounds`]
//!   (`HeldWorkspaceFence`),
//! - an optional Cartesian TCP linear speed limit via the geometric Jacobian
//!   (`HeldTcpSpeed`), and
//! - an optional velocity-extrapolated braking-horizon lookahead
//!   `clamp_to_limits(q + v * braking_horizon)` (`HeldBrakingLookahead`).
//!
//! When any check fails, the shield never passes the unsafe setpoint downstream:
//! it holds the last safe configuration (or clamped measured state) at zero
//! velocity and acceleration, and emits a structured [`CuSafetyReport`] with
//! edge transitions from [`CuCollisionMonitor`](repose_collision::CuCollisionMonitor).
//!
//! # Example
//! ```rust
//! use cu29::units::si::angle::radian;
//! use cu29::units::si::f64::Angle;
//! use cu29::units::si::length::meter;
//! use repose_kinematics::{CuJointAcceleration, CuJointPosition, CuJointVelocity, CuRobotPreset};
//! use repose_safety::{CuSafetyShieldParams, CuSafetyShieldTask, CuSafetyVerdict, CuSafetyReport};
//! use repose_trajectory::{CuFollowStatus, CuTrajectoryFollowOutput};
//!
//! let params = CuSafetyShieldParams::<6> {
//!     chain: Some(CuRobotPreset::IgusRebel.into()),
//!     collision_model: Some(CuRobotPreset::IgusRebel.into()),
//!     ..Default::default()
//! };
//! let mut shield = CuSafetyShieldTask::<6>::from_params(params).unwrap();
//! let positions = [CuJointPosition::Revolute(Angle::new::<radian>(0.0)); 6];
//! let setpoint = CuTrajectoryFollowOutput {
//!     positions,
//!     velocities: positions.map(|p| CuJointVelocity::from_coordinate(p, 0.0)),
//!     accelerations: positions.map(|p| CuJointAcceleration::from_coordinate(p, 0.0)),
//!     status: CuFollowStatus::Running,
//! };
//! let mut out_sp = CuTrajectoryFollowOutput::default();
//! let mut out_rep = CuSafetyReport::default();
//! shield.evaluate(Some(&setpoint), None, None, &mut out_sp, &mut out_rep).unwrap();
//! assert_eq!(out_rep.verdict, CuSafetyVerdict::Clear);
//! assert!(out_rep.min_clearance.get::<meter>() > 0.0);
//! ```

#![cfg_attr(not(feature = "std"), no_std)]

mod params;
mod report;
mod task;

pub use params::CuSafetyShieldParams;
pub use params::CuWorkspaceBounds;
pub use params::DEFAULT_NEAR_MISS_MARGIN_M;
pub use report::CuSafetyReport;
pub use report::CuSafetyVerdict;
pub use report::CuSafetyWitness;
pub use task::CuSafetyShieldTask;

#[cfg(test)]
mod tests {
    use super::*;
    use bincode::de::DecoderImpl;
    use bincode::de::read::SliceReader;
    use bincode::enc::EncoderImpl;
    use bincode::enc::write::SliceWriter;
    use cu_spatial_payloads::Transform3D;
    use cu29::prelude::*;
    use cu29::units::si::angle::radian;
    use cu29::units::si::f64::Angle;
    use cu29::units::si::f64::Length;
    use cu29::units::si::f64::Time;
    use cu29::units::si::f64::Velocity;
    use cu29::units::si::length::meter;
    use cu29::units::si::time::second;
    use cu29::units::si::velocity::meter_per_second;
    use repose_collision::CuCollisionShape;
    use repose_collision::CuMonitorTransition;
    use repose_collision::CuPlanningScene;
    use repose_collision::CuSceneExclusions;
    use repose_collision::CuSceneRevision;
    use repose_collision::CuSphere;
    use repose_kinematics::CuJointAcceleration;
    use repose_kinematics::CuJointPosition;
    use repose_kinematics::CuJointStatePayload;
    use repose_kinematics::CuJointVelocity;
    use repose_kinematics::CuRobotPreset;
    use repose_kinematics::CuRobotStatus;
    use repose_kinematics::igus_rebel_chain;
    use repose_trajectory::CuFollowStatus;
    use repose_trajectory::CuTrajectoryFollowOutput;

    fn rebel_params() -> CuSafetyShieldParams<6> {
        CuSafetyShieldParams {
            chain: Some(CuRobotPreset::IgusRebel.into()),
            collision_model: Some(CuRobotPreset::IgusRebel.into()),
            ..Default::default()
        }
    }

    fn rad_pos(q: [f64; 6]) -> [CuJointPosition; 6] {
        q.map(|v| CuJointPosition::Revolute(Angle::new::<radian>(v)))
    }

    fn sp_with_vel(
        q: [f64; 6],
        v: [f64; 6],
        status: CuFollowStatus,
    ) -> CuTrajectoryFollowOutput<6> {
        let positions = rad_pos(q);
        CuTrajectoryFollowOutput {
            positions,
            velocities: [0, 1, 2, 3, 4, 5]
                .map(|j| CuJointVelocity::from_coordinate(positions[j], v[j])),
            accelerations: positions.map(|p| CuJointAcceleration::from_coordinate(p, 0.0)),
            status,
        }
    }

    fn state_at(q: [f64; 6]) -> CuJointStatePayload<6> {
        let positions = rad_pos(q);
        CuJointStatePayload {
            status: CuRobotStatus::Ready,
            positions,
            velocities: positions.map(|p| CuJointVelocity::from_coordinate(p, 0.0)),
        }
    }

    fn frozen_bytes(task: &CuSafetyShieldTask<6>, buf: &mut [u8]) -> usize {
        let mut encoder = EncoderImpl::new(SliceWriter::new(buf), bincode::config::standard());
        task.freeze(&mut encoder).unwrap();
        encoder.into_writer().bytes_written()
    }

    #[test]
    fn test_clear_passthrough_on_igus_rebel_home() {
        let (ctx, _) = CuContext::new_mock_clock();
        let mut task = CuSafetyShieldTask::<6>::from_params(rebel_params()).unwrap();
        let sp = sp_with_vel([0.0; 6], [0.0; 6], CuFollowStatus::Running);
        let st = state_at([0.0; 6]);
        let scene = CuPlanningScene::new();

        let sp_msg = CuMsg::new(Some(sp.clone()));
        let mut st_msg = CuMsg::new(Some(st));
        st_msg.tov = Tov::Time(CuTime::from_secs(10));
        let sc_msg = CuMsg::new(Some(scene));
        let mut out = (CuMsg::new(None), CuMsg::new(None));

        task.process(&ctx, &(&sp_msg, &st_msg, &sc_msg), &mut out)
            .unwrap();

        let emitted_sp = out.0.payload().unwrap();
        let report = out.1.payload().unwrap();
        assert_eq!(emitted_sp, &sp);
        assert_eq!(report.verdict, CuSafetyVerdict::Clear);
        assert_eq!(report.transition, CuMonitorTransition::StillClear);
        assert!(!report.near_miss);
        assert!(report.min_clearance.get::<meter>() > 0.02);
        assert_eq!(report.tcp_speed.get::<meter_per_second>(), 0.0);
        assert_eq!(report.scene_revision, CuSceneRevision::INITIAL);
    }

    #[test]
    fn test_obstacle_collision_episodes_and_safe_hold() {
        let mut task = CuSafetyShieldTask::<6>::from_params(rebel_params()).unwrap();
        let mut out_sp = CuTrajectoryFollowOutput::default();
        let mut out_rep = CuSafetyReport::default();

        // Cycle 0: clear safe pose at q = [0.3, 0, 0, 0, 0, 0].
        let q_safe = [0.3, 0.0, 0.0, 0.0, 0.0, 0.0];
        let sp_safe = sp_with_vel(
            q_safe,
            [0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
            CuFollowStatus::Running,
        );
        task.evaluate(Some(&sp_safe), None, None, &mut out_sp, &mut out_rep)
            .unwrap();
        assert_eq!(out_rep.verdict, CuSafetyVerdict::Clear);
        assert_eq!(out_rep.transition, CuMonitorTransition::StillClear);
        assert_eq!(out_sp, sp_safe);

        // Cycle 1: obstacle placed on end-effector at q = 0 -> HeldCollision + EnteredCollision.
        let mut scene = CuPlanningScene::new();
        let m = Length::new::<meter>;
        let obs =
            CuCollisionShape::Sphere(CuSphere::new([m(0.0), m(0.0), m(0.87)], m(0.08)).unwrap());
        let obs_id = scene.add_obstacle(obs, None).unwrap();

        let sp_bad = sp_with_vel(
            [0.0; 6],
            [0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
            CuFollowStatus::Running,
        );
        task.evaluate(Some(&sp_bad), None, Some(&scene), &mut out_sp, &mut out_rep)
            .unwrap();
        assert_eq!(out_rep.verdict, CuSafetyVerdict::HeldCollision);
        assert_eq!(out_rep.transition, CuMonitorTransition::EnteredCollision);
        assert_eq!(out_rep.scene_revision, scene.revision());
        assert!(matches!(
            out_rep.witness,
            CuSafetyWitness::Environment { .. }
        ));
        assert_eq!(out_sp.positions, sp_safe.positions);
        assert_eq!(out_sp.status, CuFollowStatus::Finished);
        assert!(out_sp.velocities.iter().all(|v| v.si_value() == 0.0));
        assert!(out_sp.accelerations.iter().all(|a| a.si_value() == 0.0));

        // Cycle 2: still in collision -> StillInCollision, still holding last_safe.
        task.evaluate(Some(&sp_bad), None, Some(&scene), &mut out_sp, &mut out_rep)
            .unwrap();
        assert_eq!(out_rep.verdict, CuSafetyVerdict::HeldCollision);
        assert_eq!(out_rep.transition, CuMonitorTransition::StillInCollision);
        assert_eq!(out_sp.positions, sp_safe.positions);

        // Cycle 3: obstacle removed -> Clear + ClearedCollision, passes setpoint through.
        scene.remove_obstacle(obs_id).unwrap();
        task.evaluate(Some(&sp_bad), None, Some(&scene), &mut out_sp, &mut out_rep)
            .unwrap();
        assert_eq!(out_rep.verdict, CuSafetyVerdict::Clear);
        assert_eq!(out_rep.transition, CuMonitorTransition::ClearedCollision);
        assert_eq!(out_sp, sp_bad);
    }

    #[test]
    fn test_workspace_fence_violation() {
        let chain = igus_rebel_chain();
        let q_low = [0.0, 2.05, 0.0, 0.0, 0.0, 0.0];
        let mut ee = Transform3D::<f64>::identity();
        chain.forward_kinematics(&rad_pos(q_low), &mut ee).unwrap();
        let z_low = ee.to_matrix()[2][3];

        let m = Length::new::<meter>;
        let mut params = rebel_params();
        params.workspace = Some(
            CuWorkspaceBounds::new(
                [m(-0.8), m(-0.8), m(z_low + 0.02)],
                [m(0.8), m(0.8), m(0.95)],
            )
            .unwrap(),
        );
        let mut task = CuSafetyShieldTask::<6>::from_params(params).unwrap();
        let st = state_at([0.0; 6]);
        let sp = sp_with_vel(q_low, [0.0; 6], CuFollowStatus::Running);

        let mut out_sp = CuTrajectoryFollowOutput::default();
        let mut out_rep = CuSafetyReport::default();
        task.evaluate(Some(&sp), Some(&st), None, &mut out_sp, &mut out_rep)
            .unwrap();

        assert_eq!(out_rep.verdict, CuSafetyVerdict::HeldWorkspaceFence);
        assert_eq!(out_rep.transition, CuMonitorTransition::EnteredCollision);
        assert_eq!(
            out_rep.witness,
            CuSafetyWitness::WorkspaceAxis {
                axis: 2,
                upper: false
            }
        );
        assert!((out_rep.min_clearance.get::<meter>() - (-0.02)).abs() < 1e-12);
        // Held at seeded measured state (home).
        assert_eq!(out_sp.positions, st.positions);
        assert_eq!(out_sp.status, CuFollowStatus::Finished);
    }

    #[test]
    fn test_braking_horizon_lookahead_prevents_table_impact() {
        let chain = igus_rebel_chain();
        let q_above = [0.0, 1.9, 0.0, 0.0, 0.0, 0.0];
        let mut ee = Transform3D::<f64>::identity();
        chain
            .forward_kinematics(&rad_pos(q_above), &mut ee)
            .unwrap();
        let z_above = ee.to_matrix()[2][3];
        let z_table = z_above - 0.02; // exactly 20 mm above table floor

        let m = Length::new::<meter>;
        let mut params = rebel_params();
        params.workspace = Some(
            CuWorkspaceBounds::new([m(-0.8), m(-0.8), m(z_table)], [m(0.8), m(0.8), m(0.95)])
                .unwrap(),
        );
        params.braking_horizon = Some(Time::new::<second>(0.15));
        params.near_miss_margin = m(0.03);

        let mut task = CuSafetyShieldTask::<6>::from_params(params).unwrap();
        let st = state_at(q_above);
        let mut out_sp = CuTrajectoryFollowOutput::default();
        let mut out_rep = CuSafetyReport::default();

        // Stationary 20 mm above table: Clear, near_miss flagged.
        let sp_hover = sp_with_vel(q_above, [0.0; 6], CuFollowStatus::Running);
        task.evaluate(Some(&sp_hover), Some(&st), None, &mut out_sp, &mut out_rep)
            .unwrap();
        assert_eq!(out_rep.verdict, CuSafetyVerdict::Clear);
        assert!(out_rep.near_miss);
        assert_eq!(out_sp, sp_hover);

        // Moving downward at +1.0 rad/s on joint 2: q_brake hits table -> HeldBrakingLookahead!
        let sp_diving = sp_with_vel(
            q_above,
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            CuFollowStatus::Running,
        );
        task.evaluate(Some(&sp_diving), Some(&st), None, &mut out_sp, &mut out_rep)
            .unwrap();
        assert_eq!(out_rep.verdict, CuSafetyVerdict::HeldBrakingLookahead);
        assert_eq!(out_rep.transition, CuMonitorTransition::EnteredCollision);
        assert_eq!(
            out_rep.witness,
            CuSafetyWitness::WorkspaceAxis {
                axis: 2,
                upper: false
            }
        );
        assert!(out_rep.min_clearance.get::<meter>() < 0.0);
        assert_eq!(out_sp.positions, sp_hover.positions);
        assert!(out_sp.velocities.iter().all(|v| v.si_value() == 0.0));
    }

    #[test]
    fn test_near_miss_flags_without_halting() {
        let mut params = rebel_params();
        // igus ReBeL model has safety_margin = 0.005 m. Set near_miss_margin = 0.03 m
        // -> threshold is 0.035 m.
        params.near_miss_margin = Length::new::<meter>(0.03);
        let mut task = CuSafetyShieldTask::<6>::from_params(params).unwrap();

        let m = Length::new::<meter>;
        // Flange sphere is at z ≈ 0.87 + 0.02 = 0.89 with radius 0.032 m.
        // Place sphere at z = 0.89 + 0.032 + 0.05 + 0.015 = 0.987 m -> surface dist ≈ 0.015 m.
        let mut scene = CuPlanningScene::new();
        scene
            .add_obstacle(
                CuCollisionShape::Sphere(
                    CuSphere::new([m(0.0), m(0.0), m(0.987)], m(0.05)).unwrap(),
                ),
                None,
            )
            .unwrap();

        let sp = sp_with_vel([0.0; 6], [0.0; 6], CuFollowStatus::Running);
        let mut out_sp = CuTrajectoryFollowOutput::default();
        let mut out_rep = CuSafetyReport::default();
        task.evaluate(Some(&sp), None, Some(&scene), &mut out_sp, &mut out_rep)
            .unwrap();

        assert_eq!(out_rep.verdict, CuSafetyVerdict::Clear);
        assert!(out_rep.near_miss);
        assert!(out_rep.min_clearance.get::<meter>() >= 0.005);
        assert!(out_rep.min_clearance.get::<meter>() < 0.035);
        assert!(matches!(
            out_rep.witness,
            CuSafetyWitness::Environment {
                robot_shape: 5,
                obstacle_index: 0
            }
        ));
        assert_eq!(out_sp, sp);
    }

    #[test]
    fn test_max_tcp_speed_violation() {
        let mut params = rebel_params();
        params.max_tcp_speed = Some(Velocity::new::<meter_per_second>(0.2));
        let mut task = CuSafetyShieldTask::<6>::from_params(params).unwrap();

        let q = [0.0, 0.5, 0.0, 0.0, 0.0, 0.0];
        let st = state_at(q);
        // Joint 2 spinning at 1.5 rad/s gives ~0.9 m/s TCP speed (> 0.2 m/s).
        let sp = sp_with_vel(q, [0.0, 1.5, 0.0, 0.0, 0.0, 0.0], CuFollowStatus::Running);
        let mut out_sp = CuTrajectoryFollowOutput::default();
        let mut out_rep = CuSafetyReport::default();

        task.evaluate(Some(&sp), Some(&st), None, &mut out_sp, &mut out_rep)
            .unwrap();
        assert_eq!(out_rep.verdict, CuSafetyVerdict::HeldTcpSpeed);
        assert!(out_rep.tcp_speed.get::<meter_per_second>() > 0.5);
        assert_eq!(out_sp.positions, st.positions);
        assert_eq!(out_sp.status, CuFollowStatus::Finished);
    }

    #[test]
    fn test_freezable_fixed_size_and_round_trip() {
        let mut buf = [0u8; 4096];
        let mut task = CuSafetyShieldTask::<6>::from_params(rebel_params()).unwrap();

        // 1. Idle state (no last_safe, not in collision).
        let len_idle = frozen_bytes(&task, &mut buf);

        // 2. Clear state (last_safe populated, not in collision).
        let mut out_sp = CuTrajectoryFollowOutput::default();
        let mut out_rep = CuSafetyReport::default();
        let q_safe = [0.25, -0.2, 0.3, 0.0, 0.1, 0.0];
        let sp_safe = sp_with_vel(q_safe, [0.0; 6], CuFollowStatus::Running);
        task.evaluate(Some(&sp_safe), None, None, &mut out_sp, &mut out_rep)
            .unwrap();
        let len_clear = frozen_bytes(&task, &mut buf);
        assert_eq!(len_clear, len_idle);

        // 3. Held state (last_safe populated, in collision).
        let sp_oob = sp_with_vel(
            [10.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0; 6],
            CuFollowStatus::Running,
        );
        task.evaluate(Some(&sp_oob), None, None, &mut out_sp, &mut out_rep)
            .unwrap();
        assert_eq!(out_rep.verdict, CuSafetyVerdict::HeldJointLimits);
        assert_eq!(out_rep.witness, CuSafetyWitness::Joint { index: 0 });
        let len_held = frozen_bytes(&task, &mut buf);
        assert_eq!(len_held, len_idle);

        // Round-trip restore into a fresh task.
        let mut restored = CuSafetyShieldTask::<6>::from_params(rebel_params()).unwrap();
        let mut decoder = DecoderImpl::new(
            SliceReader::new(&buf[..len_held]),
            bincode::config::standard(),
            (),
        );
        restored.thaw(&mut decoder).unwrap();
        assert!(restored.monitor().in_collision());
        assert_eq!(restored.last_safe(), Some(&rad_pos(q_safe)));

        // Next clear cycle on restored task must emit ClearedCollision!
        restored
            .evaluate(Some(&sp_safe), None, None, &mut out_sp, &mut out_rep)
            .unwrap();
        assert_eq!(out_rep.verdict, CuSafetyVerdict::Clear);
        assert_eq!(out_rep.transition, CuMonitorTransition::ClearedCollision);
    }

    #[test]
    fn test_missing_and_invalid_scene_and_no_anchor_fallback() {
        let mut params = rebel_params();
        params.require_scene = true;
        let mut task = CuSafetyShieldTask::<6>::from_params(params).unwrap();

        let sp = sp_with_vel([0.0; 6], [0.0; 6], CuFollowStatus::Running);
        let mut out_sp = CuTrajectoryFollowOutput::default();
        let mut out_rep = CuSafetyReport::default();

        // Missing scene with no state or prior last_safe -> ErrorEmptyTrajectory hold.
        task.evaluate(Some(&sp), None, None, &mut out_sp, &mut out_rep)
            .unwrap();
        assert_eq!(out_rep.verdict, CuSafetyVerdict::HeldMissingScene);
        assert_eq!(out_sp.status, CuFollowStatus::ErrorEmptyTrajectory);

        // Invalid scene (attached shape referencing out-of-range frame 99).
        let mut bad_scene = CuPlanningScene::new();
        let m = Length::new::<meter>;
        bad_scene
            .add_attached(
                Some(99),
                CuCollisionShape::Sphere(CuSphere::new([m(0.0); 3], m(0.05)).unwrap()),
                None,
            )
            .unwrap();
        task.evaluate(Some(&sp), None, Some(&bad_scene), &mut out_sp, &mut out_rep)
            .unwrap();
        assert_eq!(out_rep.verdict, CuSafetyVerdict::HeldInvalidScene);

        let _ = CuSceneExclusions::<4>::new();
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_ron_config_and_validation() {
        let cfg: ComponentConfig = ron::from_str(
            r#"#![enable(unwrap_newtypes)] {
                "chain": "igus_rebel",
                "collision_model": "igus_rebel",
                "workspace": {"min": [-0.6, -0.6, 0.02], "max": [0.6, 0.6, 0.9]},
                "braking_horizon": 0.1,
                "near_miss_margin": 0.025,
                "max_tcp_speed": 1.2,
                "require_scene": true,
            }"#,
        )
        .unwrap();
        let task = CuSafetyShieldTask::<6>::new(Some(&cfg), ()).unwrap();
        assert!(task.params().require_scene);
        assert_eq!(task.params().near_miss_margin.get::<meter>(), 0.025);

        let bad_ws: ComponentConfig = ron::from_str(
            r#"#![enable(unwrap_newtypes)] {
                "chain": "igus_rebel",
                "collision_model": "igus_rebel",
                "workspace": {"min": [0.5, -0.6, 0.02], "max": [0.2, 0.6, 0.9]},
            }"#,
        )
        .unwrap();
        assert!(CuSafetyShieldTask::<6>::new(Some(&bad_ws), ()).is_err());

        let unknown_field: ComponentConfig = ron::from_str(
            r#"#![enable(unwrap_newtypes)] {
                "chain": "igus_rebel",
                "collision_model": "igus_rebel",
                "braking_horizon_s": 0.1,
            }"#,
        )
        .unwrap();
        assert!(CuSafetyShieldTask::<6>::new(Some(&unknown_field), ()).is_err());
    }
}
