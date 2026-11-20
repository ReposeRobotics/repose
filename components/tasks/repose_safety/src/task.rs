//! Real-time [`CuSafetyShieldTask`] implementation.

use crate::params::CuSafetyShieldParams;
use crate::report::CuSafetyReport;
use crate::report::CuSafetyVerdict;
use crate::report::CuSafetyWitness;
use bincode::Decode;
use bincode::Encode;
use bincode::de::Decoder;
use bincode::enc::Encoder;
use bincode::error::DecodeError;
use bincode::error::EncodeError;
use cu_spatial_payloads::Transform3D;
use cu29::prelude::*;
use cu29::units::si::f64::Length;
use cu29::units::si::f64::Velocity;
use cu29::units::si::length::meter;
use cu29::units::si::time::second;
use cu29::units::si::velocity::meter_per_second;
use repose_collision::CuCollisionEnv;
use repose_collision::CuCollisionMonitor;
use repose_collision::CuCollisionReport;
use repose_collision::CuPlanningScene;
use repose_collision::CuRobotCollisionModel;
use repose_collision::CuSceneRevision;
use repose_collision::DEFAULT_MAX_ATTACHED;
use repose_collision::DEFAULT_MAX_CONTACTS;
use repose_collision::DEFAULT_MAX_OBSTACLES;
use repose_kinematics::CuJointAcceleration;
use repose_kinematics::CuJointLimits;
use repose_kinematics::CuJointPosition;
use repose_kinematics::CuJointStatePayload;
use repose_kinematics::CuJointType;
use repose_kinematics::CuJointVelocity;
use repose_kinematics::CuKinematicChain;
use repose_trajectory::CuFollowStatus;
use repose_trajectory::CuTrajectoryFollowOutput;

/// Real-time safety shield `CuTask` sitting between a setpoint source and
/// `CuControllerTask`.
///
/// Evaluates joint limits, self/environment/attached collision margins, optional
/// Cartesian workspace bounds, optional TCP linear speed limit, and optional
/// velocity-extrapolated braking-horizon lookahead. Any violation holds the last
/// safe configuration (or clamped measured state) with zero velocity and
/// acceleration, never passing an unsafe setpoint downstream.
#[derive(Reflect)]
#[reflect(no_field_bounds, from_reflect = false)]
pub struct CuSafetyShieldTask<
    const N: usize,
    const M: usize = N,
    const MAX_OBS: usize = DEFAULT_MAX_OBSTACLES,
    const MAX_ATTACHED: usize = DEFAULT_MAX_ATTACHED,
    const MAX_CONTACTS: usize = DEFAULT_MAX_CONTACTS,
> {
    #[reflect(ignore)]
    params: CuSafetyShieldParams<N, M>,
    #[reflect(ignore)]
    chain: CuKinematicChain<N>,
    #[reflect(ignore)]
    model: CuRobotCollisionModel<N, M>,
    monitor: CuCollisionMonitor,
    #[reflect(ignore)]
    last_safe: Option<[CuJointPosition; N]>,
    #[reflect(ignore)]
    scratch_report: CuCollisionReport<MAX_CONTACTS>,
}

impl<
    const N: usize,
    const M: usize,
    const MAX_OBS: usize,
    const MAX_ATTACHED: usize,
    const MAX_CONTACTS: usize,
> CuSafetyShieldTask<N, M, MAX_OBS, MAX_ATTACHED, MAX_CONTACTS>
{
    /// Builds and validates a safety shield from [`CuSafetyShieldParams`].
    pub fn from_params(params: CuSafetyShieldParams<N, M>) -> CuResult<Self> {
        params.validate()?;
        let chain = params.resolve_chain()?;
        let model = params.resolve_collision_model()?;
        Ok(Self {
            params,
            chain,
            model,
            monitor: CuCollisionMonitor::new(),
            last_safe: None,
            scratch_report: CuCollisionReport::default(),
        })
    }

    /// Borrows the shield parameters.
    pub const fn params(&self) -> &CuSafetyShieldParams<N, M> {
        &self.params
    }

    /// Borrows the resolved kinematic chain.
    pub const fn chain(&self) -> &CuKinematicChain<N> {
        &self.chain
    }

    /// Borrows the resolved robot collision model.
    pub const fn model(&self) -> &CuRobotCollisionModel<N, M> {
        &self.model
    }

    /// Borrows the internal edge-detecting collision monitor.
    pub const fn monitor(&self) -> &CuCollisionMonitor {
        &self.monitor
    }

    /// Returns the currently latched safe joint positions, if any.
    pub const fn last_safe(&self) -> Option<&[CuJointPosition; N]> {
        self.last_safe.as_ref()
    }

    /// Zero positions matching each joint's coordinate variant in the chain.
    fn zero_positions(&self) -> [CuJointPosition; N] {
        core::array::from_fn(|j| self.chain.joints()[j].limits.lower().with_si_value(0.0))
    }

    /// Returns measured joint positions clamped to chain limits if every coordinate
    /// is finite and matches its joint type.
    fn valid_clamped_measured(
        &self,
        state: Option<&CuJointStatePayload<N>>,
    ) -> Option<[CuJointPosition; N]> {
        let state = state?;
        let mut clamped = state.positions;
        self.chain.clamp_to_limits(&mut clamped).ok()?;
        Some(clamped)
    }

    /// Writes a stationary safe hold (or `ErrorEmptyTrajectory` zeros when no
    /// safe or measured position exists) into `out_setpoint`.
    fn write_safe_hold(&self, out_setpoint: &mut CuTrajectoryFollowOutput<N>) {
        if let Some(pos) = self.last_safe {
            *out_setpoint = CuTrajectoryFollowOutput {
                positions: pos,
                velocities: pos.map(|q| CuJointVelocity::from_coordinate(q, 0.0)),
                accelerations: pos.map(|q| CuJointAcceleration::from_coordinate(q, 0.0)),
                status: CuFollowStatus::Finished,
            };
        } else {
            let zeros = self.zero_positions();
            *out_setpoint = CuTrajectoryFollowOutput {
                positions: zeros,
                velocities: zeros.map(|q| CuJointVelocity::from_coordinate(q, 0.0)),
                accelerations: zeros.map(|q| CuJointAcceleration::from_coordinate(q, 0.0)),
                status: CuFollowStatus::ErrorEmptyTrajectory,
            };
        }
    }

    /// Validates an active setpoint's joint positions, velocities, and accelerations,
    /// returning the first offending joint index if invalid.
    fn check_joint_limits(&self, sp: &CuTrajectoryFollowOutput<N>) -> Option<u16> {
        for j in 0..N {
            let joint = &self.chain.joints()[j];
            if !joint.limits.contains(sp.positions[j]) {
                return Some(j as u16);
            }
            let v = sp.velocities[j];
            let a = sp.accelerations[j];
            if !v.si_value().is_finite() || !a.si_value().is_finite() {
                return Some(j as u16);
            }
            let kinds_match = matches!(
                (joint.joint_type, v, a),
                (
                    CuJointType::Revolute,
                    CuJointVelocity::Revolute(_),
                    CuJointAcceleration::Revolute(_),
                ) | (
                    CuJointType::Prismatic,
                    CuJointVelocity::Prismatic(_),
                    CuJointAcceleration::Prismatic(_),
                ) | (
                    CuJointType::Fixed,
                    CuJointVelocity::Fixed,
                    CuJointAcceleration::Fixed,
                )
            );
            if !kinds_match {
                return Some(j as u16);
            }
        }
        None
    }

    /// Evaluates one control cycle in place without heap allocation.
    pub fn evaluate(
        &mut self,
        setpoint: Option<&CuTrajectoryFollowOutput<N>>,
        state: Option<&CuJointStatePayload<N>>,
        scene: Option<&CuPlanningScene<MAX_OBS, MAX_ATTACHED>>,
        out_setpoint: &mut CuTrajectoryFollowOutput<N>,
        out_report: &mut CuSafetyReport,
    ) -> CuResult<()> {
        let scene_revision = scene.map_or(CuSceneRevision::INITIAL, CuPlanningScene::revision);

        if self.last_safe.is_none()
            && let Some(measured) = self.valid_clamped_measured(state)
        {
            self.last_safe = Some(measured);
        }

        let Some(sp) = setpoint else {
            let positions = self.last_safe.unwrap_or_else(|| self.zero_positions());
            *out_setpoint = CuTrajectoryFollowOutput {
                positions,
                velocities: positions.map(|q| CuJointVelocity::from_coordinate(q, 0.0)),
                accelerations: positions.map(|q| CuJointAcceleration::from_coordinate(q, 0.0)),
                status: CuFollowStatus::Idle,
            };
            *out_report = CuSafetyReport {
                verdict: CuSafetyVerdict::Clear,
                transition: self.monitor.update(false),
                near_miss: false,
                min_clearance: Length::new::<meter>(f64::INFINITY),
                tcp_speed: Velocity::new::<meter_per_second>(0.0),
                witness: CuSafetyWitness::None,
                scene_revision,
            };
            return Ok(());
        };

        if matches!(
            sp.status,
            CuFollowStatus::Idle | CuFollowStatus::ErrorEmptyTrajectory
        ) {
            *out_setpoint = sp.clone();
            *out_report = CuSafetyReport {
                verdict: CuSafetyVerdict::Clear,
                transition: self.monitor.update(false),
                near_miss: false,
                min_clearance: Length::new::<meter>(f64::INFINITY),
                tcp_speed: Velocity::new::<meter_per_second>(0.0),
                witness: CuSafetyWitness::None,
                scene_revision,
            };
            return Ok(());
        }

        if let Some(joint_idx) = self.check_joint_limits(sp) {
            self.write_safe_hold(out_setpoint);
            *out_report = CuSafetyReport {
                verdict: CuSafetyVerdict::HeldJointLimits,
                transition: self.monitor.update(true),
                near_miss: false,
                min_clearance: Length::new::<meter>(f64::INFINITY),
                tcp_speed: Velocity::new::<meter_per_second>(0.0),
                witness: CuSafetyWitness::Joint { index: joint_idx },
                scene_revision,
            };
            return Ok(());
        }

        let mut jac = [[0.0_f64; N]; 6];
        self.chain.geometric_jacobian(&sp.positions, &mut jac)?;
        let mut vx = 0.0_f64;
        let mut vy = 0.0_f64;
        let mut vz = 0.0_f64;
        for (j, vel) in sp.velocities.iter().enumerate() {
            let vj = vel.si_value();
            vx += jac[0][j] * vj;
            vy += jac[1][j] * vj;
            vz += jac[2][j] * vj;
        }
        let tcp_speed_mps = libm::sqrt(vx * vx + vy * vy + vz * vz);
        let tcp_speed = Velocity::new::<meter_per_second>(tcp_speed_mps);

        if self.params.require_scene && scene.is_none() {
            self.write_safe_hold(out_setpoint);
            *out_report = CuSafetyReport {
                verdict: CuSafetyVerdict::HeldMissingScene,
                transition: self.monitor.update(true),
                near_miss: false,
                min_clearance: Length::new::<meter>(f64::INFINITY),
                tcp_speed,
                witness: CuSafetyWitness::None,
                scene_revision,
            };
            return Ok(());
        }

        let env = scene.map_or(CuCollisionEnv::EMPTY, CuPlanningScene::collision_env);
        let robot_shape_count = self.model.shape_count();

        let mut frames = [Transform3D::<f64>::identity(); N];
        let mut ee = Transform3D::<f64>::identity();
        self.chain
            .forward_frames(&sp.positions, &mut frames, &mut ee)?;

        let Ok(closest) =
            self.model
                .check_frames_with_closest(&frames, &env, &mut self.scratch_report)
        else {
            self.write_safe_hold(out_setpoint);
            *out_report = CuSafetyReport {
                verdict: CuSafetyVerdict::HeldInvalidScene,
                transition: self.monitor.update(true),
                near_miss: false,
                min_clearance: Length::new::<meter>(f64::INFINITY),
                tcp_speed,
                witness: CuSafetyWitness::None,
                scene_revision,
            };
            return Ok(());
        };

        let collision_dist_m = self.scratch_report.min_signed_distance().get::<meter>();
        let collision_witness = closest
            .map(|c| CuSafetyWitness::from_contact(c, robot_shape_count))
            .unwrap_or(CuSafetyWitness::None);
        let in_collision = !self.scratch_report.is_collision_free();

        let (fence_dist_m, fence_witness) = if let Some(ws) = &self.params.workspace {
            let mat = ee.to_matrix();
            ws.evaluate_point([mat[0][3], mat[1][3], mat[2][3]])
        } else {
            (f64::INFINITY, CuSafetyWitness::None)
        };
        let in_fence = fence_dist_m < 0.0;

        let (mut min_clearance_m, mut witness) = if fence_dist_m < collision_dist_m {
            (fence_dist_m, fence_witness)
        } else {
            (collision_dist_m, collision_witness)
        };

        let mut verdict = if in_collision {
            witness = collision_witness;
            min_clearance_m = collision_dist_m;
            CuSafetyVerdict::HeldCollision
        } else if in_fence {
            witness = fence_witness;
            min_clearance_m = fence_dist_m;
            CuSafetyVerdict::HeldWorkspaceFence
        } else if self.params.max_tcp_speed.is_some_and(|limit| {
            !tcp_speed_mps.is_finite() || tcp_speed_mps > limit.get::<meter_per_second>()
        }) {
            CuSafetyVerdict::HeldTcpSpeed
        } else {
            CuSafetyVerdict::Clear
        };

        if verdict == CuSafetyVerdict::Clear
            && let Some(horizon) = self.params.braking_horizon
        {
            let dt = horizon.get::<second>();
            let mut q_brake = sp.positions;
            let mut extrapolated_ok = true;
            for (j, qb) in q_brake.iter_mut().enumerate() {
                let raw = sp.positions[j].si_value() + sp.velocities[j].si_value() * dt;
                if raw.is_finite() {
                    *qb = qb.with_si_value(raw);
                } else {
                    match self.chain.joints()[j].limits {
                        CuJointLimits::Revolute { lower, upper, .. } => {
                            let val = if raw.is_sign_positive() {
                                upper.get::<cu29::units::si::angle::radian>()
                            } else {
                                lower.get::<cu29::units::si::angle::radian>()
                            };
                            *qb = qb.with_si_value(val);
                        }
                        CuJointLimits::Prismatic { lower, upper, .. } => {
                            let val = if raw.is_sign_positive() {
                                upper.get::<meter>()
                            } else {
                                lower.get::<meter>()
                            };
                            *qb = qb.with_si_value(val);
                        }
                        CuJointLimits::Continuous { .. } | CuJointLimits::Fixed => {
                            verdict = CuSafetyVerdict::HeldBrakingLookahead;
                            witness = CuSafetyWitness::Joint { index: j as u16 };
                            extrapolated_ok = false;
                            break;
                        }
                    }
                }
            }
            if extrapolated_ok {
                self.chain.clamp_to_limits(&mut q_brake)?;
                let mut brake_frames = [Transform3D::<f64>::identity(); N];
                let mut brake_ee = Transform3D::<f64>::identity();
                self.chain
                    .forward_frames(&q_brake, &mut brake_frames, &mut brake_ee)?;
                let Ok(brake_closest) = self.model.check_frames_with_closest(
                    &brake_frames,
                    &env,
                    &mut self.scratch_report,
                ) else {
                    self.write_safe_hold(out_setpoint);
                    *out_report = CuSafetyReport {
                        verdict: CuSafetyVerdict::HeldInvalidScene,
                        transition: self.monitor.update(true),
                        near_miss: false,
                        min_clearance: Length::new::<meter>(min_clearance_m),
                        tcp_speed,
                        witness,
                        scene_revision,
                    };
                    return Ok(());
                };
                let brake_coll_dist_m = self.scratch_report.min_signed_distance().get::<meter>();
                let brake_coll_witness = brake_closest
                    .map(|c| CuSafetyWitness::from_contact(c, robot_shape_count))
                    .unwrap_or(CuSafetyWitness::None);
                let brake_in_collision = !self.scratch_report.is_collision_free();

                let (brake_fence_dist_m, brake_fence_witness) =
                    if let Some(ws) = &self.params.workspace {
                        let mat = brake_ee.to_matrix();
                        ws.evaluate_point([mat[0][3], mat[1][3], mat[2][3]])
                    } else {
                        (f64::INFINITY, CuSafetyWitness::None)
                    };
                let brake_in_fence = brake_fence_dist_m < 0.0;

                if brake_in_collision {
                    verdict = CuSafetyVerdict::HeldBrakingLookahead;
                    witness = brake_coll_witness;
                    min_clearance_m = min_clearance_m.min(brake_coll_dist_m);
                } else if brake_in_fence {
                    verdict = CuSafetyVerdict::HeldBrakingLookahead;
                    witness = brake_fence_witness;
                    min_clearance_m = min_clearance_m.min(brake_fence_dist_m);
                } else {
                    if brake_coll_dist_m < min_clearance_m {
                        min_clearance_m = brake_coll_dist_m;
                        witness = brake_coll_witness;
                    }
                    if brake_fence_dist_m < min_clearance_m {
                        min_clearance_m = brake_fence_dist_m;
                        witness = brake_fence_witness;
                    }
                }
            }
        }

        let in_violation = verdict != CuSafetyVerdict::Clear;
        let transition = self.monitor.update(in_violation);
        if in_violation {
            self.write_safe_hold(out_setpoint);
        } else {
            self.last_safe = Some(sp.positions);
            *out_setpoint = sp.clone();
        }

        let near_miss_threshold_m =
            self.model.safety_margin().get::<meter>() + self.params.near_miss_margin.get::<meter>();
        let near_miss = !in_violation && min_clearance_m < near_miss_threshold_m;

        *out_report = CuSafetyReport {
            verdict,
            transition,
            near_miss,
            min_clearance: Length::new::<meter>(min_clearance_m),
            tcp_speed,
            witness,
            scene_revision,
        };
        Ok(())
    }
}

/// Fixed-size snapshot: Copper sizes the keyframe buffer from the first frozen
/// snapshot, so `in_collision` and `last_safe` always encode to the exact same
/// byte count regardless of `Some`/`None`.
impl<
    const N: usize,
    const M: usize,
    const MAX_OBS: usize,
    const MAX_ATTACHED: usize,
    const MAX_CONTACTS: usize,
> Freezable for CuSafetyShieldTask<N, M, MAX_OBS, MAX_ATTACHED, MAX_CONTACTS>
{
    fn freeze<E: Encoder>(&self, encoder: &mut E) -> Result<(), EncodeError> {
        Encode::encode(&self.monitor.in_collision(), encoder)?;
        Encode::encode(&self.last_safe.is_some(), encoder)?;
        let pos = self.last_safe.unwrap_or_else(|| self.zero_positions());
        Encode::encode(&pos, encoder)
    }

    fn thaw<D: Decoder>(&mut self, decoder: &mut D) -> Result<(), DecodeError> {
        let in_collision: bool = Decode::decode(decoder)?;
        let has_last_safe: bool = Decode::decode(decoder)?;
        let pos: [CuJointPosition; N] = Decode::decode(decoder)?;
        if in_collision {
            let _ = self.monitor.update(true);
        } else {
            self.monitor.reset();
        }
        self.last_safe = has_last_safe.then_some(pos);
        Ok(())
    }
}

impl<
    const N: usize,
    const M: usize,
    const MAX_OBS: usize,
    const MAX_ATTACHED: usize,
    const MAX_CONTACTS: usize,
> CuTask for CuSafetyShieldTask<N, M, MAX_OBS, MAX_ATTACHED, MAX_CONTACTS>
{
    type Resources<'r> = ();
    type Input<'m> = input_msg!(
        'm,
        CuTrajectoryFollowOutput<N>,
        CuJointStatePayload<N>,
        CuPlanningScene<MAX_OBS, MAX_ATTACHED>
    );
    type Output<'m> = output_msg!(CuTrajectoryFollowOutput<N>, CuSafetyReport);

    fn new(cfg: Option<&ComponentConfig>, _res: Self::Resources<'_>) -> CuResult<Self>
    where
        Self: Sized,
    {
        let params: CuSafetyShieldParams<N, M> = cfg
            .map(ComponentConfig::deserialize_into::<CuSafetyShieldParams<N, M>>)
            .transpose()?
            .unwrap_or_default();
        Self::from_params(params)
    }

    fn process<'i, 'o>(
        &mut self,
        _ctx: &CuContext,
        input: &Self::Input<'i>,
        output: &mut Self::Output<'o>,
    ) -> CuResult<()> {
        let tov = if !matches!(input.0.tov, Tov::None) {
            input.0.tov
        } else if !matches!(input.1.tov, Tov::None) {
            input.1.tov
        } else {
            input.2.tov
        };
        output.0.tov = tov;
        output.1.tov = tov;

        let mut out_setpoint = CuTrajectoryFollowOutput::default();
        let mut out_report = CuSafetyReport::default();
        self.evaluate(
            input.0.payload(),
            input.1.payload(),
            input.2.payload(),
            &mut out_setpoint,
            &mut out_report,
        )?;
        output.0.set_payload(out_setpoint);
        output.1.set_payload(out_report);
        Ok(())
    }
}
