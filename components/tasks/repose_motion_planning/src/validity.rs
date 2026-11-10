//! Scene-aware validity checking shared by every planner.
//!
//! [`CuSceneValidity`] captures one [`CuPlanningScene`] snapshot per planning
//! job and answers the three questions any planner asks, sampling-based or
//! not: is a configuration free, is a joint-space segment free, and how much
//! clearance does a configuration have.

use crate::CuPathConstraintSet;
use crate::MAX_ATTACHED_SHAPES;
use crate::MAX_EXCLUSIONS;
use crate::MAX_OBSTACLES;
use cu_spatial_payloads::Transform3D;
use cu29::prelude::CuError;
use cu29::prelude::CuResult;
use cu29::units::si::length::meter;
use repose_collision::CuCollisionEnv;
use repose_collision::CuCollisionReport;
use repose_collision::CuPlanningScene;
use repose_collision::CuRobotCollisionModel;
use repose_collision::CuSceneExclusions;
use repose_collision::CuSceneRevision;
use repose_kinematics::CuJointMetric;
use repose_kinematics::CuJointPosition;
use repose_kinematics::CuJointResolution;
use repose_kinematics::CuKinematicChain;

/// Collision queries against one captured planning-scene snapshot.
///
/// Every query checks the robot against itself, the scene obstacles, and the
/// scene's attached shapes in one pass, honouring the captured stable-ID
/// exclusions. An optional [`CuPathConstraintSet`] further rejects
/// configurations whose tool pose violates a constraint.
#[derive(Clone)]
pub struct CuSceneValidity<const N: usize, const M: usize = N> {
    chain: CuKinematicChain<N>,
    model: CuRobotCollisionModel<N, M>,
    resolution: CuJointResolution,
    scene: CuPlanningScene<MAX_OBSTACLES, MAX_ATTACHED_SHAPES>,
    exclusions: CuSceneExclusions<MAX_EXCLUSIONS>,
    constraints: Option<CuPathConstraintSet>,
    /// `false` until a valid scene has been captured.
    captured: bool,
}

impl<const N: usize, const M: usize> CuSceneValidity<N, M> {
    /// Creates a checker with an empty scene; `resolution` sets the segment
    /// sampling step.
    pub fn new(
        chain: CuKinematicChain<N>,
        model: CuRobotCollisionModel<N, M>,
        resolution: CuJointResolution,
    ) -> Self {
        Self {
            chain,
            model,
            resolution,
            scene: CuPlanningScene::new(),
            exclusions: CuSceneExclusions::new(),
            constraints: None,
            captured: true,
        }
    }

    /// The kinematic chain the checker evaluates.
    pub const fn chain(&self) -> &CuKinematicChain<N> {
        &self.chain
    }

    /// Segment sampling resolution.
    pub const fn resolution(&self) -> CuJointResolution {
        self.resolution
    }

    /// Revision of the captured scene.
    pub const fn scene_revision(&self) -> CuSceneRevision {
        self.scene.revision()
    }

    /// Exclusions captured with the scene.
    pub const fn exclusions(&self) -> &CuSceneExclusions<MAX_EXCLUSIONS> {
        &self.exclusions
    }

    /// The path constraints configurations must satisfy, if any.
    pub const fn path_constraints(&self) -> Option<&CuPathConstraintSet> {
        self.constraints.as_ref()
    }

    /// Sets the path constraints every accepted configuration must satisfy,
    /// on top of joint limits and collisions. Fails without changing the
    /// checker when `constraints` is invalid.
    pub fn set_path_constraints(&mut self, constraints: &CuPathConstraintSet) -> CuResult<()> {
        constraints.validate()?;
        if constraints.is_empty() {
            return Err(CuError::from(
                "repose-motion-planning: path constraint set must not be empty",
            ));
        }
        self.constraints = Some(*constraints);
        Ok(())
    }

    /// Clears the path constraints, restoring unconstrained checking.
    pub fn clear_path_constraints(&mut self) {
        self.constraints = None;
    }

    /// Captures a scene snapshot for the next job. Returns `false` (and
    /// treats every configuration as blocked) when the resolution, the
    /// scene's attached shapes, or the stable-ID exclusions are invalid for
    /// this robot.
    pub fn capture(
        &mut self,
        scene: &CuPlanningScene<MAX_OBSTACLES, MAX_ATTACHED_SHAPES>,
        exclusions: &CuSceneExclusions<MAX_EXCLUSIONS>,
    ) -> bool {
        self.scene = scene.clone();
        self.exclusions = exclusions.clone();
        self.captured = self.resolution.metric().validate().is_ok()
            && self
                .scene
                .attached_shapes()
                .iter()
                .all(|a| a.frame.is_none_or(|frame| frame < N))
            && exclusions
                .validate(&self.scene, self.model.shape_count())
                .is_ok();
        self.captured
    }

    /// Obstacles, attached shapes, and exclusions of the captured scene.
    fn env(&self) -> CuCollisionEnv<'_> {
        self.scene.collision_env().with_exclusions(&self.exclusions)
    }

    /// `true` when `q` is within joint limits and free of self, environment,
    /// and attached-object collisions. When path constraints are set, `q`
    /// must additionally satisfy every one of them.
    pub fn is_configuration_free(&self, q: &[CuJointPosition; N]) -> bool {
        self.captured
            && self
                .model
                .is_configuration_collision_free(&self.chain, q, &self.env())
                .unwrap_or(false)
            && self.path_constraints_satisfied(q)
    }

    /// `true` when the tool pose of `q` satisfies every path constraint, or
    /// when no constraints are set. Unevaluable configurations fail.
    fn path_constraints_satisfied(&self, q: &[CuJointPosition; N]) -> bool {
        let Some(constraints) = &self.constraints else {
            return true;
        };
        let mut tcp = Transform3D::<f64>::identity();
        if self.chain.forward_kinematics(q, &mut tcp).is_err() {
            return false;
        }
        constraints.is_satisfied_by_pose(&tcp)
    }

    /// `true` when the joint-space line from `a` to `b` is free at the
    /// configured resolution (discrete sampling, not swept volumes).
    ///
    /// Samples both endpoints (`s = 0` and `s = steps`); use
    /// [`Self::is_segment_interior_free`] instead when the caller has
    /// already proven both endpoints free.
    pub fn is_segment_free(&self, a: &[CuJointPosition; N], b: &[CuJointPosition; N]) -> bool {
        let metric = self.resolution.metric();
        let mut max_delta = 0.0_f64;
        for j in 0..N {
            let joint = &self.chain.joints()[j];
            if joint.validate_coordinate(a[j]).is_err() || joint.validate_coordinate(b[j]).is_err()
            {
                return false;
            }
            max_delta = max_delta.max(libm::fabs(
                metric.normalized(b[j]) - metric.normalized(a[j]),
            ));
        }
        if !max_delta.is_finite() || max_delta > 1_000_000.0 {
            return false;
        }
        let steps = (libm::ceil(max_delta) as usize).max(1);
        for s in 0..=steps {
            let alpha = s as f64 / steps as f64;
            let mut q = [CuJointPosition::default(); N];
            for j in 0..N {
                match a[j].interpolate(b[j], alpha) {
                    Ok(value) => q[j] = value,
                    Err(_) => return false,
                }
            }
            if !self.is_configuration_free(&q) {
                return false;
            }
        }
        true
    }

    /// `true` when the interior of the joint-space line from `a` to `b`
    /// is free at the configured resolution, skipping the `s = 0` and
    /// `s = steps` endpoint evaluations.
    ///
    /// Boolean-exact with [`Self::is_segment_free`] when both endpoints
    /// are already known free (joint limits valid and collision-free).
    /// Only use at call sites that guarantee both endpoints; otherwise
    /// keep [`Self::is_segment_free`].
    pub fn is_segment_interior_free(
        &self,
        a: &[CuJointPosition; N],
        b: &[CuJointPosition; N],
    ) -> bool {
        let metric = self.resolution.metric();
        let mut max_delta = 0.0_f64;
        for j in 0..N {
            let joint = &self.chain.joints()[j];
            if joint.validate_coordinate(a[j]).is_err() || joint.validate_coordinate(b[j]).is_err()
            {
                return false;
            }
            max_delta = max_delta.max(libm::fabs(
                metric.normalized(b[j]) - metric.normalized(a[j]),
            ));
        }
        if !max_delta.is_finite() || max_delta > 1_000_000.0 {
            return false;
        }
        let steps = (libm::ceil(max_delta) as usize).max(1);
        for s in 1..steps {
            let alpha = s as f64 / steps as f64;
            let mut q = [CuJointPosition::default(); N];
            for j in 0..N {
                match a[j].interpolate(b[j], alpha) {
                    Ok(value) => q[j] = value,
                    Err(_) => return false,
                }
            }
            if !self.is_configuration_free(&q) {
                return false;
            }
        }
        true
    }

    /// Smallest signed surface distance in meters between the robot (and
    /// attached objects) and itself or the scene; negative means penetration.
    /// `None` when `q` cannot be evaluated.
    pub fn clearance(&self, q: &[CuJointPosition; N]) -> Option<f64> {
        if !self.captured {
            return None;
        }
        let mut report = CuCollisionReport::<0>::default();
        self.model
            .check_configuration(&self.chain, q, &self.env(), &mut report)
            .ok()?;
        Some(report.min_signed_distance().get::<meter>())
    }
}

/// Joint-space L2 distance under `metric` (dimensionless, metric-scaled).
pub fn joint_distance<const N: usize>(
    a: &[CuJointPosition; N],
    b: &[CuJointPosition; N],
    metric: CuJointMetric,
) -> f64 {
    let mut sum = 0.0_f64;
    for j in 0..N {
        let d = metric.normalized(b[j]) - metric.normalized(a[j]);
        sum += d * d;
    }
    libm::sqrt(sum)
}
