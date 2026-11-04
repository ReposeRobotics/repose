//! Pose targets resolved against the same scene used by joint planners.

use crate::CuJointPlanPath;
use crate::CuJointPlanRequest;
use crate::CuJointPlanner;
use crate::CuJointPlannerParams;
use crate::CuJointPlannerTask;
use crate::CuPlanStatus;
use crate::CuSceneValidity;
use crate::MAX_ATTACHED_SHAPES;
use crate::MAX_EXCLUSIONS;
use crate::MAX_OBSTACLES;
use crate::planner_resources;
use bincode::Decode;
use bincode::Encode;
use cu_rng::prelude::*;
use cu_spatial_payloads::Transform3D;
use cu29::cutask_anytime::AnytimeStatus;
use cu29::cutask_anytime::CuAnytimeTask;
use cu29::cutask_anytime::Quality;
use cu29::prelude::*;
use cu29::units::si::f64::Angle;
use cu29::units::si::f64::Length;
use repose_kinematics::CuIkParams;
use repose_kinematics::CuJointResolution;
use serde::Deserialize;
use serde::Serialize;

/// TCP pose in the chain base frame. Rotation is roll, pitch, yaw, applied
/// X then Y then Z (`Rz * Ry * Rx` for column vectors).
#[derive(
    Default, Debug, Clone, Copy, PartialEq, Encode, Decode, Serialize, Deserialize, Reflect,
)]
#[serde(deny_unknown_fields)]
pub struct CuTcpPose {
    pub translation: [Length; 3],
    pub rotation: [Angle; 3],
}
impl CuTcpPose {
    pub(crate) fn transform(&self) -> Transform3D<f64> {
        Transform3D::<f64>::from_translation_euler_xyz(self.translation, self.rotation)
    }
}

/// Request shared by pose-goal joint planning and straight TCP motion.
#[derive(Debug, Clone, PartialEq, Encode, Decode, Serialize, Deserialize, Reflect)]
#[reflect(no_field_bounds, from_reflect = false)]
pub struct CuPosePlanRequest<const N: usize> {
    pub scene: repose_collision::CuPlanningScene<MAX_OBSTACLES, MAX_ATTACHED_SHAPES>,
    pub exclusions: repose_collision::CuSceneExclusions<MAX_EXCLUSIONS>,
    #[serde(with = "repose_kinematics::serde_arrays::position_array")]
    pub start: [repose_kinematics::CuJointPosition; N],
    pub goal: CuTcpPose,
}

impl<const N: usize> Default for CuPosePlanRequest<N> {
    fn default() -> Self {
        Self {
            scene: repose_collision::CuPlanningScene::new(),
            exclusions: repose_collision::CuSceneExclusions::new(),
            start: [repose_kinematics::CuJointPosition::default(); N],
            goal: CuTcpPose::default(),
        }
    }
}

/// Bounded IK search: measured start first, then seeded joint-limit samples.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Reflect)]
#[serde(default, deny_unknown_fields)]
pub struct CuPoseIkParams {
    pub ik: CuIkParams,
    /// Maximum IK attempts, including the measured start (1 through 64).
    pub attempts: u32,
}
impl Default for CuPoseIkParams {
    fn default() -> Self {
        Self {
            ik: CuIkParams::default(),
            attempts: 8,
        }
    }
}
impl CuPoseIkParams {
    /// Checks solver settings and the bounded attempt budget.
    pub fn validate(&self) -> CuResult<()> {
        self.ik.validate()?;
        if !(1..=64).contains(&self.attempts) {
            return Err(CuError::from("pose goal: attempts must be in 1..=64"));
        }
        Ok(())
    }
}

pub(crate) fn resolve_pose<const N: usize, const M: usize>(
    validity: &mut CuSceneValidity<N, M>,
    request: &CuPosePlanRequest<N>,
    params: &CuPoseIkParams,
    seed: u64,
    out: &mut CuJointPlanRequest<N>,
) -> CuResult<CuPlanStatus> {
    out.scene = request.scene.clone();
    out.exclusions = request.exclusions.clone();
    out.start = request.start;
    if !validity.capture(&out.scene, &out.exclusions) {
        return Ok(CuPlanStatus::InvalidRequest);
    }
    if !validity.is_configuration_free(&out.start) {
        return Ok(CuPlanStatus::InvalidStart);
    }
    let target = request.goal.transform();
    if target.to_matrix().iter().flatten().any(|v| !v.is_finite()) {
        return Ok(CuPlanStatus::InvalidRequest);
    }
    let mut rng = CuRng::from_seed(seed);
    let mut reached = false;
    for attempt in 0..params.attempts {
        let initial = if attempt == 0 {
            out.start
        } else {
            crate::common::sample_uniform(validity.chain(), &mut rng)
        };
        let status = validity.chain().inverse_kinematics_dls(
            &target,
            &initial,
            &params.ik,
            &mut out.goal,
        )?;
        if status.is_converged() {
            reached = true;
            if validity.is_configuration_free(&out.goal) {
                return Ok(CuPlanStatus::Searching);
            }
        }
    }
    Ok(if reached {
        CuPlanStatus::InvalidGoal
    } else {
        CuPlanStatus::NoIkSolution
    })
}

/// Configuration of [`CuPosePlannerTask`].
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(
    deny_unknown_fields,
    bound(
        serialize = "P: Serialize",
        deserialize = "P: Deserialize<'de> + Default"
    )
)]
pub struct CuPosePlannerParams<P, const N: usize, const M: usize = N> {
    pub planner: CuJointPlannerParams<P, N, M>,
    #[serde(default)]
    pub goal: CuPoseIkParams,
}

impl<P, const N: usize, const M: usize> CuPosePlannerParams<P, N, M> {
    /// Checks robot models, planner budgets, and bounded IK settings.
    /// Algorithm-specific planner parameters are checked when the planner is built.
    pub fn validate(&self) -> CuResult<()> {
        self.goal.validate()?;
        self.planner.validate()
    }
}

/// Resolves a TCP goal, then runs any joint planner through its anytime lifecycle.
/// Results preserve scene revisions and exclusions; failed IK publishes an empty path.
#[derive(Reflect)]
#[reflect(no_field_bounds, from_reflect = false)]
pub struct CuPosePlannerTask<P, const N: usize, const M: usize = N> {
    #[reflect(ignore)]
    inner: CuJointPlannerTask<P, N, M>,
    #[reflect(ignore)]
    validity: CuSceneValidity<N, M>,
    #[reflect(ignore)]
    params: CuPoseIkParams,
    failed: bool,
}
impl<P: CuJointPlanner<N, M>, const N: usize, const M: usize> CuPosePlannerTask<P, N, M> {
    /// Builds the planner and IK resolver from the same robot models.
    pub fn from_params(params: CuPosePlannerParams<P::Params, N, M>, seed: u64) -> CuResult<Self> {
        params.validate()?;
        let (chain, model) = params.planner.resolve_models()?;
        Ok(Self {
            inner: CuJointPlannerTask::from_params(params.planner, seed)?,
            validity: CuSceneValidity::new(chain, model, CuJointResolution::default()),
            params: params.goal,
            failed: false,
        })
    }
}
impl<P, const N: usize, const M: usize> Freezable for CuPosePlannerTask<P, N, M> {
    fn freeze<E: bincode::enc::Encoder>(
        &self,
        e: &mut E,
    ) -> Result<(), bincode::error::EncodeError> {
        self.inner.freeze(e)?;
        Encode::encode(&self.failed, e)
    }
    fn thaw<D: bincode::de::Decoder>(
        &mut self,
        d: &mut D,
    ) -> Result<(), bincode::error::DecodeError> {
        self.inner.thaw(d)?;
        self.failed = Decode::decode(d)?;
        Ok(())
    }
}
impl<P: CuJointPlanner<N, M> + Send + Sync + 'static, const N: usize, const M: usize> CuAnytimeTask
    for CuPosePlannerTask<P, N, M>
{
    type Input<'m> = input_msg!(CuPosePlanRequest<N>);
    type Output<'m> = output_msg!(CuJointPlanPath<N>);
    type Resources<'r> = planner_resources::Resources;
    type Quality = Quality;
    fn new(cfg: Option<&ComponentConfig>, resources: Self::Resources<'_>) -> CuResult<Self> {
        let cfg = cfg.ok_or_else(|| CuError::from("pose planner: missing config"))?;
        let params = cfg.deserialize_into::<CuPosePlannerParams<P::Params, N, M>>()?;
        let Owned(mut rng) = resources.rng;
        Self::from_params(params, rng.random())
    }
    fn base<'i, 'o>(
        &mut self,
        _ctx: &CuContext,
        input: &Self::Input<'i>,
        output: &mut Self::Output<'o>,
    ) -> CuResult<AnytimeStatus<Quality>> {
        output.clear_payload();
        output.tov = input.tov;
        self.failed = true;
        let Some(request) = input.payload() else {
            return Ok(AnytimeStatus::Aborted);
        };
        let mut joint = CuJointPlanRequest::default();
        let status = resolve_pose(
            &mut self.validity,
            request,
            &self.params,
            self.inner.next_seed(),
            &mut joint,
        )?;
        if status != CuPlanStatus::Searching {
            let path = output
                .payload_mut()
                .get_or_insert_with(CuJointPlanPath::default);
            crate::common::write_header(path, &self.validity, status, 0);
            return Ok(AnytimeStatus::Aborted);
        }
        self.failed = false;
        self.inner.begin_request(&joint, output)
    }
    fn refine<'o>(
        &mut self,
        ctx: &CuContext,
        output: &mut Self::Output<'o>,
    ) -> CuResult<AnytimeStatus<Quality>> {
        if self.failed {
            Ok(AnytimeStatus::Aborted)
        } else {
            self.inner.refine(ctx, output)
        }
    }
}

/// Configured TCP goal and scene for [`CuPoseGoalTask`].
#[derive(Debug, Clone, Serialize, Deserialize, Reflect)]
#[serde(deny_unknown_fields)]
#[reflect(from_reflect = false)]
pub struct CuPoseGoalParams {
    pub goal: CuTcpPose,
    #[serde(default)]
    pub scene: repose_collision::CuPlanningScene<MAX_OBSTACLES, MAX_ATTACHED_SHAPES>,
    #[serde(default)]
    pub exclusions: repose_collision::CuSceneExclusions<MAX_EXCLUSIONS>,
}

impl CuPoseGoalParams {
    /// Checks that the goal contains only finite coordinates.
    /// Scene references are checked against the robot when planning begins.
    pub fn validate(&self) -> CuResult<()> {
        if self
            .goal
            .transform()
            .to_matrix()
            .iter()
            .flatten()
            .any(|v| !v.is_finite())
        {
            return Err(CuError::from("pose goal: non-finite target"));
        }
        Ok(())
    }
}

/// Emits one pose request from the first ready measured joint state.
/// Connect its output to either [`CuPosePlannerTask`] or `CuCartesianPlanTask`.
#[derive(Reflect)]
#[reflect(no_field_bounds, from_reflect = false)]
pub struct CuPoseGoalTask<const N: usize> {
    params: CuPoseGoalParams,
    requested: bool,
}
impl<const N: usize> CuPoseGoalTask<N> {
    /// Creates a one-shot request source with a configured scene snapshot.
    pub fn from_params(params: CuPoseGoalParams) -> CuResult<Self> {
        params.validate()?;
        Ok(Self {
            params,
            requested: false,
        })
    }
}
impl<const N: usize> Freezable for CuPoseGoalTask<N> {
    fn freeze<E: bincode::enc::Encoder>(
        &self,
        e: &mut E,
    ) -> Result<(), bincode::error::EncodeError> {
        Encode::encode(&self.requested, e)
    }
    fn thaw<D: bincode::de::Decoder>(
        &mut self,
        d: &mut D,
    ) -> Result<(), bincode::error::DecodeError> {
        self.requested = Decode::decode(d)?;
        Ok(())
    }
}
impl<const N: usize> CuTask for CuPoseGoalTask<N> {
    type Input<'m> = input_msg!(repose_kinematics::CuJointStatePayload<N>);
    type Output<'m> = output_msg!(CuPosePlanRequest<N>);
    type Resources<'r> = ();
    fn new(cfg: Option<&ComponentConfig>, _resources: ()) -> CuResult<Self> {
        Self::from_params(
            cfg.ok_or_else(|| CuError::from("pose goal: missing config"))?
                .deserialize_into::<CuPoseGoalParams>()?,
        )
    }
    fn process<'i, 'o>(
        &mut self,
        _ctx: &CuContext,
        input: &Self::Input<'i>,
        output: &mut Self::Output<'o>,
    ) -> CuResult<()> {
        output.clear_payload();
        output.tov = input.tov;
        if self.requested {
            return Ok(());
        }
        if let Some(state) = input
            .payload()
            .filter(|s| s.status == repose_kinematics::CuRobotStatus::Ready)
        {
            let request = output
                .payload_mut()
                .get_or_insert_with(CuPosePlanRequest::default);
            request.start = state.positions;
            request.goal = self.params.goal;
            request.scene = self.params.scene.clone();
            request.exclusions = self.params.exclusions.clone();
            self.requested = true;
        }
        Ok(())
    }
}
