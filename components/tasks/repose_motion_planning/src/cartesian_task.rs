//! Scene-aware straight TCP requests and timed trajectory results.

use crate::CuJointPlanRequest;
use crate::CuPlanStatus;
use crate::CuPoseIkParams;
use crate::CuPosePlanRequest;
use crate::CuSceneValidity;
use crate::MAX_EXCLUSIONS;
use crate::planner_resources;
use crate::pose::resolve_pose;
use bincode::Decode;
use bincode::Encode;
use cu_rng::prelude::*;
use cu29::prelude::*;
use cu29::units::si::f64::Length;
use cu29::units::si::length::meter;
use repose_collision::CuCollisionModelSpec;
use repose_kinematics::CuChainSpec;
use repose_trajectory::CuCartesianBlendParams;
use repose_trajectory::CuCartesianBlendPath;
use repose_trajectory::CuCartesianBlendStatus;
use repose_trajectory::CuCartesianWaypoint;
use repose_trajectory::CuPathTimingParams;
use repose_trajectory::CuTimingStatus;
use repose_trajectory::CuTrajectoryPayload;
use repose_trajectory::blend_cartesian_path;
use repose_trajectory::parameterize_cartesian_path;
use serde::Deserialize;
use serde::Serialize;

/// Configuration of [`CuCartesianPlanTask`].
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CuCartesianPlanParams<const N: usize, const M: usize = N> {
    pub chain: CuChainSpec<N>,
    pub collision_model: CuCollisionModelSpec<M>,
    #[serde(default)]
    pub goal: CuPoseIkParams,
    #[serde(default)]
    pub path: CuCartesianBlendParams,
    pub timing: CuPathTimingParams<N>,
}
impl<const N: usize, const M: usize> CuCartesianPlanParams<N, M> {
    /// Checks the robot models, IK, geometry, and timing parameters.
    pub fn validate(&self) -> CuResult<()> {
        self.chain.resolve()?;
        self.collision_model.resolve::<N>()?;
        self.goal.validate()?;
        self.path.validate()?;
        self.timing.validate()
    }
}

/// A timed Cartesian move with the scene identity and explicit outcome.
/// The trajectory is empty on failure. Geometry and timing are checked by discrete
/// sampling, not continuous swept-volume collision detection.
#[derive(Default, Debug, Clone, PartialEq, Encode, Decode, Serialize, Deserialize, Reflect)]
#[reflect(no_field_bounds, from_reflect = false)]
pub struct CuCartesianPlanResult<const N: usize, const CAP: usize> {
    pub trajectory: CuTrajectoryPayload<N, CAP>,
    pub status: CuPlanStatus,
    pub scene_revision: repose_collision::CuSceneRevision,
    pub exclusions: repose_collision::CuSceneExclusions<MAX_EXCLUSIONS>,
}

/// Straight TCP motion from measured joints to a pose, with shortest-arc orientation.
/// Uses the request's scene (including attachments and exclusions) for IK endpoint,
/// Cartesian samples, and timing checks. Each input message starts one bounded job.
/// `PATH` bounds geometric samples and `CAP` bounds trajectory points.
#[derive(Reflect)]
#[reflect(no_field_bounds, from_reflect = false)]
pub struct CuCartesianPlanTask<
    const N: usize,
    const PATH: usize,
    const CAP: usize,
    const M: usize = N,
> {
    #[reflect(ignore)]
    params: CuCartesianPlanParams<N, M>,
    #[reflect(ignore)]
    chain: repose_kinematics::CuKinematicChain<N>,
    #[reflect(ignore)]
    model: repose_collision::CuRobotCollisionModel<N, M>,
    #[reflect(ignore)]
    validity: CuSceneValidity<N, M>,
    #[reflect(ignore)]
    path: CuCartesianBlendPath<N, PATH>,
    seed: u64,
    jobs: u64,
}
impl<const N: usize, const PATH: usize, const CAP: usize, const M: usize>
    CuCartesianPlanTask<N, PATH, CAP, M>
{
    /// Creates a bounded Cartesian planning task; seed determines IK retries.
    pub fn from_params(params: CuCartesianPlanParams<N, M>, seed: u64) -> CuResult<Self> {
        params.validate()?;
        if PATH < 2 || CAP < 2 {
            return Err(CuError::from(
                "Cartesian planning: capacities must be at least two",
            ));
        }
        let chain = params.chain.resolve()?;
        let model = params.collision_model.resolve::<N>()?;
        let validity = CuSceneValidity::new(chain, model, params.path.collision_step);
        Ok(Self {
            params,
            chain,
            model,
            validity,
            path: CuCartesianBlendPath::default(),
            seed,
            jobs: 0,
        })
    }
    /// Plans and times a move in place. Failed results carry no trajectory.
    pub fn plan(
        &mut self,
        request: &CuPosePlanRequest<N>,
        out: &mut CuCartesianPlanResult<N, CAP>,
    ) -> CuResult<()> {
        out.trajectory.clear();
        out.scene_revision = request.scene.revision();
        out.exclusions = request.exclusions.clone();
        out.status = CuPlanStatus::InvalidRequest;
        self.jobs = self.jobs.wrapping_add(1);
        let mut joint = CuJointPlanRequest::default();
        out.status = resolve_pose(
            &mut self.validity,
            request,
            &self.params.goal,
            self.seed.wrapping_add(self.jobs),
            &mut joint,
        )?;
        if out.status != CuPlanStatus::Searching {
            return Ok(());
        }
        let env = joint
            .scene
            .collision_env()
            .with_exclusions(&joint.exclusions);
        let waypoints = [
            CuCartesianWaypoint::new(joint.start, Length::new::<meter>(0.0)),
            CuCartesianWaypoint::new(joint.goal, Length::new::<meter>(0.0)),
        ];
        out.status = match blend_cartesian_path(
            &self.chain,
            &self.model,
            &env,
            &waypoints,
            &self.params.path,
            &mut self.path,
        )? {
            CuCartesianBlendStatus::Success => CuPlanStatus::Searching,
            CuCartesianBlendStatus::CapacityExceeded => CuPlanStatus::CapacityExceeded,
            CuCartesianBlendStatus::InverseKinematicsFailed { .. } => CuPlanStatus::NoIkSolution,
            CuCartesianBlendStatus::Collision { .. } => CuPlanStatus::Exhausted,
            _ => CuPlanStatus::InvalidRequest,
        };
        if out.status != CuPlanStatus::Searching {
            return Ok(());
        }
        out.status = match parameterize_cartesian_path(
            &self.chain,
            &self.model,
            &env,
            &self.path,
            &self.params.timing,
            &mut out.trajectory,
        )? {
            CuTimingStatus::Success => CuPlanStatus::Converged,
            CuTimingStatus::CapacityExceeded => CuPlanStatus::CapacityExceeded,
            CuTimingStatus::Collision { .. } => CuPlanStatus::Exhausted,
            _ => CuPlanStatus::InvalidRequest,
        };
        Ok(())
    }
}
impl<const N: usize, const PATH: usize, const CAP: usize, const M: usize> Freezable
    for CuCartesianPlanTask<N, PATH, CAP, M>
{
    fn freeze<E: bincode::enc::Encoder>(
        &self,
        e: &mut E,
    ) -> Result<(), bincode::error::EncodeError> {
        Encode::encode(&self.seed, e)?;
        Encode::encode(&self.jobs, e)
    }
    fn thaw<D: bincode::de::Decoder>(
        &mut self,
        d: &mut D,
    ) -> Result<(), bincode::error::DecodeError> {
        self.seed = Decode::decode(d)?;
        self.jobs = Decode::decode(d)?;
        Ok(())
    }
}
impl<const N: usize, const PATH: usize, const CAP: usize, const M: usize> CuTask
    for CuCartesianPlanTask<N, PATH, CAP, M>
{
    type Input<'m> = input_msg!(CuPosePlanRequest<N>);
    type Output<'m> = output_msg!(CuCartesianPlanResult<N, CAP>);
    type Resources<'r> = planner_resources::Resources;
    fn new(cfg: Option<&ComponentConfig>, resources: Self::Resources<'_>) -> CuResult<Self> {
        let cfg = cfg.ok_or_else(|| CuError::from("Cartesian planner: missing config"))?;
        let Owned(mut rng) = resources.rng;
        Self::from_params(
            cfg.deserialize_into::<CuCartesianPlanParams<N, M>>()?,
            rng.random(),
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
        if let Some(request) = input.payload() {
            self.plan(
                request,
                output
                    .payload_mut()
                    .get_or_insert_with(CuCartesianPlanResult::default),
            )?;
        }
        Ok(())
    }
}

/// Passes successful Cartesian trajectories to the standard trajectory follower.
#[derive(Default, Reflect)]
pub struct CuCartesianTrajectoryTask<const N: usize, const CAP: usize>;
impl<const N: usize, const CAP: usize> Freezable for CuCartesianTrajectoryTask<N, CAP> {}
impl<const N: usize, const CAP: usize> CuTask for CuCartesianTrajectoryTask<N, CAP> {
    type Input<'m> = input_msg!(CuCartesianPlanResult<N, CAP>);
    type Output<'m> = output_msg!(CuTrajectoryPayload<N, CAP>);
    type Resources<'r> = ();
    fn new(_cfg: Option<&ComponentConfig>, _resources: ()) -> CuResult<Self> {
        Ok(Self)
    }
    fn process<'i, 'o>(
        &mut self,
        _ctx: &CuContext,
        input: &Self::Input<'i>,
        output: &mut Self::Output<'o>,
    ) -> CuResult<()> {
        output.clear_payload();
        output.tov = input.tov;
        if let Some(result) = input.payload().filter(|r| r.status.has_path()) {
            output.set_payload(result.trajectory.clone());
        }
        Ok(())
    }
}
