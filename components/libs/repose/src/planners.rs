//! The one registry of Repose joint planners: [`CuPlannerKind`] names every
//! planner (its RON spelling) and [`CuAnyPlanner`] runs any of them through
//! the [`CuJointPlanner`] contract. Adding a planner means implementing the
//! trait and adding one variant here; applications select it by name.

#[cfg(not(feature = "std"))]
extern crate alloc;
#[cfg(not(feature = "std"))]
use alloc::boxed::Box;

use cu29::prelude::*;
use repose_collision::CuRobotCollisionModel;
use repose_joint_rrt_star::CuJointRrtStar;
use repose_kinematics::CuKinematicChain;
use repose_motion_planning::CuJointAStar;
use repose_motion_planning::CuJointDirect;
use repose_motion_planning::CuJointPathOptimizer;
use repose_motion_planning::CuJointPlanPath;
use repose_motion_planning::CuJointPlanRequest;
use repose_motion_planning::CuJointPlanner;
use repose_motion_planning::CuJointPrm;
use repose_motion_planning::CuJointRrtConnect;
use repose_motion_planning::CuJointStomp;
use repose_motion_planning::CuPlanStatus;
use serde::Deserialize;
use serde::Serialize;

/// Every Repose joint planner, by RON name.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, Reflect)]
#[serde(rename_all = "snake_case")]
pub enum CuPlannerKind {
    /// Bidirectional RRT, feasibility-first (`rrt_connect`).
    #[default]
    RrtConnect,
    /// Asymptotically optimal RRT* with rewiring and shortcutting (`rrt_star`).
    RrtStar,
    /// Probabilistic roadmap (`prm`).
    Prm,
    /// A* on a joint-space lattice (`a_star`).
    AStar,
    /// Stochastic trajectory optimization (`stomp`).
    Stomp,
    /// Gradient path optimizer, CHOMP-style (`path_optimizer`).
    PathOptimizer,
    /// Straight joint line, collision-checked (`direct`).
    Direct,
}

impl CuPlannerKind {
    /// Every variant, in declaration order.
    pub const ALL: [Self; 7] = [
        Self::RrtConnect,
        Self::RrtStar,
        Self::Prm,
        Self::AStar,
        Self::Stomp,
        Self::PathOptimizer,
        Self::Direct,
    ];
}

/// Any Repose joint planner with default parameters, chosen at runtime.
pub enum CuAnyPlanner<const N: usize, const M: usize = N> {
    RrtConnect(Box<CuJointRrtConnect<N, M>>),
    RrtStar(Box<CuJointRrtStar<N, M>>),
    Prm(Box<CuJointPrm<N, M>>),
    AStar(Box<CuJointAStar<N, M>>),
    Stomp(Box<CuJointStomp<N, M>>),
    PathOptimizer(Box<CuJointPathOptimizer<N, M>>),
    Direct(Box<CuJointDirect<N, M>>),
}

macro_rules! each {
    ($self:expr, $p:ident => $body:expr) => {
        match $self {
            Self::RrtConnect($p) => $body,
            Self::RrtStar($p) => $body,
            Self::Prm($p) => $body,
            Self::AStar($p) => $body,
            Self::Stomp($p) => $body,
            Self::PathOptimizer($p) => $body,
            Self::Direct($p) => $body,
        }
    };
}

impl<const N: usize, const M: usize> CuAnyPlanner<N, M> {
    /// Builds the planner `kind` for `chain` and `model`.
    pub fn new(
        kind: CuPlannerKind,
        chain: CuKinematicChain<N>,
        model: CuRobotCollisionModel<N, M>,
    ) -> CuResult<Self> {
        Ok(match kind {
            CuPlannerKind::RrtConnect => Self::RrtConnect(Box::new(CuJointPlanner::new(
                chain,
                model,
                Default::default(),
            )?)),
            CuPlannerKind::RrtStar => Self::RrtStar(Box::new(CuJointPlanner::new(
                chain,
                model,
                Default::default(),
            )?)),
            CuPlannerKind::Prm => Self::Prm(Box::new(CuJointPlanner::new(
                chain,
                model,
                Default::default(),
            )?)),
            CuPlannerKind::AStar => Self::AStar(Box::new(CuJointPlanner::new(
                chain,
                model,
                Default::default(),
            )?)),
            CuPlannerKind::Stomp => Self::Stomp(Box::new(CuJointPlanner::new(
                chain,
                model,
                Default::default(),
            )?)),
            CuPlannerKind::PathOptimizer => Self::PathOptimizer(Box::new(CuJointPlanner::new(
                chain,
                model,
                Default::default(),
            )?)),
            CuPlannerKind::Direct => Self::Direct(Box::new(CuJointPlanner::new(
                chain,
                model,
                Default::default(),
            )?)),
        })
    }

    /// Which planner this is.
    pub fn kind(&self) -> CuPlannerKind {
        match self {
            Self::RrtConnect(_) => CuPlannerKind::RrtConnect,
            Self::RrtStar(_) => CuPlannerKind::RrtStar,
            Self::Prm(_) => CuPlannerKind::Prm,
            Self::AStar(_) => CuPlannerKind::AStar,
            Self::Stomp(_) => CuPlannerKind::Stomp,
            Self::PathOptimizer(_) => CuPlannerKind::PathOptimizer,
            Self::Direct(_) => CuPlannerKind::Direct,
        }
    }

    /// See [`CuJointPlanner::begin`].
    pub fn begin(&mut self, request: &CuJointPlanRequest<N>, seed: u64) -> CuResult<CuPlanStatus> {
        each!(self, p => p.begin(request, seed))
    }

    /// See [`CuJointPlanner::step`].
    pub fn step(&mut self, budget: u32) -> CuResult<CuPlanStatus> {
        each!(self, p => p.step(budget))
    }

    /// See [`CuJointPlanner::write_best`].
    pub fn write_best(&self, out: &mut CuJointPlanPath<N>) -> CuPlanStatus {
        each!(self, p => p.write_best(out))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cu29::units::si::angle::degree;
    use cu29::units::si::f64::Angle;
    use repose_collision::igus_rebel_collision_model;
    use repose_kinematics::CuJointPosition;
    use repose_kinematics::igus_rebel_chain;

    fn deg(values: [f64; 6]) -> [CuJointPosition; 6] {
        values.map(|d| CuJointPosition::Revolute(Angle::new::<degree>(d)))
    }

    #[test]
    fn test_every_planner_builds_runs_and_round_trips() {
        let request = CuJointPlanRequest {
            start: deg([-1.1, 45.5, 60.0, -3.3, -80.9, 28.7]),
            goal: deg([0.0, 30.0, 60.0, 0.0, 60.0, 0.0]),
            ..Default::default()
        };
        for kind in CuPlannerKind::ALL {
            let name = ron::to_string(&kind).unwrap();
            assert_eq!(ron::from_str::<CuPlannerKind>(&name).unwrap(), kind);
            let mut planner =
                CuAnyPlanner::<6>::new(kind, igus_rebel_chain(), igus_rebel_collision_model())
                    .unwrap();
            assert_eq!(planner.kind(), kind);
            let mut status = planner.begin(&request, 1).unwrap();
            for _ in 0..400 {
                if status.is_final() {
                    break;
                }
                status = planner.step(64).unwrap();
            }
            let mut path = CuJointPlanPath::<6>::default();
            let best = planner.write_best(&mut path);
            assert!(best.has_path() || best.is_final(), "{kind:?}: {best:?}");
        }
        for (name, kind) in [
            ("rrt_connect", CuPlannerKind::RrtConnect),
            ("a_star", CuPlannerKind::AStar),
        ] {
            assert_eq!(ron::from_str::<CuPlannerKind>(name).unwrap(), kind);
        }
    }
}
