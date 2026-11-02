//! The one registry of Repose joint planners: [`CuPlannerKind`] names every
//! planner (its RON spelling) and [`CuAnyPlanner`] runs any of them through
//! the [`CuJointPlanner`] contract. Adding a planner means implementing the
//! trait and adding one line to the `planners!` list below; applications
//! select it by name.

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

/// Generates [`CuPlannerKind`], [`CuAnyPlanner`] and all their dispatch from
/// one list, so a planner is added in exactly one line.
macro_rules! planners {
    ($($(#[$doc:meta])* $variant:ident => $ty:ident),+ $(,)?) => {
        /// Every Repose joint planner, by RON name (snake_case of the variant).
        #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, Reflect)]
        #[serde(rename_all = "snake_case")]
        pub enum CuPlannerKind {
            #[default]
            $($(#[$doc])* $variant,)+
        }

        impl CuPlannerKind {
            /// Every variant, in declaration order.
            pub const ALL: &'static [Self] = &[$(Self::$variant),+];
        }

        /// Any Repose joint planner with default parameters, chosen at runtime.
        pub enum CuAnyPlanner<const N: usize, const M: usize = N> {
            $($(#[$doc])* $variant(Box<$ty<N, M>>),)+
        }

        impl<const N: usize, const M: usize> CuAnyPlanner<N, M> {
            /// Builds the planner `kind` for `chain` and `model`.
            pub fn new(
                kind: CuPlannerKind,
                chain: CuKinematicChain<N>,
                model: CuRobotCollisionModel<N, M>,
            ) -> CuResult<Self> {
                Ok(match kind {
                    $(CuPlannerKind::$variant => Self::$variant(Box::new(
                        CuJointPlanner::new(chain, model, Default::default())?,
                    )),)+
                })
            }

            /// Which planner this is.
            pub fn kind(&self) -> CuPlannerKind {
                match self {
                    $(Self::$variant(_) => CuPlannerKind::$variant,)+
                }
            }

            /// See [`CuJointPlanner::begin`].
            pub fn begin(&mut self, request: &CuJointPlanRequest<N>, seed: u64) -> CuResult<CuPlanStatus> {
                match self {
                    $(Self::$variant(p) => p.begin(request, seed),)+
                }
            }

            /// See [`CuJointPlanner::step`].
            pub fn step(&mut self, budget: u32) -> CuResult<CuPlanStatus> {
                match self {
                    $(Self::$variant(p) => p.step(budget),)+
                }
            }

            /// See [`CuJointPlanner::write_best`].
            pub fn write_best(&self, out: &mut CuJointPlanPath<N>) -> CuPlanStatus {
                match self {
                    $(Self::$variant(p) => p.write_best(out),)+
                }
            }
        }
    };
}

// The single list of planners. The first entry is the default.
planners! {
    /// Bidirectional RRT, feasibility-first (`rrt_connect`).
    RrtConnect => CuJointRrtConnect,
    /// Asymptotically optimal RRT* with rewiring and shortcutting (`rrt_star`).
    RrtStar => CuJointRrtStar,
    /// Probabilistic roadmap (`prm`).
    Prm => CuJointPrm,
    /// A* on a joint-space lattice (`a_star`).
    AStar => CuJointAStar,
    /// Stochastic trajectory optimization (`stomp`).
    Stomp => CuJointStomp,
    /// Gradient path optimizer, CHOMP-style (`path_optimizer`).
    PathOptimizer => CuJointPathOptimizer,
    /// Straight joint line, collision-checked (`direct`).
    Direct => CuJointDirect,
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
        for &kind in CuPlannerKind::ALL {
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
