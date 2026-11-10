//! The one registry of Repose joint planners: [`CuPlannerChoice`] names every
//! planner (its RON spelling) and [`CuAnyPlanner`] is a [`CuJointPlanner`]
//! that runs whichever one is chosen, directly or inside
//! `CuJointPlannerTask<CuAnyPlanner<N>, N>` with `"params": "rrt_star"`. Adding a planner means implementing the
//! trait and adding one line to the `planners!` list below; applications
//! select it by name.

#[cfg(not(feature = "std"))]
extern crate alloc;
#[cfg(not(feature = "std"))]
use alloc::boxed::Box;

use cu29::cutask_anytime::Quality;
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
use repose_motion_planning::CuPathConstraintSet;
use repose_motion_planning::CuPlanStatus;
use repose_motion_planning::CuPlannerKind;
use serde::Deserialize;
use serde::Serialize;

/// Generates [`CuPlannerChoice`], [`CuAnyPlanner`] and all their dispatch from
/// one list, so a planner is added in exactly one line.
macro_rules! planners {
    ($($(#[$doc:meta])* $variant:ident => $ty:ident),+ $(,)?) => {
        /// Every Repose joint planner, by RON name (snake_case of the variant).
        #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, Reflect)]
        #[serde(rename_all = "snake_case")]
        pub enum CuPlannerChoice {
            #[default]
            $($(#[$doc])* $variant,)+
        }

        impl CuPlannerChoice {
            /// Every variant, in declaration order.
            pub const ALL: &'static [Self] = &[$(Self::$variant),+];
        }

        /// Any Repose joint planner with default parameters, chosen at runtime.
        pub enum CuAnyPlanner<const N: usize, const M: usize = N> {
            $($(#[$doc])* $variant(Box<$ty<N, M>>),)+
        }

        impl<const N: usize, const M: usize> CuAnyPlanner<N, M> {
            /// Which planner this is.
            pub fn choice(&self) -> CuPlannerChoice {
                match self {
                    $(Self::$variant(_) => CuPlannerChoice::$variant,)+
                }
            }
        }

        impl<const N: usize, const M: usize> CuJointPlanner<N, M> for CuAnyPlanner<N, M> {
            /// Varies with the choice; reported as sampling.
            const KIND: CuPlannerKind = CuPlannerKind::Sampling;

            /// The planner to run, by name.
            type Params = CuPlannerChoice;

            fn new(
                chain: CuKinematicChain<N>,
                collision_model: CuRobotCollisionModel<N, M>,
                params: CuPlannerChoice,
            ) -> CuResult<Self> {
                Ok(match params {
                    $(CuPlannerChoice::$variant => Self::$variant(Box::new(
                        CuJointPlanner::new(chain, collision_model, Default::default())?,
                    )),)+
                })
            }

            fn begin(&mut self, request: &CuJointPlanRequest<N>, seed: u64) -> CuResult<CuPlanStatus> {
                match self {
                    $(Self::$variant(p) => p.begin(request, seed),)+
                }
            }

            fn step(&mut self, budget: u32) -> CuResult<CuPlanStatus> {
                match self {
                    $(Self::$variant(p) => p.step(budget),)+
                }
            }

            fn write_best(&self, out: &mut CuJointPlanPath<N>) -> CuPlanStatus {
                match self {
                    $(Self::$variant(p) => p.write_best(out),)+
                }
            }

            fn quality(&self) -> Quality {
                match self {
                    $(Self::$variant(p) => p.quality(),)+
                }
            }

            fn set_path_constraints(
                &mut self,
                constraints: &CuPathConstraintSet,
            ) -> CuResult<()> {
                match self {
                    $(Self::$variant(p) => p.set_path_constraints(constraints),)+
                }
            }

            fn clear_path_constraints(&mut self) {
                match self {
                    $(Self::$variant(p) => p.clear_path_constraints(),)+
                }
            }
        }
    };
}

repose_motion_planning::impl_planner_type_path!(CuAnyPlanner, "repose::planners::CuAnyPlanner");

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
        for &kind in CuPlannerChoice::ALL {
            let name = ron::to_string(&kind).unwrap();
            assert_eq!(ron::from_str::<CuPlannerChoice>(&name).unwrap(), kind);
            let mut planner =
                CuAnyPlanner::<6>::new(igus_rebel_chain(), igus_rebel_collision_model(), kind)
                    .unwrap();
            assert_eq!(planner.choice(), kind);
            assert!(planner.quality().value >= 0.0);
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
            ("rrt_connect", CuPlannerChoice::RrtConnect),
            ("a_star", CuPlannerChoice::AStar),
        ] {
            assert_eq!(ron::from_str::<CuPlannerChoice>(name).unwrap(), kind);
        }
    }

    #[test]
    fn test_planner_task_selects_by_name_in_ron() {
        use repose_motion_planning::CuJointPlannerParams;
        use repose_motion_planning::CuJointPlannerTask;
        // The node config exactly as an application's copperconfig.ron writes it.
        let config: ComponentConfig = ron::from_str(
            r#"#![enable(unwrap_newtypes)] {"chain": "igus_rebel", "collision_model": "igus_rebel", "params": "rrt_star"}"#,
        )
        .unwrap();
        let params: CuJointPlannerParams<CuPlannerChoice, 6, 6> =
            config.deserialize_into().unwrap();
        assert_eq!(params.params, CuPlannerChoice::RrtStar);
        assert!(CuJointPlannerTask::<CuAnyPlanner<6>, 6>::from_params(params, 7).is_ok());
    }
}

#[cfg(test)]
mod constraint_job_tests {
    use super::*;
    use cu29::cutask_anytime::AnytimeStatus;
    use cu29::cutask_anytime::CuAnytimeTask;
    use cu29::prelude::CuContext;
    use cu29::prelude::CuMsg;
    use cu29::units::si::angle::degree;
    use cu29::units::si::f64::Angle;
    use repose_collision::igus_rebel_collision_model;
    use repose_kinematics::CuJointPosition;
    use repose_kinematics::igus_rebel_chain;
    use repose_motion_planning::{
        CuJointPlannerTask, CuOrientationConstraint, CuPathConstraint, CuPathConstraintSet,
        CuToolAxis,
    };

    fn check_constraint_changes(kind: CuPlannerChoice) {
        let start = [-1.1, 45.5, 60.0, -3.3, -80.9, 28.7]
            .map(|d| CuJointPosition::Revolute(Angle::new::<degree>(d)));
        let mut goal = start;
        goal[0] = CuJointPosition::Revolute(Angle::new::<degree>(-1.0));
        let request = CuJointPlanRequest {
            start,
            goal,
            ..Default::default()
        };
        let mut constraints = CuPathConstraintSet::new();
        constraints
            .push(CuPathConstraint::Orientation(
                CuOrientationConstraint::new(
                    CuToolAxis::Z,
                    [0.0, 0.0, 1.0],
                    Angle::new::<degree>(1.0),
                )
                .unwrap(),
            ))
            .unwrap();
        let mut permissive = CuPathConstraintSet::new();
        permissive
            .push(CuPathConstraint::Orientation(
                CuOrientationConstraint::new(
                    CuToolAxis::Z,
                    [0.0, 0.0, 1.0],
                    Angle::new::<degree>(180.0),
                )
                .unwrap(),
            ))
            .unwrap();
        {
            let mut planner =
                CuAnyPlanner::<6>::new(igus_rebel_chain(), igus_rebel_collision_model(), kind)
                    .unwrap();
            let mut path = CuJointPlanPath::default();
            for clear in [false, true] {
                planner.set_path_constraints(&permissive).unwrap();
                let mut status = planner.begin(&request, 1).unwrap();
                for _ in 0..400 {
                    if status.is_final() {
                        break;
                    }
                    status = planner.step(64).unwrap();
                }
                assert!(planner.write_best(&mut path).has_path(), "{kind:?}");
                assert!(
                    planner
                        .set_path_constraints(&CuPathConstraintSet::new())
                        .is_err()
                );
                assert!(planner.write_best(&mut path).has_path());
                if clear {
                    planner.clear_path_constraints();
                } else {
                    planner.set_path_constraints(&constraints).unwrap();
                }
                assert_eq!(
                    planner.write_best(&mut path),
                    CuPlanStatus::InvalidRequest,
                    "{kind:?}"
                );
                assert_eq!(path.len, 0);
                assert_eq!(planner.step(1).unwrap(), CuPlanStatus::InvalidRequest);
                assert_eq!(planner.quality().value, 0.0);
            }
            // Changing constraints between begin and step also discards endpoint assumptions.
            planner.begin(&request, 1).unwrap();
            planner.set_path_constraints(&constraints).unwrap();
            assert_eq!(planner.step(1).unwrap(), CuPlanStatus::InvalidRequest);
            assert_eq!(
                planner.begin(&request, 1).unwrap(),
                CuPlanStatus::InvalidStart
            );
            planner.clear_path_constraints();
            let mut task = CuJointPlannerTask::from_planner(planner, 256, 256, 1).unwrap();
            let (ctx, _clock) = CuContext::new_mock_clock();
            let input = CuMsg::new(Some(request.clone()));
            let mut output = CuMsg::new(None);
            for clear in [false, true] {
                task.set_path_constraints(&permissive).unwrap();
                task.base(&ctx, &input, &mut output).unwrap();
                for _ in 0..400 {
                    if task.status().is_final() {
                        break;
                    }
                    task.refine(&ctx, &mut output).unwrap();
                }
                assert!(output.payload().unwrap().status.has_path(), "{kind:?}");
                if clear {
                    task.clear_path_constraints();
                } else {
                    task.set_path_constraints(&constraints).unwrap();
                }
                assert_eq!(task.status(), CuPlanStatus::default());
                assert_eq!(task.debug_state().effort, 0);
                assert_eq!(task.debug_state().published_cost, f64::INFINITY);
                assert_eq!(task.debug_state().published_quality.value, 0.0);
                assert!(matches!(
                    task.refine(&ctx, &mut output).unwrap(),
                    AnytimeStatus::Aborted
                ));
                assert!(!output.payload().is_some_and(|p| p.status.has_path()));
            }
        }
    }
    macro_rules! constraint_job_test {
        ($name:ident, $kind:ident) => {
            #[test]
            fn $name() {
                check_constraint_changes(CuPlannerChoice::$kind);
            }
        };
    }
    constraint_job_test!(test_discard_rrt_connect_job, RrtConnect);
    constraint_job_test!(test_discard_rrt_star_job, RrtStar);
    constraint_job_test!(test_discard_prm_job, Prm);
    constraint_job_test!(test_discard_astar_job, AStar);
    constraint_job_test!(test_discard_stomp_job, Stomp);
    constraint_job_test!(test_discard_optimizer_job, PathOptimizer);
    constraint_job_test!(test_discard_direct_job, Direct);
}
