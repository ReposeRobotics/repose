//! Joint-path optimizer: non-sampling, deterministic, anytime.
//!
//! A CHOMP-style planner. The path is a fixed number of waypoints, seeded on
//! the straight joint-space line, and every iteration takes one gradient step
//! on
//!
//! ```text
//! cost = w_smooth · ½ Σ |x[i+1] − x[i]|²  +  w_obstacle · Σ max(0, ε − c(x[i]))² / ε
//! ```
//!
//! in metric-normalized joint coordinates, where `c` is the signed clearance
//! of the robot and attached objects against the captured scene and `ε` the
//! target clearance. The obstacle gradient is a forward finite difference of
//! the clearance, so any [`CuSceneValidity`] scene works without analytic
//! distance gradients. After each step the whole path is checked segment by
//! segment; the shortest collision-free path seen so far is the published
//! result. No randomness is involved: the same request always yields the
//! same path.

use crate::CuJointPlanPath;
use crate::CuJointPlanRequest;
use crate::CuJointPlanner;
use crate::CuJointPlannerTask;
use crate::CuPathConstraintSet;
use crate::CuPlanStatus;
use crate::CuPlannerKind;
use crate::MAX_WAYPOINTS;
use crate::common::check_endpoints;
use crate::common::write_header;
use crate::validity::CuSceneValidity;
use crate::validity::joint_distance;
use bincode::Decode;
use bincode::Encode;
use cu29::cutask_anytime::Quality;
use cu29::cutask_anytime::quality_from_f32;
use cu29::prelude::*;
use cu29::units::si::f64::Length;
use cu29::units::si::length::meter;
use repose_collision::CuRobotCollisionModel;
use repose_kinematics::CuJointMetric;
use repose_kinematics::CuJointPosition;
use repose_kinematics::CuJointPositionsPayload;
use repose_kinematics::CuJointResolution;
use repose_kinematics::CuKinematicChain;
use serde::Deserialize;
use serde::Serialize;

/// Parameters of [`CuJointPathOptimizer`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Encode, Decode, Reflect)]
#[serde(default, deny_unknown_fields)]
pub struct CuJointPathOptimizerParams {
    /// Path waypoints including start and goal (`3..=MAX_WAYPOINTS`).
    pub waypoints: u32,
    /// Gradient step per unit of cost gradient (dimensionless, metric-scaled).
    pub learning_rate: f64,
    /// Clearance below which the obstacle term pushes the path away.
    pub clearance: Length,
    /// Weight of the obstacle term (dimensionless, `>= 0`).
    pub obstacle_weight: f64,
    /// Weight of the smoothness term (dimensionless, `>= 0`).
    pub smoothness_weight: f64,
    /// Finite-difference step for the clearance gradient (dimensionless,
    /// metric-scaled).
    pub gradient_step: f64,
    /// Iterations after which the job ends.
    pub max_iterations: u32,
    /// Resolution of the segment collision checks.
    pub collision_check_step: CuJointResolution,
    /// Scaling of dimensionless joint-space distances and costs.
    pub metric: CuJointMetric,
}

impl Default for CuJointPathOptimizerParams {
    fn default() -> Self {
        Self {
            waypoints: 12,
            learning_rate: 0.05,
            clearance: Length::new::<meter>(0.05),
            obstacle_weight: 40.0,
            smoothness_weight: 1.0,
            gradient_step: 1e-3,
            max_iterations: 300,
            collision_check_step: CuJointResolution::default(),
            metric: CuJointMetric::default(),
        }
    }
}

impl CuJointPathOptimizerParams {
    /// Checks every parameter, the metric, and the collision-check resolution.
    pub fn validate(&self) -> CuResult<()> {
        self.metric.validate()?;
        self.collision_check_step.metric().validate()?;
        let valid = {
            let positive = |v: f64| v.is_finite() && v > 0.0;
            (3..=MAX_WAYPOINTS as u32).contains(&self.waypoints)
                && positive(self.learning_rate)
                && positive(self.clearance.get::<meter>())
                && self.obstacle_weight.is_finite()
                && self.obstacle_weight >= 0.0
                && self.smoothness_weight.is_finite()
                && self.smoothness_weight >= 0.0
                && positive(self.gradient_step)
                && self.max_iterations > 0
        };
        if valid {
            Ok(())
        } else {
            Err(CuError::from(
                "repose-motion-planning: invalid path optimizer parameters",
            ))
        }
    }
}

/// Deterministic anytime joint-path optimizer.
///
/// Built with [`CuJointPlanner::new`] from robot models and [`CuJointPathOptimizerParams`];
/// runs in [`CuJointPathOptimizerTask`].
#[derive(Clone)]
pub struct CuJointPathOptimizer<const N: usize, const M: usize = N> {
    validity: CuSceneValidity<N, M>,
    params: CuJointPathOptimizerParams,
    path: [[CuJointPosition; N]; MAX_WAYPOINTS],
    best: [[CuJointPosition; N]; MAX_WAYPOINTS],
    count: usize,
    best_cost: f64,
    straight_cost: f64,
    iterations: u32,
    status: CuPlanStatus,
}

/// [`CuJointPlannerTask`] running [`CuJointPathOptimizer`].
pub type CuJointPathOptimizerTask<const N: usize, const M: usize = N> =
    CuJointPlannerTask<CuJointPathOptimizer<N, M>, N, M>;

impl<const N: usize, const M: usize> CuJointPathOptimizer<N, M> {
    /// The validated planner parameters.
    pub const fn params(&self) -> &CuJointPathOptimizerParams {
        &self.params
    }

    fn build(
        chain: CuKinematicChain<N>,
        model: CuRobotCollisionModel<N, M>,
        params: CuJointPathOptimizerParams,
    ) -> Self {
        Self {
            validity: CuSceneValidity::new(chain, model, params.collision_check_step),
            params,
            path: [[CuJointPosition::default(); N]; MAX_WAYPOINTS],
            best: [[CuJointPosition::default(); N]; MAX_WAYPOINTS],
            count: 0,
            best_cost: f64::INFINITY,
            straight_cost: 0.0,
            iterations: 0,
            status: CuPlanStatus::InvalidRequest,
        }
    }

    /// Iterations run on the current job.
    pub const fn iterations(&self) -> u32 {
        self.iterations
    }

    fn penalty(&self, q: &[CuJointPosition; N]) -> f64 {
        self.penalty_of(self.validity.clearance(q))
    }

    fn penalty_of(&self, clearance: Option<f64>) -> f64 {
        let eps = self.params.clearance.get::<meter>();
        match clearance {
            Some(c) if c >= eps => 0.0,
            Some(c) => self.params.obstacle_weight * (eps - c) * (eps - c) / eps,
            // Unevaluable configurations are as bad as deep penetration.
            None => self.params.obstacle_weight * eps,
        }
    }

    fn path_cost(&self, path: &[[CuJointPosition; N]]) -> f64 {
        if path.len() < 2 {
            return 0.0;
        }
        let metric = self.params.metric;
        let scales: [f64; N] = core::array::from_fn(|j| metric.scale(path[0][j]));
        let mut prev: [f64; N] = core::array::from_fn(|j| path[0][j].si_value() / scales[j]);
        let mut total = 0.0_f64;
        for wp in &path[1..] {
            let mut sum = 0.0_f64;
            let mut curr = [0.0_f64; N];
            for j in 0..N {
                let c = wp[j].si_value() / scales[j];
                curr[j] = c;
                let d = c - prev[j];
                sum += d * d;
            }
            total += libm::sqrt(sum);
            prev = curr;
        }
        total
    }

    fn path_is_free(&self) -> bool {
        (1..self.count - 1).all(|i| self.validity.is_configuration_free(&self.path[i]))
            && (0..self.count - 1).all(|i| {
                self.validity
                    .is_segment_interior_free(&self.path[i], &self.path[i + 1])
            })
    }

    fn record_if_better(&mut self) {
        if self.count < 2 {
            return;
        }
        let cost = self.path_cost(&self.path[..self.count]);
        if !cost.is_finite() || cost >= self.best_cost {
            return;
        }
        if !self.path_is_free() {
            return;
        }
        self.best_cost = cost;
        self.best[..self.count].copy_from_slice(&self.path[..self.count]);
    }

    /// One gradient step on every interior waypoint; returns the largest
    /// normalized joint update.
    #[allow(clippy::needless_range_loop)]
    fn descend(&mut self) -> f64 {
        if self.count < 3 {
            return 0.0;
        }
        let metric = self.params.metric;
        let h = self.params.gradient_step;
        let eps = self.params.clearance.get::<meter>();
        let obstacle_weight = self.params.obstacle_weight;
        let smoothness_weight = self.params.smoothness_weight;
        let learning_rate = self.params.learning_rate;
        let joints = self.validity.chain().joints();

        let mut is_fixed = [false; N];
        let mut scale = [1.0_f64; N];
        let mut inv_scale = [1.0_f64; N];
        let mut lo = [0.0_f64; N];
        let mut hi = [0.0_f64; N];
        for j in 0..N {
            if matches!(self.path[0][j], CuJointPosition::Fixed) {
                is_fixed[j] = true;
            } else {
                let s = metric.scale(self.path[0][j]);
                let limits = joints[j].limits;
                scale[j] = s;
                inv_scale[j] = 1.0 / s;
                lo[j] = limits.lower().si_value() / s;
                hi[j] = limits.upper().si_value() / s;
            }
        }

        let mut largest = 0.0_f64;
        let mut prev_norm: [f64; N] = core::array::from_fn(|j| {
            if is_fixed[j] {
                0.0
            } else {
                self.path[0][j].si_value() * inv_scale[j]
            }
        });
        let mut curr_norm: [f64; N] = core::array::from_fn(|j| {
            if is_fixed[j] {
                0.0
            } else {
                self.path[1][j].si_value() * inv_scale[j]
            }
        });

        for i in 1..self.count - 1 {
            let q = self.path[i];
            let next_q = self.path[i + 1];
            let (flat, base) = if obstacle_weight == 0.0 {
                (true, 0.0)
            } else {
                let clearance = self.validity.clearance(&q);
                // Well clear of every obstacle a finite-difference probe stays in
                // the flat zero-penalty region, so the clearance queries are skipped.
                let flat = clearance.is_some_and(|c| c > 2.0 * eps);
                (
                    flat,
                    if flat {
                        0.0
                    } else {
                        self.penalty_of(clearance)
                    },
                )
            };
            let mut next_norm = [0.0_f64; N];
            let mut updated_q = q;
            let mut probe = q;
            for j in 0..N {
                if is_fixed[j] {
                    continue;
                }
                let x = curr_norm[j];
                let prev = prev_norm[j];
                let after = next_q[j].si_value() * inv_scale[j];
                next_norm[j] = after;
                let mut gradient = smoothness_weight * (2.0 * x - prev - after);
                if !flat {
                    probe[j] = q[j].with_si_value((x + h) * scale[j]);
                    gradient += (self.penalty(&probe) - base) / h;
                    probe[j] = q[j];
                }
                let updated = (x - learning_rate * gradient).clamp(lo[j], hi[j]);
                largest = largest.max(libm::fabs(updated - x));
                updated_q[j] = q[j].with_si_value(updated * scale[j]);
            }
            self.path[i] = updated_q;
            prev_norm = curr_norm;
            curr_norm = next_norm;
        }
        largest
    }
}

impl<const N: usize, const M: usize> CuJointPlanner<N, M> for CuJointPathOptimizer<N, M> {
    const KIND: CuPlannerKind = CuPlannerKind::Optimization;

    type Params = CuJointPathOptimizerParams;

    fn new(
        chain: CuKinematicChain<N>,
        collision_model: CuRobotCollisionModel<N, M>,
        params: CuJointPathOptimizerParams,
    ) -> CuResult<Self> {
        params.validate()?;
        Ok(Self::build(chain, collision_model, params))
    }

    fn begin(&mut self, request: &CuJointPlanRequest<N>, _seed: u64) -> CuResult<CuPlanStatus> {
        self.iterations = 0;
        self.best_cost = f64::INFINITY;
        self.count = self.params.waypoints as usize;
        self.straight_cost = joint_distance(&request.start, &request.goal, self.params.metric);
        self.status = check_endpoints(&mut self.validity, request, self.params.metric);
        if self.status == CuPlanStatus::Searching {
            let last = (self.count - 1) as f64;
            for i in 0..self.count {
                for j in 0..N {
                    self.path[i][j] = request.start[j]
                        .interpolate(request.goal[j], i as f64 / last)
                        .map_err(|_| {
                            CuError::from("repose-motion-planning: joint kinds disagree")
                        })?;
                }
            }
            // The straight line is the cheapest possible path; when it is
            // free there is nothing to optimize.
            self.record_if_better();
            if self.best_cost.is_finite() {
                self.status = CuPlanStatus::Converged;
            }
        }
        Ok(self.status)
    }

    fn step(&mut self, budget: u32) -> CuResult<CuPlanStatus> {
        for _ in 0..budget {
            if self.status.is_final() {
                break;
            }
            let moved = self.descend();
            self.iterations += 1;
            self.record_if_better();
            let done = moved <= 1e-7 || self.iterations >= self.params.max_iterations;
            self.status = match (self.best_cost.is_finite(), done) {
                (true, true) => CuPlanStatus::Converged,
                (true, false) => CuPlanStatus::Improving,
                (false, true) => CuPlanStatus::Exhausted,
                (false, false) => CuPlanStatus::Searching,
            };
        }
        Ok(self.status)
    }

    fn write_best(&self, out: &mut CuJointPlanPath<N>) -> CuPlanStatus {
        write_header(out, &self.validity, self.status, self.iterations);
        if self.status.has_path() {
            for (slot, q) in out.waypoints.iter_mut().zip(&self.best[..self.count]) {
                *slot = CuJointPositionsPayload { positions: *q };
            }
            out.len = self.count as u32;
            out.cost = self.best_cost;
        }
        self.status
    }

    fn quality(&self) -> Quality {
        if !self.best_cost.is_finite() {
            return quality_from_f32(0.0);
        }
        if self.best_cost <= 1e-9 {
            return quality_from_f32(1.0);
        }
        quality_from_f32((self.straight_cost / self.best_cost).clamp(0.0, 1.0) as f32)
    }

    fn set_path_constraints(&mut self, constraints: &CuPathConstraintSet) -> CuResult<()> {
        self.validity.set_path_constraints(constraints)?;
        self.status = CuPlanStatus::InvalidRequest;
        self.iterations = 0;
        self.count = 0;
        self.best_cost = f64::INFINITY;
        Ok(())
    }

    fn clear_path_constraints(&mut self) {
        self.validity.clear_path_constraints();
        self.status = CuPlanStatus::InvalidRequest;
        self.iterations = 0;
        self.count = 0;
        self.best_cost = f64::INFINITY;
    }
}

crate::impl_planner_type_path!(
    CuJointPathOptimizer,
    "repose_motion_planning::CuJointPathOptimizer"
);

#[cfg(test)]
mod tests {
    use super::*;
    use cu29::units::si::angle::radian;
    use cu29::units::si::f64::Angle;
    use cu29::units::si::f64::Velocity;
    use cu29::units::si::velocity::meter_per_second;
    use repose_collision::CuCollisionModelParams;
    use repose_collision::CuCollisionShape;
    use repose_collision::CuLinkShape;
    use repose_collision::CuSphere;
    use repose_kinematics::CuJointAxis;
    use repose_kinematics::CuJointLimits;
    use repose_kinematics::CuJointSpec;
    use repose_kinematics::CuJointType;

    fn m(v: f64) -> Length {
        Length::new::<meter>(v)
    }

    fn gantry_with_fixed() -> (CuKinematicChain<4>, CuRobotCollisionModel<4, 1>) {
        let limits =
            CuJointLimits::prismatic(m(-1.0), m(1.0), Velocity::new::<meter_per_second>(1.0));
        let origin = [m(0.0); 3];
        let rpy = [Angle::new::<radian>(0.0); 3];
        let chain = CuKinematicChain::new([
            CuJointSpec::prismatic(origin, rpy, CuJointAxis::PosX, limits),
            CuJointSpec::prismatic(origin, rpy, CuJointAxis::PosY, limits),
            CuJointSpec::prismatic(origin, rpy, CuJointAxis::PosZ, limits),
            CuJointSpec {
                joint_type: CuJointType::Fixed,
                origin_translation: origin,
                origin_rpy: rpy,
                axis: CuJointAxis::PosZ,
                limits: CuJointLimits::Fixed,
            },
        ]);
        let sphere = CuCollisionShape::Sphere(CuSphere::new([m(0.0); 3], m(0.02)).unwrap());
        let model = CuRobotCollisionModel::new(
            &[CuLinkShape::new(sphere, Some(2))],
            CuCollisionModelParams {
                safety_margin: m(0.0),
                skip_adjacent_links: false,
            },
        )
        .unwrap();
        (chain, model)
    }

    fn cfg(x: f64, y: f64, z: f64) -> [CuJointPosition; 4] {
        [
            m(x).into(),
            m(y).into(),
            m(z).into(),
            CuJointPosition::Fixed,
        ]
    }

    #[test]
    fn test_record_if_better_skips_non_improving_and_rejects_colliding() {
        let (chain, model) = gantry_with_fixed();
        let params = CuJointPathOptimizerParams {
            waypoints: 5,
            ..Default::default()
        };
        let mut planner = CuJointPathOptimizer::new(chain, model, params).unwrap();
        let mut req = CuJointPlanRequest {
            start: cfg(-0.3, 0.0, 0.0),
            goal: cfg(0.3, 0.0, 0.0),
            ..Default::default()
        };
        assert_eq!(planner.begin(&req, 0).unwrap(), CuPlanStatus::Converged);
        let initial_cost = planner.best_cost;
        let initial_best = planner.best;
        assert!(libm::fabs(initial_cost - 0.6) < 1e-12);

        // Worse (longer) free path must not overwrite `best`.
        planner.path[2] = cfg(0.0, 0.25, 0.0);
        planner.record_if_better();
        assert_eq!(planner.best_cost, initial_cost);
        assert_eq!(planner.best[..5], initial_best[..5]);

        // Equal-cost path must not overwrite `best` either.
        planner.path[2] = cfg(0.05, 0.0, 0.0);
        planner.record_if_better();
        assert_eq!(planner.best_cost, initial_cost);
        assert_eq!(planner.best[..5], initial_best[..5]);

        // Non-finite path cost must be ignored safely.
        planner.path[2] = cfg(0.0, f64::NAN, 0.0);
        planner.record_if_better();
        assert_eq!(planner.best_cost, initial_cost);
        assert_eq!(planner.best[..5], initial_best[..5]);

        // Colliding lower-cost path must still be rejected.
        let obstacle = CuCollisionShape::Sphere(CuSphere::new([m(0.0); 3], m(0.1)).unwrap());
        req.scene.add_obstacle(obstacle, None).unwrap();
        assert_eq!(planner.begin(&req, 0).unwrap(), CuPlanStatus::Searching);
        planner.best_cost = 10.0;
        let sentinel = cfg(0.77, -0.88, 0.99);
        planner.best[0] = sentinel;
        // Straight line passes straight through the sphere at the origin (cost = 0.6 < 10.0).
        planner.record_if_better();
        assert_eq!(planner.best_cost, 10.0);
        assert_eq!(planner.best[0], sentinel);

        // Collision-free detour cheaper than 10.0 is accepted and only updates `..count`.
        planner.best[planner.count] = sentinel;
        planner.path[1] = cfg(-0.15, 0.2, 0.0);
        planner.path[2] = cfg(0.0, 0.25, 0.0);
        planner.path[3] = cfg(0.15, 0.2, 0.0);
        planner.record_if_better();
        assert!(planner.best_cost < 1.0);
        assert_eq!(planner.best[..5], planner.path[..5]);
        assert_eq!(planner.best[planner.count], sentinel);
    }

    #[test]
    fn test_converged_obstacle_detour_determinism_and_invariants() {
        let (chain, model) = gantry_with_fixed();
        let params = CuJointPathOptimizerParams {
            waypoints: 9,
            learning_rate: 0.05,
            clearance: m(0.06),
            obstacle_weight: 40.0,
            smoothness_weight: 1.0,
            max_iterations: 400,
            metric: CuJointMetric {
                angular_scale: Angle::new::<radian>(0.5),
                linear_scale: m(0.25),
            },
            ..Default::default()
        };
        let mut req = CuJointPlanRequest {
            start: cfg(-0.3, 0.0, 0.0),
            goal: cfg(0.3, 0.0, 0.0),
            ..Default::default()
        };
        let obstacle =
            CuCollisionShape::Sphere(CuSphere::new([m(0.0), m(0.01), m(0.0)], m(0.08)).unwrap());
        req.scene.add_obstacle(obstacle, None).unwrap();

        let mut p1 = CuJointPathOptimizer::new(chain, model, params).unwrap();
        let mut p2 = CuJointPathOptimizer::new(chain, model, params).unwrap();
        let tail_sentinel = cfg(0.42, -0.42, 0.13);
        for slot in &mut p1.path[params.waypoints as usize..] {
            *slot = tail_sentinel;
        }

        let mut s1 = p1.begin(&req, 0).unwrap();
        let mut s2 = p2.begin(&req, 99).unwrap();
        assert_eq!(s1, CuPlanStatus::Searching);
        assert_eq!(s2, CuPlanStatus::Searching);
        while !s1.is_final() {
            s1 = p1.step(32).unwrap();
        }
        while !s2.is_final() {
            s2 = p2.step(32).unwrap();
        }
        assert_eq!(s1, CuPlanStatus::Converged);
        assert_eq!(s2, CuPlanStatus::Converged);

        let mut out1 = CuJointPlanPath::default();
        let mut out2 = CuJointPlanPath::default();
        p1.write_best(&mut out1);
        p2.write_best(&mut out2);
        assert_eq!(out1, out2);
        assert_eq!(out1.len, params.waypoints);
        assert!(out1.cost.is_finite());
        assert!(out1.cost > p1.straight_cost);
        for wp in out1.waypoints() {
            assert_eq!(wp.positions[3], CuJointPosition::Fixed);
            assert!(p1.validity.is_configuration_free(&wp.positions));
        }
        for slot in &p1.path[params.waypoints as usize..] {
            assert_eq!(*slot, tail_sentinel);
        }
    }
}
