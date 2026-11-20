//! STOMP: stochastic trajectory optimization, anytime and seeded.
//!
//! Like [`CuJointPathOptimizer`](crate::CuJointPathOptimizer) the
//! path is a fixed number of waypoints seeded on the straight joint line, but
//! updates need no gradients. Each iteration draws `rollouts` smooth noisy
//! copies of the path, scores every waypoint of every rollout by its clearance
//! penalty plus a local acceleration (smoothness) cost, weights the noise per
//! waypoint by `exp(-sensitivity · normalized cost)`, and adds the smoothed
//! weighted noise to the path. That copes with costs that are flat or
//! discontinuous where a gradient would vanish. The shortest collision-free
//! path seen so far is the result. All randomness comes from the job seed.

use crate::CuJointPlanPath;
use crate::CuJointPlanRequest;
use crate::CuJointPlanner;
use crate::CuJointPlannerTask;
use crate::CuPathConstraintSet;
use crate::CuPlanStatus;
use crate::CuPlannerKind;
use crate::MAX_WAYPOINTS;
use crate::common::check_endpoints;
use crate::common::length_quality;
use crate::common::write_header;
use crate::validity::CuSceneValidity;
use crate::validity::joint_distance;
use bincode::Decode;
use bincode::Encode;
use cu_rng::prelude::*;
use cu29::cutask_anytime::Quality;
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

/// Rollout capacity per iteration.
pub const STOMP_MAX_ROLLOUTS: usize = 16;

/// Parameters of [`CuJointStomp`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Encode, Decode, Reflect)]
#[serde(default, deny_unknown_fields)]
pub struct CuJointStompParams {
    /// Path waypoints including start and goal (`3..=MAX_WAYPOINTS`).
    pub waypoints: u32,
    /// Noisy rollouts per iteration (`2..=STOMP_MAX_ROLLOUTS`).
    pub rollouts: u32,
    /// Standard deviation of the exploration noise (dimensionless,
    /// metric-scaled).
    pub noise: f64,
    /// Exponent scale turning normalized rollout costs into weights
    /// (dimensionless).
    pub sensitivity: f64,
    /// Clearance below which the obstacle cost applies.
    pub clearance: Length,
    /// Weight of the obstacle cost (dimensionless, `>= 0`).
    pub obstacle_weight: f64,
    /// Weight of the smoothness cost (dimensionless, `>= 0`).
    pub smoothness_weight: f64,
    /// Iterations after which the job ends.
    pub max_iterations: u32,
    /// Resolution of the segment collision checks.
    pub collision_check_step: CuJointResolution,
    /// Scaling of dimensionless joint-space distances and costs.
    pub metric: CuJointMetric,
}

impl Default for CuJointStompParams {
    fn default() -> Self {
        Self {
            waypoints: 12,
            rollouts: 8,
            noise: 0.15,
            sensitivity: 10.0,
            clearance: Length::new::<meter>(0.05),
            obstacle_weight: 40.0,
            smoothness_weight: 1.0,
            max_iterations: 200,
            collision_check_step: CuJointResolution::default(),
            metric: CuJointMetric::default(),
        }
    }
}

impl CuJointStompParams {
    /// Checks every parameter, the metric, and the collision-check resolution.
    pub fn validate(&self) -> CuResult<()> {
        self.metric.validate()?;
        self.collision_check_step.metric().validate()?;
        let valid = {
            let positive = |v: f64| v.is_finite() && v > 0.0;
            (3..=MAX_WAYPOINTS as u32).contains(&self.waypoints)
                && (2..=STOMP_MAX_ROLLOUTS as u32).contains(&self.rollouts)
                && positive(self.noise)
                && positive(self.sensitivity)
                && positive(self.clearance.get::<meter>())
                && self.obstacle_weight.is_finite()
                && self.obstacle_weight >= 0.0
                && self.smoothness_weight.is_finite()
                && self.smoothness_weight >= 0.0
                && self.max_iterations > 0
        };
        if valid {
            Ok(())
        } else {
            Err(CuError::from(
                "repose-motion-planning: invalid STOMP parameters",
            ))
        }
    }
}

/// Stochastic trajectory optimizer.
///
/// Built with [`CuJointPlanner::new`] from robot models and [`CuJointStompParams`];
/// runs in [`CuJointStompTask`].
pub struct CuJointStomp<const N: usize, const M: usize = N> {
    validity: CuSceneValidity<N, M>,
    params: CuJointStompParams,
    path: [[f64; N]; MAX_WAYPOINTS],
    kinds: [CuJointPosition; N],
    scales: [f64; N],
    lo: [f64; N],
    hi: [f64; N],
    is_fixed: [bool; N],
    best: [[CuJointPosition; N]; MAX_WAYPOINTS],
    count: usize,
    best_cost: f64,
    straight_cost: f64,
    rng: CuRng,
    iterations: u32,
    status: CuPlanStatus,
}

/// Approximately standard normal sample (Irwin–Hall with four uniforms).
fn normal(rng: &mut CuRng) -> f64 {
    let sum: f64 = (0..4).map(|_| rng.random::<f64>()).sum();
    (sum - 2.0) * libm::sqrt(3.0)
}

/// In-place `[1/4, 1/2, 1/4]` smoothing along the path, endpoints fixed at zero.
fn smooth<const N: usize>(values: &mut [[f64; N]], count: usize) {
    if count < 3 {
        if count >= 1 {
            values[0] = [0.0; N];
            values[count - 1] = [0.0; N];
        }
        return;
    }
    let mut prev = values[0];
    for i in 1..count - 1 {
        let curr = values[i];
        let next = values[i + 1];
        for j in 0..N {
            values[i][j] = 0.25 * prev[j] + 0.5 * curr[j] + 0.25 * next[j];
        }
        prev = curr;
    }
    values[0] = [0.0; N];
    values[count - 1] = [0.0; N];
}

/// [`CuJointPlannerTask`] running [`CuJointStomp`].
pub type CuJointStompTask<const N: usize, const M: usize = N> =
    CuJointPlannerTask<CuJointStomp<N, M>, N, M>;

impl<const N: usize, const M: usize> CuJointStomp<N, M> {
    /// The validated planner parameters.
    pub const fn params(&self) -> &CuJointStompParams {
        &self.params
    }

    fn build(
        chain: CuKinematicChain<N>,
        model: CuRobotCollisionModel<N, M>,
        params: CuJointStompParams,
    ) -> Self {
        Self {
            validity: CuSceneValidity::new(chain, model, params.collision_check_step),
            params,
            path: [[0.0; N]; MAX_WAYPOINTS],
            kinds: [CuJointPosition::default(); N],
            scales: [1.0; N],
            lo: [0.0; N],
            hi: [0.0; N],
            is_fixed: [false; N],
            best: [[CuJointPosition::default(); N]; MAX_WAYPOINTS],
            count: 0,
            best_cost: f64::INFINITY,
            straight_cost: 0.0,
            rng: CuRng::from_seed(0),
            iterations: 0,
            status: CuPlanStatus::InvalidRequest,
        }
    }

    /// Iterations run on the current job.
    pub const fn iterations(&self) -> u32 {
        self.iterations
    }

    fn configuration(&self, x: &[f64; N]) -> [CuJointPosition; N] {
        core::array::from_fn(|j| self.kinds[j].with_si_value(x[j] * self.scales[j]))
    }

    fn obstacle_cost(&self, x: &[f64; N], eps: f64, obstacle_weight: f64) -> f64 {
        if obstacle_weight == 0.0 {
            return 0.0;
        }
        match self.validity.clearance(&self.configuration(x)) {
            Some(c) if c >= eps => 0.0,
            Some(c) => obstacle_weight * (eps - c) * (eps - c) / eps,
            None => obstacle_weight * eps,
        }
    }

    fn path_cost(&self) -> f64 {
        let mut total = 0.0_f64;
        for w in self.path[..self.count].windows(2) {
            let mut sum = 0.0_f64;
            for ((&a, &b), &fixed) in w[0].iter().zip(&w[1]).zip(&self.is_fixed) {
                if !fixed {
                    let d = b - a;
                    sum += d * d;
                }
            }
            total += libm::sqrt(sum);
        }
        total
    }

    fn record_if_better(&mut self) {
        if self.count < 2 {
            return;
        }
        let cost = self.path_cost();
        if !cost.is_finite() || cost >= self.best_cost {
            return;
        }
        let mut q = [[CuJointPosition::default(); N]; MAX_WAYPOINTS];
        q[0] = self.configuration(&self.path[0]);
        for (slot, x) in q[1..self.count - 1]
            .iter_mut()
            .zip(&self.path[1..self.count - 1])
        {
            let qi = self.configuration(x);
            if !self.validity.is_configuration_free(&qi) {
                return;
            }
            *slot = qi;
        }
        q[self.count - 1] = self.configuration(&self.path[self.count - 1]);
        if !(0..self.count - 1).all(|i| self.validity.is_segment_interior_free(&q[i], &q[i + 1])) {
            return;
        }
        self.best_cost = cost;
        self.best[..self.count].copy_from_slice(&q[..self.count]);
    }

    // Rollout, waypoint and joint indices address several parallel arrays;
    // explicit indices read clearer than zipped iterators here.
    #[allow(clippy::needless_range_loop)]
    fn iterate(&mut self) {
        let count = self.count;
        if count < 3 {
            return;
        }
        let k = self.params.rollouts as usize;
        let noise_scale = self.params.noise;
        let sensitivity = self.params.sensitivity;
        let smoothness_weight = self.params.smoothness_weight;
        let obstacle_weight = self.params.obstacle_weight;
        let eps = self.params.clearance.get::<meter>();
        let mut noise = [[[0.0_f64; N]; MAX_WAYPOINTS]; STOMP_MAX_ROLLOUTS];
        let mut costs = [[0.0_f64; MAX_WAYPOINTS]; STOMP_MAX_ROLLOUTS];
        for r in 0..k {
            for i in 1..count - 1 {
                for j in 0..N {
                    noise[r][i][j] = noise_scale * normal(&mut self.rng);
                }
            }
            smooth(&mut noise[r], count);
            for i in 1..count - 1 {
                let mut x = self.path[i];
                for j in 0..N {
                    x[j] = if self.is_fixed[j] {
                        0.0
                    } else {
                        (x[j] + noise[r][i][j]).clamp(self.lo[j], self.hi[j])
                    };
                }
                let mut acceleration = 0.0;
                for j in 0..N {
                    let a = (self.path[i - 1][j] + noise[r][i - 1][j]) - 2.0 * x[j]
                        + (self.path[i + 1][j] + noise[r][i + 1][j]);
                    acceleration += a * a;
                }
                costs[r][i] =
                    self.obstacle_cost(&x, eps, obstacle_weight) + smoothness_weight * acceleration;
            }
        }
        let mut update = [[0.0_f64; N]; MAX_WAYPOINTS];
        for i in 1..count - 1 {
            let (lo, hi) = (0..k).fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), r| {
                (lo.min(costs[r][i]), hi.max(costs[r][i]))
            });
            let span = (hi - lo).max(1e-12);
            let mut weights = [0.0_f64; STOMP_MAX_ROLLOUTS];
            let mut total = 0.0;
            for r in 0..k {
                weights[r] = libm::exp(-sensitivity * (costs[r][i] - lo) / span);
                total += weights[r];
            }
            for r in 0..k {
                let weight = weights[r] / total;
                for j in 0..N {
                    update[i][j] += weight * noise[r][i][j];
                }
            }
        }
        smooth(&mut update, count);
        for i in 1..count - 1 {
            let slot = &mut self.path[i];
            for j in 0..N {
                slot[j] = if self.is_fixed[j] {
                    0.0
                } else {
                    (slot[j] + update[i][j]).clamp(self.lo[j], self.hi[j])
                };
            }
        }
    }
}

impl<const N: usize, const M: usize> CuJointPlanner<N, M> for CuJointStomp<N, M> {
    const KIND: CuPlannerKind = CuPlannerKind::Optimization;

    type Params = CuJointStompParams;

    fn new(
        chain: CuKinematicChain<N>,
        collision_model: CuRobotCollisionModel<N, M>,
        params: CuJointStompParams,
    ) -> CuResult<Self> {
        params.validate()?;
        Ok(Self::build(chain, collision_model, params))
    }

    fn begin(&mut self, request: &CuJointPlanRequest<N>, seed: u64) -> CuResult<CuPlanStatus> {
        self.rng.reseed(seed);
        self.iterations = 0;
        self.best_cost = f64::INFINITY;
        self.count = self.params.waypoints as usize;
        self.kinds = request.start;
        self.straight_cost = joint_distance(&request.start, &request.goal, self.params.metric);
        self.status = check_endpoints(&mut self.validity, request, self.params.metric);
        if self.status != CuPlanStatus::Searching {
            return Ok(self.status);
        }
        let last = (self.count - 1) as f64;
        let joints = self.validity.chain().joints();
        for (j, spec) in joints.iter().enumerate() {
            let fixed = matches!(self.kinds[j], CuJointPosition::Fixed);
            let scale = self.params.metric.scale(self.kinds[j]);
            self.is_fixed[j] = fixed;
            self.scales[j] = scale;
            if fixed {
                self.lo[j] = 0.0;
                self.hi[j] = 0.0;
                for row in &mut self.path[..self.count] {
                    row[j] = 0.0;
                }
            } else {
                let limits = spec.limits;
                self.lo[j] = limits.lower().si_value() / scale;
                self.hi[j] = limits.upper().si_value() / scale;
                let a = request.start[j].si_value() / scale;
                let b = request.goal[j].si_value() / scale;
                let delta = b - a;
                for (i, row) in self.path[..self.count].iter_mut().enumerate() {
                    let t = i as f64 / last;
                    row[j] = a + t * delta;
                }
            }
        }
        // A free straight line is already the shortest path.
        self.record_if_better();
        if self.best_cost.is_finite() {
            self.status = CuPlanStatus::Converged;
        }
        Ok(self.status)
    }

    fn step(&mut self, budget: u32) -> CuResult<CuPlanStatus> {
        for _ in 0..budget {
            if self.status.is_final() {
                break;
            }
            self.iterate();
            self.iterations += 1;
            self.record_if_better();
            let done = self.iterations >= self.params.max_iterations;
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
        length_quality(self.straight_cost, self.best_cost)
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

crate::impl_planner_type_path!(CuJointStomp, "repose_motion_planning::CuJointStomp");

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
        let params = CuJointStompParams {
            waypoints: 5,
            ..Default::default()
        };
        let mut planner = CuJointStomp::new(chain, model, params).unwrap();
        let mut req = CuJointPlanRequest {
            start: cfg(-0.3, 0.0, 0.0),
            goal: cfg(0.3, 0.0, 0.0),
            ..Default::default()
        };
        assert_eq!(planner.begin(&req, 7).unwrap(), CuPlanStatus::Converged);
        let initial_cost = planner.best_cost;
        let initial_best = planner.best;
        assert!(libm::fabs(initial_cost - 0.6) < 1e-12);

        // Worse (longer) path must not overwrite `best`.
        planner.path[2] = [0.0, 0.25, 0.0, 0.0];
        planner.record_if_better();
        assert_eq!(planner.best_cost, initial_cost);
        assert_eq!(planner.best[..5], initial_best[..5]);

        // Equal-cost path must not overwrite `best`.
        planner.path[2] = [0.05, 0.0, 0.0, 0.0];
        planner.record_if_better();
        assert_eq!(planner.best_cost, initial_cost);
        assert_eq!(planner.best[..5], initial_best[..5]);

        // Non-finite path cost must be ignored safely.
        planner.path[2] = [0.0, f64::NAN, 0.0, 0.0];
        planner.record_if_better();
        assert_eq!(planner.best_cost, initial_cost);
        assert_eq!(planner.best[..5], initial_best[..5]);

        // Colliding lower-cost path must still be rejected.
        let obstacle = CuCollisionShape::Sphere(CuSphere::new([m(0.0); 3], m(0.1)).unwrap());
        req.scene.add_obstacle(obstacle, None).unwrap();
        assert_eq!(planner.begin(&req, 7).unwrap(), CuPlanStatus::Searching);
        planner.best_cost = 10.0;
        let sentinel = cfg(0.77, -0.88, 0.99);
        planner.best[0] = sentinel;
        planner.record_if_better();
        assert_eq!(planner.best_cost, 10.0);
        assert_eq!(planner.best[0], sentinel);

        // Free detour cheaper than 10.0 is accepted and only writes `..count`.
        planner.best[planner.count] = sentinel;
        planner.path[1] = [-0.15, 0.2, 0.0, 0.0];
        planner.path[2] = [0.0, 0.25, 0.0, 0.0];
        planner.path[3] = [0.15, 0.2, 0.0, 0.0];
        planner.record_if_better();
        assert!(planner.best_cost < 1.0);
        assert_eq!(planner.best[2], cfg(0.0, 0.25, 0.0));
        assert_eq!(planner.best[planner.count], sentinel);
    }

    #[test]
    fn test_converged_obstacle_detour_seeded_determinism_and_invariants() {
        let (chain, model) = gantry_with_fixed();
        let params = CuJointStompParams {
            waypoints: 9,
            rollouts: 8,
            noise: 0.4,
            sensitivity: 10.0,
            clearance: m(0.06),
            obstacle_weight: 40.0,
            smoothness_weight: 1.0,
            max_iterations: 150,
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

        let mut p1 = CuJointStomp::new(chain, model, params).unwrap();
        let mut p2 = CuJointStomp::new(chain, model, params).unwrap();
        let path_tail = [0.123, -0.456, 0.789, 0.321];
        let best_tail = cfg(0.42, -0.42, 0.13);
        for slot in &mut p1.path[params.waypoints as usize..] {
            *slot = path_tail;
        }
        for slot in &mut p1.best[params.waypoints as usize..] {
            *slot = best_tail;
        }

        let mut s1 = p1.begin(&req, 42).unwrap();
        let mut s2 = p2.begin(&req, 42).unwrap();
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
            assert_eq!(*slot, path_tail);
        }
        for slot in &p1.best[params.waypoints as usize..] {
            assert_eq!(*slot, best_tail);
        }
    }
}
