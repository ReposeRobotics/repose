//! PRM: sampling-based probabilistic roadmap.
//!
//! The roadmap starts with the start and goal configurations. Each work unit
//! adds one uniform sample; free samples connect to their nearest roadmap
//! neighbours within `connection_radius` through free segments. After every
//! quantum that changed the roadmap, Dijkstra finds the shortest roadmap
//! path, which is shortcut by line of sight; the job improves as the roadmap
//! densifies and converges when it is full. The roadmap is rebuilt for every
//! job from the job seed, so replay reproduces it.

use crate::CuJointPlanPath;
use crate::CuJointPlanRequest;
use crate::CuJointPlanner;
use crate::CuJointPlannerTask;
use crate::CuPathConstraintSet;
use crate::CuPlanStatus;
use crate::CuPlannerKind;
use crate::common::check_endpoints;
use crate::common::length_quality;
use crate::common::sample_uniform;
use crate::common::write_header;
use crate::common::write_shortcut;
use crate::validity::CuSceneValidity;
use crate::validity::joint_distance;
use crate::validity::joint_distance_sq;
use bincode::Decode;
use bincode::Encode;
use cu_rng::prelude::*;
use cu29::cutask_anytime::Quality;
use cu29::prelude::*;
use repose_collision::CuRobotCollisionModel;
use repose_kinematics::CuJointMetric;
use repose_kinematics::CuJointPosition;
use repose_kinematics::CuJointResolution;
use repose_kinematics::CuKinematicChain;
use serde::Deserialize;
use serde::Serialize;

/// Roadmap node capacity, including start and goal.
pub const PRM_MAX_NODES: usize = 1024;
/// Longest roadmap path (in nodes) that can be reconstructed.
const PRM_MAX_PATH: usize = 256;
/// Edge capacity per roadmap node.
pub const PRM_MAX_NEIGHBORS: usize = 10;

/// Parameters of [`CuJointPrm`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Encode, Decode, Reflect)]
#[serde(default, deny_unknown_fields)]
pub struct CuJointPrmParams {
    /// Joint-space radius for roadmap edges (dimensionless, metric-scaled).
    pub connection_radius: f64,
    /// Neighbours tried per new node (`1..=PRM_MAX_NEIGHBORS`).
    pub neighbors: u32,
    /// Roadmap capacity (`3..=PRM_MAX_NODES`).
    pub max_nodes: u32,
    /// Resolution of the segment collision checks.
    pub collision_check_step: CuJointResolution,
    /// Scaling of dimensionless joint-space distances and costs.
    pub metric: CuJointMetric,
}

impl Default for CuJointPrmParams {
    fn default() -> Self {
        Self {
            connection_radius: 2.0,
            neighbors: 8,
            max_nodes: PRM_MAX_NODES as u32,
            collision_check_step: CuJointResolution::default(),
            metric: CuJointMetric::default(),
        }
    }
}

impl CuJointPrmParams {
    /// Checks every parameter, the metric, and the collision-check resolution.
    pub fn validate(&self) -> CuResult<()> {
        self.metric.validate()?;
        self.collision_check_step.metric().validate()?;
        let valid = {
            self.connection_radius.is_finite()
                && self.connection_radius > 0.0
                && (1..=PRM_MAX_NEIGHBORS as u32).contains(&self.neighbors)
                && (3..=PRM_MAX_NODES as u32).contains(&self.max_nodes)
        };
        if valid {
            Ok(())
        } else {
            Err(CuError::from(
                "repose-motion-planning: invalid PRM parameters",
            ))
        }
    }
}

/// Probabilistic roadmap planner (single query per job).
///
/// Built with [`CuJointPlanner::new`] from robot models and [`CuJointPrmParams`];
/// runs in [`CuJointPrmTask`].
pub struct CuJointPrm<const N: usize, const M: usize = N> {
    validity: CuSceneValidity<N, M>,
    params: CuJointPrmParams,
    nodes: [[CuJointPosition; N]; PRM_MAX_NODES],
    edges: [[u16; PRM_MAX_NEIGHBORS]; PRM_MAX_NODES],
    edge_costs: [[f64; PRM_MAX_NEIGHBORS]; PRM_MAX_NODES],
    degree: [u8; PRM_MAX_NODES],
    len: usize,
    rng: CuRng,
    straight_cost: f64,
    samples: u32,
    status: CuPlanStatus,
    best: CuJointPlanPath<N>,
}

const START: usize = 0;
const GOAL: usize = 1;

/// [`CuJointPlannerTask`] running [`CuJointPrm`].
pub type CuJointPrmTask<const N: usize, const M: usize = N> =
    CuJointPlannerTask<CuJointPrm<N, M>, N, M>;

impl<const N: usize, const M: usize> CuJointPrm<N, M> {
    /// The validated planner parameters.
    pub const fn params(&self) -> &CuJointPrmParams {
        &self.params
    }

    fn build(
        chain: CuKinematicChain<N>,
        model: CuRobotCollisionModel<N, M>,
        params: CuJointPrmParams,
    ) -> Self {
        Self {
            validity: CuSceneValidity::new(chain, model, params.collision_check_step),
            params,
            nodes: [[CuJointPosition::default(); N]; PRM_MAX_NODES],
            edges: [[0; PRM_MAX_NEIGHBORS]; PRM_MAX_NODES],
            edge_costs: [[0.0; PRM_MAX_NEIGHBORS]; PRM_MAX_NODES],
            degree: [0; PRM_MAX_NODES],
            len: 0,
            rng: CuRng::from_seed(0),
            straight_cost: 0.0,
            samples: 0,
            status: CuPlanStatus::InvalidRequest,
            best: CuJointPlanPath::default(),
        }
    }

    /// Roadmap size of the current job.
    pub const fn roadmap_size(&self) -> usize {
        self.len
    }

    fn link(&mut self, a: usize, b: usize, dist: f64) {
        let cap = PRM_MAX_NEIGHBORS.min(self.params.neighbors as usize);
        let deg_a = self.degree[a] as usize;
        let deg_b = self.degree[b] as usize;
        if deg_a < cap && deg_b < cap {
            self.edges[a][deg_a] = b as u16;
            self.edge_costs[a][deg_a] = dist;
            self.degree[a] += 1;
            self.edges[b][deg_b] = a as u16;
            self.edge_costs[b][deg_b] = dist;
            self.degree[b] += 1;
        }
    }

    /// Connects node `idx` to its nearest free neighbours; returns whether
    /// any edge was added.
    fn connect(&mut self, idx: usize) -> bool {
        let k = (self.params.neighbors as usize).min(PRM_MAX_NEIGHBORS);
        let radius_sq = self.params.connection_radius * self.params.connection_radius;
        let mut nearest = [(usize::MAX, f64::INFINITY); PRM_MAX_NEIGHBORS];
        for other in 0..self.len {
            if other == idx {
                continue;
            }
            let d_sq = joint_distance_sq(&self.nodes[idx], &self.nodes[other], self.params.metric);
            if d_sq > radius_sq || d_sq >= nearest[k - 1].1 {
                continue;
            }
            let mut slot = k - 1;
            while slot > 0 && nearest[slot - 1].1 > d_sq {
                nearest[slot] = nearest[slot - 1];
                slot -= 1;
            }
            nearest[slot] = (other, d_sq);
        }
        let mut added = false;
        for &(other, d_sq) in nearest[..k].iter().take_while(|n| n.0 != usize::MAX) {
            if (self.degree[idx] as usize) >= k {
                break;
            }
            if (self.degree[other] as usize) >= k {
                continue;
            }
            if self
                .validity
                .is_segment_interior_free(&self.nodes[idx], &self.nodes[other])
            {
                let before = self.degree[idx];
                self.link(idx, other, libm::sqrt(d_sq));
                added |= self.degree[idx] != before;
            }
        }
        added
    }

    #[inline]
    fn heap_less(dist: &[f64; PRM_MAX_NODES], a: usize, b: usize) -> bool {
        dist[a] < dist[b] || (dist[a] == dist[b] && a < b)
    }

    /// Dijkstra from start to goal; on a shorter path, shortcuts it into `best`.
    fn search(&mut self) {
        let mut dist = [f64::INFINITY; PRM_MAX_NODES];
        let mut prev = [u16::MAX; PRM_MAX_NODES];
        let mut heap = [0u16; PRM_MAX_NODES];
        let mut heap_pos = [u16::MAX; PRM_MAX_NODES];
        let mut heap_len = 1usize;

        dist[START] = 0.0;
        heap[0] = START as u16;
        heap_pos[START] = 0;

        while heap_len > 0 {
            let current = heap[0] as usize;
            heap_pos[current] = u16::MAX;
            heap_len -= 1;
            if heap_len > 0 {
                let last = heap[heap_len] as usize;
                let mut pos = 0usize;
                loop {
                    let left = (pos << 1) + 1;
                    if left >= heap_len {
                        break;
                    }
                    let right = left + 1;
                    let mut best = left;
                    if right < heap_len
                        && Self::heap_less(&dist, heap[right] as usize, heap[left] as usize)
                    {
                        best = right;
                    }
                    let best_node = heap[best] as usize;
                    if Self::heap_less(&dist, best_node, last) {
                        heap[pos] = best_node as u16;
                        heap_pos[best_node] = pos as u16;
                        pos = best;
                    } else {
                        break;
                    }
                }
                heap[pos] = last as u16;
                heap_pos[last] = pos as u16;
            }
            if current == GOAL {
                break;
            }
            for e in 0..self.degree[current] as usize {
                let next = self.edges[current][e] as usize;
                let d = dist[current] + self.edge_costs[current][e];
                if d < dist[next] {
                    dist[next] = d;
                    prev[next] = current as u16;
                    let mut pos = if heap_pos[next] == u16::MAX {
                        let p = heap_len;
                        heap_len += 1;
                        p
                    } else {
                        heap_pos[next] as usize
                    };
                    while pos > 0 {
                        let parent_pos = (pos - 1) >> 1;
                        let parent_node = heap[parent_pos] as usize;
                        if Self::heap_less(&dist, next, parent_node) {
                            heap[pos] = parent_node as u16;
                            heap_pos[parent_node] = pos as u16;
                            pos = parent_pos;
                        } else {
                            break;
                        }
                    }
                    heap[pos] = next as u16;
                    heap_pos[next] = pos as u16;
                }
            }
        }
        if !dist[GOAL].is_finite() || dist[GOAL] >= self.best.cost {
            return;
        }
        let mut raw = [[CuJointPosition::default(); N]; PRM_MAX_PATH];
        let mut len = 0;
        let mut at = GOAL;
        loop {
            if len >= PRM_MAX_PATH {
                if !self.best.cost.is_finite() {
                    self.status = CuPlanStatus::CapacityExceeded;
                }
                return;
            }
            raw[len] = self.nodes[at];
            len += 1;
            if at == START {
                break;
            }
            at = prev[at] as usize;
        }
        raw[..len].reverse();
        let mut candidate = CuJointPlanPath::default();
        if write_shortcut(
            &self.validity,
            &raw[..len],
            self.params.metric,
            &mut candidate,
        ) {
            if candidate.cost < self.best.cost {
                self.best = candidate;
            }
        } else if !self.best.cost.is_finite() {
            self.status = CuPlanStatus::CapacityExceeded;
        }
    }

    fn refresh_status(&mut self) {
        if self.status.is_final() {
            return;
        }
        let full = self.len >= self.params.max_nodes as usize;
        let solved = self.best.cost.is_finite();
        let optimal = solved && self.best.cost <= self.straight_cost + 1e-9;
        self.status = match (solved, full || optimal) {
            (true, true) => CuPlanStatus::Converged,
            (true, false) => CuPlanStatus::Improving,
            (false, true) => CuPlanStatus::Exhausted,
            (false, false) => CuPlanStatus::Searching,
        };
    }
}

impl<const N: usize, const M: usize> CuJointPlanner<N, M> for CuJointPrm<N, M> {
    const KIND: CuPlannerKind = CuPlannerKind::Sampling;

    type Params = CuJointPrmParams;

    fn new(
        chain: CuKinematicChain<N>,
        collision_model: CuRobotCollisionModel<N, M>,
        params: CuJointPrmParams,
    ) -> CuResult<Self> {
        params.validate()?;
        Ok(Self::build(chain, collision_model, params))
    }

    fn begin(&mut self, request: &CuJointPlanRequest<N>, seed: u64) -> CuResult<CuPlanStatus> {
        self.rng.reseed(seed);
        self.samples = 0;
        self.len = 0;
        self.degree = [0; PRM_MAX_NODES];
        self.best = CuJointPlanPath::default();
        self.straight_cost = joint_distance(&request.start, &request.goal, self.params.metric);
        self.status = check_endpoints(&mut self.validity, request, self.params.metric);
        if self.status != CuPlanStatus::Searching {
            return Ok(self.status);
        }
        self.nodes[START] = request.start;
        self.nodes[GOAL] = request.goal;
        self.len = 2;
        if self
            .validity
            .is_segment_interior_free(&request.start, &request.goal)
        {
            self.link(START, GOAL, self.straight_cost);
            self.search();
        }
        self.refresh_status();
        Ok(self.status)
    }

    fn step(&mut self, budget: u32) -> CuResult<CuPlanStatus> {
        let mut changed = false;
        for _ in 0..budget {
            if self.status.is_final() || self.len >= self.params.max_nodes as usize {
                break;
            }
            self.samples += 1;
            let q = sample_uniform(self.validity.chain(), &mut self.rng);
            if !self.validity.is_configuration_free(&q) {
                continue;
            }
            let idx = self.len;
            self.nodes[idx] = q;
            self.degree[idx] = 0;
            self.len += 1;
            changed |= self.connect(idx);
        }
        if changed {
            self.search();
        }
        self.refresh_status();
        Ok(self.status)
    }

    fn write_best(&self, out: &mut CuJointPlanPath<N>) -> CuPlanStatus {
        write_header(out, &self.validity, self.status, self.samples);
        if self.status.has_path() {
            out.waypoints = self.best.waypoints.clone();
            out.len = self.best.len;
            out.cost = self.best.cost;
        }
        self.status
    }

    fn quality(&self) -> Quality {
        length_quality(
            self.straight_cost,
            if self.status.has_path() {
                self.best.cost
            } else {
                f64::INFINITY
            },
        )
    }

    fn set_path_constraints(&mut self, constraints: &CuPathConstraintSet) -> CuResult<()> {
        self.validity.set_path_constraints(constraints)?;
        self.status = CuPlanStatus::InvalidRequest;
        self.samples = 0;
        self.len = 0;
        self.best = CuJointPlanPath::default();
        Ok(())
    }

    fn clear_path_constraints(&mut self) {
        self.validity.clear_path_constraints();
        self.status = CuPlanStatus::InvalidRequest;
        self.samples = 0;
        self.len = 0;
        self.best = CuJointPlanPath::default();
    }
}

crate::impl_planner_type_path!(CuJointPrm, "repose_motion_planning::CuJointPrm");

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(not(feature = "std"))]
    use alloc::boxed::Box;
    use cu29::units::si::angle::radian;
    use cu29::units::si::angular_velocity::radian_per_second;
    use cu29::units::si::f64::Angle;
    use cu29::units::si::f64::AngularVelocity;
    use cu29::units::si::f64::Length;
    use cu29::units::si::f64::Velocity;
    use cu29::units::si::length::meter;
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

    fn rad(v: f64) -> Angle {
        Angle::new::<radian>(v)
    }

    fn planar_chain() -> (CuKinematicChain<2>, CuRobotCollisionModel<2, 1>) {
        let limits =
            CuJointLimits::prismatic(m(-5.0), m(5.0), Velocity::new::<meter_per_second>(1.0));
        let chain = CuKinematicChain::new([
            CuJointSpec::prismatic([m(0.0); 3], [rad(0.0); 3], CuJointAxis::PosX, limits),
            CuJointSpec::prismatic([m(0.0); 3], [rad(0.0); 3], CuJointAxis::PosY, limits),
        ]);
        let shape = CuLinkShape::new(
            CuCollisionShape::Sphere(CuSphere::new([m(0.0); 3], m(0.05)).unwrap()),
            Some(1),
        );
        let model =
            CuRobotCollisionModel::new(&[shape], CuCollisionModelParams::default()).unwrap();
        (chain, model)
    }

    fn xy(x: f64, y: f64) -> [CuJointPosition; 2] {
        [
            CuJointPosition::Prismatic(m(x)),
            CuJointPosition::Prismatic(m(y)),
        ]
    }

    fn mixed_chain() -> (CuKinematicChain<3>, CuRobotCollisionModel<3, 1>) {
        let rev_limits = CuJointLimits::revolute(
            rad(-4.0),
            rad(4.0),
            AngularVelocity::new::<radian_per_second>(1.0),
        );
        let pris_limits =
            CuJointLimits::prismatic(m(-4.0), m(4.0), Velocity::new::<meter_per_second>(1.0));
        let chain = CuKinematicChain::new([
            CuJointSpec::revolute([m(0.0); 3], [rad(0.0); 3], CuJointAxis::PosZ, rev_limits),
            CuJointSpec::prismatic([m(0.0); 3], [rad(0.0); 3], CuJointAxis::PosX, pris_limits),
            CuJointSpec {
                joint_type: CuJointType::Fixed,
                origin_translation: [m(0.0); 3],
                origin_rpy: [rad(0.0); 3],
                axis: CuJointAxis::PosZ,
                limits: CuJointLimits::Fixed,
            },
        ]);
        let model = CuRobotCollisionModel::new(&[], CuCollisionModelParams::default()).unwrap();
        (chain, model)
    }

    fn rpf(r: f64, p: f64) -> [CuJointPosition; 3] {
        [
            CuJointPosition::Revolute(rad(r)),
            CuJointPosition::Prismatic(m(p)),
            CuJointPosition::Fixed,
        ]
    }

    #[test]
    fn test_connect_skips_full_degree_neighbors_without_changing_selection() {
        let (chain, model) = planar_chain();
        let params = CuJointPrmParams {
            connection_radius: 10.0,
            neighbors: 2,
            max_nodes: 64,
            ..Default::default()
        };
        let mut prm = Box::new(CuJointPrm::new(chain, model, params).unwrap());
        let mut request = CuJointPlanRequest {
            start: xy(-4.0, 0.0),
            goal: xy(4.0, 0.0),
            ..Default::default()
        };
        // Obstacle blocks start->goal and sits on 5->2 so skipping full-degree
        // node 2 avoids collision-checking it while keeping k-nearest selection identical.
        request
            .scene
            .add_obstacle(
                CuCollisionShape::Sphere(CuSphere::new([m(0.1), m(0.0), m(0.0)], m(0.02)).unwrap()),
                None,
            )
            .unwrap();
        assert_eq!(prm.begin(&request, 1).unwrap(), CuPlanStatus::Searching);

        prm.nodes[2] = xy(0.2, 0.0);
        prm.nodes[3] = xy(0.0, 0.5);
        prm.nodes[4] = xy(0.2, 4.0);
        prm.nodes[5] = xy(0.0, 0.0);
        prm.nodes[6] = xy(0.0, 0.8);
        prm.len = 7;

        // Fill node 2 to cap (neighbors = 2).
        let d23 = joint_distance(&prm.nodes[2], &prm.nodes[3], params.metric);
        let d24 = joint_distance(&prm.nodes[2], &prm.nodes[4], params.metric);
        prm.link(2, 3, d23);
        prm.link(2, 4, d24);
        assert_eq!(prm.degree[2], 2);

        // Nearest 2 nodes to node 5 are [2 (dist 0.2), 3 (dist 0.5)].
        // Node 2 is full so it must be skipped, and node 6 (dist 0.8) must NOT
        // replace it in the k-nearest candidate list.
        assert!(prm.connect(5));
        assert_eq!(prm.degree[5], 1);
        assert_eq!(prm.edges[5][0], 3);
        assert_eq!(prm.degree[2], 2);
        assert_eq!(prm.degree[6], 0);

        // Now verify `idx` reaching cap breaks early and multi-step rollouts
        // preserve symmetric cached edge costs.
        let mut rollout = Box::new(CuJointPrm::new(chain, model, params).unwrap());
        assert_eq!(rollout.begin(&request, 7).unwrap(), CuPlanStatus::Searching);
        rollout.step(30).unwrap();
        for u in 0..rollout.len {
            assert!(rollout.degree[u] as usize <= 2);
            for e in 0..rollout.degree[u] as usize {
                let v = rollout.edges[u][e] as usize;
                let expected = joint_distance(&rollout.nodes[u], &rollout.nodes[v], params.metric);
                assert!(libm::fabs(rollout.edge_costs[u][e] - expected) < 1e-12);
                let has_reverse = (0..rollout.degree[v] as usize).any(|re| {
                    rollout.edges[v][re] as usize == u
                        && rollout.edge_costs[v][re] == rollout.edge_costs[u][e]
                });
                assert!(has_reverse, "missing symmetric reverse edge {v} -> {u}");
            }
        }
    }

    #[test]
    fn test_search_min_heap_shortest_path_tie_breaking_and_unreachable() {
        let (chain, model) = planar_chain();
        let params = CuJointPrmParams {
            connection_radius: 10.0,
            neighbors: 4,
            max_nodes: 32,
            ..Default::default()
        };
        let mut prm = Box::new(CuJointPrm::new(chain, model, params).unwrap());
        let mut request = CuJointPlanRequest {
            start: xy(-1.0, 0.0),
            goal: xy(1.0, 0.0),
            ..Default::default()
        };
        // Blocks direct 0->1, 0->3, and 2->1 while leaving 0->2->3->1 and 0->4->1 free.
        request
            .scene
            .add_obstacle(
                CuCollisionShape::Sphere(CuSphere::new([m(0.0); 3], m(0.45)).unwrap()),
                None,
            )
            .unwrap();

        // 1. Unreachable goal leaves `best` empty.
        assert_eq!(prm.begin(&request, 1).unwrap(), CuPlanStatus::Searching);
        prm.nodes[2] = xy(-0.4, 0.6);
        prm.nodes[3] = xy(0.4, 0.6);
        prm.len = 4;
        let d02 = joint_distance(&prm.nodes[START], &prm.nodes[2], params.metric);
        let d31 = joint_distance(&prm.nodes[3], &prm.nodes[GOAL], params.metric);
        prm.link(START, 2, d02);
        prm.link(3, GOAL, d31);
        prm.search();
        assert!(!prm.best.cost.is_finite());
        assert_eq!(prm.best.len, 0);

        // 2. Shorter 3-hop path (0->2->3->1) beats 2-hop path (0->4->1), exercising decrease_key.
        assert_eq!(prm.begin(&request, 1).unwrap(), CuPlanStatus::Searching);
        prm.nodes[2] = xy(-0.4, 0.6);
        prm.nodes[3] = xy(0.4, 0.6);
        prm.nodes[4] = xy(0.0, -1.5);
        prm.len = 5;
        let d04 = joint_distance(&prm.nodes[START], &prm.nodes[4], params.metric);
        let d41 = joint_distance(&prm.nodes[4], &prm.nodes[GOAL], params.metric);
        let d23 = joint_distance(&prm.nodes[2], &prm.nodes[3], params.metric);
        prm.link(START, 4, d04);
        prm.link(4, GOAL, d41);
        prm.link(START, 2, d02);
        prm.link(2, 3, d23);
        prm.link(3, GOAL, d31);
        prm.search();
        let expected_cost = d02 + d23 + d31;
        assert!(expected_cost < d04 + d41);
        assert!(libm::fabs(prm.best.cost - expected_cost) < 1e-12);
        assert_eq!(prm.best.len, 4);
        assert_eq!(prm.best.waypoints[1].positions, prm.nodes[2]);
        assert_eq!(prm.best.waypoints[2].positions, prm.nodes[3]);

        // 3. Equal-length paths break ties deterministically by smaller node index.
        for (upper_first, expected_y) in [(true, 1.0), (false, -1.0)] {
            assert_eq!(prm.begin(&request, 1).unwrap(), CuPlanStatus::Searching);
            let (p2, p3) = if upper_first {
                (xy(0.0, 1.0), xy(0.0, -1.0))
            } else {
                (xy(0.0, -1.0), xy(0.0, 1.0))
            };
            prm.nodes[2] = p2;
            prm.nodes[3] = p3;
            prm.len = 4;
            for node in [2, 3] {
                let da = joint_distance(&prm.nodes[START], &prm.nodes[node], params.metric);
                let db = joint_distance(&prm.nodes[node], &prm.nodes[GOAL], params.metric);
                prm.link(START, node, da);
                prm.link(node, GOAL, db);
            }
            prm.search();
            assert_eq!(prm.best.len, 3);
            assert_eq!(prm.best.waypoints[1].positions, xy(0.0, expected_y));
        }
    }

    #[test]
    fn test_squared_distance_filter_mixed_joints_and_radii() {
        let (chain, model) = mixed_chain();
        let metric = CuJointMetric {
            angular_scale: rad(0.5),
            linear_scale: m(0.25),
        };
        let a = rpf(0.0, 0.0);
        let b = rpf(0.5, 0.25);
        let c = rpf(0.6, 0.25);
        let dab = joint_distance(&a, &b, metric);
        let dac = joint_distance(&a, &c, metric);
        assert!(libm::fabs(joint_distance_sq(&a, &b, metric) - dab * dab) < 1e-12);
        assert!(libm::fabs(joint_distance_sq(&a, &c, metric) - dac * dac) < 1e-12);

        let params = CuJointPrmParams {
            connection_radius: 1.5,
            neighbors: 4,
            max_nodes: 16,
            metric,
            ..Default::default()
        };
        let mut prm = Box::new(CuJointPrm::new(chain, model, params).unwrap());
        let request = CuJointPlanRequest {
            start: a,
            goal: rpf(2.0, 0.0), // normalized distance 4.0 > 1.5
            ..Default::default()
        };
        assert_eq!(prm.begin(&request, 1).unwrap(), CuPlanStatus::Converged);

        prm.nodes[2] = b; // normalized distance sqrt(1^2 + 1^2) = sqrt(2) <= 1.5
        prm.nodes[3] = c; // normalized distance sqrt(1.2^2 + 1^2) = sqrt(2.44) > 1.5
        prm.nodes[4] = a;
        prm.len = 5;

        assert!(prm.connect(4));
        assert_eq!(prm.degree[4], 2);
        assert_eq!(prm.edges[4][0] as usize, START);
        assert_eq!(prm.edges[4][1] as usize, 2);
        assert_eq!(prm.degree[3], 0);
        assert!(libm::fabs(prm.edge_costs[4][1] - dab) < 1e-12);
    }
}
