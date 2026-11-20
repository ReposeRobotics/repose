//! A* lattice search: search-based, deterministic, resolution-complete.
//!
//! Configurations live on a grid anchored at the start with spacing
//! `resolution` (normalized joint units). Each expansion moves one joint by
//! one grid step in either direction through a free segment, at cost
//! `resolution`; the heuristic is the straight-line distance to the goal,
//! which is admissible and consistent for these moves. When an expanded node
//! lies within one grid cell diagonal of the goal and sees it through a free
//! segment, the lattice path plus that last segment is shortcut by line of
//! sight and the job converges. Exhausting the open set or the node capacity
//! ends the job without a path. No randomness is involved.

use crate::CuJointPlanPath;
use crate::CuJointPlanRequest;
use crate::CuJointPlanner;
use crate::CuJointPlannerTask;
use crate::CuPathConstraintSet;
use crate::CuPlanStatus;
use crate::CuPlannerKind;
use crate::common::check_endpoints;
use crate::common::length_quality;
use crate::common::write_header;
use crate::common::write_shortcut;
use crate::validity::CuSceneValidity;
use crate::validity::joint_distance;
use bincode::Decode;
use bincode::Encode;
use cu29::cutask_anytime::Quality;
use cu29::prelude::*;
use repose_collision::CuRobotCollisionModel;
use repose_kinematics::CuJointMetric;
use repose_kinematics::CuJointPosition;
use repose_kinematics::CuJointResolution;
use repose_kinematics::CuKinematicChain;
use serde::Deserialize;
use serde::Serialize;

/// Lattice node capacity.
pub const ASTAR_MAX_NODES: usize = 2048;
/// Longest lattice path (in grid steps) that can be reconstructed.
const ASTAR_MAX_PATH: usize = 512;
const HASH_SLOTS: usize = 2 * ASTAR_MAX_NODES;
const HASH_MASK: usize = HASH_SLOTS - 1;
const EMPTY: u16 = u16::MAX;

const _: () = assert!(HASH_SLOTS.is_power_of_two());
const _: () = assert!(ASTAR_MAX_NODES < EMPTY as usize);

/// Parameters of [`CuJointAStar`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Encode, Decode, Reflect)]
#[serde(default, deny_unknown_fields)]
pub struct CuJointAStarParams {
    /// Grid spacing (dimensionless, metric-scaled joint units).
    pub resolution: f64,
    /// Node capacity (`2..=ASTAR_MAX_NODES`).
    pub max_nodes: u32,
    /// Resolution of the segment collision checks.
    pub collision_check_step: CuJointResolution,
    /// Scaling of dimensionless joint-space distances and costs.
    pub metric: CuJointMetric,
}

impl Default for CuJointAStarParams {
    fn default() -> Self {
        Self {
            resolution: 0.25,
            max_nodes: ASTAR_MAX_NODES as u32,
            collision_check_step: CuJointResolution::default(),
            metric: CuJointMetric::default(),
        }
    }
}

impl CuJointAStarParams {
    /// Checks every parameter, the metric, and the collision-check resolution.
    pub fn validate(&self) -> CuResult<()> {
        self.metric.validate()?;
        self.collision_check_step.metric().validate()?;
        let valid = {
            self.resolution.is_finite()
                && self.resolution > 0.0
                && (2..=ASTAR_MAX_NODES as u32).contains(&self.max_nodes)
        };
        if valid {
            Ok(())
        } else {
            Err(CuError::from(
                "repose-motion-planning: invalid A* parameters",
            ))
        }
    }
}

/// A* search over a joint-space lattice.
///
/// Built with [`CuJointPlanner::new`] from robot models and [`CuJointAStarParams`];
/// runs in [`CuJointAStarTask`].
#[derive(Clone)]
pub struct CuJointAStar<const N: usize, const M: usize = N> {
    validity: CuSceneValidity<N, M>,
    params: CuJointAStarParams,
    start: [CuJointPosition; N],
    goal: [CuJointPosition; N],
    cells: [[i16; N]; ASTAR_MAX_NODES],
    cost: [f64; ASTAR_MAX_NODES],
    estimate: [f64; ASTAR_MAX_NODES],
    parent: [u16; ASTAR_MAX_NODES],
    closed: [bool; ASTAR_MAX_NODES],
    table: [u16; HASH_SLOTS],
    heap: [u16; ASTAR_MAX_NODES],
    heap_pos: [u16; ASTAR_MAX_NODES],
    len: usize,
    heap_len: usize,
    goal_reach: f64,
    straight_cost: f64,
    expansions: u32,
    status: CuPlanStatus,
    best: CuJointPlanPath<N>,
}

/// [`CuJointPlannerTask`] running [`CuJointAStar`].
pub type CuJointAStarTask<const N: usize, const M: usize = N> =
    CuJointPlannerTask<CuJointAStar<N, M>, N, M>;

impl<const N: usize, const M: usize> CuJointAStar<N, M> {
    /// The validated planner parameters.
    pub const fn params(&self) -> &CuJointAStarParams {
        &self.params
    }

    fn build(
        chain: CuKinematicChain<N>,
        model: CuRobotCollisionModel<N, M>,
        params: CuJointAStarParams,
    ) -> Self {
        Self {
            validity: CuSceneValidity::new(chain, model, params.collision_check_step),
            params,
            start: [CuJointPosition::default(); N],
            goal: [CuJointPosition::default(); N],
            cells: [[0; N]; ASTAR_MAX_NODES],
            cost: [0.0; ASTAR_MAX_NODES],
            estimate: [0.0; ASTAR_MAX_NODES],
            parent: [EMPTY; ASTAR_MAX_NODES],
            closed: [false; ASTAR_MAX_NODES],
            table: [EMPTY; HASH_SLOTS],
            heap: [EMPTY; ASTAR_MAX_NODES],
            heap_pos: [EMPTY; ASTAR_MAX_NODES],
            len: 0,
            heap_len: 0,
            goal_reach: 0.0,
            straight_cost: 0.0,
            expansions: 0,
            status: CuPlanStatus::InvalidRequest,
            best: CuJointPlanPath::default(),
        }
    }

    /// Nodes created on the current job.
    pub const fn lattice_size(&self) -> usize {
        self.len
    }

    fn configuration(&self, cell: &[i16; N]) -> [CuJointPosition; N] {
        let metric = self.params.metric;
        core::array::from_fn(|j| {
            let q = self.start[j];
            let offset = cell[j] as f64 * self.params.resolution * metric.scale(q);
            q.with_si_value(q.si_value() + offset)
        })
    }

    fn hash(cell: &[i16; N]) -> usize {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for &c in cell {
            h ^= c as u16 as u64;
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
        (h as usize) & HASH_MASK
    }

    fn find(&self, cell: &[i16; N]) -> Option<usize> {
        let mut slot = Self::hash(cell);
        loop {
            match self.table[slot] {
                EMPTY => return None,
                idx if self.cells[idx as usize] == *cell => return Some(idx as usize),
                _ => slot = (slot + 1) & HASH_MASK,
            }
        }
    }

    #[inline]
    fn heap_less(&self, a: u16, b: u16) -> bool {
        let ea = self.estimate[a as usize];
        let eb = self.estimate[b as usize];
        match (ea.is_nan(), eb.is_nan()) {
            (false, false) => ea < eb || (ea == eb && a < b),
            (false, true) => true,
            (true, false) => false,
            (true, true) => a < b,
        }
    }

    fn heap_sift_up(&mut self, mut pos: usize) {
        let node = self.heap[pos];
        while pos > 0 {
            let parent_pos = (pos - 1) >> 1;
            let parent_node = self.heap[parent_pos];
            if !self.heap_less(node, parent_node) {
                break;
            }
            self.heap[pos] = parent_node;
            self.heap_pos[parent_node as usize] = pos as u16;
            pos = parent_pos;
        }
        self.heap[pos] = node;
        self.heap_pos[node as usize] = pos as u16;
    }

    fn heap_sift_down(&mut self, mut pos: usize) {
        let len = self.heap_len;
        let node = self.heap[pos];
        loop {
            let left = (pos << 1) + 1;
            if left >= len {
                break;
            }
            let right = left + 1;
            let mut best_child_pos = left;
            let mut best_child = self.heap[left];
            if right < len {
                let right_child = self.heap[right];
                if self.heap_less(right_child, best_child) {
                    best_child_pos = right;
                    best_child = right_child;
                }
            }
            if !self.heap_less(best_child, node) {
                break;
            }
            self.heap[pos] = best_child;
            self.heap_pos[best_child as usize] = pos as u16;
            pos = best_child_pos;
        }
        self.heap[pos] = node;
        self.heap_pos[node as usize] = pos as u16;
    }

    fn heap_push(&mut self, node: u16) {
        let pos = self.heap_len;
        if pos >= ASTAR_MAX_NODES || node as usize >= ASTAR_MAX_NODES {
            return;
        }
        self.heap[pos] = node;
        self.heap_pos[node as usize] = pos as u16;
        self.heap_len = pos + 1;
        self.heap_sift_up(pos);
    }

    fn heap_pop(&mut self) -> Option<usize> {
        if self.heap_len == 0 {
            return None;
        }
        let root = self.heap[0];
        self.heap_pos[root as usize] = EMPTY;
        self.heap_len -= 1;
        if self.heap_len > 0 {
            let last = self.heap[self.heap_len];
            self.heap[0] = last;
            self.heap_pos[last as usize] = 0;
            self.heap_sift_down(0);
        }
        Some(root as usize)
    }

    fn heap_decrease_key(&mut self, idx: usize) {
        if idx >= ASTAR_MAX_NODES {
            return;
        }
        let pos = self.heap_pos[idx] as usize;
        if pos < self.heap_len && self.heap[pos] as usize == idx {
            self.heap_sift_up(pos);
        }
    }

    fn insert(
        &mut self,
        cell: [i16; N],
        q: &[CuJointPosition; N],
        cost: f64,
        parent: u16,
    ) -> usize {
        let idx = self.len;
        self.cells[idx] = cell;
        self.cost[idx] = cost;
        self.estimate[idx] = cost + joint_distance(q, &self.goal, self.params.metric);
        self.parent[idx] = parent;
        self.closed[idx] = false;
        let mut slot = Self::hash(&cell);
        while self.table[slot] != EMPTY {
            slot = (slot + 1) & HASH_MASK;
        }
        self.table[slot] = idx as u16;
        self.len += 1;
        self.heap_push(idx as u16);
        idx
    }

    fn finish(&mut self, last: usize) {
        let mut raw = [[CuJointPosition::default(); N]; ASTAR_MAX_PATH];
        let mut len = 0;
        raw[len] = self.goal;
        len += 1;
        let mut at = last;
        loop {
            if len >= ASTAR_MAX_PATH {
                self.status = CuPlanStatus::CapacityExceeded;
                return;
            }
            raw[len] = self.configuration(&self.cells[at]);
            len += 1;
            if self.parent[at] == EMPTY {
                break;
            }
            at = self.parent[at] as usize;
        }
        raw[..len].reverse();
        self.status = if write_shortcut(
            &self.validity,
            &raw[..len],
            self.params.metric,
            &mut self.best,
        ) {
            CuPlanStatus::Converged
        } else {
            CuPlanStatus::CapacityExceeded
        };
    }

    fn expand(&mut self) {
        let Some(current) = self.heap_pop() else {
            self.status = CuPlanStatus::Exhausted;
            return;
        };
        self.closed[current] = true;
        self.expansions += 1;
        let cell = self.cells[current];
        let q = self.configuration(&cell);
        if joint_distance(&q, &self.goal, self.params.metric) <= self.goal_reach
            && self.validity.is_segment_interior_free(&q, &self.goal)
        {
            self.finish(current);
            return;
        }
        let cost = self.cost[current] + self.params.resolution;
        for j in 0..N {
            let start_j = self.start[j];
            if matches!(start_j, CuJointPosition::Fixed) {
                continue;
            }
            let start_si_j = start_j.si_value();
            let scale_j = self.params.metric.scale(start_j);
            for direction in [-1i16, 1] {
                let mut next = cell;
                next[j] = match next[j].checked_add(direction) {
                    Some(v) => v,
                    None => continue,
                };
                match self.find(&next) {
                    Some(idx) if self.closed[idx] || cost >= self.cost[idx] => {}
                    Some(idx) => {
                        let mut q_next = q;
                        let offset = next[j] as f64 * self.params.resolution * scale_j;
                        q_next[j] = start_j.with_si_value(start_si_j + offset);
                        if self.validity.is_segment_free(&q, &q_next) {
                            let new_estimate = self.estimate[idx] + (cost - self.cost[idx]);
                            if !new_estimate.is_nan() {
                                self.estimate[idx] = new_estimate;
                            }
                            self.cost[idx] = cost;
                            self.parent[idx] = current as u16;
                            self.heap_decrease_key(idx);
                        }
                    }
                    None => {
                        if self.len >= self.params.max_nodes as usize {
                            continue;
                        }
                        let mut q_next = q;
                        let offset = next[j] as f64 * self.params.resolution * scale_j;
                        q_next[j] = start_j.with_si_value(start_si_j + offset);
                        if self.validity.is_segment_free(&q, &q_next) {
                            self.insert(next, &q_next, cost, current as u16);
                        }
                    }
                }
            }
        }
        if self.heap_len == 0 {
            self.status = CuPlanStatus::Exhausted;
        }
    }
}

impl<const N: usize, const M: usize> CuJointPlanner<N, M> for CuJointAStar<N, M> {
    const KIND: CuPlannerKind = CuPlannerKind::Search;

    type Params = CuJointAStarParams;

    fn new(
        chain: CuKinematicChain<N>,
        collision_model: CuRobotCollisionModel<N, M>,
        params: CuJointAStarParams,
    ) -> CuResult<Self> {
        params.validate()?;
        Ok(Self::build(chain, collision_model, params))
    }

    fn begin(&mut self, request: &CuJointPlanRequest<N>, _seed: u64) -> CuResult<CuPlanStatus> {
        self.start = request.start;
        self.goal = request.goal;
        self.expansions = 0;
        self.len = 0;
        self.heap_len = 0;
        self.table = [EMPTY; HASH_SLOTS];
        self.best = CuJointPlanPath::default();
        self.straight_cost = joint_distance(&request.start, &request.goal, self.params.metric);
        let active = self
            .start
            .iter()
            .filter(|q| !matches!(q, CuJointPosition::Fixed))
            .count()
            .max(1);
        // Every point is within half a cell diagonal of the lattice; one full
        // diagonal leaves room for the free-segment check to succeed.
        self.goal_reach = self.params.resolution * libm::sqrt(active as f64);
        self.status = check_endpoints(&mut self.validity, request, self.params.metric);
        if self.status == CuPlanStatus::Searching {
            let start_cell = [0; N];
            let q_start = self.configuration(&start_cell);
            self.insert(start_cell, &q_start, 0.0, EMPTY);
        }
        Ok(self.status)
    }

    fn step(&mut self, budget: u32) -> CuResult<CuPlanStatus> {
        for _ in 0..budget {
            if self.status.is_final() {
                break;
            }
            self.expand();
        }
        Ok(self.status)
    }

    fn write_best(&self, out: &mut CuJointPlanPath<N>) -> CuPlanStatus {
        write_header(out, &self.validity, self.status, self.expansions);
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
        self.expansions = 0;
        self.len = 0;
        self.heap_len = 0;
        self.best = CuJointPlanPath::default();
        Ok(())
    }

    fn clear_path_constraints(&mut self) {
        self.validity.clear_path_constraints();
        self.status = CuPlanStatus::InvalidRequest;
        self.expansions = 0;
        self.len = 0;
        self.heap_len = 0;
        self.best = CuJointPlanPath::default();
    }
}

crate::impl_planner_type_path!(CuJointAStar, "repose_motion_planning::CuJointAStar");

#[cfg(test)]
mod tests {
    use super::*;
    use cu29::units::si::angle::radian;
    use cu29::units::si::f64::Angle;
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

    fn m(v: f64) -> Length {
        Length::new::<meter>(v)
    }

    fn gantry_robot() -> (CuKinematicChain<3>, CuRobotCollisionModel<3, 1>) {
        let limits =
            CuJointLimits::prismatic(m(-1.0), m(1.0), Velocity::new::<meter_per_second>(1.0));
        let origin = [m(0.0); 3];
        let rpy = [Angle::new::<radian>(0.0); 3];
        let chain = CuKinematicChain::new([
            CuJointSpec::prismatic(origin, rpy, CuJointAxis::PosX, limits),
            CuJointSpec::prismatic(origin, rpy, CuJointAxis::PosY, limits),
            CuJointSpec::prismatic(origin, rpy, CuJointAxis::PosZ, limits),
        ]);
        let sphere = CuCollisionShape::Sphere(CuSphere::new([m(0.0); 3], m(0.05)).unwrap());
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

    fn check_heap_invariants<const N: usize, const M: usize>(planner: &CuJointAStar<N, M>) {
        for pos in 0..planner.heap_len {
            let node = planner.heap[pos] as usize;
            assert!(node < ASTAR_MAX_NODES);
            assert_eq!(planner.heap_pos[node] as usize, pos);
            if pos > 0 {
                let parent = (pos - 1) >> 1;
                assert!(!planner.heap_less(planner.heap[pos], planner.heap[parent]));
            }
        }
    }

    #[test]
    fn test_heap_ordering_and_tie_breaking() {
        let (chain, model) = gantry_robot();
        let mut planner = CuJointAStar::new(chain, model, CuJointAStarParams::default()).unwrap();

        // Populate estimates with ties, -0.0/+0.0, infinity, and NaN.
        let estimates = [3.0, 1.0, 2.0, 1.0, 2.0, 0.0, -0.0, f64::NAN, f64::INFINITY];
        for (i, &est) in estimates.iter().enumerate() {
            planner.estimate[i] = est;
        }
        // Push in scrambled order so tie-breaking must select smaller node index.
        for idx in [7u16, 4, 8, 3, 2, 6, 1, 5, 0] {
            planner.heap_push(idx);
            check_heap_invariants(&planner);
        }
        assert_eq!(planner.heap_len, estimates.len());

        let mut popped = [u16::MAX; 9];
        for slot in &mut popped {
            let idx = planner.heap_pop().unwrap() as u16;
            assert_eq!(planner.heap_pos[idx as usize], EMPTY);
            check_heap_invariants(&planner);
            *slot = idx;
        }
        assert_eq!(planner.heap_pop(), None);
        assert_eq!(planner.heap_len, 0);

        // Expected order: (0.0/-0.0: 5, 6), (1.0: 1, 3), (2.0: 2, 4), (3.0: 0), (inf: 8), (NaN: 7)
        assert_eq!(popped, [5, 6, 1, 3, 2, 4, 0, 8, 7]);
    }

    #[test]
    fn test_heap_decrease_key() {
        let (chain, model) = gantry_robot();
        let mut planner = CuJointAStar::new(chain, model, CuJointAStarParams::default()).unwrap();

        let estimates = [5.0, 4.0, 3.0, 6.0, 7.0, 2.0];
        for (i, &est) in estimates.iter().enumerate() {
            planner.estimate[i] = est;
            planner.heap_push(i as u16);
        }
        check_heap_invariants(&planner);

        // Decrease node 4 from 7.0 -> 2.0 (ties with node 5, index 4 < 5 so 4 wins).
        planner.estimate[4] = 2.0;
        planner.heap_decrease_key(4);
        check_heap_invariants(&planner);

        // Decrease node 3 from 6.0 -> 1.0 (becomes new root).
        planner.estimate[3] = 1.0;
        planner.heap_decrease_key(3);
        check_heap_invariants(&planner);

        // Pop root (3), then verify decrease_key on popped or out-of-range node is a no-op.
        assert_eq!(planner.heap_pop(), Some(3));
        planner.estimate[3] = 0.0;
        planner.heap_decrease_key(3);
        planner.heap_decrease_key(ASTAR_MAX_NODES + 1);
        check_heap_invariants(&planner);

        let mut rest = [usize::MAX; 5];
        for slot in &mut rest {
            *slot = planner.heap_pop().unwrap();
            check_heap_invariants(&planner);
        }
        assert_eq!(rest, [4, 5, 2, 1, 0]);
        assert_eq!(planner.heap_pop(), None);
    }

    #[test]
    fn test_astar_matches_linear_scan_expansion_order() {
        let (chain, model) = gantry_robot();
        let mut planner = CuJointAStar::new(
            chain,
            model,
            CuJointAStarParams {
                resolution: 0.15,
                max_nodes: ASTAR_MAX_NODES as u32,
                ..Default::default()
            },
        )
        .unwrap();

        let mut req = CuJointPlanRequest {
            start: [m(-0.5), m(0.0), m(0.0)].map(CuJointPosition::Prismatic),
            goal: [m(0.5), m(0.0), m(0.0)].map(CuJointPosition::Prismatic),
            ..Default::default()
        };
        // Block the straight X line at the origin so A* detours and explores many nodes.
        req.scene
            .add_obstacle(
                CuCollisionShape::Sphere(CuSphere::new([m(0.0); 3], m(0.2)).unwrap()),
                None,
            )
            .unwrap();

        let mut status = planner.begin(&req, 0).unwrap();
        assert_eq!(status, CuPlanStatus::Searching);

        let mut iterations = 0;
        while !status.is_final() {
            check_heap_invariants(&planner);

            // Replicate the exact original linear scan over 0..self.len.
            let mut expected = usize::MAX;
            for i in 0..planner.len {
                if !planner.closed[i]
                    && (expected == usize::MAX || planner.estimate[i] < planner.estimate[expected])
                {
                    expected = i;
                }
            }
            assert_ne!(expected, usize::MAX);
            assert_eq!(planner.heap[0] as usize, expected);

            status = planner.step(1).unwrap();
            assert!(planner.closed[expected]);
            iterations += 1;
            assert!(iterations < 10_000);
        }

        assert_eq!(status, CuPlanStatus::Converged);
        let mut path = CuJointPlanPath::<3>::default();
        assert_eq!(planner.write_best(&mut path), CuPlanStatus::Converged);
        assert!(path.len >= 3);
        assert!(path.cost.is_finite());

        // Re-running begin() cleanly resets heap_len and produces identical results.
        let status2 = planner.begin(&req, 99).unwrap();
        assert_eq!(status2, CuPlanStatus::Searching);
        assert_eq!(planner.heap_len, 1);
        assert_eq!(planner.lattice_size(), 1);
        let final_status = planner.step(10_000).unwrap();
        assert_eq!(final_status, CuPlanStatus::Converged);
        let mut path2 = CuJointPlanPath::<3>::default();
        planner.write_best(&mut path2);
        assert_eq!(path.len, path2.len);
        assert_eq!(path.cost, path2.cost);
        assert_eq!(path.waypoints(), path2.waypoints());
    }
}
