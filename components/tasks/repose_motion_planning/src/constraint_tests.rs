//! Path-constraint enforcement across the built-in planners.
use super::*;
use cu_rng::prelude::*;
use cu_spatial_payloads::Transform3D;
use cu29::cutask_anytime::CuAnytimeTask;
use cu29::units::si::angle::degree;
use cu29::units::si::angle::radian;
use cu29::units::si::f64::Angle;
use cu29::units::si::f64::Length;
use cu29::units::si::length::meter;
use repose_collision::CuCollisionShape;
use repose_collision::CuSphere;
use repose_collision::igus_rebel_collision_model;
use repose_kinematics::CuJointPosition;
use repose_kinematics::CuJointResolution;
use repose_kinematics::igus_rebel_chain;

const START: [f64; 6] = [-1.1, 45.5, 60.0, -3.3, -80.9, 28.7];
const GOAL: [f64; 6] = [58.9, 45.5, 60.0, -3.3, -80.9, 28.7];
const SEED: u64 = 1;

fn deg(values: [f64; 6]) -> [CuJointPosition; 6] {
    values.map(|d| CuJointPosition::Revolute(Angle::new::<degree>(d)))
}

fn tool_pose(q: &[CuJointPosition; 6]) -> Transform3D<f64> {
    let mut tcp = Transform3D::<f64>::identity();
    igus_rebel_chain().forward_kinematics(q, &mut tcp).unwrap();
    tcp
}

/// Base-yaw move with a sphere on the straight-line tool path, so every
/// planner must detour. Start and goal tool Z axes tilt 24.7 degrees from
/// world up, 24.1 degrees from each other.
fn request() -> CuJointPlanRequest<6> {
    let mut request = CuJointPlanRequest {
        start: deg(START),
        goal: deg(GOAL),
        ..Default::default()
    };
    let mid = deg(core::array::from_fn(|j| 0.5 * (START[j] + GOAL[j])));
    let tcp = tool_pose(&mid);
    request
        .scene
        .add_obstacle(
            CuCollisionShape::Sphere(
                CuSphere::new(tcp.translation(), Length::new::<meter>(0.1)).unwrap(),
            ),
            None,
        )
        .unwrap();
    request
}

/// The same move without obstacles: the straight joint line is free.
fn clear_request() -> CuJointPlanRequest<6> {
    CuJointPlanRequest {
        start: deg(START),
        goal: deg(GOAL),
        ..Default::default()
    }
}

/// Tool Z within 30 degrees of world up. Start and goal (24.7 degrees) satisfy
/// it; every unconstrained detour around the sphere violates it.
fn orientation_constraints() -> CuPathConstraintSet {
    let mut set = CuPathConstraintSet::new();
    set.push(CuPathConstraint::Orientation(
        CuOrientationConstraint::new(CuToolAxis::Z, [0.0, 0.0, 1.0], Angle::new::<degree>(30.0))
            .unwrap(),
    ))
    .unwrap();
    set
}

/// Drives `planner` from `begin` to a final status (bounded).
fn plan<P: CuJointPlanner<6>>(
    planner: &mut P,
    request: &CuJointPlanRequest<6>,
    seed: u64,
) -> (CuPlanStatus, CuJointPlanPath<6>) {
    let mut status = planner.begin(request, seed).unwrap();
    let mut steps = 0;
    while !status.is_final() && steps < 200 {
        status = planner.step(64).unwrap();
        steps += 1;
    }
    let mut path = CuJointPlanPath::<6>::default();
    planner.write_best(&mut path);
    (status, path)
}

/// Samples the planner's own segment resolution for one segment.
fn planner_steps(a: &[CuJointPosition; 6], b: &[CuJointPosition; 6]) -> usize {
    let metric = CuJointResolution::default().metric();
    let mut max_delta = 0.0_f64;
    for j in 0..6 {
        max_delta = max_delta.max(libm::fabs(
            metric.normalized(b[j]) - metric.normalized(a[j]),
        ));
    }
    (libm::ceil(max_delta) as usize).max(1)
}

/// Checks every path segment at eight times the planner's sampling density.
/// Returns whether any sample violates, with violating and total counts.
fn dense_violation(path: &CuJointPlanPath<6>, set: &CuPathConstraintSet) -> (bool, usize, usize) {
    let mut bad = 0usize;
    let mut total = 0usize;
    for window in path.waypoints().windows(2) {
        let (a, b) = (window[0].positions, window[1].positions);
        let steps = planner_steps(&a, &b) * 8;
        for s in 0..=steps {
            let q: [CuJointPosition; 6] =
                core::array::from_fn(|j| a[j].interpolate(b[j], s as f64 / steps as f64).unwrap());
            total += 1;
            if !set.is_satisfied_by_pose(&tool_pose(&q)) {
                bad += 1;
            }
        }
    }
    (bad > 0, bad, total)
}

/// The orientation scenario for one planner: unconstrained planning detours
/// out of the cone, constrained planning stays inside it.
fn check_orientation<P: CuJointPlanner<6>>(mut unconstrained: P, mut constrained: P) {
    let request = request();
    let set = orientation_constraints();
    let (status, path) = plan(&mut unconstrained, &request, SEED);
    assert!(status.has_path(), "unconstrained status: {status:?}");
    let (violated, bad, total) = dense_violation(&path, &set);
    assert!(
        violated,
        "unconstrained path stays in the cone ({bad}/{total}): the test proves nothing"
    );
    constrained.set_path_constraints(&set).unwrap();
    let (status, path) = plan(&mut constrained, &request, SEED);
    assert!(status.has_path(), "constrained status: {status:?}");
    let (violated, bad, total) = dense_violation(&path, &set);
    assert!(
        !violated,
        "constrained path leaves the cone ({bad}/{total})"
    );
}

#[test]
fn test_orientation_rrt_connect_stays_in_cone() {
    let chain = igus_rebel_chain();
    let model = igus_rebel_collision_model();
    check_orientation(
        CuJointRrtConnect::new(chain, model, Default::default()).unwrap(),
        CuJointRrtConnect::new(chain, model, Default::default()).unwrap(),
    );
}

#[test]
fn test_orientation_prm_stays_in_cone() {
    let chain = igus_rebel_chain();
    let model = igus_rebel_collision_model();
    check_orientation(
        CuJointPrm::new(chain, model, Default::default()).unwrap(),
        CuJointPrm::new(chain, model, Default::default()).unwrap(),
    );
}

#[test]
fn test_orientation_a_star_stays_in_cone() {
    let chain = igus_rebel_chain();
    let model = igus_rebel_collision_model();
    check_orientation(
        CuJointAStar::new(chain, model, Default::default()).unwrap(),
        CuJointAStar::new(chain, model, Default::default()).unwrap(),
    );
}

#[test]
fn test_orientation_stomp_stays_in_cone() {
    let chain = igus_rebel_chain();
    let model = igus_rebel_collision_model();
    check_orientation(
        CuJointStomp::new(chain, model, Default::default()).unwrap(),
        CuJointStomp::new(chain, model, Default::default()).unwrap(),
    );
}

#[test]
fn test_orientation_optimizer_stays_in_cone() {
    let chain = igus_rebel_chain();
    let model = igus_rebel_collision_model();
    check_orientation(
        CuJointPathOptimizer::new(chain, model, Default::default()).unwrap(),
        CuJointPathOptimizer::new(chain, model, Default::default()).unwrap(),
    );
}

/// Box around the base-yaw tool path. Start and goal (z 0.412) are inside;
/// every unconstrained detour around the sphere leaves it.
fn region_constraints() -> CuPathConstraintSet {
    let mut set = CuPathConstraintSet::new();
    set.push(CuPathConstraint::PositionRegion(
        CuPositionRegionConstraint::new(
            [0.15, -0.1, 0.3].map(Length::new::<meter>),
            [0.6, 0.55, 0.58].map(Length::new::<meter>),
        )
        .unwrap(),
    ))
    .unwrap();
    set
}

/// The region scenario for one planner: unconstrained planning detours out
/// of the box, constrained planning stays inside it.
fn check_region<P: CuJointPlanner<6>>(mut unconstrained: P, mut constrained: P) {
    let request = request();
    let set = region_constraints();
    let (status, path) = plan(&mut unconstrained, &request, SEED);
    assert!(status.has_path(), "unconstrained status: {status:?}");
    let (violated, bad, total) = dense_violation(&path, &set);
    assert!(
        violated,
        "unconstrained path stays in the box ({bad}/{total}): the test proves nothing"
    );
    constrained.set_path_constraints(&set).unwrap();
    let (status, path) = plan(&mut constrained, &request, SEED);
    assert!(status.has_path(), "constrained status: {status:?}");
    let (violated, bad, total) = dense_violation(&path, &set);
    assert!(!violated, "constrained path leaves the box ({bad}/{total})");
}

#[test]
fn test_region_rrt_connect_stays_in_box() {
    let chain = igus_rebel_chain();
    let model = igus_rebel_collision_model();
    check_region(
        CuJointRrtConnect::new(chain, model, Default::default()).unwrap(),
        CuJointRrtConnect::new(chain, model, Default::default()).unwrap(),
    );
}

#[test]
fn test_region_prm_stays_in_box() {
    let chain = igus_rebel_chain();
    let model = igus_rebel_collision_model();
    check_region(
        CuJointPrm::new(chain, model, Default::default()).unwrap(),
        CuJointPrm::new(chain, model, Default::default()).unwrap(),
    );
}

#[test]
fn test_region_a_star_stays_in_box() {
    let chain = igus_rebel_chain();
    let model = igus_rebel_collision_model();
    check_region(
        CuJointAStar::new(chain, model, Default::default()).unwrap(),
        CuJointAStar::new(chain, model, Default::default()).unwrap(),
    );
}

/// The optimizers cannot steer toward satisfaction: their descent only sees
/// collisions and smoothness. When no descent path is both free and
/// satisfying, the constrained job fails explicitly with `Exhausted` and an
/// empty path rather than returning a violating one.
fn check_region_optimizer_fails_explicitly<P: CuJointPlanner<6>>(
    mut unconstrained: P,
    mut constrained: P,
) {
    let request = request();
    let set = region_constraints();
    let (status, path) = plan(&mut unconstrained, &request, SEED);
    assert!(status.has_path(), "unconstrained status: {status:?}");
    let (violated, bad, total) = dense_violation(&path, &set);
    assert!(
        violated,
        "unconstrained path stays in the box ({bad}/{total}): the test proves nothing"
    );
    constrained.set_path_constraints(&set).unwrap();
    let (status, path) = plan(&mut constrained, &request, SEED);
    assert_eq!(status, CuPlanStatus::Exhausted);
    assert!(!status.has_path());
    assert_eq!(path.len, 0);
}

#[test]
fn test_region_stomp_fails_explicitly_without_satisfying_descent() {
    let chain = igus_rebel_chain();
    let model = igus_rebel_collision_model();
    check_region_optimizer_fails_explicitly(
        CuJointStomp::new(chain, model, Default::default()).unwrap(),
        CuJointStomp::new(chain, model, Default::default()).unwrap(),
    );
}

#[test]
fn test_region_optimizer_fails_explicitly_without_satisfying_descent() {
    let chain = igus_rebel_chain();
    let model = igus_rebel_collision_model();
    check_region_optimizer_fails_explicitly(
        CuJointPathOptimizer::new(chain, model, Default::default()).unwrap(),
        CuJointPathOptimizer::new(chain, model, Default::default()).unwrap(),
    );
}

#[test]
fn test_direct_accepts_satisfying_line_and_rejects_violating_line() {
    let chain = igus_rebel_chain();
    let model = igus_rebel_collision_model();
    // The straight line cannot detour, so enforcement is rejection: a line
    // that satisfies the constraints converges with a clean dense check.
    let mut set = orientation_constraints();
    set.push(CuPathConstraint::PositionRegion(
        CuPositionRegionConstraint::new(
            [0.15, -0.1, 0.3].map(Length::new::<meter>),
            [0.6, 0.55, 0.58].map(Length::new::<meter>),
        )
        .unwrap(),
    ))
    .unwrap();
    let mut planner = CuJointDirect::new(chain, model, Default::default()).unwrap();
    planner.set_path_constraints(&set).unwrap();
    let (status, path) = plan(&mut planner, &clear_request(), SEED);
    assert_eq!(status, CuPlanStatus::Converged);
    let (violated, bad, total) = dense_violation(&path, &set);
    assert!(!violated, "direct path leaves the region ({bad}/{total})");
    // The same move with the sphere blocking the line is exhausted, with or
    // without constraints: the direct planner never detours.
    let mut planner = CuJointDirect::new(chain, model, Default::default()).unwrap();
    planner.set_path_constraints(&set).unwrap();
    let (status, _) = plan(&mut planner, &request(), SEED);
    assert_eq!(status, CuPlanStatus::Exhausted);
    // A line whose interior leaves the constrained region is exhausted:
    // the joint-space straight line bulges outside its endpoint TCP box.
    let far = CuJointPlanRequest {
        start: deg([-1.1, 45.5, 60.0, -3.3, -80.9, 28.7]),
        goal: deg([90.0, 30.0, 60.0, 0.0, 60.0, 0.0]),
        ..Default::default()
    };
    let (ta, tb) = (tool_pose(&far.start), tool_pose(&far.goal));
    let (tra, trb) = (ta.translation(), tb.translation());
    let mut hull = CuPathConstraintSet::new();
    hull.push(CuPathConstraint::PositionRegion(
        CuPositionRegionConstraint::new(
            core::array::from_fn(|i| if tra[i] <= trb[i] { tra[i] } else { trb[i] }),
            core::array::from_fn(|i| if tra[i] >= trb[i] { tra[i] } else { trb[i] }),
        )
        .unwrap(),
    ))
    .unwrap();
    let mut planner = CuJointDirect::new(chain, model, Default::default()).unwrap();
    let (status, path) = plan(&mut planner, &far, SEED);
    assert_eq!(status, CuPlanStatus::Converged);
    let (violated, bad, total) = dense_violation(&path, &hull);
    assert!(
        violated,
        "straight line stays in its endpoint box ({bad}/{total}): the test proves nothing"
    );
    let mut planner = CuJointDirect::new(chain, model, Default::default()).unwrap();
    planner.set_path_constraints(&hull).unwrap();
    let (status, path) = plan(&mut planner, &far, SEED);
    assert_eq!(status, CuPlanStatus::Exhausted);
    assert!(!status.has_path());
    assert_eq!(path.len, 0);
    // A line whose start violates the constraints fails explicitly with the
    // existing invalid-start status rather than returning a violating path.
    let mut tight = CuPathConstraintSet::new();
    tight
        .push(CuPathConstraint::Orientation(
            CuOrientationConstraint::new(
                CuToolAxis::Z,
                [0.0, 0.0, 1.0],
                Angle::new::<degree>(10.0),
            )
            .unwrap(),
        ))
        .unwrap();
    let mut planner = CuJointDirect::new(chain, model, Default::default()).unwrap();
    planner.set_path_constraints(&tight).unwrap();
    let (status, path) = plan(&mut planner, &clear_request(), SEED);
    assert_eq!(status, CuPlanStatus::InvalidStart);
    assert!(!status.has_path());
    assert_eq!(path.len, 0);
}

/// Start or goal configurations that violate the constraints fail with the
/// existing invalid-start / invalid-goal statuses.
fn check_invalid_endpoints<P: CuJointPlanner<6>>(make: impl Fn() -> P) {
    // The start tool Z tilts 24.7 degrees from world up: a 10-degree cone
    // rejects the start.
    let mut tight = CuPathConstraintSet::new();
    tight
        .push(CuPathConstraint::Orientation(
            CuOrientationConstraint::new(
                CuToolAxis::Z,
                [0.0, 0.0, 1.0],
                Angle::new::<degree>(10.0),
            )
            .unwrap(),
        ))
        .unwrap();
    let mut planner = make();
    planner.set_path_constraints(&tight).unwrap();
    assert_eq!(
        planner.begin(&clear_request(), SEED).unwrap(),
        CuPlanStatus::InvalidStart
    );
    // The goal tool Z tilts 24.1 degrees from the start tool Z: a 10-degree
    // cone around the start direction accepts the start, rejects the goal.
    let rotation = tool_pose(&deg(START)).rotation();
    let mut goal_out = CuPathConstraintSet::new();
    goal_out
        .push(CuPathConstraint::Orientation(
            CuOrientationConstraint::new(
                CuToolAxis::Z,
                [rotation[0][2], rotation[1][2], rotation[2][2]],
                Angle::new::<degree>(10.0),
            )
            .unwrap(),
        ))
        .unwrap();
    let mut planner = make();
    planner.set_path_constraints(&goal_out).unwrap();
    assert_eq!(
        planner.begin(&clear_request(), SEED).unwrap(),
        CuPlanStatus::InvalidGoal
    );
    // A box around the goal tool point only: the start is outside.
    let goal_translation = tool_pose(&deg(GOAL)).translation();
    let mut start_out = CuPathConstraintSet::new();
    start_out
        .push(CuPathConstraint::PositionRegion(
            CuPositionRegionConstraint::new(
                goal_translation.map(|v| v - Length::new::<meter>(0.05)),
                goal_translation.map(|v| v + Length::new::<meter>(0.05)),
            )
            .unwrap(),
        ))
        .unwrap();
    let mut planner = make();
    planner.set_path_constraints(&start_out).unwrap();
    assert_eq!(
        planner.begin(&clear_request(), SEED).unwrap(),
        CuPlanStatus::InvalidStart
    );
    // A box around the start tool point only: the goal is outside.
    let start_translation = tool_pose(&deg(START)).translation();
    let mut goal_box_out = CuPathConstraintSet::new();
    goal_box_out
        .push(CuPathConstraint::PositionRegion(
            CuPositionRegionConstraint::new(
                start_translation.map(|v| v - Length::new::<meter>(0.05)),
                start_translation.map(|v| v + Length::new::<meter>(0.05)),
            )
            .unwrap(),
        ))
        .unwrap();
    let mut planner = make();
    planner.set_path_constraints(&goal_box_out).unwrap();
    assert_eq!(
        planner.begin(&clear_request(), SEED).unwrap(),
        CuPlanStatus::InvalidGoal
    );
}

#[test]
fn test_violating_endpoints_fail_with_invalid_start_or_goal() {
    let chain = igus_rebel_chain();
    let model = igus_rebel_collision_model();
    check_invalid_endpoints(|| CuJointRrtConnect::new(chain, model, Default::default()).unwrap());
    check_invalid_endpoints(|| CuJointPrm::new(chain, model, Default::default()).unwrap());
    check_invalid_endpoints(|| CuJointAStar::new(chain, model, Default::default()).unwrap());
    check_invalid_endpoints(|| CuJointStomp::new(chain, model, Default::default()).unwrap());
    check_invalid_endpoints(|| {
        CuJointPathOptimizer::new(chain, model, Default::default()).unwrap()
    });
    check_invalid_endpoints(|| CuJointDirect::new(chain, model, Default::default()).unwrap());
}

#[test]
fn test_clearing_constraints_restores_unconstrained_plans() {
    let chain = igus_rebel_chain();
    let model = igus_rebel_collision_model();
    let request = request();
    let mut planner = CuJointRrtConnect::new(chain, model, Default::default()).unwrap();
    let (before, path_before) = plan(&mut planner, &request, SEED);
    assert!(before.has_path());
    planner
        .set_path_constraints(&orientation_constraints())
        .unwrap();
    let (status, path) = plan(&mut planner, &request, SEED);
    assert!(status.has_path());
    assert!(!dense_violation(&path, &orientation_constraints()).0);
    planner.clear_path_constraints();
    let (after, path_after) = plan(&mut planner, &request, SEED);
    assert_eq!(after, before);
    assert_eq!(path_after.waypoints, path_before.waypoints);
    assert_eq!(path_after.len, path_before.len);
    assert_eq!(path_after.cost, path_before.cost);
}

fn task_resources() -> planner_resources::Resources {
    planner_resources::Resources {
        rng: Owned(CuRng::from_seed(11)),
    }
}

/// Runs a planner task to a final status (bounded) and returns its path.
fn run_task<P: CuJointPlanner<6> + Send + Sync + 'static>(
    task: &mut CuJointPlannerTask<P, 6>,
    request: &CuJointPlanRequest<6>,
) -> CuJointPlanPath<6> {
    let (ctx, _clock) = CuContext::new_mock_clock();
    let input = CuMsg::new(Some(request.clone()));
    let mut output = CuMsg::new(None);
    task.base(&ctx, &input, &mut output).unwrap();
    for _ in 0..40 {
        if task.status().is_final() {
            break;
        }
        task.refine(&ctx, &mut output).unwrap();
    }
    output.payload().cloned().unwrap_or_default()
}

#[test]
fn test_task_reads_constraints_from_ron() {
    let config: ComponentConfig = ron::from_str(
        r#"#![enable(unwrap_newtypes)] {"chain": "igus_rebel", "collision_model": "igus_rebel",
            "params": {}, "constraints": [
                {"orientation": {"tool_axis": "z", "reference": [0.0, 0.0, 1.0], "tolerance": 0.5236}},
            ]}"#,
    )
    .unwrap();
    let mut task =
        CuJointPlannerTask::<CuJointRrtConnect<6>, 6>::new(Some(&config), task_resources())
            .unwrap();
    let path = run_task(&mut task, &request());
    assert!(path.status.has_path(), "task status: {:?}", path.status);
    assert!(!dense_violation(&path, &orientation_constraints()).0);
    // Configs without the key plan exactly as before (here: out of the cone).
    let legacy: ComponentConfig = ron::from_str(
        r#"#![enable(unwrap_newtypes)] {"chain": "igus_rebel", "collision_model": "igus_rebel", "params": {}}"#,
    )
    .unwrap();
    let mut task =
        CuJointPlannerTask::<CuJointRrtConnect<6>, 6>::new(Some(&legacy), task_resources())
            .unwrap();
    let path = run_task(&mut task, &request());
    assert!(path.status.has_path(), "task status: {:?}", path.status);
    assert!(dense_violation(&path, &orientation_constraints()).0);
}

#[test]
fn test_task_rejects_invalid_constraints_and_typos() {
    for bad in [
        // Negative tolerance.
        r#"{"chain": "igus_rebel", "collision_model": "igus_rebel",
            "constraints": [{"orientation": {"tool_axis": "z", "reference": [0.0, 0.0, 1.0], "tolerance": -0.5}}]}"#,
        // Zero reference direction.
        r#"{"chain": "igus_rebel", "collision_model": "igus_rebel",
            "constraints": [{"orientation": {"tool_axis": "z", "reference": [0.0, 0.0, 0.0], "tolerance": 0.5}}]}"#,
        // Empty set.
        r#"{"chain": "igus_rebel", "collision_model": "igus_rebel", "constraints": []}"#,
        // Misspelled key still fails: unknown fields are denied.
        r#"{"chain": "igus_rebel", "collision_model": "igus_rebel", "constrain": []}"#,
        // Misspelled planner parameter still fails.
        r#"{"chain": "igus_rebel", "collision_model": "igus_rebel", "params": {"step_siz": 0.3}}"#,
    ] {
        let config: ComponentConfig =
            ron::from_str(&format!("#![enable(unwrap_newtypes)] {bad}")).unwrap();
        assert!(
            CuJointPlannerTask::<CuJointRrtConnect<6>, 6>::new(Some(&config), task_resources())
                .is_err(),
            "{bad}"
        );
    }
}

/// Converts a base-to-tool matrix into the TCP pose goal form.
fn tcp_pose_from_matrix(matrix: [[f64; 4]; 4]) -> CuTcpPose {
    let r = [
        [matrix[0][0], matrix[0][1], matrix[0][2]],
        [matrix[1][0], matrix[1][1], matrix[1][2]],
        [matrix[2][0], matrix[2][1], matrix[2][2]],
    ];
    CuTcpPose {
        translation: [matrix[0][3], matrix[1][3], matrix[2][3]].map(Length::new::<meter>),
        rotation: [
            Angle::new::<radian>(libm::atan2(r[2][1], r[2][2])),
            Angle::new::<radian>(libm::atan2(
                -r[2][0],
                libm::sqrt(r[0][0] * r[0][0] + r[1][0] * r[1][0]),
            )),
            Angle::new::<radian>(libm::atan2(r[1][0], r[0][0])),
        ],
    }
}

#[test]
fn test_pose_planner_honours_constraints() {
    let joint = request();
    let goal = tcp_pose_from_matrix(tool_pose(&joint.goal).to_matrix());
    // The pose goal reproduces the joint goal's tool pose.
    let round_trip = goal.transform().to_matrix();
    let target = tool_pose(&joint.goal).to_matrix();
    for row in 0..4 {
        for col in 0..4 {
            assert!(libm::fabs(round_trip[row][col] - target[row][col]) < 1e-9);
        }
    }
    let request = CuPosePlanRequest {
        scene: joint.scene.clone(),
        exclusions: joint.exclusions.clone(),
        start: joint.start,
        goal,
    };
    let config: ComponentConfig = ron::from_str(
        r#"#![enable(unwrap_newtypes)] {"planner": {"chain": "igus_rebel", "collision_model": "igus_rebel",
            "params": {}, "constraints": [
                {"orientation": {"tool_axis": "z", "reference": [0.0, 0.0, 1.0], "tolerance": 0.5236}},
            ]}, "goal": {"attempts": 8}}"#,
    )
    .unwrap();
    let mut task =
        CuPosePlannerTask::<CuJointRrtConnect<6>, 6>::new(Some(&config), task_resources()).unwrap();
    let (ctx, _clock) = CuContext::new_mock_clock();
    let input = CuMsg::new(Some(request.clone()));
    let mut output = CuMsg::new(None);
    task.base(&ctx, &input, &mut output).unwrap();
    for _ in 0..40 {
        let status = task.refine(&ctx, &mut output).unwrap();
        if matches!(
            status,
            cu29::cutask_anytime::AnytimeStatus::Converged(_)
                | cu29::cutask_anytime::AnytimeStatus::Aborted
        ) {
            break;
        }
    }
    let path = output.payload().cloned().unwrap_or_default();
    assert!(path.status.has_path(), "pose status: {:?}", path.status);
    assert!(!dense_violation(&path, &orientation_constraints()).0);
}

fn excluding_start_constraints() -> CuPathConstraintSet {
    let mut set = CuPathConstraintSet::new();
    set.push(CuPathConstraint::Orientation(
        CuOrientationConstraint::new(CuToolAxis::Z, [0.0, 0.0, 1.0], Angle::new::<degree>(1.0))
            .unwrap(),
    ))
    .unwrap();
    set
}

#[test]
fn test_direct_constraint_change_discards_short_pending_move() {
    let mut request = clear_request();
    request.goal = request.start;
    request.goal[0] = CuJointPosition::Revolute(Angle::new::<degree>(START[0] + 0.001));
    let mut planner = CuJointDirect::new(
        igus_rebel_chain(),
        igus_rebel_collision_model(),
        Default::default(),
    )
    .unwrap();
    assert_eq!(planner.begin(&request, 1).unwrap(), CuPlanStatus::Searching);
    planner
        .set_path_constraints(&excluding_start_constraints())
        .unwrap();
    assert_eq!(planner.step(1).unwrap(), CuPlanStatus::InvalidRequest);
    let mut path = CuJointPlanPath::default();
    assert!(!planner.write_best(&mut path).has_path());
    assert_eq!(path.len, 0);
}

#[test]
fn test_task_constraint_change_replaces_published_result() {
    let planner = CuJointDirect::new(
        igus_rebel_chain(),
        igus_rebel_collision_model(),
        Default::default(),
    )
    .unwrap();
    let mut task = CuJointPlannerTask::from_planner(planner, 1, 1, 1).unwrap();
    let (ctx, _clock) = CuContext::new_mock_clock();
    let input = CuMsg::new(Some(clear_request()));
    let mut output = CuMsg::new(None);
    task.base(&ctx, &input, &mut output).unwrap();
    assert!(output.payload().unwrap().status.has_path());
    task.set_path_constraints(&excluding_start_constraints())
        .unwrap();
    task.refine(&ctx, &mut output).unwrap();
    assert!(!output.payload().is_some_and(|path| path.status.has_path()));
}

#[test]
fn test_pose_task_constraint_changes_discard_published_result() {
    let config: ComponentConfig = ron::from_str(
        r#"#![enable(unwrap_newtypes)] {"planner": {"chain": "igus_rebel", "collision_model": "igus_rebel", "params": {}}}"#
    ).unwrap();
    let mut task =
        CuPosePlannerTask::<CuJointDirect<6>, 6>::new(Some(&config), task_resources()).unwrap();
    let joint = clear_request();
    let request = CuPosePlanRequest {
        start: joint.start,
        goal: tcp_pose_from_matrix(tool_pose(&joint.start).to_matrix()),
        scene: joint.scene,
        exclusions: joint.exclusions,
    };
    let (ctx, _clock) = CuContext::new_mock_clock();
    let input = CuMsg::new(Some(request));
    let mut output = CuMsg::new(None);
    for clear in [false, true] {
        task.clear_path_constraints();
        task.base(&ctx, &input, &mut output).unwrap();
        assert!(output.payload().unwrap().status.has_path());
        if clear {
            task.clear_path_constraints();
        } else {
            task.set_path_constraints(&excluding_start_constraints())
                .unwrap();
        }
        task.refine(&ctx, &mut output).unwrap();
        assert!(!output.payload().is_some_and(|path| path.status.has_path()));
    }
}
