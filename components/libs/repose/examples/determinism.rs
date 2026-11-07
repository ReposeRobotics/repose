//! Prints one hash per scenario for the determinism check (`just determinism`).
//!
//! Every scenario uses the benchmark inputs on the 6-DOF igus ReBeL with a
//! fixed seed, runs twice in this process, and fails if the two outputs
//! differ. `just determinism` then runs this binary in several separate
//! processes and fails if any process prints a different hash.
//!
//! Outputs are hashed through their `Debug` text with 64-bit FNV-1a. `f64`
//! prints in its shortest round-trip form, so any change in the bits changes
//! the hash. The hashes are only compared between runs of the same binary;
//! they are not stable across targets, compilers, or dependency versions.

use core::fmt;
use core::fmt::Write;

use cu_spatial_payloads::Transform3D;
use cu29::prelude::CuError;
use cu29::prelude::CuResult;
use cu29::units::si::angle::degree;
use cu29::units::si::angular_acceleration::radian_per_second_squared;
use cu29::units::si::angular_jerk::radian_per_second_cubed;
use cu29::units::si::angular_velocity::radian_per_second;
use cu29::units::si::f64::Angle;
use cu29::units::si::f64::AngularAcceleration;
use cu29::units::si::f64::AngularJerk;
use cu29::units::si::f64::AngularVelocity;
use cu29::units::si::f64::Length;
use cu29::units::si::f64::Time;
use cu29::units::si::length::meter;
use cu29::units::si::time::second;
use repose::collision::CuCapsule;
use repose::collision::CuCollisionEnv;
use repose::collision::CuCollisionReport;
use repose::collision::CuCollisionShape;
use repose::collision::CuPlanningScene;
use repose::collision::CuSphere;
use repose::collision::igus_rebel_collision_model;
use repose::kinematics::CuIkParams;
use repose::kinematics::CuJointAcceleration;
use repose::kinematics::CuJointJerk;
use repose::kinematics::CuJointPosition;
use repose::kinematics::CuJointVelocity;
use repose::kinematics::igus_rebel_chain;
use repose::motion_planning::CuJointPlanPath;
use repose::motion_planning::CuJointPlanRequest;
use repose::motion_planning::CuJointPlanner;
use repose::planners::CuAnyPlanner;
use repose::planners::CuPlannerChoice;
use repose::trajectory::CuJointTrajectoryLimits;
use repose::trajectory::CuTimingOverrides;
use repose::trajectory::CuTimingParams;
use repose::trajectory::CuTimingStatus;
use repose::trajectory::CuTrajectoryPayload;
use repose::trajectory::CuTrajectoryProfile;
use repose::trajectory::parameterize;

const START: [f64; 6] = [-1.1, 45.5, 60.0, -3.3, -80.9, 28.7];
const GOAL: [f64; 6] = [0.0, 30.0, 60.0, 0.0, 60.0, 0.0];
/// Planner goal, far enough from `START` that the mid-point obstacle blocks the straight line.
const PLAN_GOAL: [f64; 6] = [90.0, 30.0, 60.0, 0.0, 60.0, 0.0];
const SEED: u64 = 1;
/// Upper bound on `step` calls, so every planner does a fixed amount of work.
const MAX_STEPS: u32 = 64;
const STEP_BUDGET: u32 = 64;
const CAPACITY: usize = 1024;
const PROFILES: [CuTrajectoryProfile; 4] = [
    CuTrajectoryProfile::Trapezoidal,
    CuTrajectoryProfile::SCurve,
    CuTrajectoryProfile::MinimumJerk,
    CuTrajectoryProfile::Cubic,
];

/// A named scenario returning the hash of its output.
type Scenario = (String, Box<dyn Fn() -> CuResult<u64>>);

/// 64-bit FNV-1a over everything written to it.
struct Fnv1a(u64);

impl Fnv1a {
    fn new() -> Self {
        Self(0xcbf2_9ce4_8422_2325)
    }
}

impl Write for Fnv1a {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        for byte in s.bytes() {
            self.0 ^= u64::from(byte);
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
        }
        Ok(())
    }
}

fn hash_debug(value: &impl fmt::Debug) -> CuResult<u64> {
    let mut hasher = Fnv1a::new();
    write!(hasher, "{value:?}")
        .map_err(|_| CuError::from("formatting a scenario output failed"))?;
    Ok(hasher.0)
}

fn deg(values: [f64; 6]) -> [CuJointPosition; 6] {
    values.map(|d| CuJointPosition::Revolute(Angle::new::<degree>(d)))
}

fn m(v: [f64; 3]) -> [Length; 3] {
    v.map(Length::new::<meter>)
}

fn forward_kinematics() -> CuResult<u64> {
    let mut tcp = Transform3D::<f64>::identity();
    igus_rebel_chain().forward_kinematics(&deg(START), &mut tcp)?;
    hash_debug(&tcp)
}

fn geometric_jacobian() -> CuResult<u64> {
    let mut jac = [[0.0_f64; 6]; 6];
    igus_rebel_chain().geometric_jacobian(&deg(START), &mut jac)?;
    hash_debug(&jac)
}

fn inverse_kinematics_dls() -> CuResult<u64> {
    let chain = igus_rebel_chain();
    let mut target = Transform3D::<f64>::identity();
    chain.forward_kinematics(&deg(GOAL), &mut target)?;
    let mut solution = deg(START);
    let status = chain.inverse_kinematics_dls(
        &target,
        &deg(START),
        &CuIkParams::default(),
        &mut solution,
    )?;
    hash_debug(&(status, solution))
}

/// A table, two posts and a ring of spheres around the robot (the collision benchmark scene).
fn cluttered_scene() -> CuResult<CuPlanningScene> {
    let mut scene = CuPlanningScene::new();
    let radius = Length::new::<meter>(0.05);
    for (a, b) in [
        ([-1.5, -1.5, -0.2], [1.5, -1.5, -0.2]),
        ([-1.2, 1.2, 0.0], [-1.2, 1.2, 1.5]),
        ([1.2, 1.2, 0.0], [1.2, 1.2, 1.5]),
    ] {
        let capsule = CuCapsule::new(m(a), m(b), radius)?;
        scene.add_obstacle(CuCollisionShape::Capsule(capsule), None)?;
    }
    for i in 0..8 {
        let a = core::f64::consts::TAU * f64::from(i) / 8.0;
        let sphere = CuSphere::new(
            m([1.6 * a.cos(), 1.6 * a.sin(), 1.8]),
            Length::new::<meter>(0.08),
        )?;
        scene.add_obstacle(CuCollisionShape::Sphere(sphere), None)?;
    }
    Ok(scene)
}

fn check_configuration(q: &[CuJointPosition; 6], env: &CuCollisionEnv) -> CuResult<u64> {
    let mut report = CuCollisionReport::<16>::default();
    igus_rebel_collision_model().check_configuration(&igus_rebel_chain(), q, env, &mut report)?;
    hash_debug(&report)
}

fn check_configuration_self() -> CuResult<u64> {
    check_configuration(&deg(START), &CuCollisionEnv::EMPTY)
}

fn check_configuration_scene() -> CuResult<u64> {
    check_configuration(&deg(START), &cluttered_scene()?.collision_env())
}

/// The planning mid-point against its blocking sphere: a report with real contacts and depths.
fn check_configuration_contact() -> CuResult<u64> {
    let request = plan_request()?;
    check_configuration(&plan_mid(), &request.scene.collision_env())
}

fn segment_collision_free_scene() -> CuResult<u64> {
    let scene = cluttered_scene()?;
    let free = igus_rebel_collision_model().is_segment_collision_free(
        &igus_rebel_chain(),
        &deg(START),
        &deg(GOAL),
        Angle::new::<degree>(2.0),
        &scene.collision_env(),
    )?;
    hash_debug(&free)
}

/// Half-way along the straight joint line from `START` to `PLAN_GOAL`.
fn plan_mid() -> [CuJointPosition; 6] {
    deg(core::array::from_fn(|j| 0.5 * (START[j] + PLAN_GOAL[j])))
}

/// Start and goal with a sphere on the tool path of the straight joint line
/// (the planning benchmark request).
fn plan_request() -> CuResult<CuJointPlanRequest<6>> {
    let mut request = CuJointPlanRequest {
        start: deg(START),
        goal: deg(PLAN_GOAL),
        ..Default::default()
    };
    let mut tcp = Transform3D::<f64>::identity();
    igus_rebel_chain().forward_kinematics(&plan_mid(), &mut tcp)?;
    let sphere = CuSphere::new(tcp.translation(), Length::new::<meter>(0.1))?;
    request
        .scene
        .add_obstacle(CuCollisionShape::Sphere(sphere), None)?;
    Ok(request)
}

fn plan(choice: CuPlannerChoice) -> CuResult<u64> {
    let request = plan_request()?;
    let mut planner =
        CuAnyPlanner::<6>::new(igus_rebel_chain(), igus_rebel_collision_model(), choice)?;
    let mut status = planner.begin(&request, SEED)?;
    let mut steps = 0;
    while !status.is_final() && steps < MAX_STEPS {
        status = planner.step(STEP_BUDGET)?;
        steps += 1;
    }
    let mut path = Box::new(CuJointPlanPath::<6>::default());
    planner.write_best(&mut path);
    hash_debug(&(status, steps, &path))
}

fn timing_params(profile: CuTrajectoryProfile) -> CuTimingParams<6> {
    CuTimingParams {
        limits: CuJointTrajectoryLimits::new(
            [CuJointVelocity::Revolute(AngularVelocity::new::<radian_per_second>(1.0)); 6],
            [CuJointAcceleration::Revolute(AngularAcceleration::new::<radian_per_second_squared>(
                2.0,
            )); 6],
        )
        .with_jerks([CuJointJerk::Revolute(AngularJerk::new::<radian_per_second_cubed>(10.0)); 6]),
        profile,
        sample_period: Time::new::<second>(0.01),
    }
}

fn waypoints() -> [[CuJointPosition; 6]; 5] {
    [
        deg(START),
        deg([10.0, 40.0, 55.0, 5.0, -40.0, 20.0]),
        deg([20.0, 35.0, 50.0, 10.0, 0.0, 10.0]),
        deg([10.0, 30.0, 55.0, 5.0, 30.0, 5.0]),
        deg(GOAL),
    ]
}

fn timed_trajectory(
    profile: CuTrajectoryProfile,
) -> CuResult<Box<CuTrajectoryPayload<6, CAPACITY>>> {
    let mut traj = Box::new(CuTrajectoryPayload::<6, CAPACITY>::new());
    let status = parameterize(
        &waypoints(),
        &timing_params(profile),
        CuTimingOverrides::default(),
        &mut traj,
    )?;
    if status != CuTimingStatus::Success {
        return Err(CuError::from(format!(
            "timing with {profile:?} failed: {status:?}"
        )));
    }
    Ok(traj)
}

fn parameterize_profile(profile: CuTrajectoryProfile) -> CuResult<u64> {
    hash_debug(&timed_trajectory(profile)?)
}

/// Samples the S-curve trajectory at 1 kHz, as a follower task would.
fn sample_quintic_1khz() -> CuResult<u64> {
    let traj = timed_trajectory(CuTrajectoryProfile::SCurve)?;
    let samples = (traj.duration().get::<second>() * 1000.0) as usize;
    let mut positions = [CuJointPosition::default(); 6];
    let mut velocities = [CuJointVelocity::default(); 6];
    let mut hasher = Fnv1a::new();
    for i in 0..samples {
        traj.sample_quintic(
            Time::new::<second>(i as f64 * 1e-3),
            &mut positions,
            &mut velocities,
        )?;
        write!(hasher, "{positions:?}{velocities:?}")
            .map_err(|_| CuError::from("formatting a trajectory sample failed"))?;
    }
    Ok(hasher.0)
}

/// Runs a scenario twice in this process and returns its hash if both runs agree.
fn run_twice(name: &str, scenario: impl Fn() -> CuResult<u64>) -> CuResult<u64> {
    let run_a = scenario()?;
    let run_b = scenario()?;
    if run_a != run_b {
        return Err(CuError::from(format!(
            "{name}: in-process runs differ ({run_a:016x} != {run_b:016x})"
        )));
    }
    Ok(run_a)
}

fn main() -> CuResult<()> {
    let mut scenarios: Vec<Scenario> = vec![
        ("forward_kinematics".into(), Box::new(forward_kinematics)),
        ("geometric_jacobian".into(), Box::new(geometric_jacobian)),
        (
            "inverse_kinematics_dls".into(),
            Box::new(inverse_kinematics_dls),
        ),
        (
            "check_configuration_self".into(),
            Box::new(check_configuration_self),
        ),
        (
            "check_configuration_scene".into(),
            Box::new(check_configuration_scene),
        ),
        (
            "check_configuration_contact".into(),
            Box::new(check_configuration_contact),
        ),
        (
            "segment_collision_free_scene".into(),
            Box::new(segment_collision_free_scene),
        ),
    ];
    for &choice in CuPlannerChoice::ALL {
        scenarios.push((format!("plan/{choice:?}"), Box::new(move || plan(choice))));
    }
    for profile in PROFILES {
        scenarios.push((
            format!("parameterize/{profile:?}"),
            Box::new(move || parameterize_profile(profile)),
        ));
    }
    scenarios.push(("sample_quintic_1khz".into(), Box::new(sample_quintic_1khz)));

    for (name, scenario) in &scenarios {
        let hash = run_twice(name, scenario)?;
        println!("{name} {hash:016x}");
    }
    Ok(())
}
