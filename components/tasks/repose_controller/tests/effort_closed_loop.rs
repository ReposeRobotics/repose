//! Simulator-free closed-loop check of effort mode on the igus ReBeL chain.
//!
//! The plant is the rigid-body model of the built-in `igus_rebel` chain under
//! gravity. Its forward dynamics come from the crate's own recursive
//! Newton-Euler `inverse_dynamics`: column `j` of the mass matrix is the
//! inverse dynamics of a unit acceleration on joint `j` with no velocity and
//! no gravity, the bias is the inverse dynamics at zero acceleration, and
//! `M(q) qdd = tau - bias` is solved each step. The state is integrated with
//! semi-implicit Euler at 2 kHz while `CuControllerTask` runs at 1 kHz with a
//! zero-order hold, tracking a quintic joint move followed by a hold.
//!
//! What this does not model: joint friction, gear and rotor inertia, torque
//! limits, sensor noise, latency, and model error. The controller's model is
//! the plant's model, so the feedforward is exact when the setpoint carries
//! the reference acceleration and misses only the inertial term when it does
//! not (the `acceleration feedforward` comparison below). The link inertias are hand estimates, not
//! igus data (see [`rebel_inertia`]).

use cu29::prelude::*;
use cu29::units::si::acceleration::meter_per_second_squared;
use cu29::units::si::angle::radian;
use cu29::units::si::angular_acceleration::radian_per_second_squared;
use cu29::units::si::angular_velocity::radian_per_second;
use cu29::units::si::f64::Acceleration;
use cu29::units::si::f64::Angle;
use cu29::units::si::f64::AngularAcceleration;
use cu29::units::si::f64::AngularVelocity;
use cu29::units::si::f64::Length;
use cu29::units::si::f64::Mass;
use cu29::units::si::f64::MomentOfInertia;
use cu29::units::si::length::meter;
use cu29::units::si::mass::kilogram;
use cu29::units::si::moment_of_inertia::kilogram_square_meter;
use repose_controller::CuControllerMode;
use repose_controller::CuControllerParams;
use repose_controller::CuControllerStatus;
use repose_controller::CuControllerTask;
use repose_controller::CuEffortGains;
use repose_controller::CuEffortModel;
use repose_controller::CuJointCommandPayload;
use repose_kinematics::CuChainInertia;
use repose_kinematics::CuInverseDynamicsInput;
use repose_kinematics::CuJointAcceleration;
use repose_kinematics::CuJointEffort;
use repose_kinematics::CuJointPosition;
use repose_kinematics::CuJointStatePayload;
use repose_kinematics::CuJointVelocity;
use repose_kinematics::CuKinematicChain;
use repose_kinematics::CuLinkInertia;
use repose_kinematics::CuRobotStatus;
use repose_kinematics::igus_rebel_chain;
use repose_kinematics::standard_gravity;
use repose_trajectory::CuFollowStatus;
use repose_trajectory::CuTrajectoryFollowOutput;

const N: usize = 6;
/// Controller period: 1 kHz.
const CONTROL_DT: f64 = 1e-3;
/// Plant integration steps per controller period.
const SUBSTEPS: usize = 2;
const MOVE_TIME: f64 = 2.0;
const HOLD_TIME: f64 = 1.0;
const START: [f64; N] = [0.0; N];
const GOAL: [f64; N] = [0.8, 0.9, 0.7, 0.5, -0.6, 0.4];

/// A solid cylinder of radius `r` and length `h` along the link's `z` axis,
/// centred `h / 2` up the link.
fn cylinder(mass: f64, r: f64, h: f64) -> CuLinkInertia {
    let transverse = mass * (3.0 * r * r + h * h) / 12.0;
    let axial = mass * r * r / 2.0;
    CuLinkInertia::new(
        Mass::new::<kilogram>(mass),
        [0.0, 0.0, h / 2.0].map(Length::new::<meter>),
        [transverse, transverse, axial, 0.0, 0.0, 0.0]
            .map(MomentOfInertia::new::<kilogram_square_meter>),
    )
    .unwrap()
}

/// Rough inertial parameters for the ReBeL chain: each link is a solid
/// cylinder spanning the distance to the next joint, with masses that add up
/// to about 6.8 kg of moving structure. These are estimates for this test,
/// not manufacturer data; the repository ships no ReBeL inertia preset.
fn rebel_inertia() -> CuChainInertia<N> {
    CuChainInertia::new([
        cylinder(1.6, 0.05, 0.106),
        cylinder(1.9, 0.045, 0.2415),
        cylinder(1.3, 0.045, 0.1575),
        cylinder(1.0, 0.04, 0.142),
        cylinder(0.7, 0.04, 0.0768),
        cylinder(0.3, 0.035, 0.04),
    ])
    .unwrap()
}

fn positions(q: &[f64; N]) -> [CuJointPosition; N] {
    q.map(|v| CuJointPosition::Revolute(Angle::new::<radian>(v)))
}

fn velocities(qd: &[f64; N]) -> [CuJointVelocity; N] {
    qd.map(|v| CuJointVelocity::Revolute(AngularVelocity::new::<radian_per_second>(v)))
}

fn accelerations(qdd: &[f64; N]) -> [CuJointAcceleration; N] {
    qdd.map(|v| {
        CuJointAcceleration::Revolute(AngularAcceleration::new::<radian_per_second_squared>(v))
    })
}

fn inverse_dynamics(
    chain: &CuKinematicChain<N>,
    inertia: &CuChainInertia<N>,
    q: &[f64; N],
    qd: &[f64; N],
    qdd: &[f64; N],
    gravity: [Acceleration; 3],
) -> [f64; N] {
    let (q, qd, qdd) = (positions(q), velocities(qd), accelerations(qdd));
    let mut efforts = [CuJointEffort::default(); N];
    chain
        .inverse_dynamics(
            inertia,
            &CuInverseDynamicsInput::new(&q, &qd, &qdd, gravity),
            &mut efforts,
        )
        .unwrap();
    efforts.map(CuJointEffort::si_value)
}

/// Solves `a x = b` by Gaussian elimination with partial pivoting.
fn solve(mut a: [[f64; N]; N], mut b: [f64; N]) -> [f64; N] {
    for col in 0..N {
        let pivot = (col..N)
            .max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs()))
            .unwrap();
        a.swap(col, pivot);
        b.swap(col, pivot);
        for row in col + 1..N {
            let factor = a[row][col] / a[col][col];
            let pivot_row = a[col];
            for (value, pivot_value) in a[row].iter_mut().zip(pivot_row).skip(col) {
                *value -= factor * pivot_value;
            }
            b[row] -= factor * b[col];
        }
    }
    let mut x = [0.0; N];
    for row in (0..N).rev() {
        let tail: f64 = (row + 1..N).map(|k| a[row][k] * x[k]).sum();
        x[row] = (b[row] - tail) / a[row][row];
    }
    x
}

/// Joint accelerations of the rigid-body plant under `tau` and gravity.
fn forward_dynamics(
    chain: &CuKinematicChain<N>,
    inertia: &CuChainInertia<N>,
    q: &[f64; N],
    qd: &[f64; N],
    tau: &[f64; N],
) -> [f64; N] {
    let no_gravity = [Acceleration::new::<meter_per_second_squared>(0.0); 3];
    let mut mass = [[0.0; N]; N];
    for j in 0..N {
        let mut unit = [0.0; N];
        unit[j] = 1.0;
        let column = inverse_dynamics(chain, inertia, q, &[0.0; N], &unit, no_gravity);
        for i in 0..N {
            mass[i][j] = column[i];
        }
    }
    let bias = inverse_dynamics(chain, inertia, q, qd, &[0.0; N], standard_gravity());
    let mut rhs = [0.0; N];
    for i in 0..N {
        rhs[i] = tau[i] - bias[i];
    }
    solve(mass, rhs)
}

/// Quintic rest-to-rest move from `START` to `GOAL`, then a hold:
/// positions, velocities and accelerations.
fn reference(t: f64) -> ([f64; N], [f64; N], [f64; N]) {
    let s = (t / MOVE_TIME).clamp(0.0, 1.0);
    let blend = s * s * s * (10.0 - 15.0 * s + 6.0 * s * s);
    let rate = 30.0 * s * s * (1.0 - s) * (1.0 - s) / MOVE_TIME;
    let curvature = if t > 0.0 && t < MOVE_TIME {
        60.0 * s * (1.0 - s) * (1.0 - 2.0 * s) / (MOVE_TIME * MOVE_TIME)
    } else {
        0.0
    };
    let mut q = [0.0; N];
    let mut qd = [0.0; N];
    let mut qdd = [0.0; N];
    for j in 0..N {
        q[j] = START[j] + (GOAL[j] - START[j]) * blend;
        qd[j] = (GOAL[j] - START[j]) * rate;
        qdd[j] = (GOAL[j] - START[j]) * curvature;
    }
    (q, qd, qdd)
}

#[derive(Debug)]
struct Report {
    /// Largest joint tracking error over the whole run (rad); infinite when
    /// the loop diverged.
    max_error: f64,
    /// RMS over time of the largest joint error (rad).
    rms_error: f64,
    /// Joint error at the end of the hold (rad).
    final_error: [f64; N],
    /// Whether every command stayed out of `Fault`.
    never_faulted: bool,
}

/// What the controller is given to feed forward.
#[derive(Clone, Copy, PartialEq)]
enum Feedforward {
    /// Pure PD, no model.
    None,
    /// Model, but the setpoint carries zero acceleration (the old follower).
    ModelZeroAcceleration,
    /// Model and the reference acceleration in the setpoint.
    ModelWithAcceleration,
}

fn simulate(gains: CuEffortGains<N>, feedforward: Feedforward) -> Report {
    let with_model = feedforward != Feedforward::None;
    let chain = igus_rebel_chain();
    let inertia = rebel_inertia();
    let mut task = CuControllerTask::<N>::from_params(CuControllerParams {
        mode: CuControllerMode::Effort,
        // Wide open, so the comparison measures tracking instead of ending
        // in a tracking fault.
        tracking_error_limit: 10.0,
        effort_gains: gains,
        effort_model: with_model.then(|| CuEffortModel::new(chain, inertia)),
        ..CuControllerParams::default()
    })
    .unwrap();
    let (ctx, _clock) = CuContext::new_mock_clock();

    let mut q = START;
    let mut qd = [0.0; N];
    let steps = ((MOVE_TIME + HOLD_TIME) / CONTROL_DT).round() as usize;
    let dt = CONTROL_DT / SUBSTEPS as f64;
    let mut max_error = 0.0f64;
    let mut sum_squares = 0.0;
    let mut never_faulted = true;
    for step in 0..steps {
        let (q_des, qd_des, qdd_des) = reference(step as f64 * CONTROL_DT);
        let qdd_sent = if feedforward == Feedforward::ModelWithAcceleration {
            qdd_des
        } else {
            [0.0; N]
        };
        let setpoint = CuMsg::new(Some(CuTrajectoryFollowOutput {
            positions: positions(&q_des),
            velocities: velocities(&qd_des),
            accelerations: accelerations(&qdd_sent),
            status: CuFollowStatus::Running,
        }));
        let state = CuMsg::new(Some(CuJointStatePayload {
            status: CuRobotStatus::Ready,
            positions: positions(&q),
            velocities: velocities(&qd),
        }));
        let mut output = CuMsg::<CuJointCommandPayload<N>>::default();
        task.process(&ctx, &(&setpoint, &state), &mut output)
            .unwrap();
        let command = output.payload().unwrap();
        never_faulted &= command.status != CuControllerStatus::Fault;
        let tau = command.efforts.map(|e| e.si_value());

        let error = (0..N)
            .map(|j| (q_des[j] - q[j]).abs())
            .fold(0.0f64, f64::max);
        max_error = max_error.max(error);
        sum_squares += error * error;
        if !error.is_finite() || error > 1e3 {
            // Diverged: stop integrating, the report says so.
            max_error = f64::INFINITY;
            break;
        }

        for _ in 0..SUBSTEPS {
            let qdd = forward_dynamics(&chain, &inertia, &q, &qd, &tau);
            for j in 0..N {
                qd[j] += qdd[j] * dt;
                q[j] += qd[j] * dt;
            }
        }
    }
    let mut final_error = [0.0; N];
    for j in 0..N {
        final_error[j] = (GOAL[j] - q[j]).abs();
    }
    Report {
        max_error,
        rms_error: (sum_squares / steps as f64).sqrt(),
        final_error,
        never_faulted,
    }
}

/// Gains scaled to each joint's inertia: stiff base joints, soft wrist.
const PER_JOINT: CuEffortGains<N> = CuEffortGains {
    kp: [100.0, 300.0, 100.0, 15.0, 15.0, 2.0],
    kv: [4.0, 10.0, 4.0, 0.3, 0.3, 0.05],
};

#[test]
fn feedforward_removes_gravity_sag_at_equal_gains() {
    let pd = simulate(PER_JOINT, Feedforward::None);
    let model = simulate(PER_JOINT, Feedforward::ModelZeroAcceleration);
    let largest = |errors: &[f64; N]| errors.iter().copied().fold(0.0f64, f64::max);

    assert!(pd.never_faulted && model.never_faulted);
    // Pure PD settles where kp * error balances gravity: about 0.066 rad on
    // joint 3 (6.6 N m against kp = 100).
    assert!(largest(&pd.final_error) > 0.03, "{pd:?}");
    // With the model the hold error is gone and the error during the move,
    // left by the acceleration term when the setpoint carries none, is far
    // smaller.
    assert!(largest(&model.final_error) < 1e-3, "{model:?}");
    assert!(model.max_error < pd.max_error / 5.0, "{model:?} vs {pd:?}");
    assert!(model.rms_error < pd.rms_error / 5.0, "{model:?} vs {pd:?}");
}

#[test]
fn model_based_per_joint_control_beats_scalar_pd() {
    let model = simulate(PER_JOINT, Feedforward::ModelWithAcceleration);
    assert!(model.never_faulted);
    assert!(model.max_error < 0.02, "{model:?}");

    // The default scalar gains (kp = 20, kv = 0.8) on every joint: the
    // damping is beyond the 1 kHz zero-order-hold limit `kv * T / I < 2` of
    // the last wrist joint (I is about 1.8e-4 kg m² in this model), so the
    // loop diverges.
    let default_pd = simulate(CuEffortGains::default(), Feedforward::None);
    assert!(default_pd.max_error.is_infinite(), "{default_pd:?}");

    // A scalar pair the wrist tolerates is far too soft for the shoulder.
    let soft_pd = simulate(CuEffortGains::uniform(20.0, 0.3), Feedforward::None);
    assert!(soft_pd.never_faulted);
    assert!(
        model.max_error < soft_pd.max_error / 50.0,
        "{model:?} vs {soft_pd:?}"
    );
}

#[test]
fn acceleration_feedforward_cuts_tracking_error() {
    let pd = simulate(PER_JOINT, Feedforward::None);
    let without = simulate(PER_JOINT, Feedforward::ModelZeroAcceleration);
    let with = simulate(PER_JOINT, Feedforward::ModelWithAcceleration);
    let largest = |errors: &[f64; N]| errors.iter().copied().fold(0.0f64, f64::max);
    for (name, r) in [
        ("pure PD", &pd),
        ("model, zero acceleration", &without),
        ("model + acceleration", &with),
    ] {
        println!(
            "{name:<26} max {:.3e} rad  rms {:.3e} rad  final {:.3e} rad",
            r.max_error,
            r.rms_error,
            largest(&r.final_error)
        );
    }
    assert!(pd.never_faulted && without.never_faulted && with.never_faulted);
    // The plant is the controller's model, so the full feedforward leaves only
    // the 1 kHz zero-order-hold and integration error.
    assert!(
        with.max_error < without.max_error / 20.0,
        "{with:?} vs {without:?}"
    );
    assert!(
        with.rms_error < without.rms_error / 20.0,
        "{with:?} vs {without:?}"
    );
    assert!(with.max_error < 1e-3, "{with:?}");
    assert!(largest(&with.final_error) < 1e-3, "{with:?}");
}
