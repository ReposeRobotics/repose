//! Execution-monitor overhead: the [`CuCollisionMonitor`] policy layer plus
//! collision checking, versus the raw collision check alone.
//!
//! The monitor is a stateful wrapper: callers feed one observation per control
//! cycle and get back a transition. This measures what that wrapper costs on
//! top of an already-running collision check.

use cu29::units::si::angle::degree;
use cu29::units::si::f64::Angle;
use divan::Bencher;
use divan::black_box;
use repose::collision::CuCollisionEnv;
use repose::collision::CuCollisionMonitor;
use repose::collision::CuCollisionReport;
use repose::collision::igus_rebel_collision_model;
use repose::kinematics::CuJointPosition;
use repose::kinematics::igus_rebel_chain;

fn main() {
    divan::main();
}

fn deg(values: [f64; 6]) -> [CuJointPosition; 6] {
    values.map(|d| CuJointPosition::Revolute(Angle::new::<degree>(d)))
}

const START: [f64; 6] = [-1.1, 45.5, 60.0, -3.3, -80.9, 28.7];

#[divan::bench]
fn monitor_overhead_raw_check(bencher: Bencher) {
    let chain = igus_rebel_chain();
    let model = igus_rebel_collision_model();
    let env = CuCollisionEnv::EMPTY;
    let q = deg(START);
    let mut report = CuCollisionReport::<16>::default();
    bencher.bench_local(|| {
        model
            .check_configuration(&chain, black_box(&q), &env, black_box(&mut report))
            .unwrap();
    });
}

#[divan::bench]
fn monitor_overhead_with_monitor(bencher: Bencher) {
    let chain = igus_rebel_chain();
    let model = igus_rebel_collision_model();
    let env = CuCollisionEnv::EMPTY;
    let q = deg(START);
    let mut report = CuCollisionReport::<16>::default();
    let mut monitor = CuCollisionMonitor::new();
    bencher.bench_local(|| {
        model
            .check_configuration(&chain, black_box(&q), &env, black_box(&mut report))
            .unwrap();
        let clear = report.is_collision_free();
        monitor.update(!clear);
    });
}
