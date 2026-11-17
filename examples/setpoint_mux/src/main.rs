//! Classical transit → learned policy → safe fallback → classical retract,
//! closed loop on an in-process plant, recorded to `logs/setpoint_mux.copper`.
//! The graph is in `copperconfig.ron`; [`tasks`] holds the application nodes.
//!
//! The plant is simulated, so the run uses simulated time: Copper's mock
//! clock advances exactly one period per cycle (paced to wall-clock time for
//! the background planner). Every `ctx.now()` inside a cycle is therefore
//! reproducible, which is what lets `setpoint_mux_resim` replay the log
//! bit-exactly.

pub mod tasks;

use cu29::prelude::*;
use std::sync::atomic::Ordering;

#[copper_runtime(config = "copperconfig.ron")]
struct SetpointMuxDemo {}

fn main() -> CuResult<()> {
    let logs = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("logs");
    std::fs::create_dir_all(&logs)
        .map_err(|e| CuError::new_with_cause("cannot create logs/", e))?;
    let log_base = logs.join("setpoint_mux.copper");
    cu29::replay::remove_log_family(&log_base)?;
    let (clock, clock_mock) = RobotClock::mock();
    let mut app = SetpointMuxDemo::builder()
        .with_clock(clock)
        .with_log_path(&log_base, None)?
        .build()?
        .start()
        .map_err(|fault| fault.error)?;
    let period = std::time::Duration::from_millis(10);
    while !tasks::DONE.load(Ordering::SeqCst) {
        let started = std::time::Instant::now();
        clock_mock.increment(CuDuration::from(period));
        app.run_one_iteration()?;
        if let Some(rest) = period.checked_sub(started.elapsed()) {
            std::thread::sleep(rest);
        }
    }
    app.stop().map_err(|fault| fault.error)?;
    eprintln!("setpoint-mux-demo: log written to {}", log_base.display());
    if tasks::SUCCESS.load(Ordering::SeqCst) {
        Ok(())
    } else {
        Err(CuError::from(
            "setpoint-mux-demo: scenario failed, see result.json",
        ))
    }
}
