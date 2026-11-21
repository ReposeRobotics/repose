//! End-to-end Physical AI Simplex Safety Demo on a 6-DOF igus ReBeL arm.
//!
//! Runs two policy checkpoints through the exact same Copper safety stack
//! (`CuJointRrtStarTask` -> `CuAuditedSetpointMuxTask` -> `CuSafetyShieldTask`
//! -> `CuControllerTask` -> `PlantBridge`) and records cycle-exact `.copper`
//! logs:
//! - `logs/simplex_v1.copper` — rogue policy (`v1`) stopped by `CuSafetyShieldTask`
//!   before colliding with the workspace obstacle, followed by RRT* retract.
//! - `logs/simplex_v2.copper` — retrained safe policy (`v2`) navigating cleanly
//!   around the obstacle with zero shield interventions.

pub mod tasks;

use cu29::prelude::*;
use std::path::Path;
use std::sync::atomic::Ordering;

#[copper_runtime(config = "copperconfig.ron")]
struct SimplexSafetyDemo {}

fn run_checkpoint(label: &str, mode: u8, log_base: &Path) -> CuResult<()> {
    tasks::POLICY_MODE.store(mode, Ordering::SeqCst);
    tasks::DONE.store(false, Ordering::SeqCst);
    tasks::SUCCESS.store(false, Ordering::SeqCst);

    cu29::replay::remove_log_family(log_base)?;
    let (clock, clock_mock) = RobotClock::mock();
    let mut app = SimplexSafetyDemo::builder()
        .with_clock(clock)
        .with_log_path(log_base, None)?
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
    eprintln!(
        "simplex-safety-demo ({label}): log written to {}",
        log_base.display()
    );
    if tasks::SUCCESS.load(Ordering::SeqCst) {
        Ok(())
    } else {
        Err(CuError::from(format!(
            "simplex-safety-demo ({label}): scenario check failed"
        )))
    }
}

fn main() -> CuResult<()> {
    let logs = Path::new(env!("CARGO_MANIFEST_DIR")).join("logs");
    std::fs::create_dir_all(&logs)
        .map_err(|e| CuError::new_with_cause("cannot create logs/", e))?;

    let arg = std::env::args().nth(1);
    match arg.as_deref() {
        Some("v1") => run_checkpoint(
            "v1_rogue",
            tasks::POLICY_MODE_V1_ROGUE,
            &logs.join("simplex_v1.copper"),
        ),
        Some("v2") => run_checkpoint(
            "v2_safe",
            tasks::POLICY_MODE_V2_SAFE,
            &logs.join("simplex_v2.copper"),
        ),
        None | Some("both") => {
            run_checkpoint(
                "v1_rogue",
                tasks::POLICY_MODE_V1_ROGUE,
                &logs.join("simplex_v1.copper"),
            )?;
            run_checkpoint(
                "v2_safe",
                tasks::POLICY_MODE_V2_SAFE,
                &logs.join("simplex_v2.copper"),
            )
        }
        Some(other) => Err(CuError::from(format!(
            "unknown argument '{other}'; expected 'v1', 'v2', or 'both'"
        ))),
    }
}
