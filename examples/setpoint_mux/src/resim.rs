//! Deterministic resimulation of a recorded demo log.
//!
//! Recorded external inputs (the plant joint state) and the background
//! planner's outputs are injected from the log; every other task, including
//! `CuSetpointMuxTask` and the controller, executes again. The resim log is
//! then compared with the recording: every mux and controller output must be
//! bit-identical.
//!
//! ```text
//! cargo run --bin setpoint_mux_resim -- --log-base logs/setpoint_mux.copper
//! ```

// Replay harness: drives the raw lifecycle API on purpose (keyframe locking
// and forced timestamps are not exposed by the typestate handles).
#![allow(deprecated)]

pub mod tasks;

use cu29::prelude::app::CuSimApplication;
use cu29::prelude::*;
use cu29::replay::ReplayCli;
use cu29::replay::ReplayDefaults;
use cu29::replay::ensure_log_family_exists;
use cu29::replay::remove_log_family;
use cu29_export::copperlists_reader;
use cu29_export::keyframes_reader;
use cu29_unifiedlog::memmap::MmapSectionStorage;
use cu29_unifiedlog::memmap::MmapUnifiedLoggerWrite;
use std::path::Path;

const DEFAULT_LOG_BASE: &str = "logs/setpoint_mux.copper";
const DEFAULT_REPLAY_LOG_BASE: &str = "logs/setpoint_mux_resim.copper";

#[copper_runtime(config = "copperconfig.ron", sim_mode = true)]
struct SetpointMuxReSim {}

type Recorded = CopperList<default::CuStampedDataSet>;

fn default_callback(_step: default::SimStep) -> SimOverride {
    SimOverride::ExecuteByRuntime
}

fn open_reader(log_base: &Path, log_type: UnifiedLogType) -> CuResult<UnifiedLoggerIOReader> {
    let UnifiedLogger::Read(reader) = UnifiedLoggerBuilder::new()
        .file_base_name(log_base)
        .build()
        .map_err(|e| CuError::new_with_cause("failed to open log", e))?
    else {
        return Err(CuError::from("log could not be opened for reading"));
    };
    Ok(UnifiedLoggerIOReader::new(reader, log_type))
}

fn replay_step(step: default::SimStep<'_>, recorded: &Recorded) -> SimOverride {
    use CuTaskCallbackState::Process;
    match step {
        // The anytime planner runs in the background: its results arrive on
        // wall-clock-dependent cycles, so they are replayed as recorded.
        default::SimStep::Planner(Process(_, output)) => {
            *output = recorded.msgs.get_planner_output().clone();
            SimOverride::ExecutedBySim
        }
        step => default::recorded_debug_replay_step(step, recorded),
    }
}

fn resim(log_base: &Path, replay_log_base: &Path) -> CuResult<usize> {
    remove_log_family(replay_log_base)?;
    let (clock, clock_mock) = RobotClock::mock();
    let mut app = SetpointMuxReSim::builder()
        .with_clock(clock)
        .with_log_path(replay_log_base, None)?
        .with_sim_callback(&mut default_callback)
        .build()?
        .into_inner();
    app.start_all_tasks(&mut default_callback)?;

    let mut keyframes = open_reader(log_base, UnifiedLogType::FrozenTasks)?;
    let mut keyframes = keyframes_reader(&mut keyframes).peekable();
    if let Some(first) = keyframes.peek() {
        <SetpointMuxReSim as CuSimApplication<MmapSectionStorage, MmapUnifiedLoggerWrite>>::restore_keyframe(&mut app, first)?;
        clock_mock.set_value(first.timestamp.as_nanos());
    }

    let mut lists = open_reader(log_base, UnifiedLogType::CopperList)?;
    let mut count = 0;
    for recorded in copperlists_reader::<default::CuStampedDataSet>(&mut lists) {
        if let Some(kf) = keyframes.peek()
            && kf.culistid == recorded.id
        {
            app.copper_runtime_mut().lock_keyframe(kf);
            app.copper_runtime_mut()
                .set_forced_keyframe_timestamp(kf.timestamp);
            keyframes.next();
        }
        // The cycle time: the latest process start in the copperlist (the
        // earliest can be a background planner result from a previous cycle).
        let cycle_time = recorded
            .msgs
            .cumsgs()
            .into_iter()
            .filter_map(|msg| Option::<CuTime>::from(msg.metadata().process_time().start))
            .max();
        if let Some(ts) = cycle_time {
            clock_mock.set_value(ts.as_nanos());
        }
        let mut callback = |step: default::SimStep<'_>| replay_step(step, &recorded);
        app.run_one_iteration(&mut callback)?;
        count += 1;
    }
    app.stop_all_tasks(&mut default_callback)?;
    Ok(count)
}

fn compare(log_base: &Path, replay_log_base: &Path) -> CuResult<()> {
    let mut a = open_reader(log_base, UnifiedLogType::CopperList)?;
    let mut b = open_reader(replay_log_base, UnifiedLogType::CopperList)?;
    let mut checked = 0usize;
    let mut first_mismatch = None;
    for (ra, rb) in copperlists_reader::<default::CuStampedDataSet>(&mut a)
        .zip(copperlists_reader::<default::CuStampedDataSet>(&mut b))
    {
        let mux_same = ra.msgs.get_mux_output().payload() == rb.msgs.get_mux_output().payload();
        let ctl_same =
            ra.msgs.get_controller_output().payload() == rb.msgs.get_controller_output().payload();
        if !(mux_same && ctl_same) && first_mismatch.is_none() {
            first_mismatch = Some(ra.id);
        }
        checked += 1;
    }
    match first_mismatch {
        None => {
            eprintln!(
                "setpoint-mux-resim: {checked} copperlists, mux and controller outputs bit-identical"
            );
            Ok(())
        }
        Some(id) => Err(CuError::from(format!(
            "setpoint-mux-resim: replay diverged at copperlist {id} ({checked} compared)"
        ))),
    }
}

fn run() -> CuResult<()> {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let cli = ReplayCli::parse(ReplayDefaults::new(
        manifest.join(DEFAULT_LOG_BASE),
        manifest.join(DEFAULT_REPLAY_LOG_BASE),
    ));
    ensure_log_family_exists(&cli.log_base)?;
    let count = resim(&cli.log_base, &cli.replay_log_base)?;
    eprintln!(
        "setpoint-mux-resim: replayed {count} copperlists into {}",
        cli.replay_log_base.display()
    );
    compare(&cli.log_base, &cli.replay_log_base)
}

fn main() -> CuResult<()> {
    // Copperlists hold 1024-point trajectories; decoding them needs more than
    // the default main-thread stack.
    std::thread::Builder::new()
        .stack_size(256 * 1024 * 1024)
        .spawn(run)
        .map_err(|e| CuError::new_with_cause("cannot spawn resim thread", e))?
        .join()
        .unwrap_or_else(|_| Err(CuError::from("resim panicked")))
}
