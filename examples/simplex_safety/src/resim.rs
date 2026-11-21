//! Deterministic resimulation of a recorded `simplex_safety` `.copper` log.
//!
//! External plant states and background RRT* planner results are injected from
//! the recording; `CuAuditedSetpointMuxTask`, `CuSafetyShieldTask`, and
//! `CuControllerTask` re-execute and are checked for bit-identical outputs.
//!
//! ```text
//! cargo run --bin simplex_safety_resim -- --log-base logs/simplex_v1.copper
//! ```

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

const DEFAULT_LOG_BASE: &str = "logs/simplex_v1.copper";
const DEFAULT_REPLAY_LOG_BASE: &str = "logs/simplex_v1_resim.copper";

#[copper_runtime(config = "copperconfig.ron", sim_mode = true)]
struct SimplexSafetyReSim {}

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
    let mut app = SimplexSafetyReSim::builder()
        .with_clock(clock)
        .with_log_path(replay_log_base, None)?
        .with_sim_callback(&mut default_callback)
        .build()?
        .into_inner();
    app.start_all_tasks(&mut default_callback)?;

    let mut keyframes = open_reader(log_base, UnifiedLogType::FrozenTasks)?;
    let mut keyframes = keyframes_reader(&mut keyframes).peekable();
    if let Some(first) = keyframes.peek() {
        <SimplexSafetyReSim as CuSimApplication<MmapSectionStorage, MmapUnifiedLoggerWrite>>::restore_keyframe(&mut app, first)?;
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
        let mux_match = ra.msgs.get_mux_outputs().0.payload()
            == rb.msgs.get_mux_outputs().0.payload()
            && ra.msgs.get_mux_outputs().1.payload() == rb.msgs.get_mux_outputs().1.payload();
        let shield_match = ra.msgs.get_shield_outputs().0.payload()
            == rb.msgs.get_shield_outputs().0.payload()
            && ra.msgs.get_shield_outputs().1.payload() == rb.msgs.get_shield_outputs().1.payload();
        let ctrl_match =
            ra.msgs.get_controller_output().payload() == rb.msgs.get_controller_output().payload();
        if !(mux_match && shield_match && ctrl_match) && first_mismatch.is_none() {
            first_mismatch = Some(ra.id);
        }
        checked += 1;
    }
    if let Some(id) = first_mismatch {
        return Err(CuError::from(format!(
            "simplex-safety-resim: mismatch at copperlist {id}"
        )));
    }
    eprintln!(
        "simplex-safety-resim: {checked} cycles bit-identical across mux, shield, and controller"
    );
    Ok(())
}

fn run() -> CuResult<()> {
    let defaults = ReplayDefaults::new(DEFAULT_LOG_BASE, DEFAULT_REPLAY_LOG_BASE);
    let cli = ReplayCli::parse(defaults);
    ensure_log_family_exists(&cli.log_base)?;
    let count = resim(&cli.log_base, &cli.replay_log_base)?;
    eprintln!("simplex-safety-resim: replayed {count} cycles");
    compare(&cli.log_base, &cli.replay_log_base)
}

fn main() -> CuResult<()> {
    let handle = std::thread::Builder::new()
        .stack_size(256 * 1024 * 1024)
        .spawn(run)
        .map_err(|e| CuError::new_with_cause("cannot spawn resim thread", e))?;
    handle
        .join()
        .map_err(|_| CuError::from("resim thread panicked"))?
}
