//! Converts the demo's Copper log into a Rerun recording. Offline: the demo
//! itself never logs to Rerun.
//!
//! ```text
//! cargo run --features rrd --bin setpoint_mux_to_rrd                   # logs/setpoint_mux.copper -> logs/setpoint_mux.rrd
//! cargo run --features rrd --bin setpoint_mux_to_rrd -- in.copper out.rrd
//! rerun logs/setpoint_mux.rrd
//! ```
//!
//! Entities, on the `cu_time` timeline (Copper clock) and the `copperlist`
//! sequence:
//! - `joints/<n>/{measured,follower,policy,mux,command}`: joint `n` position
//!   in radians, for the plant, both mux inputs, the mux output and the
//!   controller command, so each handoff blend is visible per joint
//! - `velocity/<n>`: mux setpoint velocity, radians per second
//! - `world/arm/measured`, `world/arm/command`: the arm as a line through its
//!   joint frames, from the planner's kinematic chain in `copperconfig.ron`
//! - `world/tcp/<k>`: measured tool path, in segments of up to 256 points
//! - `events`: follower, policy, mux and controller status changes, as text

pub mod tasks;

use cu_spatial_payloads::Transform3D;
use cu29::bincode::config::standard;
use cu29::bincode::decode_from_std_read;
use cu29::bincode::error::DecodeError;
use cu29::copperlist::CopperList;
use cu29::prelude::*;
use repose_controller::CuJointCommandPayload;
use repose_kinematics::CuChainSpec;
use repose_kinematics::CuJointPosition;
use repose_kinematics::CuJointStatePayload;
use repose_kinematics::CuKinematicChain;
use repose_trajectory::CuTrajectoryFollowOutput;
use rerun::RecordingStream;
use rerun::blueprint::Blueprint;
use rerun::blueprint::Horizontal;
use rerun::blueprint::Spatial3DView;
use rerun::blueprint::TextLogView;
use rerun::blueprint::TimeSeriesView;
use rerun::blueprint::Vertical;
use serde::Deserialize;
use std::io::Read;
use std::path::Path;
use std::path::PathBuf;

gen_cumsgs!("copperconfig.ron");

const N: usize = 6;
const DEFAULT_LOG_BASE: &str = "logs/setpoint_mux.copper";
/// Points per logged tool-path segment; bounds the work per cycle.
const TCP_SEGMENT_POINTS: usize = 256;

const MEASURED_COLOR: [u8; 3] = [80, 170, 255];
const FOLLOWER_COLOR: [u8; 3] = [120, 200, 120];
const POLICY_COLOR: [u8; 3] = [200, 120, 220];
const MUX_COLOR: [u8; 3] = [255, 150, 60];
const COMMAND_COLOR: [u8; 3] = [230, 80, 80];

/// The plotted joint series: entity suffix and color.
const SERIES: [(&str, [u8; 3]); 5] = [
    ("measured", MEASURED_COLOR),
    ("follower", FOLLOWER_COLOR),
    ("policy", POLICY_COLOR),
    ("mux", MUX_COLOR),
    ("command", COMMAND_COLOR),
];

/// The measured tool path, logged as consecutive bounded segments.
#[derive(Default)]
struct ToolPath {
    segment: usize,
    points: Vec<[f32; 3]>,
}

impl ToolPath {
    fn log(&mut self, rec: &RecordingStream, tool: [f32; 3]) -> CuResult<()> {
        if self.points.len() == TCP_SEGMENT_POINTS {
            // Start the next segment where this one ended, so the path stays joined.
            let last = self.points[TCP_SEGMENT_POINTS - 1];
            self.points.clear();
            self.points.push(last);
            self.segment += 1;
        }
        self.points.push(tool);
        rec.log(
            format!("world/tcp/{}", self.segment),
            &rerun::LineStrips3D::new([self.points.clone()])
                .with_colors([MEASURED_COLOR])
                .with_radii([0.002]),
        )
        .map_err(rerun_error)
    }
}

/// Counts the bytes read, to tell a clean end of log from a truncated record.
struct CountingReader<R> {
    inner: R,
    bytes: u64,
}

impl<R: Read> Read for CountingReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let n = self.inner.read(buf)?;
        self.bytes += n as u64;
        Ok(n)
    }
}

/// Next copperlist, `None` at a clean end of the log. A malformed or
/// truncated record, or a read error, is an error (`copperlists_reader`
/// would end silently and hide a damaged log).
fn next_copperlist<R: Read>(src: &mut CountingReader<R>) -> CuResult<Option<CopperList<CuMsgs>>> {
    let start = src.bytes;
    match decode_from_std_read::<CopperList<CuMsgs>, _, _>(src, standard()) {
        Ok(list) => Ok(Some(list)),
        Err(e) => {
            let at_end = match &e {
                DecodeError::UnexpectedEnd { .. } => true,
                DecodeError::Io { inner, .. } => inner.kind() == std::io::ErrorKind::UnexpectedEof,
                _ => false,
            };
            if at_end && src.bytes == start {
                Ok(None)
            } else {
                Err(CuError::from(format!(
                    "cannot decode the copperlist at byte {start} of the log: {e}"
                )))
            }
        }
    }
}

/// Whether `a` and `b` are the same file. On Unix this compares device and
/// inode, so hard links and symlinks both count; elsewhere it compares
/// canonical paths.
fn same_file(a: &Path, b: &Path) -> bool {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        match (std::fs::metadata(a), std::fs::metadata(b)) {
            (Ok(a), Ok(b)) => a.dev() == b.dev() && a.ino() == b.ino(),
            _ => false,
        }
    }
    #[cfg(not(unix))]
    {
        match (std::fs::canonicalize(a), std::fs::canonicalize(b)) {
            (Ok(a), Ok(b)) => a == b,
            _ => false,
        }
    }
}

/// Rejects an output that is the input log or one of its slabs
/// (`<stem>_<k>.<ext>`), by file identity so links count: saving would
/// truncate the log being read.
fn ensure_distinct_output(input: &Path, output: &Path) -> CuResult<()> {
    if !output.exists() {
        // The output does not exist yet, so it cannot be an input file.
        return Ok(());
    }
    let stem = input.file_stem().unwrap_or_default().to_string_lossy();
    let ext = input.extension().map(|e| e.to_string_lossy());
    let slab = |k: usize| {
        let name = match &ext {
            Some(ext) => format!("{stem}_{k}.{ext}"),
            None => format!("{stem}_{k}"),
        };
        input.with_file_name(name)
    };
    let candidates =
        std::iter::once(input.to_path_buf()).chain((0..).map(slab).take_while(|p| p.exists()));
    for candidate in candidates {
        if same_file(&candidate, output) {
            return Err(CuError::from(format!(
                "output {} is the Copper log {} itself; choose a different output",
                output.display(),
                candidate.display()
            )));
        }
    }
    Ok(())
}

/// The `planner` task config; only the chain is needed here.
#[derive(Deserialize)]
struct ChainConfig {
    chain: CuChainSpec<N>,
}

fn rerun_error(e: impl std::error::Error + Send + Sync + 'static) -> CuError {
    CuError::new_with_cause("setpoint-mux-to-rrd: Rerun", e)
}

fn load_chain() -> CuResult<CuKinematicChain<N>> {
    let config_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("copperconfig.ron");
    let config = read_configuration(&config_path.to_string_lossy())?;
    let params: ChainConfig = config
        .find_task_node(None, "planner")
        .and_then(|node| node.get_instance_config())
        .ok_or_else(|| CuError::from("no `planner` task config in copperconfig.ron"))?
        .deserialize_into()?;
    params.chain.resolve()
}

fn translation(t: &Transform3D<f64>) -> [f32; 3] {
    // Row-major homogeneous matrix: the translation is the last column.
    let m = t.to_matrix();
    [m[0][3] as f32, m[1][3] as f32, m[2][3] as f32]
}

/// Logs the arm as a line through base, joint frames and tool; returns the
/// tool position. A pose the chain rejects (non-finite) is skipped.
fn log_arm(
    rec: &RecordingStream,
    entity: &str,
    chain: &CuKinematicChain<N>,
    q: &[CuJointPosition; N],
    color: [u8; 3],
) -> CuResult<Option<[f32; 3]>> {
    let mut frames = [Transform3D::<f64>::identity(); N];
    let mut tool = Transform3D::<f64>::identity();
    if chain.forward_frames(q, &mut frames, &mut tool).is_err() {
        return Ok(None);
    }
    let mut points = Vec::with_capacity(N + 2);
    points.push([0.0; 3]);
    points.extend(frames.iter().map(translation));
    points.push(translation(&tool));
    let tcp = points.last().copied();
    rec.log(
        entity,
        &rerun::LineStrips3D::new([points.clone()])
            .with_colors([color])
            .with_radii([0.006]),
    )
    .map_err(rerun_error)?;
    rec.log(
        format!("{entity}/joints"),
        &rerun::Points3D::new(points)
            .with_colors([color])
            .with_radii([0.012]),
    )
    .map_err(rerun_error)?;
    Ok(tcp)
}

fn log_positions(
    rec: &RecordingStream,
    series: &str,
    positions: &[CuJointPosition; N],
) -> CuResult<()> {
    for (i, q) in positions.iter().enumerate() {
        rec.log(
            format!("joints/{}/{series}", i + 1),
            &rerun::Scalars::single(q.si_value()),
        )
        .map_err(rerun_error)?;
    }
    Ok(())
}

fn log_setpoint(
    rec: &RecordingStream,
    series: &str,
    msg: &CuMsg<CuTrajectoryFollowOutput<N>>,
) -> CuResult<()> {
    match msg.payload() {
        Some(setpoint) => log_positions(rec, series, &setpoint.positions),
        None => Ok(()),
    }
}

/// Logs `who: current` under `events` when `current` differs from `last`.
fn log_change<T: PartialEq + core::fmt::Debug>(
    rec: &RecordingStream,
    who: &str,
    last: &mut Option<T>,
    current: Option<T>,
) -> CuResult<()> {
    if current.is_some() && current != *last {
        if let Some(value) = &current {
            rec.log("events", &rerun::TextLog::new(format!("{who}: {value:?}")))
                .map_err(rerun_error)?;
        }
        *last = current;
    }
    Ok(())
}

/// Default viewer layout: the arm in 3D beside per-joint plots and events.
fn blueprint() -> Blueprint {
    let joints = (1..=N)
        .map(|n| {
            TimeSeriesView::new(format!("Joint {n} (rad)"))
                .with_origin(format!("joints/{n}"))
                .into()
        })
        .collect::<Vec<_>>();
    let plots = Vertical::new([
        Vertical::new(joints).into(),
        TimeSeriesView::new("Mux setpoint velocity (rad/s)")
            .with_origin("velocity")
            .into(),
        TextLogView::new("Events").with_origin("events").into(),
    ])
    .with_row_shares([4.0, 1.0, 1.0]);
    Blueprint::new(
        Horizontal::new([
            Spatial3DView::new("Arm").with_origin("world").into(),
            plots.into(),
        ])
        .with_column_shares([2.0, 3.0]),
    )
}

fn log_series_styles(rec: &RecordingStream) -> CuResult<()> {
    for n in 1..=N {
        for (series, color) in SERIES {
            rec.log_static(
                format!("joints/{n}/{series}"),
                &rerun::SeriesLines::new()
                    .with_names([series])
                    .with_colors([color]),
            )
            .map_err(rerun_error)?;
        }
        rec.log_static(
            format!("velocity/{n}"),
            &rerun::SeriesLines::new().with_names([format!("joint {n}")]),
        )
        .map_err(rerun_error)?;
    }
    Ok(())
}

/// Copper time of a cycle: when the plant state was valid, else when it was
/// processed.
fn cycle_time(msg: &CuMsg<CuJointStatePayload<N>>) -> Option<CuTime> {
    match msg.tov {
        Tov::Time(t) => Some(t),
        Tov::Range(r) => Some(r.end),
        Tov::None => Option::<CuTime>::from(msg.metadata.process_time.end),
    }
}

/// Status of every reported stage, to log only the changes.
#[derive(Default)]
struct Statuses {
    follower: Option<repose_trajectory::CuFollowStatus>,
    policy: Option<repose_policy_rl::CuPolicyStatus>,
    mux: Option<repose_trajectory::CuFollowStatus>,
    controller: Option<repose_controller::CuControllerStatus>,
}

fn convert(input: &Path, output: &Path) -> CuResult<u64> {
    ensure_distinct_output(input, output)?;
    let chain = load_chain()?;
    let UnifiedLogger::Read(log) = UnifiedLoggerBuilder::new()
        .file_base_name(input)
        .build()
        .map_err(|e| CuError::new_with_cause("cannot open the Copper log", e))?
    else {
        return Err(CuError::from("expected a readable Copper log"));
    };
    let mut reader = CountingReader {
        inner: UnifiedLoggerIOReader::new(log, UnifiedLogType::CopperList),
        bytes: 0,
    };

    let rec = rerun::RecordingStreamBuilder::new("setpoint_mux")
        .with_blueprint(blueprint())
        .save(output)
        .map_err(rerun_error)?;
    rec.log_static("world", &rerun::ViewCoordinates::RIGHT_HAND_Z_UP())
        .map_err(rerun_error)?;
    log_series_styles(&rec)?;

    let mut tcp_path = ToolPath::default();
    let mut statuses = Statuses::default();
    let mut cycles = 0u64;
    let mut failure = None;
    loop {
        let list = match next_copperlist(&mut reader) {
            Ok(Some(list)) => list,
            Ok(None) => break,
            Err(e) => {
                failure = Some(e);
                break;
            }
        };
        let msgs = &list.msgs;
        let state = msgs.get_plant_rx_joint_state();
        rec.set_time_sequence("copperlist", list.id as i64);
        if let Some(t) = cycle_time(state) {
            rec.set_duration_secs("cu_time", t.as_nanos() as f64 * 1e-9);
        }

        if let Some(state) = state.payload() {
            log_positions(&rec, "measured", &state.positions)?;
            if let Some(tool) = log_arm(
                &rec,
                "world/arm/measured",
                &chain,
                &state.positions,
                MEASURED_COLOR,
            )? {
                tcp_path.log(&rec, tool)?;
            }
        }
        log_setpoint(&rec, "follower", msgs.get_follower_output())?;
        log_setpoint(&rec, "policy", msgs.get_joint_action_output())?;
        log_setpoint(&rec, "mux", msgs.get_mux_output())?;
        if let Some(mux) = msgs.get_mux_output().payload() {
            for (i, v) in mux.velocities.iter().enumerate() {
                rec.log(
                    format!("velocity/{}", i + 1),
                    &rerun::Scalars::single(v.si_value()),
                )
                .map_err(rerun_error)?;
            }
        }
        let command: &CuMsg<CuJointCommandPayload<N>> = msgs.get_controller_output();
        if let Some(command) = command.payload() {
            log_positions(&rec, "command", &command.positions)?;
            log_arm(
                &rec,
                "world/arm/command",
                &chain,
                &command.positions,
                COMMAND_COLOR,
            )?;
        }

        log_change(
            &rec,
            "follower",
            &mut statuses.follower,
            msgs.get_follower_output().payload().map(|p| p.status),
        )?;
        log_change(
            &rec,
            "policy",
            &mut statuses.policy,
            msgs.get_policy_output().payload().map(|p| p.status),
        )?;
        log_change(
            &rec,
            "mux",
            &mut statuses.mux,
            msgs.get_mux_output().payload().map(|p| p.status),
        )?;
        log_change(
            &rec,
            "controller",
            &mut statuses.controller,
            command.payload().map(|p| p.status),
        )?;
        cycles += 1;
    }
    // Keep what was converted, but do not report a cut-short log as success.
    rec.flush_blocking().map_err(rerun_error)?;
    match failure {
        Some(e) => {
            eprintln!(
                "setpoint-mux-to-rrd: the log ended early after {cycles} copperlists; {} holds only those",
                output.display()
            );
            Err(e)
        }
        None => Ok(cycles),
    }
}

fn run() -> CuResult<()> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut args = std::env::args_os().skip(1).map(PathBuf::from);
    let input = args.next().unwrap_or_else(|| root.join(DEFAULT_LOG_BASE));
    let output = args.next().unwrap_or_else(|| input.with_extension("rrd"));
    let cycles = convert(&input, &output)?;
    println!(
        "setpoint-mux-to-rrd: {cycles} copperlists from {} -> {}",
        input.display(),
        output.display()
    );
    Ok(())
}

fn main() -> CuResult<()> {
    // Copperlists hold 1024-point trajectories; decoding them needs more than
    // the default main-thread stack.
    std::thread::Builder::new()
        .stack_size(256 * 1024 * 1024)
        .spawn(run)
        .map_err(|e| CuError::new_with_cause("cannot spawn converter thread", e))?
        .join()
        .unwrap_or_else(|_| Err(CuError::from("converter panicked")))
}
