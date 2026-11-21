//! Offline black-box incident triage, counterfactual checkpoint scorecard, and
//! 3D telemetry exporter for `simplex_safety_demo`.
//!
//! Reads `logs/simplex_v1.copper` and `logs/simplex_v2.copper`, feeds every
//! recorded control cycle through [`CuIncidentDetector`], runs
//! [`CuCheckpointDiff::compare`] to verify the counterfactual safety gate, and
//! writes:
//! - `logs/incident_v1.json` & `logs/incident_v2.json`
//! - `logs/scorecard_diff.json`
//! - `logs/telemetry_v1.json` & `logs/telemetry_v2.json` (3D joint/TCP frames +
//!   live shield/mux/incident telemetry for visualization & video rendering)

pub mod tasks;

use cu_spatial_payloads::Transform3D;
use cu29::prelude::*;
use cu29::units::si::f64::Length;
use cu29::units::si::length::meter;
use cu29::units::si::velocity::meter_per_second;
use cu29_export::copperlists_reader;
use repose_incident::CuCheckpointDiff;
use repose_incident::CuCycleSample;
use repose_incident::CuIncidentDetector;
use repose_incident::CuIncidentDetectorParams;
use repose_incident::CuIncidentSummary;
use repose_kinematics::igus_rebel_chain;
use repose_policy_rl::CuPolicyStatus;
use repose_safety::CuSafetyVerdict;
use repose_trajectory::CuSetpointSource;
use serde::Serialize;
use std::path::Path;

gen_cumsgs!("copperconfig.ron");

const OBSTACLE_CENTER: [f64; 3] = [0.34, 0.14, 0.36];
const OBSTACLE_RADIUS: f64 = 0.085;
const WORKSPACE_MIN: [f64; 3] = [-0.65, -0.65, 0.05];
const WORKSPACE_MAX: [f64; 3] = [0.65, 0.65, 0.95];

#[derive(Serialize)]
struct CycleTelemetry {
    cycle: u64,
    time_s: f64,
    links_xyz: Vec<[f64; 3]>,
    tcp_xyz: [f64; 3],
    desired_tcp_xyz: [f64; 3],
    joints_rad: [f64; 6],
    min_clearance_m: Option<f64>,
    tcp_speed_m_s: f64,
    verdict: String,
    near_miss: bool,
    mux_source: String,
    mux_blending: bool,
    mux_reason: String,
    policy_status: String,
}

#[derive(Serialize)]
struct RunTelemetry {
    label: String,
    model_hash: String,
    obstacle_center: [f64; 3],
    obstacle_radius: f64,
    workspace_min: [f64; 3],
    workspace_max: [f64; 3],
    summary: CuIncidentSummary,
    cycles: Vec<CycleTelemetry>,
}

fn open_copperlists(log_base: &Path) -> CuResult<UnifiedLoggerIOReader> {
    let UnifiedLogger::Read(reader) = UnifiedLoggerBuilder::new()
        .file_base_name(log_base)
        .build()
        .map_err(|e| CuError::new_with_cause("failed to open log", e))?
    else {
        return Err(CuError::from("log could not be opened for reading"));
    };
    Ok(UnifiedLoggerIOReader::new(
        reader,
        UnifiedLogType::CopperList,
    ))
}

fn tf_xyz(tf: &Transform3D<f64>) -> [f64; 3] {
    let m = tf.to_matrix();
    [m[0][3], m[1][3], m[2][3]]
}

fn verdict_label(v: CuSafetyVerdict) -> &'static str {
    match v {
        CuSafetyVerdict::Clear => "clear",
        CuSafetyVerdict::HeldCollision => "held_collision",
        CuSafetyVerdict::HeldBrakingLookahead => "held_braking_lookahead",
        CuSafetyVerdict::HeldWorkspaceFence => "held_workspace_fence",
        CuSafetyVerdict::HeldTcpSpeed => "held_tcp_speed",
        CuSafetyVerdict::HeldJointLimits => "held_joint_limits",
        CuSafetyVerdict::HeldMissingScene => "held_missing_scene",
        CuSafetyVerdict::HeldInvalidScene => "held_invalid_scene",
    }
}

fn source_label(s: CuSetpointSource) -> &'static str {
    match s {
        CuSetpointSource::Classical => "classical",
        CuSetpointSource::Policy => "policy",
        CuSetpointSource::Hold => "hold",
    }
}

fn policy_label(s: CuPolicyStatus) -> &'static str {
    match s {
        CuPolicyStatus::Idle => "idle",
        CuPolicyStatus::Fresh => "fresh",
        CuPolicyStatus::Held => "held",
        CuPolicyStatus::Fault => "fault",
    }
}

fn analyze_run(
    label: &str,
    model_hash: u64,
    log_base: &Path,
) -> CuResult<(CuIncidentSummary, RunTelemetry)> {
    let chain = igus_rebel_chain();
    let mut detector = CuIncidentDetector::<6>::from_params(CuIncidentDetectorParams {
        model_hash: Some(model_hash),
        near_miss_threshold: Some(Length::new::<meter>(0.04)),
        ..CuIncidentDetectorParams::default()
    })?;

    let mut reader = open_copperlists(log_base)?;
    let mut cycles = Vec::new();
    let mut frames = [Transform3D::<f64>::default(); 6];
    let mut ee = Transform3D::<f64>::default();
    let mut desired_ee = Transform3D::<f64>::default();

    for cl in copperlists_reader::<CuMsgs>(&mut reader) {
        let cycle = cl.id;
        let time_ns = cycle.saturating_mul(10_000_000);
        let mut sample = CuCycleSample::<6>::new(cycle, time_ns);

        let state = cl.msgs.get_plant_rx_joint_state().payload();
        let policy = cl.msgs.get_policy_output().payload();
        let mux_out = cl.msgs.get_mux_outputs();
        let mux_sp = mux_out.0.payload();
        let mux_rep = mux_out.1.payload();
        let shield_out = cl.msgs.get_shield_outputs();
        let shield_sp = shield_out.0.payload();
        let shield_rep = shield_out.1.payload();
        let cmd = cl.msgs.get_controller_output().payload();

        if let Some(st) = state {
            sample.with_joint_state(st);
        }
        if let Some(pol) = policy {
            sample.with_policy_action(pol);
        }
        if let Some(sp) = shield_sp {
            sample.with_follow_output(sp);
        }
        if let Some(rep) = shield_rep {
            sample.with_safety_report(rep);
        }
        if let Some(c) = cmd {
            sample.with_controller_command(c);
        }

        detector.observe(&sample)?;

        if let Some(st) = state {
            chain.forward_frames(&st.positions, &mut frames, &mut ee)?;
            let mut links_xyz = Vec::with_capacity(8);
            links_xyz.push([0.0, 0.0, 0.0]);
            for f in &frames {
                links_xyz.push(tf_xyz(f));
            }
            let tcp_xyz = tf_xyz(&ee);
            links_xyz.push(tcp_xyz);

            let desired_tcp_xyz = if let Some(sp) = mux_sp {
                chain.forward_kinematics(&sp.positions, &mut desired_ee)?;
                tf_xyz(&desired_ee)
            } else {
                tcp_xyz
            };

            let clearance = shield_rep
                .map(|r| r.min_clearance.get::<meter>())
                .filter(|v| v.is_finite());

            cycles.push(CycleTelemetry {
                cycle,
                time_s: time_ns as f64 / 1e9,
                links_xyz,
                tcp_xyz,
                desired_tcp_xyz,
                joints_rad: st.positions.map(|p| p.si_value()),
                min_clearance_m: clearance,
                tcp_speed_m_s: shield_rep
                    .map(|r| r.tcp_speed.get::<meter_per_second>())
                    .unwrap_or(0.0),
                verdict: shield_rep
                    .map(|r| verdict_label(r.verdict).to_string())
                    .unwrap_or_else(|| "clear".to_string()),
                near_miss: shield_rep.map(|r| r.near_miss).unwrap_or(false),
                mux_source: mux_rep
                    .map(|r| source_label(r.source).to_string())
                    .unwrap_or_else(|| "classical".to_string()),
                mux_blending: mux_rep.map(|r| r.blending).unwrap_or(false),
                mux_reason: mux_rep
                    .map(|r| format!("{:?}", r.reason))
                    .unwrap_or_else(|| "None".to_string()),
                policy_status: policy
                    .map(|p| policy_label(p.status).to_string())
                    .unwrap_or_else(|| "idle".to_string()),
            });
        }
    }

    let summary = detector.finish();
    let telemetry = RunTelemetry {
        label: label.to_string(),
        model_hash: format!("0x{model_hash:016x}"),
        obstacle_center: OBSTACLE_CENTER,
        obstacle_radius: OBSTACLE_RADIUS,
        workspace_min: WORKSPACE_MIN,
        workspace_max: WORKSPACE_MAX,
        summary: summary.clone(),
        cycles,
    };
    Ok((summary, telemetry))
}

fn run_triage() -> CuResult<()> {
    let logs = Path::new(env!("CARGO_MANIFEST_DIR")).join("logs");
    let (v1_summary, v1_tel) = analyze_run(
        "v1_rogue",
        0x526f_6775_655f_7631,
        &logs.join("simplex_v1.copper"),
    )?;
    let (v2_summary, v2_tel) = analyze_run(
        "v2_safe",
        0x5361_6665_5f76_3200,
        &logs.join("simplex_v2.copper"),
    )?;

    let diff = CuCheckpointDiff::compare(&v1_summary, &v2_summary);

    std::fs::write(logs.join("incident_v1.json"), v1_summary.to_json_pretty()?)
        .map_err(|e| CuError::new_with_cause("write incident_v1.json", e))?;
    std::fs::write(logs.join("incident_v2.json"), v2_summary.to_json_pretty()?)
        .map_err(|e| CuError::new_with_cause("write incident_v2.json", e))?;
    std::fs::write(
        logs.join("scorecard_diff.json"),
        serde_json::to_string_pretty(&diff)
            .map_err(|e| CuError::new_with_cause("serialize diff", e))?,
    )
    .map_err(|e| CuError::new_with_cause("write scorecard_diff.json", e))?;
    std::fs::write(
        logs.join("telemetry_v1.json"),
        serde_json::to_string(&v1_tel)
            .map_err(|e| CuError::new_with_cause("serialize telemetry_v1", e))?,
    )
    .map_err(|e| CuError::new_with_cause("write telemetry_v1.json", e))?;
    std::fs::write(
        logs.join("telemetry_v2.json"),
        serde_json::to_string(&v2_tel)
            .map_err(|e| CuError::new_with_cause("serialize telemetry_v2", e))?,
    )
    .map_err(|e| CuError::new_with_cause("write telemetry_v2.json", e))?;

    println!("=== Simplex Safety Black-Box Triage ===");
    println!(
        "v1 (rogue): cycles={} incidents={} collision_episodes={} near_miss_episodes={} policy_faults={} min_clearance={:?}",
        v1_summary.total_cycles,
        v1_summary.incidents.len(),
        v1_summary.collision_episodes,
        v1_summary.near_miss_episodes,
        v1_summary.policy_faults,
        v1_summary.min_clearance_m,
    );
    for inc in &v1_summary.incidents {
        let w = inc.window();
        println!(
            "  incident #{}: trigger={:?} blame={:?} window=[{:.2}s..={:.2}s] min_clearance={:?}",
            inc.id,
            inc.primary_trigger,
            inc.blame,
            w.start_ns as f64 / 1e9,
            w.end_ns as f64 / 1e9,
            inc.min_clearance_m
        );
    }
    println!(
        "v2 (safe):  cycles={} incidents={} collision_episodes={} near_miss_episodes={} policy_faults={} min_clearance={:?}",
        v2_summary.total_cycles,
        v2_summary.incidents.len(),
        v2_summary.collision_episodes,
        v2_summary.near_miss_episodes,
        v2_summary.policy_faults,
        v2_summary.min_clearance_m,
    );
    println!(
        "Counterfactual Gate (v1 -> v2): passed_gate={} collision_delta={} near_miss_delta={} min_clearance_delta={:?}",
        diff.passed_gate(),
        diff.collision_delta,
        diff.near_miss_delta,
        diff.min_clearance_delta_m,
    );

    if v1_summary.incidents.is_empty() {
        return Err(CuError::from("expected at least 1 incident in v1_rogue"));
    }
    if !v2_summary.incidents.is_empty() {
        return Err(CuError::from("expected 0 incidents in v2_safe"));
    }
    if !diff.passed_gate() {
        return Err(CuError::from("expected counterfactual diff to pass gate"));
    }
    Ok(())
}

fn main() -> CuResult<()> {
    let handle = std::thread::Builder::new()
        .stack_size(256 * 1024 * 1024)
        .spawn(run_triage)
        .map_err(|e| CuError::new_with_cause("spawn triage thread", e))?;
    handle
        .join()
        .map_err(|_| CuError::from("triage thread panicked"))?
}
