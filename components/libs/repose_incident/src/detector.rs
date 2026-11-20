//! Black-box per-cycle incident detection, root-cause attribution, and windowed manifest generation.

use cu29::prelude::*;
use cu29::units::si::f64::Length;
use cu29::units::si::f64::Time;
use cu29::units::si::length::meter;
use cu29::units::si::time::second;
use repose_collision::CuCollisionMonitor;
use repose_collision::CuMonitorTransition;
use repose_collision::CuSceneRevision;
use repose_controller::CuControllerStatus;
use repose_kinematics::CuJointPosition;
use repose_kinematics::CuJointVelocity;
use repose_kinematics::CuRobotStatus;
use repose_policy_rl::CuPolicyStatus;
use repose_trajectory::CuFollowStatus;
use serde::Deserialize;
use serde::Serialize;

const DEFAULT_PRE_WINDOW_S: f64 = 3.0;
const DEFAULT_POST_WINDOW_S: f64 = 2.0;
const DEFAULT_COOLDOWN_S: f64 = 1.0;
const DEFAULT_NEAR_MISS_THRESHOLD_M: f64 = 0.02;
const DEFAULT_CLAMP_BURST_CYCLES: usize = 10;

/// Maximum duration representable in `u64` nanoseconds (`~584` years) without
/// saturating conversions.
const MAX_DURATION_S: f64 = (u64::MAX as f64) / 1e9;

/// Lightweight per-cycle snapshot fed into [`CuIncidentDetector`].
///
/// Built by a `.copper` logreader, simulation harness, or live ring-buffer
/// observer from the recorded task payloads of one control cycle.
#[derive(Default, Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Reflect)]
#[reflect(no_field_bounds, from_reflect = false)]
pub struct CuCycleSample<const N: usize> {
    /// Monotonic cycle index (typically the `CopperList` ID).
    pub cycle: u64,
    /// Cycle timestamp in nanoseconds.
    pub time_ns: u64,
    /// Health reported by the robot or simulator bridge.
    pub robot_status: Option<CuRobotStatus>,
    /// Policy inference status for this cycle.
    pub policy_status: Option<CuPolicyStatus>,
    /// Trajectory follower / action adapter status for this cycle.
    pub follow_status: Option<CuFollowStatus>,
    /// Joint controller status for this cycle.
    pub controller_status: Option<CuControllerStatus>,
    /// Minimum signed surface clearance reported by collision checking or a
    /// safety shield. Non-finite values (such as `+inf` when no pairs were
    /// checked) are ignored when tracking minimum clearance.
    pub min_clearance: Option<Length>,
    /// Whether collision checking reported contact / safety-margin violation.
    pub in_collision: bool,
    /// Whether the action adapter or safety shield clamped/limited the action
    /// this cycle (step, translation, rotation, or singularity damping).
    pub action_clamped: bool,
    /// Monotonic planning scene revision observed on this cycle.
    pub scene_revision: Option<CuSceneRevision>,
    /// Measured or commanded joint positions for single-cycle jump detection.
    #[serde(default, with = "crate::serde_arrays::option_position_array")]
    pub positions: Option<[CuJointPosition; N]>,
    /// Measured or commanded joint velocities for this cycle.
    #[serde(default, with = "crate::serde_arrays::option_velocity_array")]
    pub velocities: Option<[CuJointVelocity; N]>,
}

impl<const N: usize> CuCycleSample<N> {
    /// Creates an empty cycle sample at `cycle` and `time_ns`.
    pub const fn new(cycle: u64, time_ns: u64) -> Self {
        Self {
            cycle,
            time_ns,
            robot_status: None,
            policy_status: None,
            follow_status: None,
            controller_status: None,
            min_clearance: None,
            in_collision: false,
            action_clamped: false,
            scene_revision: None,
            positions: None,
            velocities: None,
        }
    }
}

/// Specific edge or threshold event that initiated or joined an incident.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Reflect)]
#[serde(rename_all = "snake_case")]
pub enum CuIncidentTrigger {
    /// Transitioned from collision-free into collision (`CuCollisionMonitor`).
    CollisionEntered,
    /// Clearance dropped below `near_miss_threshold` without colliding.
    NearMiss { clearance_m: f64 },
    /// Policy status transitioned into [`CuPolicyStatus::Fault`].
    PolicyFault,
    /// Consecutive clamped cycles reached `clamp_burst_cycles`.
    ActionClampedBurst,
    /// Controller status transitioned into [`CuControllerStatus::Fault`].
    ControllerFault,
    /// Robot bridge status transitioned out of [`CuRobotStatus::Ready`].
    RobotFault,
    /// Single-cycle joint position jump exceeded `max_position_step`.
    SetpointDiscontinuity { joint: u16, step_si: f64 },
}

impl CuIncidentTrigger {
    const fn priority(self) -> u8 {
        match self {
            Self::PolicyFault => 7,
            Self::ActionClampedBurst => 6,
            Self::SetpointDiscontinuity { .. } => 5,
            Self::CollisionEntered => 4,
            Self::NearMiss { .. } => 3,
            Self::RobotFault => 2,
            Self::ControllerFault => 1,
        }
    }
}

/// Deterministic root-cause attribution for a coalesced incident episode.
///
/// Priority when multiple signals fire in the same cooldown window:
/// `LearnedPolicy` > `EnvironmentChange` > `TrajectoryOrCollision` >
/// `HardwareOrBridge` (when robot is not `Ready`) > `ControllerOrTracking`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Reflect)]
#[serde(rename_all = "snake_case")]
pub enum CuIncidentBlame {
    /// Policy faulted, triggered a clamp burst, or drove a setpoint discontinuity.
    LearnedPolicy,
    /// Collision or near-miss immediately following a `CuSceneRevision` increment.
    EnvironmentChange,
    /// Collision or near-miss on an unchanged scene without a policy signal.
    TrajectoryOrCollision,
    /// Controller tracking or watchdog fault while the robot remained `Ready`.
    ControllerOrTracking,
    /// Robot bridge reported `Fault`, `EmergencyStop`, or `NotReady`.
    HardwareOrBridge,
}

/// Parameters of [`CuIncidentDetector`].
///
/// ```ron
/// {
///     "pre_window": 3.0,
///     "post_window": 2.0,
///     "cooldown": 1.0,
///     "near_miss_threshold": 0.02,
///     "clamp_burst_cycles": 10,
/// }
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Reflect)]
#[serde(default, deny_unknown_fields)]
pub struct CuIncidentDetectorParams {
    /// Context window before the trigger timestamp (default `3.0` s).
    pub pre_window: Time,
    /// Recovery/hold window after the trigger timestamp (default `2.0` s).
    pub post_window: Time,
    /// Episode coalescing window (default `1.0` s): triggers within this
    /// duration of an episode's opening trigger merge into one [`CuIncidentRecord`].
    pub cooldown: Time,
    /// Clearance threshold below which non-colliding cycles trigger
    /// [`CuIncidentTrigger::NearMiss`] (default `Some(0.02 m)`). `None` disables
    /// near-miss detection.
    pub near_miss_threshold: Option<Length>,
    /// Maximum allowed single-cycle joint step in SI units (`rad` for revolute,
    /// `m` for prismatic; dimensionless across mixed joint kinds). `None`
    /// disables [`CuIncidentTrigger::SetpointDiscontinuity`].
    pub max_position_step: Option<f64>,
    /// Consecutive clamped cycles required to trigger
    /// [`CuIncidentTrigger::ActionClampedBurst`] (default `10`; `0` disables).
    pub clamp_burst_cycles: usize,
    /// Optional pinned ONNX FNV-1a model hash carried into manifests and
    /// incident records for provenance.
    pub model_hash: Option<u64>,
}

impl Default for CuIncidentDetectorParams {
    fn default() -> Self {
        Self {
            pre_window: Time::new::<second>(DEFAULT_PRE_WINDOW_S),
            post_window: Time::new::<second>(DEFAULT_POST_WINDOW_S),
            cooldown: Time::new::<second>(DEFAULT_COOLDOWN_S),
            near_miss_threshold: Some(Length::new::<meter>(DEFAULT_NEAR_MISS_THRESHOLD_M)),
            max_position_step: None,
            clamp_burst_cycles: DEFAULT_CLAMP_BURST_CYCLES,
            model_hash: None,
        }
    }
}

impl CuIncidentDetectorParams {
    /// Checks that all windows, thresholds, and limits are valid and finite.
    pub fn validate(&self) -> CuResult<()> {
        validate_non_negative_time(self.pre_window, "pre_window")?;
        validate_non_negative_time(self.post_window, "post_window")?;
        validate_non_negative_time(self.cooldown, "cooldown")?;

        if let Some(threshold) = self.near_miss_threshold {
            let m = threshold.get::<meter>();
            if !m.is_finite() || m < 0.0 {
                return Err(CuError::from(
                    "repose-incident: 'near_miss_threshold' must be finite and >= 0",
                ));
            }
        }
        if let Some(step) = self.max_position_step
            && (!step.is_finite() || step <= 0.0)
        {
            return Err(CuError::from(
                "repose-incident: 'max_position_step' must be positive and finite",
            ));
        }
        Ok(())
    }
}

fn validate_non_negative_time(t: Time, name: &str) -> CuResult<()> {
    let s = t.get::<second>();
    if !s.is_finite() || s < 0.0 || s > MAX_DURATION_S {
        return Err(CuError::from(format!(
            "repose-incident: '{name}' must be finite, >= 0, and representable in nanoseconds"
        )));
    }
    Ok(())
}

fn to_nanos(t: Time) -> u64 {
    let ns = t.get::<second>() * 1e9;
    if ns <= 0.0 { 0 } else { (ns + 0.5) as u64 }
}

/// Time slice `[start_ns, end_ns]` for keyframe clip extraction around an incident.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Reflect)]
pub struct CuIncidentWindow {
    /// Inclusive window start timestamp in nanoseconds.
    pub start_ns: u64,
    /// Inclusive window end timestamp in nanoseconds.
    pub end_ns: u64,
}

impl CuIncidentWindow {
    /// Creates a window from explicit nanosecond bounds.
    pub const fn new(start_ns: u64, end_ns: u64) -> Self {
        Self { start_ns, end_ns }
    }

    /// Window span in seconds (`0.0` if `end_ns <= start_ns`).
    pub fn duration_s(&self) -> f64 {
        self.end_ns.saturating_sub(self.start_ns) as f64 * 1e-9
    }

    /// Returns `true` when `time_ns` lies inside `[start_ns, end_ns]`.
    pub const fn contains_ns(&self, time_ns: u64) -> bool {
        time_ns >= self.start_ns && time_ns <= self.end_ns
    }
}

/// One coalesced incident episode with keyframe clip bounds and root-cause blame.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Reflect)]
pub struct CuIncidentRecord {
    /// Sequential 1-based episode identifier within the run.
    pub id: u64,
    /// Highest-priority trigger observed in this episode.
    pub primary_trigger: CuIncidentTrigger,
    /// Deterministic root-cause attribution across all triggers in the episode.
    pub blame: CuIncidentBlame,
    /// Cycle index of the trigger that opened the episode.
    pub trigger_cycle: u64,
    /// Timestamp (ns) of the trigger that opened the episode.
    pub trigger_time_ns: u64,
    /// Start of the extraction window (`trigger_time_ns.saturating_sub(pre_window_ns)`).
    pub window_start_ns: u64,
    /// End of the extraction window (`trigger_time_ns.saturating_add(post_window_ns)`).
    pub window_end_ns: u64,
    /// Latest planning scene revision observed when the episode triggered.
    pub scene_revision: Option<u64>,
    /// Pinned model hash from [`CuIncidentDetectorParams::model_hash`].
    pub model_hash: Option<u64>,
    /// Minimum surface clearance (meters) observed during the episode.
    pub min_clearance_m: Option<f64>,
    /// Peak single-cycle joint position step (SI rad or m) during the episode.
    pub peak_position_step_si: f64,
    /// Number of triggers coalesced into this episode.
    pub event_count: usize,
}

impl CuIncidentRecord {
    /// Returns the extraction [`CuIncidentWindow`] for this record.
    pub const fn window(&self) -> CuIncidentWindow {
        CuIncidentWindow::new(self.window_start_ns, self.window_end_ns)
    }
}

/// Run-wide aggregate summary and incident list for `ci.getrepose.dev` and Flightdeck.
#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize, Reflect)]
pub struct CuIncidentSummary {
    /// Total cycles observed by the detector.
    pub total_cycles: u64,
    /// Elapsed run duration in seconds (`(last_time_ns - first_time_ns) * 1e-9`).
    pub duration_s: f64,
    /// Pinned model hash from [`CuIncidentDetectorParams::model_hash`].
    pub model_hash: Option<u64>,
    /// Number of collision-entry episodes (`CuCollisionMonitor::EnteredCollision`).
    pub collision_episodes: usize,
    /// Number of near-miss-entry episodes.
    pub near_miss_episodes: usize,
    /// Number of policy fault edges (`CuPolicyStatus::Fault`).
    pub policy_faults: usize,
    /// Number of controller fault edges (`CuControllerStatus::Fault`).
    pub controller_faults: usize,
    /// Total cycles where `action_clamped` was `true`.
    pub clamped_cycles: u64,
    /// Minimum finite surface clearance (meters) across the entire run.
    pub min_clearance_m: Option<f64>,
    /// Peak single-cycle joint position step (SI rad or m) across the run.
    pub peak_position_step_si: f64,
    /// Coalesced incident episodes ordered by trigger time.
    pub incidents: Vec<CuIncidentRecord>,
}

impl CuIncidentSummary {
    /// Serializes the summary as pretty-printed JSON (`incident_report.json`).
    pub fn to_json_pretty(&self) -> CuResult<String> {
        serde_json::to_string_pretty(self).map_err(|e| {
            CuError::new_with_cause("repose-incident: failed to serialize incident summary", e)
        })
    }
}

#[derive(Default, Debug, Clone, Copy)]
struct EpisodeFlags {
    has_policy_signal: bool,
    has_collision_signal: bool,
    has_env_change: bool,
    has_robot_fault: bool,
}

impl EpisodeFlags {
    fn record_trigger(&mut self, trigger: CuIncidentTrigger, recent_scene_change: bool) {
        match trigger {
            CuIncidentTrigger::PolicyFault
            | CuIncidentTrigger::ActionClampedBurst
            | CuIncidentTrigger::SetpointDiscontinuity { .. } => {
                self.has_policy_signal = true;
            }
            CuIncidentTrigger::CollisionEntered | CuIncidentTrigger::NearMiss { .. } => {
                self.has_collision_signal = true;
                if recent_scene_change {
                    self.has_env_change = true;
                }
            }
            CuIncidentTrigger::RobotFault => {
                self.has_robot_fault = true;
            }
            CuIncidentTrigger::ControllerFault => {}
        }
    }

    const fn resolve_blame(self) -> CuIncidentBlame {
        if self.has_policy_signal {
            CuIncidentBlame::LearnedPolicy
        } else if self.has_collision_signal {
            if self.has_env_change {
                CuIncidentBlame::EnvironmentChange
            } else {
                CuIncidentBlame::TrajectoryOrCollision
            }
        } else if self.has_robot_fault {
            CuIncidentBlame::HardwareOrBridge
        } else {
            CuIncidentBlame::ControllerOrTracking
        }
    }
}

/// Streaming black-box incident detector over per-cycle [`CuCycleSample`] frames.
///
/// Edge-detects collision entry, near-misses, policy/controller/robot faults,
/// clamp bursts, and setpoint discontinuities, coalesces cascading triggers
/// within [`CuIncidentDetectorParams::cooldown`], attributes deterministic
/// root-cause blame, and builds a [`CuIncidentSummary`].
///
/// # Example
/// ```rust
/// use cu29::units::si::f64::Length;
/// use cu29::units::si::length::meter;
/// use repose_controller::CuControllerStatus;
/// use repose_incident::{
///     CuCycleSample, CuIncidentBlame, CuIncidentDetector, CuIncidentDetectorParams,
/// };
/// use repose_kinematics::CuRobotStatus;
/// use repose_policy_rl::CuPolicyStatus;
///
/// let mut detector = CuIncidentDetector::<6>::from_params(CuIncidentDetectorParams::default())?;
/// let mut s0 = CuCycleSample::new(0, 0);
/// s0.robot_status = Some(CuRobotStatus::Ready);
/// s0.policy_status = Some(CuPolicyStatus::Fresh);
/// s0.controller_status = Some(CuControllerStatus::Running);
/// s0.min_clearance = Some(Length::new::<meter>(0.12));
/// detector.observe(&s0)?;
///
/// let mut s1 = CuCycleSample::new(1, 20_000_000);
/// s1.robot_status = Some(CuRobotStatus::Ready);
/// s1.policy_status = Some(CuPolicyStatus::Fault);
/// s1.controller_status = Some(CuControllerStatus::Fault);
/// s1.min_clearance = Some(Length::new::<meter>(0.10));
/// detector.observe(&s1)?;
///
/// let summary = detector.finish();
/// assert_eq!(summary.incidents.len(), 1);
/// assert_eq!(summary.incidents[0].blame, CuIncidentBlame::LearnedPolicy);
/// # Ok::<(), cu29::CuError>(())
/// ```
#[derive(Debug, Clone)]
pub struct CuIncidentDetector<const N: usize> {
    params: CuIncidentDetectorParams,
    pre_window_ns: u64,
    post_window_ns: u64,
    cooldown_ns: u64,
    collision_monitor: CuCollisionMonitor,
    in_near_miss: bool,
    in_policy_fault: bool,
    in_controller_fault: bool,
    in_robot_fault: bool,
    consecutive_clamped: usize,
    prev_positions: Option<[CuJointPosition; N]>,
    prev_scene_revision: Option<CuSceneRevision>,
    last_scene_increment: Option<(u64, u64)>,
    total_cycles: u64,
    first_time_ns: Option<u64>,
    last_time_ns: Option<u64>,
    collision_episodes: usize,
    near_miss_episodes: usize,
    policy_faults: usize,
    controller_faults: usize,
    clamped_cycles: u64,
    min_clearance_m: Option<f64>,
    peak_position_step_si: f64,
    next_id: u64,
    active_flags: EpisodeFlags,
    incidents: Vec<CuIncidentRecord>,
}

impl<const N: usize> Default for CuIncidentDetector<N> {
    fn default() -> Self {
        let params = CuIncidentDetectorParams::default();
        Self::with_valid_params(params)
    }
}

impl<const N: usize> CuIncidentDetector<N> {
    /// Creates a detector from validated parameters.
    pub fn from_params(params: CuIncidentDetectorParams) -> CuResult<Self> {
        params.validate()?;
        Ok(Self::with_valid_params(params))
    }

    fn with_valid_params(params: CuIncidentDetectorParams) -> Self {
        Self {
            pre_window_ns: to_nanos(params.pre_window),
            post_window_ns: to_nanos(params.post_window),
            cooldown_ns: to_nanos(params.cooldown),
            params,
            collision_monitor: CuCollisionMonitor::new(),
            in_near_miss: false,
            in_policy_fault: false,
            in_controller_fault: false,
            in_robot_fault: false,
            consecutive_clamped: 0,
            prev_positions: None,
            prev_scene_revision: None,
            last_scene_increment: None,
            total_cycles: 0,
            first_time_ns: None,
            last_time_ns: None,
            collision_episodes: 0,
            near_miss_episodes: 0,
            policy_faults: 0,
            controller_faults: 0,
            clamped_cycles: 0,
            min_clearance_m: None,
            peak_position_step_si: 0.0,
            next_id: 1,
            active_flags: EpisodeFlags::default(),
            incidents: Vec::new(),
        }
    }

    /// Returns the detector configuration.
    pub const fn params(&self) -> &CuIncidentDetectorParams {
        &self.params
    }

    /// Resets accumulated run state and incidents while keeping [`Self::params`].
    pub fn reset(&mut self) {
        *self = Self::with_valid_params(self.params);
    }

    /// Processes a batch of samples in order and returns the run summary.
    pub fn summarize<'a>(
        params: CuIncidentDetectorParams,
        samples: impl IntoIterator<Item = &'a CuCycleSample<N>>,
    ) -> CuResult<CuIncidentSummary> {
        let mut detector = Self::from_params(params)?;
        detector.observe_all(samples)?;
        Ok(detector.finish())
    }

    /// Feeds an iterator of samples into the detector in order.
    pub fn observe_all<'a>(
        &mut self,
        samples: impl IntoIterator<Item = &'a CuCycleSample<N>>,
    ) -> CuResult<()> {
        for sample in samples {
            self.observe(sample)?;
        }
        Ok(())
    }

    /// Feeds one cycle sample into the detector.
    ///
    /// Fails if `sample.time_ns` moves backwards relative to the previous sample.
    pub fn observe(&mut self, sample: &CuCycleSample<N>) -> CuResult<()> {
        if let Some(prev_time) = self.last_time_ns
            && sample.time_ns < prev_time
        {
            return Err(CuError::from(
                "repose-incident: sample time_ns must be monotonically non-decreasing",
            ));
        }

        self.total_cycles = self.total_cycles.saturating_add(1);
        if self.first_time_ns.is_none() {
            self.first_time_ns = Some(sample.time_ns);
        }
        self.last_time_ns = Some(sample.time_ns);

        if let Some(curr_rev) = sample.scene_revision {
            if let Some(prev_rev) = self.prev_scene_revision
                && curr_rev > prev_rev
            {
                self.last_scene_increment = Some((sample.cycle, sample.time_ns));
            }
            self.prev_scene_revision = Some(curr_rev);
        }

        let cycle_clearance_m = sample
            .min_clearance
            .map(|c| c.get::<meter>())
            .filter(|m| m.is_finite());
        if let Some(m) = cycle_clearance_m {
            self.min_clearance_m = Some(self.min_clearance_m.map_or(m, |cur| cur.min(m)));
        }

        let (cycle_peak_step_si, step_discontinuity) = self.measure_position_step(sample);
        self.peak_position_step_si = self.peak_position_step_si.max(cycle_peak_step_si);

        let mut triggers: Vec<CuIncidentTrigger> = Vec::new();

        let policy_fault_now = sample.policy_status == Some(CuPolicyStatus::Fault);
        if policy_fault_now && !self.in_policy_fault {
            self.policy_faults = self.policy_faults.saturating_add(1);
            triggers.push(CuIncidentTrigger::PolicyFault);
        }
        self.in_policy_fault = policy_fault_now;

        if sample.action_clamped {
            self.clamped_cycles = self.clamped_cycles.saturating_add(1);
            self.consecutive_clamped = self.consecutive_clamped.saturating_add(1);
            if self.params.clamp_burst_cycles > 0
                && self.consecutive_clamped == self.params.clamp_burst_cycles
            {
                triggers.push(CuIncidentTrigger::ActionClampedBurst);
            }
        } else {
            self.consecutive_clamped = 0;
        }

        if let Some((joint, step_si)) = step_discontinuity {
            triggers.push(CuIncidentTrigger::SetpointDiscontinuity { joint, step_si });
        }

        if self.collision_monitor.update(sample.in_collision)
            == CuMonitorTransition::EnteredCollision
        {
            self.collision_episodes = self.collision_episodes.saturating_add(1);
            triggers.push(CuIncidentTrigger::CollisionEntered);
        }

        let near_miss_now = if sample.in_collision {
            false
        } else if let (Some(threshold), Some(c_m)) =
            (self.params.near_miss_threshold, cycle_clearance_m)
        {
            c_m < threshold.get::<meter>()
        } else {
            false
        };
        if near_miss_now && !self.in_near_miss {
            self.near_miss_episodes = self.near_miss_episodes.saturating_add(1);
            if let Some(clearance_m) = cycle_clearance_m {
                triggers.push(CuIncidentTrigger::NearMiss { clearance_m });
            }
        }
        self.in_near_miss = near_miss_now;

        let robot_fault_now =
            matches!(sample.robot_status, Some(status) if status != CuRobotStatus::Ready);
        if robot_fault_now && !self.in_robot_fault {
            triggers.push(CuIncidentTrigger::RobotFault);
        }
        self.in_robot_fault = robot_fault_now;

        let controller_fault_now = sample.controller_status == Some(CuControllerStatus::Fault);
        if controller_fault_now && !self.in_controller_fault {
            self.controller_faults = self.controller_faults.saturating_add(1);
            triggers.push(CuIncidentTrigger::ControllerFault);
        }
        self.in_controller_fault = controller_fault_now;

        let recent_scene_change = self.has_recent_scene_change(sample);

        if let Some(first_trigger) = triggers.first().copied() {
            let coalesce = self.incidents.last().is_some_and(|last| {
                sample.time_ns <= last.trigger_time_ns.saturating_add(self.cooldown_ns)
            });

            if coalesce {
                for &t in &triggers {
                    self.active_flags.record_trigger(t, recent_scene_change);
                }
                let blame = self.active_flags.resolve_blame();
                if let Some(last) = self.incidents.last_mut() {
                    for &t in &triggers {
                        if t.priority() > last.primary_trigger.priority() {
                            last.primary_trigger = t;
                        }
                    }
                    last.blame = blame;
                    last.event_count = last.event_count.saturating_add(triggers.len());
                    if let Some(m) = cycle_clearance_m {
                        last.min_clearance_m =
                            Some(last.min_clearance_m.map_or(m, |cur| cur.min(m)));
                    }
                    last.peak_position_step_si = last.peak_position_step_si.max(cycle_peak_step_si);
                    if let Some(rev) = sample.scene_revision {
                        last.scene_revision = Some(rev.as_u64());
                    }
                }
            } else {
                let mut flags = EpisodeFlags {
                    has_robot_fault: robot_fault_now,
                    ..EpisodeFlags::default()
                };
                let mut primary = first_trigger;
                for &t in &triggers {
                    flags.record_trigger(t, recent_scene_change);
                    if t.priority() > primary.priority() {
                        primary = t;
                    }
                }
                self.active_flags = flags;
                let id = self.next_id;
                self.next_id = self.next_id.saturating_add(1);
                self.incidents.push(CuIncidentRecord {
                    id,
                    primary_trigger: primary,
                    blame: flags.resolve_blame(),
                    trigger_cycle: sample.cycle,
                    trigger_time_ns: sample.time_ns,
                    window_start_ns: sample.time_ns.saturating_sub(self.pre_window_ns),
                    window_end_ns: sample.time_ns.saturating_add(self.post_window_ns),
                    scene_revision: sample.scene_revision.map(CuSceneRevision::as_u64),
                    model_hash: self.params.model_hash,
                    min_clearance_m: cycle_clearance_m,
                    peak_position_step_si: cycle_peak_step_si,
                    event_count: triggers.len(),
                });
            }
        } else if let Some(last) = self.incidents.last_mut()
            && sample.time_ns <= last.trigger_time_ns.saturating_add(self.cooldown_ns)
        {
            if let Some(m) = cycle_clearance_m {
                last.min_clearance_m = Some(last.min_clearance_m.map_or(m, |cur| cur.min(m)));
            }
            last.peak_position_step_si = last.peak_position_step_si.max(cycle_peak_step_si);
        }

        Ok(())
    }

    fn measure_position_step(&mut self, sample: &CuCycleSample<N>) -> (f64, Option<(u16, f64)>) {
        let Some(curr_q) = sample.positions else {
            self.prev_positions = None;
            return (0.0, None);
        };
        let mut cycle_peak = 0.0_f64;
        let mut worst_joint = 0_u16;
        if let Some(prev_q) = &self.prev_positions {
            for (j, (prev, curr)) in prev_q.iter().zip(curr_q.iter()).enumerate() {
                let step = (curr.si_value() - prev.si_value()).abs();
                if step.is_finite() && step > cycle_peak {
                    cycle_peak = step;
                    worst_joint = u16::try_from(j).unwrap_or(u16::MAX);
                }
            }
        }
        self.prev_positions = Some(curr_q);
        let discontinuity = self
            .params
            .max_position_step
            .filter(|&limit| cycle_peak > limit)
            .map(|_| (worst_joint, cycle_peak));
        (cycle_peak, discontinuity)
    }

    fn has_recent_scene_change(&self, sample: &CuCycleSample<N>) -> bool {
        self.last_scene_increment
            .is_some_and(|(rev_cycle, rev_time_ns)| {
                let dt_ns = sample.time_ns.saturating_sub(rev_time_ns);
                let dc = sample.cycle.saturating_sub(rev_cycle);
                dt_ns <= self.cooldown_ns && (self.cooldown_ns > 0 || dc <= 1)
            })
    }

    /// Borrows the coalesced incident records detected so far.
    pub fn incidents(&self) -> &[CuIncidentRecord] {
        &self.incidents
    }

    /// Builds a snapshot [`CuIncidentSummary`] without consuming the detector.
    pub fn summary(&self) -> CuIncidentSummary {
        CuIncidentSummary {
            total_cycles: self.total_cycles,
            duration_s: self.duration_s(),
            model_hash: self.params.model_hash,
            collision_episodes: self.collision_episodes,
            near_miss_episodes: self.near_miss_episodes,
            policy_faults: self.policy_faults,
            controller_faults: self.controller_faults,
            clamped_cycles: self.clamped_cycles,
            min_clearance_m: self.min_clearance_m,
            peak_position_step_si: self.peak_position_step_si,
            incidents: self.incidents.clone(),
        }
    }

    /// Consumes the detector and returns the final [`CuIncidentSummary`].
    pub fn finish(self) -> CuIncidentSummary {
        let duration_s = self.duration_s();
        CuIncidentSummary {
            total_cycles: self.total_cycles,
            duration_s,
            model_hash: self.params.model_hash,
            collision_episodes: self.collision_episodes,
            near_miss_episodes: self.near_miss_episodes,
            policy_faults: self.policy_faults,
            controller_faults: self.controller_faults,
            clamped_cycles: self.clamped_cycles,
            min_clearance_m: self.min_clearance_m,
            peak_position_step_si: self.peak_position_step_si,
            incidents: self.incidents,
        }
    }

    fn duration_s(&self) -> f64 {
        match (self.first_time_ns, self.last_time_ns) {
            (Some(first), Some(last)) => last.saturating_sub(first) as f64 * 1e-9,
            _ => 0.0,
        }
    }
}
