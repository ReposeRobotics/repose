//! Application-side nodes of the Simplex Safety demo:
//! - [`ObstacleSceneSource`]: publishes a fixed [`CuPlanningScene`] containing
//!   a workspace obstacle sphere for [`CuSafetyShieldTask`].
//! - [`PlantBridge`]: in-process 6-DOF servo plant with first-order lag.
//! - [`Scenario`]: drives the classical transit (`home -> pregrasp`) and safe
//!   classical retract (`current -> home`) after the learned policy finishes or
//!   faults.
//! - [`ScriptedPolicy`]: stand-in for two learned policy checkpoints:
//!   - `POLICY_MODE_V1_ROGUE (0)`: drives toward the workspace obstacle until
//!     the safety shield holds it and the policy faults.
//!   - `POLICY_MODE_V2_SAFE (1)`: steers cleanly around the obstacle and
//!     finishes with zero shield interventions.
//! - [`SimplexObserver`]: headless oracle verifying phase progression and
//!   recording summary stats.

use cu29::bincode::Decode;
use cu29::bincode::Encode;
use cu29::bincode::de::Decoder;
use cu29::bincode::enc::Encoder;
use cu29::bincode::error::DecodeError;
use cu29::bincode::error::EncodeError;
use cu29::cubridge::BridgeChannel;
use cu29::cubridge::BridgeChannelConfig;
use cu29::cubridge::BridgeChannelInfo;
use cu29::cubridge::BridgeChannelSet;
use cu29::cubridge::CuBridge;
use cu29::prelude::*;
use cu29::units::si::angle::radian;
use cu29::units::si::angular_velocity::radian_per_second;
use cu29::units::si::f64::Angle;
use cu29::units::si::f64::AngularVelocity;
use cu29::units::si::f64::Length;
use cu29::units::si::length::meter;
use repose_collision::CuCollisionShape;
use repose_collision::CuObjectId;
use repose_collision::CuPlanningScene;
use repose_collision::CuSphere;
use repose_controller::CuJointCommandPayload;
use repose_kinematics::CuJointPosition;
use repose_kinematics::CuJointStatePayload;
use repose_kinematics::CuJointVelocity;
use repose_kinematics::CuRobotStatus;
use repose_motion_planning::CuJointPlanRequest;
use repose_policy_rl::CuPolicyAction;
use repose_policy_rl::CuPolicyStatus;
use repose_safety::CuSafetyReport;
use repose_trajectory::CuSetpointMuxReport;
use repose_trajectory::CuSetpointSource;
use serde::Deserialize;
use serde::Serialize;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicU8;
use std::sync::atomic::Ordering;

/// Joint count of the demo robot (igus ReBeL).
pub const N: usize = 6;

/// Policy checkpoint selector: v1 drives toward the obstacle (rogue).
pub const POLICY_MODE_V1_ROGUE: u8 = 0;
/// Policy checkpoint selector: v2 steers safely around the obstacle.
pub const POLICY_MODE_V2_SAFE: u8 = 1;

/// Active checkpoint mode read by [`ScriptedPolicy::new`].
pub static POLICY_MODE: AtomicU8 = AtomicU8::new(POLICY_MODE_V1_ROGUE);
/// Set by [`SimplexObserver`] once the scenario has a verdict.
pub static DONE: AtomicBool = AtomicBool::new(false);
/// Verdict of the run, valid once [`DONE`] is set.
pub static SUCCESS: AtomicBool = AtomicBool::new(false);

fn rev(values: [f64; N]) -> [CuJointPosition; N] {
    values.map(|q| CuJointPosition::Revolute(Angle::new::<radian>(q)))
}

fn max_error(a: &[CuJointPosition; N], b: &[CuJointPosition; N]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(a, b)| (a.si_value() - b.si_value()).abs())
        .fold(0.0, f64::max)
}

// ---------------------------------------------------------------------------
// Obstacle Scene Source
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ObstacleSceneParams {
    obstacle_center: [f64; 3],
    obstacle_radius: f64,
}

/// Publishes a versioned [`CuPlanningScene`] containing a single workspace
/// obstacle sphere on every cycle.
#[derive(Reflect)]
#[reflect(from_reflect = false)]
pub struct ObstacleSceneSource {
    #[reflect(ignore)]
    scene: CuPlanningScene,
}

impl Freezable for ObstacleSceneSource {}

impl CuSrcTask for ObstacleSceneSource {
    type Resources<'r> = ();
    type Output<'m> = output_msg!(CuPlanningScene);

    fn new(config: Option<&ComponentConfig>, _resources: Self::Resources<'_>) -> CuResult<Self>
    where
        Self: Sized,
    {
        let p: ObstacleSceneParams = config
            .ok_or_else(|| CuError::from("scene_pub: missing config"))?
            .deserialize_into()?;
        let mut scene = CuPlanningScene::new();
        let center = p.obstacle_center.map(Length::new::<meter>);
        let radius = Length::new::<meter>(p.obstacle_radius);
        scene.add_obstacle(
            CuCollisionShape::Sphere(CuSphere::new(center, radius)?),
            Some(CuObjectId::new(1)),
        )?;
        Ok(Self { scene })
    }

    fn process(&mut self, ctx: &CuContext, output: &mut Self::Output<'_>) -> CuResult<()> {
        output.tov = Tov::Time(ctx.now());
        output.set_payload(self.scene.clone());
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Plant Bridge
// ---------------------------------------------------------------------------

#[derive(Copy, Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[repr(usize)]
#[serde(rename_all = "snake_case")]
pub enum PlantTxId {
    JointCommand,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[repr(usize)]
#[serde(rename_all = "snake_case")]
pub enum PlantRxId {
    JointState,
}

pub struct PlantTx;

impl PlantTx {
    pub const JOINT_COMMAND: BridgeChannel<PlantTxId, CuJointCommandPayload<N>> =
        BridgeChannel::with_channel(PlantTxId::JointCommand, "joint_command");
}

impl BridgeChannelSet for PlantTx {
    type Id = PlantTxId;
    const STATIC_CHANNELS: &'static [&'static dyn BridgeChannelInfo<Self::Id>] =
        &[&Self::JOINT_COMMAND];
}

pub struct PlantRx;

impl PlantRx {
    pub const JOINT_STATE: BridgeChannel<PlantRxId, CuJointStatePayload<N>> =
        BridgeChannel::with_channel(PlantRxId::JointState, "joint_state");
}

impl BridgeChannelSet for PlantRx {
    type Id = PlantRxId;
    const STATIC_CHANNELS: &'static [&'static dyn BridgeChannelInfo<Self::Id>] =
        &[&Self::JOINT_STATE];
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PlantParams {
    time_constant: f64,
}

/// In-process position-servoed 6-DOF arm with first-order lag.
#[derive(Reflect)]
pub struct PlantBridge {
    time_constant: f64,
    #[reflect(ignore)]
    q: [f64; N],
    #[reflect(ignore)]
    v: [f64; N],
    #[reflect(ignore)]
    command: Option<[f64; N]>,
    last_ns: Option<u64>,
}

impl Freezable for PlantBridge {
    fn freeze<E: Encoder>(&self, encoder: &mut E) -> Result<(), EncodeError> {
        Encode::encode(&self.q, encoder)?;
        Encode::encode(&self.v, encoder)?;
        Encode::encode(&self.command, encoder)?;
        Encode::encode(&self.last_ns, encoder)
    }

    fn thaw<D: Decoder>(&mut self, decoder: &mut D) -> Result<(), DecodeError> {
        self.q = Decode::decode(decoder)?;
        self.v = Decode::decode(decoder)?;
        self.command = Decode::decode(decoder)?;
        self.last_ns = Decode::decode(decoder)?;
        Ok(())
    }
}

impl CuBridge for PlantBridge {
    type Tx = PlantTx;
    type Rx = PlantRx;
    type Resources<'r> = ();

    fn new(
        config: Option<&ComponentConfig>,
        _tx_channels: &[BridgeChannelConfig<PlantTxId>],
        _rx_channels: &[BridgeChannelConfig<PlantRxId>],
        _resources: Self::Resources<'_>,
    ) -> CuResult<Self> {
        let p: PlantParams = config
            .ok_or_else(|| CuError::from("plant: missing config"))?
            .deserialize_into()?;
        if !p.time_constant.is_finite() || p.time_constant <= 0.0 {
            return Err(CuError::from("plant: time_constant must be positive"));
        }
        Ok(Self {
            time_constant: p.time_constant,
            q: [0.0; N],
            v: [0.0; N],
            command: None,
            last_ns: None,
        })
    }

    fn send<'a, Payload>(
        &mut self,
        _ctx: &CuContext,
        channel: &'static BridgeChannel<PlantTxId, Payload>,
        msg: &CuMsg<Payload>,
    ) -> CuResult<()>
    where
        Payload: CuMsgPayload + 'a,
    {
        if channel.id() == PlantTxId::JointCommand {
            let cmd = msg.downcast_ref::<CuJointCommandPayload<N>>()?;
            if let Some(cmd) = cmd.payload() {
                self.command = Some(cmd.positions.map(|p| p.si_value()));
            }
        }
        Ok(())
    }

    fn receive<'a, Payload>(
        &mut self,
        ctx: &CuContext,
        channel: &'static BridgeChannel<PlantRxId, Payload>,
        msg: &mut CuMsg<Payload>,
    ) -> CuResult<()>
    where
        Payload: CuMsgPayload + 'a,
    {
        if channel.id() != PlantRxId::JointState {
            return Ok(());
        }
        let now = ctx.now();
        let now_ns = now.as_nanos();
        let dt = self
            .last_ns
            .map(|prev| now_ns.saturating_sub(prev) as f64 / 1e9)
            .unwrap_or(0.0);
        self.last_ns = Some(now_ns);
        if let Some(target) = self.command
            && dt > 0.0
        {
            let alpha = 1.0 - (-dt / self.time_constant).exp();
            for ((q, v), target) in self.q.iter_mut().zip(self.v.iter_mut()).zip(target) {
                let next = *q + alpha * (target - *q);
                *v = (next - *q) / dt;
                *q = next;
            }
        }
        let slot = msg.downcast_mut::<CuJointStatePayload<N>>()?;
        slot.tov = Tov::Time(now);
        slot.set_payload(CuJointStatePayload {
            positions: rev(self.q),
            velocities: self
                .v
                .map(|w| CuJointVelocity::Revolute(AngularVelocity::new::<radian_per_second>(w))),
            status: CuRobotStatus::Ready,
        });
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Scenario
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ScenarioParams {
    pregrasp: [f64; N],
    home: [f64; N],
    retract_delay: u64,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Encode, Decode, Reflect)]
#[bincode(crate = "cu29::bincode")]
enum ScenarioPhase {
    Init,
    Transit,
    PolicyActive,
    WaitRetract(u64),
    Done,
}

/// Sends the classical transit request (`home -> pregrasp`), waits for the
/// policy phase to finish or fault, then sends a classical retract request
/// (`current -> home`).
#[derive(Reflect)]
pub struct Scenario {
    #[reflect(ignore)]
    pregrasp: [CuJointPosition; N],
    #[reflect(ignore)]
    home: [CuJointPosition; N],
    retract_delay: u64,
    phase: ScenarioPhase,
    next_id: u64,
}

impl Freezable for Scenario {
    fn freeze<E: Encoder>(&self, encoder: &mut E) -> Result<(), EncodeError> {
        Encode::encode(&self.phase, encoder)?;
        Encode::encode(&self.next_id, encoder)
    }

    fn thaw<D: Decoder>(&mut self, decoder: &mut D) -> Result<(), DecodeError> {
        self.phase = Decode::decode(decoder)?;
        self.next_id = Decode::decode(decoder)?;
        Ok(())
    }
}

impl CuTask for Scenario {
    type Resources<'r> = ();
    type Input<'m> = input_msg!('m, CuJointStatePayload<N>, CuPolicyAction<N>);
    type Output<'m> = output_msg!(CuJointPlanRequest<N>);

    fn new(config: Option<&ComponentConfig>, _resources: Self::Resources<'_>) -> CuResult<Self> {
        let p: ScenarioParams = config
            .ok_or_else(|| CuError::from("scenario: missing config"))?
            .deserialize_into()?;
        Ok(Self {
            pregrasp: rev(p.pregrasp),
            home: rev(p.home),
            retract_delay: p.retract_delay,
            phase: ScenarioPhase::Init,
            next_id: 1,
        })
    }

    fn process(
        &mut self,
        ctx: &CuContext,
        input: &Self::Input<'_>,
        output: &mut Self::Output<'_>,
    ) -> CuResult<()> {
        output.clear_payload();
        let Some(state) = input.0.payload() else {
            return Ok(());
        };
        let policy_status = input.1.payload().map(|p| p.status);
        match self.phase {
            ScenarioPhase::Init => {
                self.next_id += 1;
                output.tov = Tov::Time(ctx.now());
                output.set_payload(CuJointPlanRequest {
                    start: state.positions,
                    goal: self.pregrasp,
                    ..Default::default()
                });
                self.phase = ScenarioPhase::Transit;
            }
            ScenarioPhase::Transit => {
                if policy_status == Some(CuPolicyStatus::Fresh) {
                    self.phase = ScenarioPhase::PolicyActive;
                }
            }
            ScenarioPhase::PolicyActive => {
                if matches!(
                    policy_status,
                    Some(CuPolicyStatus::Fault | CuPolicyStatus::Idle)
                ) {
                    self.phase = ScenarioPhase::WaitRetract(0);
                }
            }
            ScenarioPhase::WaitRetract(waited) => {
                if waited >= self.retract_delay {
                    self.next_id += 1;
                    output.tov = Tov::Time(ctx.now());
                    output.set_payload(CuJointPlanRequest {
                        start: state.positions,
                        goal: self.home,
                        ..Default::default()
                    });
                    self.phase = ScenarioPhase::Done;
                } else {
                    self.phase = ScenarioPhase::WaitRetract(waited + 1);
                }
            }
            ScenarioPhase::Done => {}
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Scripted Policy (Checkpoint v1 Rogue vs Checkpoint v2 Safe)
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PolicyParams {
    pregrasp: [f64; N],
    engage_tolerance: f64,
    rogue_action: [f64; N],
    safe_action: [f64; N],
    max_policy_cycles: u64,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Encode, Decode, Reflect)]
#[bincode(crate = "cu29::bincode")]
enum PolicyPhase {
    Waiting,
    Active(u64),
    Faulted,
    Completed,
}

/// Stand-in for two learned policy checkpoints:
/// - `mode == 0` (`v1_rogue`): commands `rogue_action` straight into the
///   obstacle until `max_policy_cycles`, then faults.
/// - `mode == 1` (`v2_safe`): commands `safe_action` steering smoothly around
///   the obstacle for `max_policy_cycles`, then finishes cleanly (`Idle`).
#[derive(Reflect)]
pub struct ScriptedPolicy {
    #[reflect(ignore)]
    pregrasp: [CuJointPosition; N],
    engage_tolerance: f64,
    #[reflect(ignore)]
    rogue_action: [f32; N],
    #[reflect(ignore)]
    safe_action: [f32; N],
    max_policy_cycles: u64,
    mode: u8,
    phase: PolicyPhase,
}

impl Freezable for ScriptedPolicy {
    fn freeze<E: Encoder>(&self, encoder: &mut E) -> Result<(), EncodeError> {
        Encode::encode(&self.mode, encoder)?;
        Encode::encode(&self.phase, encoder)
    }

    fn thaw<D: Decoder>(&mut self, decoder: &mut D) -> Result<(), DecodeError> {
        self.mode = Decode::decode(decoder)?;
        self.phase = Decode::decode(decoder)?;
        Ok(())
    }
}

impl CuTask for ScriptedPolicy {
    type Resources<'r> = ();
    type Input<'m> = input_msg!(CuJointStatePayload<N>);
    type Output<'m> = output_msg!(CuPolicyAction<N>);

    fn new(config: Option<&ComponentConfig>, _resources: Self::Resources<'_>) -> CuResult<Self> {
        let p: PolicyParams = config
            .ok_or_else(|| CuError::from("policy: missing config"))?
            .deserialize_into()?;
        Ok(Self {
            pregrasp: rev(p.pregrasp),
            engage_tolerance: p.engage_tolerance,
            rogue_action: p.rogue_action.map(|v| v as f32),
            safe_action: p.safe_action.map(|v| v as f32),
            max_policy_cycles: p.max_policy_cycles,
            mode: POLICY_MODE.load(Ordering::SeqCst),
            phase: PolicyPhase::Waiting,
        })
    }

    fn process(
        &mut self,
        ctx: &CuContext,
        input: &Self::Input<'_>,
        output: &mut Self::Output<'_>,
    ) -> CuResult<()> {
        let Some(state) = input.payload() else {
            return Ok(());
        };
        let (status, values) = match self.phase {
            PolicyPhase::Waiting => {
                if max_error(&state.positions, &self.pregrasp) <= self.engage_tolerance {
                    self.phase = PolicyPhase::Active(1);
                    let action = if self.mode == POLICY_MODE_V1_ROGUE {
                        self.rogue_action
                    } else {
                        self.safe_action
                    };
                    (CuPolicyStatus::Fresh, action)
                } else {
                    (CuPolicyStatus::Idle, [0.0; N])
                }
            }
            PolicyPhase::Active(ticks) => {
                if ticks >= self.max_policy_cycles {
                    if self.mode == POLICY_MODE_V1_ROGUE {
                        self.phase = PolicyPhase::Faulted;
                        (CuPolicyStatus::Fault, [0.0; N])
                    } else {
                        self.phase = PolicyPhase::Completed;
                        (CuPolicyStatus::Idle, [0.0; N])
                    }
                } else {
                    self.phase = PolicyPhase::Active(ticks + 1);
                    let action = if self.mode == POLICY_MODE_V1_ROGUE {
                        self.rogue_action
                    } else {
                        self.safe_action
                    };
                    (CuPolicyStatus::Fresh, action)
                }
            }
            PolicyPhase::Faulted => (CuPolicyStatus::Fault, [0.0; N]),
            PolicyPhase::Completed => (CuPolicyStatus::Idle, [0.0; N]),
        };
        output.tov = Tov::Time(ctx.now());
        output.set_payload(CuPolicyAction { values, status });
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Simplex Observer
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ObserverParams {
    pregrasp: [f64; N],
    home: [f64; N],
    tolerance: f64,
    max_cycles: u64,
}

#[derive(Reflect)]
pub struct SimplexObserver {
    #[reflect(ignore)]
    pregrasp: [CuJointPosition; N],
    #[reflect(ignore)]
    home: [CuJointPosition; N],
    tolerance: f64,
    max_cycles: u64,
    mode: u8,
    cycles: u64,
    at_pregrasp: Option<u64>,
    policy_engaged: Option<u64>,
    shield_intervened: Option<u64>,
    shield_hold_cycles: u64,
    min_clearance_m: f64,
    retracted: Option<u64>,
}

impl Freezable for SimplexObserver {}

impl CuSinkTask for SimplexObserver {
    type Resources<'r> = ();
    type Input<'m> = input_msg!(
        'm,
        CuJointStatePayload<N>,
        CuSetpointMuxReport,
        CuSafetyReport,
        CuJointCommandPayload<N>
    );

    fn new(config: Option<&ComponentConfig>, _resources: Self::Resources<'_>) -> CuResult<Self> {
        let p: ObserverParams = config
            .ok_or_else(|| CuError::from("observer: missing config"))?
            .deserialize_into()?;
        Ok(Self {
            pregrasp: rev(p.pregrasp),
            home: rev(p.home),
            tolerance: p.tolerance,
            max_cycles: p.max_cycles,
            mode: POLICY_MODE.load(Ordering::SeqCst),
            cycles: 0,
            at_pregrasp: None,
            policy_engaged: None,
            shield_intervened: None,
            shield_hold_cycles: 0,
            min_clearance_m: f64::INFINITY,
            retracted: None,
        })
    }

    fn process(&mut self, _ctx: &CuContext, input: &Self::Input<'_>) -> CuResult<()> {
        if DONE.load(Ordering::SeqCst) {
            return Ok(());
        }
        self.cycles += 1;

        if let Some(report) = input.2.payload() {
            let c = report.min_clearance.get::<meter>();
            if c.is_finite() && c < self.min_clearance_m {
                self.min_clearance_m = c;
            }
            if !report.verdict.is_clear() {
                if self.shield_intervened.is_none() {
                    self.shield_intervened = Some(self.cycles);
                }
                self.shield_hold_cycles += 1;
            }
        }

        if let Some(mux) = input.1.payload()
            && self.policy_engaged.is_none()
            && mux.source == CuSetpointSource::Policy
        {
            self.policy_engaged = Some(self.cycles);
        }

        if let Some(state) = input.0.payload() {
            if self.at_pregrasp.is_none()
                && max_error(&state.positions, &self.pregrasp) <= self.tolerance
            {
                self.at_pregrasp = Some(self.cycles);
            }
            if self.policy_engaged.is_some()
                && self.cycles > self.policy_engaged.unwrap_or(0) + 160
                && max_error(&state.positions, &self.home) <= self.tolerance
            {
                self.retracted = Some(self.cycles);
                let ok = if self.mode == POLICY_MODE_V1_ROGUE {
                    self.at_pregrasp.is_some()
                        && self.policy_engaged.is_some()
                        && self.shield_intervened.is_some()
                } else {
                    self.at_pregrasp.is_some()
                        && self.policy_engaged.is_some()
                        && self.shield_intervened.is_none()
                };
                eprintln!(
                    "simplex_safety ({}): cycles={} pregrasp={:?} policy={:?} shield_first={:?} shield_holds={} min_clearance={:.4}m retracted={:?} ok={}",
                    if self.mode == POLICY_MODE_V1_ROGUE {
                        "v1_rogue"
                    } else {
                        "v2_safe"
                    },
                    self.cycles,
                    self.at_pregrasp,
                    self.policy_engaged,
                    self.shield_intervened,
                    self.shield_hold_cycles,
                    self.min_clearance_m,
                    self.retracted,
                    ok,
                );
                SUCCESS.store(ok, Ordering::SeqCst);
                DONE.store(true, Ordering::SeqCst);
                return Ok(());
            }
        }

        if self.cycles >= self.max_cycles {
            eprintln!(
                "simplex_safety ({}): TIMEOUT at cycle {} (pregrasp={:?} policy={:?} shield_first={:?} shield_holds={} min_clearance={:.4}m)",
                if self.mode == POLICY_MODE_V1_ROGUE {
                    "v1_rogue"
                } else {
                    "v2_safe"
                },
                self.cycles,
                self.at_pregrasp,
                self.policy_engaged,
                self.shield_intervened,
                self.shield_hold_cycles,
                self.min_clearance_m,
            );
            SUCCESS.store(false, Ordering::SeqCst);
            DONE.store(true, Ordering::SeqCst);
        }
        Ok(())
    }
}
