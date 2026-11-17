//! Application-side nodes of the setpoint-mux demo: an in-process plant, a
//! scripted stand-in for a learned policy, the scenario driver issuing the
//! classical plan requests, and the headless test oracle. Every other node is
//! a Repose library component.

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
use repose_controller::CuControllerStatus;
use repose_controller::CuJointCommandPayload;
use repose_kinematics::CuJointPosition;
use repose_kinematics::CuJointStatePayload;
use repose_kinematics::CuJointVelocity;
use repose_kinematics::CuRobotStatus;
use repose_motion_planning::CuJointPlanRequest;
use repose_policy_rl::CuPolicyAction;
use repose_policy_rl::CuPolicyStatus;
use repose_trajectory::CuFollowStatus;
use repose_trajectory::CuTrajectoryFollowOutput;
use serde::Deserialize;
use serde::Serialize;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;

/// Joint count of the demo robot (igus ReBeL).
pub const N: usize = 6;

/// Set by [`HandoffObserver`] once the scenario has a verdict.
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
// Plant
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
    /// First-order servo lag of every joint, in seconds.
    time_constant: f64,
}

/// In-process position-servoed arm: each joint follows the commanded
/// position with a first-order lag, starting at joint zero.
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
        _tx: &[BridgeChannelConfig<PlantTxId>],
        _rx: &[BridgeChannelConfig<PlantRxId>],
        _res: Self::Resources<'_>,
    ) -> CuResult<Self> {
        let params = config
            .ok_or_else(|| CuError::from("plant: missing config"))?
            .deserialize_into::<PlantParams>()?;
        if !(params.time_constant.is_finite() && params.time_constant > 0.0) {
            return Err(CuError::from("plant: 'time_constant' must be positive"));
        }
        Ok(Self {
            time_constant: params.time_constant,
            q: [0.0; N],
            v: [0.0; N],
            command: None,
            last_ns: None,
        })
    }

    fn send<'a, Payload>(
        &mut self,
        _ctx: &CuContext,
        _channel: &'static BridgeChannel<PlantTxId, Payload>,
        msg: &CuMsg<Payload>,
    ) -> CuResult<()>
    where
        Payload: CuMsgPayload + 'a,
    {
        let cmd = msg.downcast_ref::<CuJointCommandPayload<N>>()?;
        if let Some(cmd) = cmd.payload() {
            self.command = Some(cmd.positions.map(|q| q.si_value()));
        }
        Ok(())
    }

    fn receive<'a, Payload>(
        &mut self,
        ctx: &CuContext,
        _channel: &'static BridgeChannel<PlantRxId, Payload>,
        msg: &mut CuMsg<Payload>,
    ) -> CuResult<()>
    where
        Payload: CuMsgPayload + 'a,
    {
        let now = ctx.now().as_nanos();
        let dt = self
            .last_ns
            .map_or(0.0, |last| now.saturating_sub(last) as f64 * 1e-9);
        self.last_ns = Some(now);
        if let Some(cmd) = self.command
            && dt > 0.0
        {
            let alpha = 1.0 - (-dt / self.time_constant).exp();
            for ((q, v), target) in self.q.iter_mut().zip(self.v.iter_mut()).zip(cmd) {
                let next = *q + alpha * (target - *q);
                *v = (next - *q) / dt;
                *q = next;
            }
        }
        let state = msg.downcast_mut::<CuJointStatePayload<N>>()?;
        state.tov = Tov::Time(ctx.now());
        state.set_payload(CuJointStatePayload {
            status: CuRobotStatus::Ready,
            positions: rev(self.q),
            velocities: self
                .v
                .map(|v| CuJointVelocity::Revolute(AngularVelocity::new::<radian_per_second>(v))),
        });
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Scripted policy
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PolicyParams {
    pregrasp: [f64; N],
    engage_tolerance: f64,
    action: [f32; N],
    fault_after: u32,
}

/// Stand-in for an ONNX/VLA policy: `Idle` until the arm is at the pre-grasp
/// pose, then a constant `Fresh` action (the "insertion" motion) for
/// `fault_after` cycles, then `Fault` forever (a lost camera, a NaN, a
/// watchdog).
#[derive(Reflect)]
pub struct ScriptedPolicy {
    #[reflect(ignore)]
    pregrasp: [CuJointPosition; N],
    engage_tolerance: f64,
    #[reflect(ignore)]
    action: [f32; N],
    fault_after: u32,
    engaged_cycles: Option<u32>,
}

impl Freezable for ScriptedPolicy {
    fn freeze<E: Encoder>(&self, encoder: &mut E) -> Result<(), EncodeError> {
        Encode::encode(&self.engaged_cycles, encoder)
    }

    fn thaw<D: Decoder>(&mut self, decoder: &mut D) -> Result<(), DecodeError> {
        self.engaged_cycles = Decode::decode(decoder)?;
        Ok(())
    }
}

impl CuTask for ScriptedPolicy {
    type Resources<'r> = ();
    type Input<'m> = input_msg!(CuJointStatePayload<N>);
    type Output<'m> = output_msg!(CuPolicyAction<N>);

    fn new(cfg: Option<&ComponentConfig>, _res: Self::Resources<'_>) -> CuResult<Self> {
        let p = cfg
            .ok_or_else(|| CuError::from("policy: missing config"))?
            .deserialize_into::<PolicyParams>()?;
        Ok(Self {
            pregrasp: rev(p.pregrasp),
            engage_tolerance: p.engage_tolerance,
            action: p.action,
            fault_after: p.fault_after,
            engaged_cycles: None,
        })
    }

    fn process<'i, 'o>(
        &mut self,
        _ctx: &CuContext,
        input: &Self::Input<'i>,
        output: &mut Self::Output<'o>,
    ) -> CuResult<()> {
        output.tov = input.tov;
        if self.engaged_cycles.is_none()
            && let Some(state) = input.payload()
            && max_error(&state.positions, &self.pregrasp) < self.engage_tolerance
        {
            self.engaged_cycles = Some(0);
        }
        let action = match self.engaged_cycles {
            None => CuPolicyAction::default(),
            Some(k) if k < self.fault_after => CuPolicyAction {
                values: self.action,
                status: CuPolicyStatus::Fresh,
            },
            Some(_) => CuPolicyAction {
                values: [0.0; N],
                status: CuPolicyStatus::Fault,
            },
        };
        if let Some(k) = self.engaged_cycles.as_mut() {
            *k = k.saturating_add(1);
        }
        output.set_payload(action);
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
    retract_delay: u32,
}

/// Issues the classical plan requests: home -> pre-grasp on the first ready
/// state, and a retract to `home` from the measured state `retract_delay`
/// cycles after the policy faults.
#[derive(Reflect)]
pub struct Scenario {
    #[reflect(ignore)]
    pregrasp: [CuJointPosition; N],
    #[reflect(ignore)]
    home: [CuJointPosition; N],
    retract_delay: u32,
    transit_sent: bool,
    fault_cycles: Option<u32>,
    retract_sent: bool,
}

impl Freezable for Scenario {
    fn freeze<E: Encoder>(&self, encoder: &mut E) -> Result<(), EncodeError> {
        Encode::encode(&self.transit_sent, encoder)?;
        Encode::encode(&self.fault_cycles, encoder)?;
        Encode::encode(&self.retract_sent, encoder)
    }

    fn thaw<D: Decoder>(&mut self, decoder: &mut D) -> Result<(), DecodeError> {
        self.transit_sent = Decode::decode(decoder)?;
        self.fault_cycles = Decode::decode(decoder)?;
        self.retract_sent = Decode::decode(decoder)?;
        Ok(())
    }
}

impl CuTask for Scenario {
    type Resources<'r> = ();
    type Input<'m> = input_msg!('m, CuJointStatePayload<N>, CuPolicyAction<N>);
    type Output<'m> = output_msg!(CuJointPlanRequest<N>);

    fn new(cfg: Option<&ComponentConfig>, _res: Self::Resources<'_>) -> CuResult<Self> {
        let p = cfg
            .ok_or_else(|| CuError::from("scenario: missing config"))?
            .deserialize_into::<ScenarioParams>()?;
        Ok(Self {
            pregrasp: rev(p.pregrasp),
            home: rev(p.home),
            retract_delay: p.retract_delay,
            transit_sent: false,
            fault_cycles: None,
            retract_sent: false,
        })
    }

    fn process<'i, 'o>(
        &mut self,
        _ctx: &CuContext,
        input: &Self::Input<'i>,
        output: &mut Self::Output<'o>,
    ) -> CuResult<()> {
        output.tov = input.0.tov;
        output.clear_payload();
        let Some(state) = input
            .0
            .payload()
            .filter(|s| s.status == CuRobotStatus::Ready)
        else {
            return Ok(());
        };
        if !self.transit_sent {
            self.transit_sent = true;
            output.set_payload(CuJointPlanRequest {
                start: state.positions,
                goal: self.pregrasp,
                ..Default::default()
            });
            return Ok(());
        }
        if input
            .1
            .payload()
            .is_some_and(|a| a.status == CuPolicyStatus::Fault)
            && self.fault_cycles.is_none()
        {
            self.fault_cycles = Some(0);
        }
        if let Some(k) = self.fault_cycles.as_mut() {
            *k = k.saturating_add(1);
            if *k >= self.retract_delay && !self.retract_sent {
                self.retract_sent = true;
                output.set_payload(CuJointPlanRequest {
                    start: state.positions,
                    goal: self.home,
                    ..Default::default()
                });
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Observer
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ObserverParams {
    pregrasp: [f64; N],
    home: [f64; N],
    tolerance: f64,
    max_cycles: u32,
}

/// Scenario phases the oracle must see, in order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Reflect)]
pub enum Phase {
    Transit,
    AtPregrasp,
    PolicyMoved,
    Held,
    Retracted,
}

/// Headless oracle: checks the transit → policy → hold → retract sequence
/// and the continuity of the mux setpoint, writes `result.json`, and sets
/// [`DONE`].
#[derive(Reflect)]
pub struct HandoffObserver {
    #[reflect(ignore)]
    pregrasp: [CuJointPosition; N],
    #[reflect(ignore)]
    home: [CuJointPosition; N],
    tolerance: f64,
    max_cycles: u32,
    cycles: u32,
    phase: Phase,
    #[reflect(ignore)]
    phase_cycles: [u32; 5],
    held_cycles: u32,
    #[reflect(ignore)]
    previous: Option<CuTrajectoryFollowOutput<N>>,
    max_position_step: f64,
    max_velocity_step: f64,
    fault_cycles: u32,
}

impl Freezable for HandoffObserver {}

impl HandoffObserver {
    fn finish(&self, outcome: &str) {
        let success = outcome == "retracted";
        let body = format!(
            concat!(
                "{{\"success\":{},\"outcome\":\"{}\",\"cycles\":{},",
                "\"phase_cycles\":{{\"at_pregrasp\":{},\"policy_moved\":{},\"held\":{},\"retracted\":{}}},",
                "\"max_setpoint_position_step_rad\":{:.6},\"max_setpoint_velocity_step_rad_s\":{:.6}}}"
            ),
            success,
            outcome,
            self.cycles,
            self.phase_cycles[1],
            self.phase_cycles[2],
            self.phase_cycles[3],
            self.phase_cycles[4],
            self.max_position_step,
            self.max_velocity_step,
        );
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("result.json");
        if let Err(e) = std::fs::write(path, format!("{body}\n")) {
            eprintln!("setpoint-mux-demo: cannot write result.json: {e}");
        }
        eprintln!("setpoint-mux-demo: {body}");
        SUCCESS.store(success, Ordering::SeqCst);
        DONE.store(true, Ordering::SeqCst);
    }

    fn advance(&mut self, phase: Phase) {
        self.phase = phase;
        self.phase_cycles[phase as usize] = self.cycles;
    }
}

impl CuSinkTask for HandoffObserver {
    type Resources<'r> = ();
    type Input<'m> = input_msg!(
        'm,
        CuTrajectoryFollowOutput<N>,
        CuJointStatePayload<N>,
        CuJointCommandPayload<N>
    );

    fn new(cfg: Option<&ComponentConfig>, _res: Self::Resources<'_>) -> CuResult<Self> {
        let p = cfg
            .ok_or_else(|| CuError::from("observer: missing config"))?
            .deserialize_into::<ObserverParams>()?;
        Ok(Self {
            pregrasp: rev(p.pregrasp),
            home: rev(p.home),
            tolerance: p.tolerance,
            max_cycles: p.max_cycles,
            cycles: 0,
            phase: Phase::Transit,
            phase_cycles: [0; 5],
            held_cycles: 0,
            previous: None,
            max_position_step: 0.0,
            max_velocity_step: 0.0,
            fault_cycles: 0,
        })
    }

    fn process<'i>(&mut self, _ctx: &CuContext, input: &Self::Input<'i>) -> CuResult<()> {
        if DONE.load(Ordering::SeqCst) {
            return Ok(());
        }
        let (setpoint, state, command) = *input;
        self.cycles += 1;
        let (Some(setpoint), Some(state)) = (setpoint.payload(), state.payload()) else {
            return Ok(());
        };
        let status = command.payload().map(|c| c.status);

        // Continuity of the emitted setpoint once it carries a target.
        if setpoint.status != CuFollowStatus::Idle {
            if let Some(prev) = self.previous.as_ref() {
                for j in 0..N {
                    let dq =
                        (setpoint.positions[j].si_value() - prev.positions[j].si_value()).abs();
                    let dv =
                        (setpoint.velocities[j].si_value() - prev.velocities[j].si_value()).abs();
                    self.max_position_step = self.max_position_step.max(dq);
                    self.max_velocity_step = self.max_velocity_step.max(dv);
                }
            }
            self.previous = Some(setpoint.clone());
        }

        let at = |goal: &[CuJointPosition; N]| max_error(&state.positions, goal);
        match self.phase {
            Phase::Transit if at(&self.pregrasp) < self.tolerance => {
                self.advance(Phase::AtPregrasp);
            }
            Phase::AtPregrasp if at(&self.pregrasp) > 0.1 => self.advance(Phase::PolicyMoved),
            Phase::PolicyMoved => {
                let still = setpoint.velocities.iter().all(|v| v.si_value() == 0.0);
                self.held_cycles = if setpoint.status == CuFollowStatus::Finished && still {
                    self.held_cycles + 1
                } else {
                    0
                };
                if self.held_cycles >= 20 && at(&self.pregrasp) > 0.1 {
                    self.advance(Phase::Held);
                }
            }
            Phase::Held
                if status == Some(CuControllerStatus::Holding)
                    && at(&self.home) < self.tolerance =>
            {
                self.advance(Phase::Retracted);
                self.finish("retracted");
                return Ok(());
            }
            _ => {}
        }

        self.fault_cycles = match status {
            Some(CuControllerStatus::Fault) => self.fault_cycles + 1,
            _ => 0,
        };
        if self.fault_cycles >= 20 {
            self.finish("controller_fault");
        } else if self.cycles >= self.max_cycles {
            self.finish(match self.phase {
                Phase::Transit => "timeout_transit",
                Phase::AtPregrasp => "timeout_no_handoff",
                Phase::PolicyMoved => "timeout_no_hold",
                Phase::Held => "timeout_no_retract",
                Phase::Retracted => "retracted",
            });
        }
        Ok(())
    }
}
