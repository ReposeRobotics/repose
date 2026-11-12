//! [`CuMujocoSimBridge`]: a [`CuBridge`](cu29::prelude::CuBridge) connecting Copper to an
//! external MuJoCo simulator sidecar process over TCP.
//!
//! This crate is `std`-only: bridges are host components and this one owns a
//! blocking [`TcpStream`](std::net::TcpStream). There is no `no_std` surface.
//!
//! # Wire protocol
//!
//! TCP, line-delimited JSON, one request/response per Copper cycle. The sidecar
//! listens; the bridge is the TCP client.
//!
//! Bridge send (Tx), joint position setpoints in radians (SI values):
//!
//! ```text
//! {"type":"command","n":N,"q":[q0..qN-1],"qd":[..],"effort":[..],"mode":"position"}\n
//! ```
//!
//! Bridge receive (Rx), after the sidecar steps physics:
//!
//! ```text
//! {"type":"state","n":N,"q":[q0..qN-1],"qd":[qd0..qdN-1],"time":<sim seconds>}\n
//! ```
//!
//! Receive parses `q` into positions and `qd` into velocities. `time` is
//! ignored. `n` is optional on receive (older sidecars omit it); when present
//! it must equal `N`. `q` and `qd` must both hold exactly `N` numbers. A
//! malformed line, a joint-count mismatch between the bridge and the sidecar
//! (the error names both counts), or a missing/non-numeric `q`/`qd` array is a
//! [`CuError`](cu29::prelude::CuError); nothing panics.
//!
//! A sidecar that rejects a command answers with
//! `{"type":"error","message":"..."}\n` instead of a state line; the bridge
//! turns it into a [`CuError`](cu29::prelude::CuError) carrying the message.
//!
//! The `n` field is additive: a 6-joint bridge still works against a sidecar
//! written for the previous fixed-6 protocol (it ignores `n`), so the
//! handshake is not versioned. A bridge with `N != 6` needs a sidecar that
//! serves `N` joints.
//!
//! # Payload contract
//!
//! `CuMujocoSimBridge<N>` is generic over the joint count. The Tx
//! `joint_command` channel carries
//! [`CuJointCommandPayload<N>`](repose_controller::CuJointCommandPayload) and the
//! Rx `joint_state` channel carries
//! [`CuJointStatePayload<N>`](repose_kinematics::CuJointStatePayload): the
//! unified controller contract. `N` comes from the type in the RON graph
//! (for example `repose_mujoco_sim::CuMujocoSimBridge<6>` for the igus ReBeL)
//! and the channel `msg:` types must use the same `N`. Joints are revolute,
//! positions are SI radians, velocities SI radians per second. Command
//! positions, velocities, efforts, and mode reach the wire; controller
//! status stays on the Copper side.
//!
//! # Config
//!
//! The bridge `config:` block is a [`CuMujocoSimParams`]; unknown keys are
//! rejected.
//!
//! | Key | Type | Default | Meaning |
//! |---|---|---|---|
//! | `host` | `String` | `"127.0.0.1"` | sidecar address |
//! | `port` | `u16` | `8765` | sidecar TCP port |
//! | `read_timeout` | `Time` (s) | `0.25` | socket read (and connect/write) timeout |
//!
//! The bridge connects in
//! [`start`](cu29::prelude::CuBridge::start) and reconnects lazily in
//! `send`/`receive` when the stream is dead. Reads always carry a timeout, so
//! the runtime can never hang forever on a silent sidecar.
//!
//! # `copperconfig.ron` sketch
//!
//! ```ron
//! bridges: [
//!     (
//!         id: "mujoco",
//!         type: "repose_mujoco_sim::CuMujocoSimBridge<6>",
//!         config: {
//!             "host": "127.0.0.1",
//!             "port": 8765,
//!             "read_timeout": 0.25,
//!         },
//!         channels: [
//!             Tx(id: "joint_command"),
//!             Rx(id: "joint_state"),
//!         ],
//!     ),
//! ],
//! cnx: [
//!     ( src: "controller", dst: "mujoco/joint_command",
//!       msg: "repose_controller::CuJointCommandPayload<6>" ),
//!     ( src: "mujoco/joint_state", dst: "controller",
//!       msg: "repose_kinematics::CuJointStatePayload<6>" ),
//! ],
//! ```
//!
//! Channel routes default to `"joint_command"` (Tx) and `"joint_state"` (Rx).

use bincode::Decode;
use bincode::Encode;
use bincode::de::Decoder;
use bincode::enc::Encoder;
use bincode::error::DecodeError;
use bincode::error::EncodeError;
use cu29::prelude::*;
use cu29::units::si::angle::radian;
use cu29::units::si::angular_velocity::radian_per_second;
use cu29::units::si::f64::Angle;
use cu29::units::si::f64::AngularVelocity;
use cu29::units::si::f64::Time;
use cu29::units::si::time::second;
use repose_controller::CuJointCommandPayload;
use repose_kinematics::CuJointEffort;
use repose_kinematics::CuJointPosition;
use repose_kinematics::CuJointStatePayload;
use repose_kinematics::CuJointVelocity;
use repose_kinematics::CuRobotStatus;
use serde::Deserialize;
use serde::Serialize;
use std::io::Read;
use std::io::Write;
use std::net::TcpStream;
use std::net::ToSocketAddrs;
use std::time::Duration;

/// Parameters of [`CuMujocoSimBridge`]; the bridge `config:` block.
///
/// ```rust
/// use repose_mujoco_sim::CuMujocoSimParams;
/// use cu29::units::si::f64::Time;
/// use cu29::units::si::time::second;
///
/// let params = CuMujocoSimParams {
///     port: 9000,
///     ..CuMujocoSimParams::default()
/// };
/// assert_eq!(params.host, "127.0.0.1");
/// assert_eq!(params.read_timeout, Time::new::<second>(0.25));
/// params.validate().unwrap();
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Reflect)]
#[serde(default, deny_unknown_fields)]
pub struct CuMujocoSimParams {
    /// Sidecar host name or address.
    pub host: String,
    /// Sidecar TCP port.
    pub port: u16,
    /// Socket read (and connect/write) timeout; positive and finite.
    pub read_timeout: Time,
}

impl Default for CuMujocoSimParams {
    /// `127.0.0.1:8765`, 0.25 s timeout.
    fn default() -> Self {
        Self {
            host: "127.0.0.1".into(),
            port: 8765,
            read_timeout: Time::new::<second>(0.25),
        }
    }
}

impl CuMujocoSimParams {
    /// Checks that the host is non-empty and the timeout is positive, finite,
    /// and representable as a [`Duration`].
    pub fn validate(&self) -> CuResult<()> {
        if self.host.is_empty() {
            return Err(CuError::from("repose-mujoco-sim: 'host' must not be empty"));
        }
        Duration::try_from_secs_f64(self.read_timeout.get::<second>())
            .ok()
            .filter(|d| !d.is_zero())
            .ok_or_else(|| {
                CuError::from("repose-mujoco-sim: 'read_timeout' must be positive and finite")
            })?;
        Ok(())
    }
}
/// Largest accepted wire line; bounds the stack read buffer.
///
/// A state line carries `2 * N` numbers, so this caps `N` at roughly 25.
const MAX_LINE_BYTES: usize = 1024;

/// Transmit channel identifiers of [`CuMujocoSimBridge`].
#[derive(Copy, Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[repr(usize)]
#[serde(rename_all = "snake_case")]
pub enum MujocoTxId {
    /// Joint command channel.
    JointCommand,
}

/// Receive channel identifiers of [`CuMujocoSimBridge`].
#[derive(Copy, Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[repr(usize)]
#[serde(rename_all = "snake_case")]
pub enum MujocoRxId {
    /// Joint state channel.
    JointState,
}

/// Tx channel set of [`CuMujocoSimBridge<N>`](CuMujocoSimBridge).
pub struct MujocoTxChannels<const N: usize>;

impl<const N: usize> MujocoTxChannels<N> {
    /// `joint_command`: [`CuJointCommandPayload<N>`].
    pub const JOINT_COMMAND: BridgeChannel<MujocoTxId, CuJointCommandPayload<N>> =
        BridgeChannel::with_channel(MujocoTxId::JointCommand, "joint_command");
}

impl<const N: usize> BridgeChannelSet for MujocoTxChannels<N> {
    type Id = MujocoTxId;

    const STATIC_CHANNELS: &'static [&'static dyn BridgeChannelInfo<Self::Id>] =
        &[&Self::JOINT_COMMAND];
}

/// Rx channel set of [`CuMujocoSimBridge<N>`](CuMujocoSimBridge).
pub struct MujocoRxChannels<const N: usize>;

impl<const N: usize> MujocoRxChannels<N> {
    /// `joint_state`: [`CuJointStatePayload<N>`].
    pub const JOINT_STATE: BridgeChannel<MujocoRxId, CuJointStatePayload<N>> =
        BridgeChannel::with_channel(MujocoRxId::JointState, "joint_state");
}

impl<const N: usize> BridgeChannelSet for MujocoRxChannels<N> {
    type Id = MujocoRxId;

    const STATIC_CHANNELS: &'static [&'static dyn BridgeChannelInfo<Self::Id>] =
        &[&Self::JOINT_STATE];
}

/// `CuBridge` to an external MuJoCo simulator sidecar over TCP/JSON.
///
/// Owns one TCP session and multiplexes the command/state exchange over it.
/// `N` is the joint count; it must match the sidecar (checked on every state
/// line). `freeze`/`thaw` persist the connection config only; the socket itself is
/// re-established on `start`, so keyframe restore never resurrects a stale fd.
#[derive(Reflect)]
pub struct CuMujocoSimBridge<const N: usize> {
    #[reflect(ignore)]
    params: CuMujocoSimParams,
    #[reflect(ignore)]
    stream: Option<TcpStream>,
}

impl<const N: usize> CuMujocoSimBridge<N> {
    /// Creates a bridge from validated params (the socket connects in `start`).
    ///
    /// Fails when `N == 0` or the params are invalid.
    pub fn from_params(params: CuMujocoSimParams) -> CuResult<Self> {
        if N == 0 {
            return Err(CuError::from(
                "repose-mujoco-sim: joint count N must be at least 1",
            ));
        }
        params.validate()?;
        Ok(Self {
            params,
            stream: None,
        })
    }

    /// Params this bridge runs with.
    pub fn params(&self) -> &CuMujocoSimParams {
        &self.params
    }

    fn timeout(&self) -> Duration {
        Duration::from_secs_f64(self.params.read_timeout.get::<second>())
    }

    fn connect(&self) -> CuResult<TcpStream> {
        let addr = (self.params.host.as_str(), self.params.port)
            .to_socket_addrs()
            .map_err(|e| {
                CuError::new_with_cause("repose-mujoco-sim: cannot resolve sidecar address", e)
            })?
            .next()
            .ok_or_else(|| {
                CuError::from("repose-mujoco-sim: sidecar address resolved to nothing")
            })?;
        let stream = TcpStream::connect_timeout(&addr, self.timeout()).map_err(|e| {
            CuError::new_with_cause("repose-mujoco-sim: cannot connect to sidecar", e)
        })?;
        stream.set_read_timeout(Some(self.timeout())).map_err(|e| {
            CuError::new_with_cause("repose-mujoco-sim: cannot set read timeout", e)
        })?;
        stream
            .set_write_timeout(Some(self.timeout()))
            .map_err(|e| {
                CuError::new_with_cause("repose-mujoco-sim: cannot set write timeout", e)
            })?;
        Ok(stream)
    }

    fn ensure_connected(&mut self) -> CuResult<&mut TcpStream> {
        match self.stream {
            Some(ref mut stream) => Ok(stream),
            None => {
                let stream = self.connect()?;
                Ok(self.stream.insert(stream))
            }
        }
    }

    fn drop_stream(&mut self) {
        self.stream = None;
    }
}

/// Encodes a joint command payload as one protocol line (with `\n`).
///
/// Only the command positions reach the wire; velocities and status stay on
/// the Copper side.
pub(crate) fn encode_command_line<const N: usize>(
    payload: &CuJointCommandPayload<N>,
) -> CuResult<String> {
    let q: [f64; N] = payload.positions.map(CuJointPosition::si_value);
    let qd: [f64; N] = payload.velocities.map(CuJointVelocity::si_value);
    let effort: [f64; N] = payload.efforts.map(CuJointEffort::si_value);
    let mut line = serde_json::to_string(&serde_json::json!({
        "type": "command",
        "n": N,
        "q": q.as_slice(),
        "qd": qd.as_slice(),
        "effort": effort.as_slice(),
        "mode": payload.mode,
    }))
    .map_err(|e| CuError::new_with_cause("repose-mujoco-sim: cannot encode command", e))?;
    line.push('\n');
    Ok(line)
}

/// Decodes one sidecar state line into `out`, parsing `q` into positions and
/// `qd` into velocities (`time` is ignored).
///
/// `out` is left untouched when the line is malformed; the error is returned.
pub(crate) fn decode_state_line<const N: usize>(
    line: &str,
    out: &mut CuJointStatePayload<N>,
) -> CuResult<()> {
    let value: serde_json::Value = serde_json::from_str(line)
        .map_err(|e| CuError::new_with_cause("repose-mujoco-sim: bad JSON", e))?;
    match value.get("type").and_then(serde_json::Value::as_str) {
        Some("state") => {}
        Some("error") => {
            let message = value
                .get("message")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("no message");
            return Err(CuError::from(format!(
                "repose-mujoco-sim: sidecar rejected the command: {message}"
            )));
        }
        _ => {
            return Err(CuError::from(
                "repose-mujoco-sim: expected {\"type\":\"state\",...}",
            ));
        }
    }
    if let Some(n) = value.get("n") {
        let sidecar_n = n.as_u64();
        if sidecar_n != u64::try_from(N).ok() {
            return Err(joint_count_error("n", n.to_string(), N));
        }
    }
    let q = joint_array::<N>(&value, "q")?;
    let qd = joint_array::<N>(&value, "qd")?;
    let mut positions = [CuJointPosition::default(); N];
    for (i, v) in q.iter().enumerate() {
        let rad = v.as_f64().ok_or_else(|| {
            CuError::from("repose-mujoco-sim: state \"q\" holds non-numeric joint")
        })?;
        positions[i] = CuJointPosition::Revolute(Angle::new::<radian>(rad));
    }
    let mut velocities = [CuJointVelocity::default(); N];
    for (i, v) in qd.iter().enumerate() {
        let rad_per_s = v.as_f64().ok_or_else(|| {
            CuError::from("repose-mujoco-sim: state \"qd\" holds non-numeric joint")
        })?;
        velocities[i] =
            CuJointVelocity::Revolute(AngularVelocity::new::<radian_per_second>(rad_per_s));
    }
    out.positions = positions;
    out.velocities = velocities;
    out.status = CuRobotStatus::Ready;
    Ok(())
}

/// Returns `value[key]` as an array of exactly `N` entries.
fn joint_array<'v, const N: usize>(
    value: &'v serde_json::Value,
    key: &str,
) -> CuResult<&'v [serde_json::Value]> {
    let array = value
        .get(key)
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| {
            CuError::from(format!(
                "repose-mujoco-sim: state line misses array \"{key}\""
            ))
        })?;
    if array.len() != N {
        return Err(joint_count_error(key, array.len().to_string(), N));
    }
    Ok(array)
}

fn joint_count_error(key: &str, sidecar: String, expected: usize) -> CuError {
    CuError::from(format!(
        "repose-mujoco-sim: joint count mismatch: sidecar state \"{key}\" reports {sidecar}, \
         bridge is CuMujocoSimBridge<{expected}>"
    ))
}

impl<const N: usize> Freezable for CuMujocoSimBridge<N> {
    fn freeze<E: Encoder>(&self, encoder: &mut E) -> Result<(), EncodeError> {
        Encode::encode(&self.params.host, encoder)?;
        Encode::encode(&self.params.port, encoder)?;
        Encode::encode(&self.params.read_timeout.get::<second>(), encoder)
    }

    fn thaw<D: Decoder>(&mut self, decoder: &mut D) -> Result<(), DecodeError> {
        self.params.host = Decode::decode(decoder)?;
        self.params.port = Decode::decode(decoder)?;
        let read_timeout: f64 = Decode::decode(decoder)?;
        self.params.read_timeout = Time::new::<second>(read_timeout);
        self.stream = None;
        Ok(())
    }
}

impl<const N: usize> CuBridge for CuMujocoSimBridge<N> {
    type Tx = MujocoTxChannels<N>;
    type Rx = MujocoRxChannels<N>;
    type Resources<'r> = ();

    fn new(
        config: Option<&ComponentConfig>,
        _tx_channels: &[BridgeChannelConfig<<Self::Tx as BridgeChannelSet>::Id>],
        _rx_channels: &[BridgeChannelConfig<<Self::Rx as BridgeChannelSet>::Id>],
        _resources: Self::Resources<'_>,
    ) -> CuResult<Self>
    where
        Self: Sized,
    {
        let params = config
            .map(ComponentConfig::deserialize_into::<CuMujocoSimParams>)
            .transpose()?
            .unwrap_or_default();
        Self::from_params(params)
    }

    fn start(&mut self, _ctx: &CuContext) -> CuResult<()> {
        self.stream = Some(self.connect()?);
        Ok(())
    }

    fn send<'a, Payload>(
        &mut self,
        _ctx: &CuContext,
        channel: &'static BridgeChannel<<Self::Tx as BridgeChannelSet>::Id, Payload>,
        msg: &CuMsg<Payload>,
    ) -> CuResult<()>
    where
        Payload: CuMsgPayload + 'a,
    {
        if channel.id != MujocoTxId::JointCommand {
            return Err(CuError::from("repose-mujoco-sim: unknown Tx channel"));
        }
        let cmd = msg.downcast_ref::<CuJointCommandPayload<N>>()?;
        let Some(payload) = cmd.payload() else {
            return Ok(());
        };
        let line = encode_command_line(payload)?;
        let stream = self.ensure_connected()?;
        let result = stream
            .write_all(line.as_bytes())
            .and_then(|()| stream.flush());
        if let Err(e) = result {
            self.drop_stream();
            return Err(CuError::new_with_cause(
                "repose-mujoco-sim: command write failed",
                e,
            ));
        }
        Ok(())
    }

    fn receive<'a, Payload>(
        &mut self,
        ctx: &CuContext,
        channel: &'static BridgeChannel<<Self::Rx as BridgeChannelSet>::Id, Payload>,
        msg: &mut CuMsg<Payload>,
    ) -> CuResult<()>
    where
        Payload: CuMsgPayload + 'a,
    {
        if channel.id != MujocoRxId::JointState {
            return Err(CuError::from("repose-mujoco-sim: unknown Rx channel"));
        }
        let stream = self.ensure_connected()?;
        let mut buf = [0u8; MAX_LINE_BYTES];
        let mut len = 0usize;
        loop {
            if len >= buf.len() {
                self.drop_stream();
                return Err(CuError::from("repose-mujoco-sim: state line too long"));
            }
            let n = match stream.read(&mut buf[len..len + 1]) {
                Ok(0) => {
                    self.drop_stream();
                    return Err(CuError::from(
                        "repose-mujoco-sim: sidecar closed the connection",
                    ));
                }
                Ok(n) => n,
                Err(e) => {
                    self.drop_stream();
                    return Err(CuError::new_with_cause(
                        "repose-mujoco-sim: state read failed",
                        e,
                    ));
                }
            };
            len += n;
            if buf[len - 1] == b'\n' {
                break;
            }
        }
        let line = core::str::from_utf8(&buf[..len]).map_err(|e| {
            CuError::new_with_cause("repose-mujoco-sim: state line is not UTF-8", e)
        })?;
        let state = msg.downcast_mut::<CuJointStatePayload<N>>()?;
        state.tov = Tov::Time(ctx.now());
        let Some(payload) = state.payload_mut() else {
            let mut fresh = CuJointStatePayload::<N>::default();
            decode_state_line(line, &mut fresh)?;
            state.set_payload(fresh);
            return Ok(());
        };
        decode_state_line(line, payload)
    }

    fn stop(&mut self, _ctx: &CuContext) -> CuResult<()> {
        self.drop_stream();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use repose_controller::CuControllerMode;
    use std::net::TcpListener;
    use std::sync::mpsc::channel;
    use std::thread::spawn;

    fn test_command<const N: usize>(q: [f64; N]) -> CuJointCommandPayload<N> {
        CuJointCommandPayload {
            positions: q.map(|v| CuJointPosition::Revolute(Angle::new::<radian>(v))),
            ..Default::default()
        }
    }

    #[test]
    fn test_encode_command_line_shape() {
        let line = encode_command_line(&test_command([0.0, 0.5, -1.0, 1.5, -0.25, 3.0])).unwrap();
        assert!(line.ends_with('\n'));
        let value: serde_json::Value = serde_json::from_str(&line).unwrap();
        assert_eq!(value.get("type").and_then(|t| t.as_str()), Some("command"));
        let q = value.get("q").and_then(|q| q.as_array()).unwrap();
        assert_eq!(q.len(), 6);
        for (got, want) in q.iter().zip([0.0, 0.5, -1.0, 1.5, -0.25, 3.0]) {
            assert!((got.as_f64().unwrap() - want).abs() < 1e-12);
        }
        // Default command is position mode with zero velocity/effort.
        assert_eq!(value.get("mode").and_then(|m| m.as_str()), Some("position"));
        assert_eq!(value.get("qd").and_then(|v| v.as_array()).unwrap().len(), 6);
        assert_eq!(
            value
                .get("effort")
                .and_then(|v| v.as_array())
                .unwrap()
                .len(),
            6
        );
    }

    #[test]
    fn test_encode_command_line_carries_effort_mode() {
        let mut payload = test_command([0.0; 6]);
        payload.mode = CuControllerMode::Effort;
        payload.efforts = [CuJointEffort::Revolute(cu29::units::si::f64::Torque::new::<
            cu29::units::si::torque::newton_meter,
        >(2.5)); 6];
        let line = encode_command_line(&payload).unwrap();
        let value: serde_json::Value = serde_json::from_str(&line).unwrap();
        assert_eq!(value.get("mode").and_then(|m| m.as_str()), Some("effort"));
        let effort = value.get("effort").and_then(|v| v.as_array()).unwrap();
        assert!((effort[0].as_f64().unwrap() - 2.5).abs() < 1e-12);
    }

    #[test]
    fn test_decode_state_line_round_trip() {
        let line = "{\"type\":\"state\",\"q\":[0.1,0.2,0.3,0.4,0.5,0.6],\"qd\":[0.01,0.02,0.03,0.04,0.05,0.06],\"time\":1.25}\n";
        let mut out = CuJointStatePayload::<6>::default();
        decode_state_line(line, &mut out).unwrap();
        for (i, want) in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6].iter().enumerate() {
            assert!((out.positions[i].si_value() - want).abs() < 1e-12);
        }
        for (i, want) in [0.01, 0.02, 0.03, 0.04, 0.05, 0.06].iter().enumerate() {
            assert!((out.velocities[i].si_value() - want).abs() < 1e-12);
        }
    }

    #[test]
    fn test_decode_state_line_rejects_malformed() {
        let mut out = CuJointStatePayload::<6>::default();
        for bad in [
            "not json\n",
            "{\"type\":\"command\",\"q\":[0,0,0,0,0,0]}\n",
            "{\"type\":\"state\",\"qd\":[0,0,0,0,0,0]}\n",
            "{\"type\":\"state\",\"q\":[0,0,0,0,0]}\n",
            "{\"type\":\"state\",\"q\":[0,0,0,0,0,0,0]}\n",
            "{\"type\":\"state\",\"q\":[0,0,0,0,0,\"x\"],\"qd\":[0,0,0,0,0,0]}\n",
            "{\"type\":\"state\",\"q\":[0,0,0,0,0,0]}\n",
            "{\"type\":\"state\",\"q\":[0,0,0,0,0,0],\"qd\":[0,0,0,0,0]}\n",
            "{\"type\":\"state\",\"q\":[0,0,0,0,0,0],\"qd\":[0,0,0,0,0,\"x\"]}\n",
        ] {
            assert!(decode_state_line(bad, &mut out).is_err(), "accepted: {bad}");
        }
        // Rejected lines leave the payload untouched (still default zeros).
        for p in out.positions {
            assert_eq!(p.si_value(), 0.0);
        }
        for v in out.velocities {
            assert_eq!(v.si_value(), 0.0);
        }
    }

    #[test]
    fn test_new_defaults() {
        let bridge = CuMujocoSimBridge::<6>::new(None, &[], &[], ()).unwrap();
        assert_eq!(bridge.params().host, "127.0.0.1");
        assert_eq!(bridge.params().port, 8765);
        assert_eq!(bridge.timeout(), Duration::from_millis(250));
    }

    #[test]
    fn test_new_reads_config() {
        let cfg: ComponentConfig = ron::from_str(
            r#"#![enable(unwrap_newtypes)] {"host": "10.0.0.9", "port": 9999, "read_timeout": 0.042}"#,
        )
        .unwrap();
        let bridge = CuMujocoSimBridge::<6>::new(Some(&cfg), &[], &[], ()).unwrap();
        assert_eq!(bridge.params().host, "10.0.0.9");
        assert_eq!(bridge.params().port, 9999);
        assert_eq!(bridge.timeout(), Duration::from_millis(42));
        for bad in [
            r#"{"read_timeout_ms": 250}"#,
            r#"{"read_timeout": 0.0}"#,
            r#"{"host": ""}"#,
        ] {
            let cfg: ComponentConfig =
                ron::from_str(&format!("#![enable(unwrap_newtypes)] {bad}")).unwrap();
            assert!(
                CuMujocoSimBridge::<6>::new(Some(&cfg), &[], &[], ()).is_err(),
                "{bad}"
            );
        }
    }

    #[test]
    fn test_freeze_thaw_restores_config_without_socket() {
        let mut bridge = CuMujocoSimBridge::<6>::from_params(CuMujocoSimParams {
            host: "10.0.0.9".into(),
            port: 9999,
            read_timeout: Time::new::<second>(0.042),
        })
        .unwrap();
        let mut bytes = [0u8; 256];
        let mut encoder = bincode::enc::EncoderImpl::new(
            bincode::enc::write::SliceWriter::new(&mut bytes),
            bincode::config::standard(),
        );
        bridge.freeze(&mut encoder).unwrap();
        let len = encoder.into_writer().bytes_written();
        bridge.params.host = "changed".into();
        let mut decoder = bincode::de::DecoderImpl::new(
            bincode::de::read::SliceReader::new(&bytes[..len]),
            bincode::config::standard(),
            (),
        );
        bridge.thaw(&mut decoder).unwrap();
        assert_eq!(bridge.params().host, "10.0.0.9");
        assert_eq!(bridge.params().port, 9999);
        assert_eq!(bridge.timeout(), Duration::from_millis(42));
    }

    /// Loopback sidecar: one command line in, one state line out.
    /// Binds port 0 (ephemeral) and hands the port over a channel, so the
    /// test cannot collide with a fixed port.
    #[test]
    fn test_loopback_command_state_cycle() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let (ready_tx, ready_rx) = channel();
        let sidecar = spawn(move || {
            ready_tx.send(()).unwrap();
            let (mut sock, _) = listener.accept().unwrap();
            sock.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
            let mut buf = [0u8; MAX_LINE_BYTES];
            let mut len = 0usize;
            while !buf[..len].contains(&b'\n') {
                len += sock.read(&mut buf[len..]).unwrap();
            }
            let value: serde_json::Value =
                serde_json::from_str(core::str::from_utf8(&buf[..len]).unwrap()).unwrap();
            assert_eq!(value.get("type").and_then(|t| t.as_str()), Some("command"));
            let reply = "{\"type\":\"state\",\"q\":[0.1,0.2,0.3,0.4,0.5,0.6],\"qd\":[0,0,0,0,0,0],\"time\":0.01}\n";
            sock.write_all(reply.as_bytes()).unwrap();
            sock.flush().unwrap();
        });
        ready_rx.recv().unwrap();

        let (ctx, _) = CuContext::new_mock_clock();
        let mut bridge = CuMujocoSimBridge::<6>::from_params(CuMujocoSimParams {
            port,
            read_timeout: Time::new::<second>(5.0),
            ..CuMujocoSimParams::default()
        })
        .unwrap();
        bridge.start(&ctx).unwrap();
        let cmd = CuMsg::new(Some(test_command([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])));
        bridge
            .send(&ctx, &MujocoTxChannels::<6>::JOINT_COMMAND, &cmd)
            .unwrap();
        let mut state = CuMsg::<CuJointStatePayload<6>>::new(None);
        bridge
            .receive(&ctx, &MujocoRxChannels::<6>::JOINT_STATE, &mut state)
            .unwrap();
        let payload = state.payload().unwrap();
        for (i, want) in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6].iter().enumerate() {
            assert!((payload.positions[i].si_value() - want).abs() < 1e-12);
        }
        bridge.stop(&ctx).unwrap();
        sidecar.join().unwrap();
    }

    fn joints_json(n: usize, base: f64) -> String {
        let values: Vec<String> = (0..n).map(|i| (base + i as f64).to_string()).collect();
        format!("[{}]", values.join(","))
    }

    fn state_line(n_field: Option<usize>, q_len: usize, qd_len: usize) -> String {
        let n = n_field.map_or(String::new(), |n| format!("\"n\":{n},"));
        format!(
            "{{\"type\":\"state\",{n}\"q\":{},\"qd\":{},\"time\":0.5}}\n",
            joints_json(q_len, 0.1),
            joints_json(qd_len, 0.01)
        )
    }

    fn round_trip<const N: usize>() {
        let mut q = [0.0; N];
        for (i, v) in q.iter_mut().enumerate() {
            *v = 0.25 * i as f64 - 0.5;
        }
        let line = encode_command_line(&test_command(q)).unwrap();
        let value: serde_json::Value = serde_json::from_str(&line).unwrap();
        assert_eq!(value.get("n").and_then(|n| n.as_u64()), Some(N as u64));
        for key in ["q", "qd", "effort"] {
            assert_eq!(value[key].as_array().unwrap().len(), N, "{key}");
        }
        for (got, want) in value["q"].as_array().unwrap().iter().zip(q) {
            assert!((got.as_f64().unwrap() - want).abs() < 1e-12);
        }
        // Both with and without the optional `n` field.
        for n_field in [Some(N), None] {
            let mut out = CuJointStatePayload::<N>::default();
            decode_state_line(&state_line(n_field, N, N), &mut out).unwrap();
            for i in 0..N {
                assert!((out.positions[i].si_value() - (0.1 + i as f64)).abs() < 1e-12);
                assert!((out.velocities[i].si_value() - (0.01 + i as f64)).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn test_protocol_round_trip_for_other_joint_counts() {
        round_trip::<1>();
        round_trip::<3>();
        round_trip::<6>();
        round_trip::<7>();
    }

    fn assert_mismatch<const N: usize>(line: &str) {
        let mut out = CuJointStatePayload::<N>::default();
        let err = decode_state_line(line, &mut out).unwrap_err().to_string();
        assert!(err.contains("joint count mismatch"), "{err}");
        assert!(err.contains(&format!("CuMujocoSimBridge<{N}>")), "{err}");
        // A rejected line never touches the payload.
        assert!(out.positions.iter().all(|p| p.si_value() == 0.0));
    }

    #[test]
    fn test_joint_count_mismatch_is_a_clear_error() {
        // Sidecar serves 6 joints, bridge is N = 3 and N = 7 (and vice versa).
        assert_mismatch::<3>(&state_line(Some(6), 6, 6));
        assert_mismatch::<3>(&state_line(None, 6, 6));
        assert_mismatch::<7>(&state_line(None, 6, 6));
        assert_mismatch::<6>(&state_line(None, 3, 3));
        // Only one of the two arrays has the wrong length.
        assert_mismatch::<3>(&state_line(Some(3), 3, 2));
        assert_mismatch::<3>(&state_line(Some(3), 4, 3));
    }

    #[test]
    fn test_sidecar_error_line_becomes_cu_error() {
        let mut out = CuJointStatePayload::<3>::default();
        let err = decode_state_line(
            "{\"type\":\"error\",\"message\":\"sidecar serves 6 joints, command has 3\"}\n",
            &mut out,
        )
        .unwrap_err()
        .to_string();
        assert!(
            err.contains("sidecar serves 6 joints, command has 3"),
            "{err}"
        );
        assert!(decode_state_line("{\"type\":\"error\"}\n", &mut out).is_err());
    }

    #[test]
    fn test_zero_joint_bridge_is_rejected() {
        assert!(CuMujocoSimBridge::<0>::new(None, &[], &[], ()).is_err());
    }

    /// Loopback sidecar serving `sidecar_n` joints against a bridge with
    /// `N` joints: the command is accepted, the state line is not.
    fn loopback_mismatch<const N: usize>(sidecar_n: usize) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let sidecar = spawn(move || {
            let (mut sock, _) = listener.accept().unwrap();
            sock.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
            let mut buf = [0u8; 2 * MAX_LINE_BYTES];
            let mut len = 0usize;
            while !buf[..len].contains(&b'\n') {
                len += sock.read(&mut buf[len..]).unwrap();
            }
            let value: serde_json::Value =
                serde_json::from_str(core::str::from_utf8(&buf[..len]).unwrap()).unwrap();
            assert_eq!(value["n"].as_u64(), Some(N as u64));
            let reply = state_line(Some(sidecar_n), sidecar_n, sidecar_n);
            sock.write_all(reply.as_bytes()).unwrap();
            sock.flush().unwrap();
        });
        let (ctx, _) = CuContext::new_mock_clock();
        let mut bridge = CuMujocoSimBridge::<N>::from_params(CuMujocoSimParams {
            port,
            read_timeout: Time::new::<second>(5.0),
            ..CuMujocoSimParams::default()
        })
        .unwrap();
        bridge.start(&ctx).unwrap();
        let cmd = CuMsg::new(Some(test_command([0.0; N])));
        bridge
            .send(&ctx, &MujocoTxChannels::<N>::JOINT_COMMAND, &cmd)
            .unwrap();
        let mut state = CuMsg::<CuJointStatePayload<N>>::new(None);
        let err = bridge
            .receive(&ctx, &MujocoRxChannels::<N>::JOINT_STATE, &mut state)
            .unwrap_err()
            .to_string();
        assert!(err.contains("joint count mismatch"), "{err}");
        assert!(state.payload().is_none());
        bridge.stop(&ctx).unwrap();
        sidecar.join().unwrap();
    }

    #[test]
    fn test_loopback_joint_count_mismatch_errors_without_panic() {
        loopback_mismatch::<3>(6);
        loopback_mismatch::<7>(6);
    }

    #[test]
    fn test_loopback_non_six_cycle() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let sidecar = spawn(move || {
            let (mut sock, _) = listener.accept().unwrap();
            sock.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
            let mut buf = [0u8; 2 * MAX_LINE_BYTES];
            let mut len = 0usize;
            while !buf[..len].contains(&b'\n') {
                len += sock.read(&mut buf[len..]).unwrap();
            }
            sock.write_all(state_line(Some(3), 3, 3).as_bytes())
                .unwrap();
        });
        let (ctx, _) = CuContext::new_mock_clock();
        let mut bridge = CuMujocoSimBridge::<3>::from_params(CuMujocoSimParams {
            port,
            read_timeout: Time::new::<second>(5.0),
            ..CuMujocoSimParams::default()
        })
        .unwrap();
        bridge.start(&ctx).unwrap();
        let cmd = CuMsg::new(Some(test_command([0.1, 0.2, 0.3])));
        bridge
            .send(&ctx, &MujocoTxChannels::<3>::JOINT_COMMAND, &cmd)
            .unwrap();
        let mut state = CuMsg::<CuJointStatePayload<3>>::new(None);
        bridge
            .receive(&ctx, &MujocoRxChannels::<3>::JOINT_STATE, &mut state)
            .unwrap();
        let payload = state.payload().unwrap();
        assert!((payload.positions[2].si_value() - 2.1).abs() < 1e-12);
        bridge.stop(&ctx).unwrap();
        sidecar.join().unwrap();
    }
}
