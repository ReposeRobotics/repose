use bincode::Decode;
use bincode::Encode;
use cu29::prelude::*;
use serde::Deserialize;
use serde::Serialize;

/// Vendor-neutral session command, the payload of a bridge's `session` Tx channel.
///
/// No command moves the robot. [`EnableDrives`](Self::EnableDrives) energises
/// the drives and releases the brakes while the robot holds position.
#[derive(
    Default, Debug, Clone, Copy, PartialEq, Eq, Encode, Decode, Serialize, Deserialize, Reflect,
)]
#[serde(rename_all = "snake_case")]
pub enum CuSessionCommand {
    /// Ask the controller to accept commands from this client.
    #[default]
    AcquireControl,
    /// Hand control back to the controller or other clients.
    ReleaseControl,
    /// Clear resettable controller and drive faults.
    ClearFaults,
    /// Energise the drives.
    EnableDrives,
    /// De-energise the drives.
    DisableDrives,
}

/// Outcome of the most recent session command.
#[derive(
    Default, Debug, Clone, Copy, PartialEq, Eq, Encode, Decode, Serialize, Deserialize, Reflect,
)]
#[serde(rename_all = "snake_case")]
pub enum CuSessionResult {
    /// No command has been requested.
    #[default]
    Idle,
    /// Sent; waiting for the controller's answer.
    Pending,
    /// The controller accepted the command.
    Acknowledged,
    /// The command was refused; [`CuSessionState::rejection`] says by whom.
    Rejected,
}

/// Why the most recent command was rejected.
#[derive(
    Default, Debug, Clone, Copy, PartialEq, Eq, Encode, Decode, Serialize, Deserialize, Reflect,
)]
#[serde(rename_all = "snake_case")]
pub enum CuSessionRejection {
    /// Not rejected.
    #[default]
    None,
    /// Refused locally: the controller does not report this client in control.
    NotInControl,
    /// Refused by the controller; [`CuSessionState::detail`] holds its text.
    Controller,
}

/// Fixed-capacity ASCII text from a controller, truncated to 32 bytes.
#[derive(Default, Debug, Clone, PartialEq, Eq, Encode, Decode, Serialize, Deserialize, Reflect)]
pub struct CuSessionDetail {
    bytes: [u8; 32],
    len: u8,
}

impl CuSessionDetail {
    /// Store `text`, truncated to 32 bytes on a character boundary.
    pub fn new(text: &str) -> Self {
        let mut take = text.len().min(32);
        while !text.is_char_boundary(take) {
            take -= 1;
        }
        let mut bytes = [0; 32];
        bytes[..take].copy_from_slice(&text.as_bytes()[..take]);
        Self {
            bytes,
            len: take as u8,
        }
    }

    /// The stored text; empty when none was given.
    pub fn as_str(&self) -> &str {
        core::str::from_utf8(&self.bytes[..usize::from(self.len.min(32))]).unwrap_or("")
    }
}

/// Session state, the payload of a bridge's `session_state` Rx channel.
#[derive(Default, Debug, Clone, PartialEq, Encode, Decode, Serialize, Deserialize, Reflect)]
pub struct CuSessionState {
    /// Most recent command, absent before the first request.
    pub command: Option<CuSessionCommand>,
    /// Protocol sequence number the command was sent with.
    pub sequence: u32,
    /// Outcome of that command.
    pub result: CuSessionResult,
    /// Who rejected it, when `result` is `Rejected`.
    pub rejection: CuSessionRejection,
    /// Controller text accompanying a rejection.
    pub detail: CuSessionDetail,
    /// The controller reports this client in control.
    pub control_acquired: bool,
    /// Drives were enabled by an acknowledged `EnableDrives` and not since disabled.
    pub drives_enabled: bool,
    /// Every drive reports ready. Lags `drives_enabled` on controllers that
    /// bring axes up one at a time.
    pub drives_ready: bool,
    /// The controller reports that it would accept motion now.
    pub motion_allowed: bool,
    /// Requests refused because another command was still pending.
    pub refused: u32,
}

/// What a bridge does with a request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CuSessionDecision {
    /// Queued; fetch it with [`CuSessionTracker::take_outgoing`] and send it.
    Send,
    /// Not sent; [`CuSessionState`] records why.
    Refused,
}

/// Protocol-independent session rules shared by every bridge.
///
/// The bridge feeds it requests from Copper and events decoded from the
/// controller. It owns no I/O and never allocates.
#[derive(Default, Debug, Clone)]
pub struct CuSessionTracker {
    state: CuSessionState,
    outgoing: Option<CuSessionCommand>,
    /// Taken by the bridge but not yet confirmed with `mark_sent`.
    in_transit: bool,
    /// Whether control was already held when `AcquireControl` went out.
    held_before_acquire: bool,
    acquired_here: bool,
    changed: bool,
}

impl CuSessionTracker {
    /// Current state.
    pub fn state(&self) -> &CuSessionState {
        &self.state
    }

    /// True once since the last call if the state changed.
    pub fn take_changed(&mut self) -> bool {
        core::mem::take(&mut self.changed)
    }

    /// True while a command is queued or awaiting its answer.
    pub fn is_busy(&self) -> bool {
        self.outgoing.is_some() || self.in_transit || self.state.result == CuSessionResult::Pending
    }

    /// Handle one request from Copper.
    pub fn request(&mut self, command: CuSessionCommand) -> CuSessionDecision {
        self.changed = true;
        if self.is_busy() {
            self.state.refused = self.state.refused.saturating_add(1);
            return CuSessionDecision::Refused;
        }
        if command != CuSessionCommand::AcquireControl && !self.state.control_acquired {
            self.state.command = Some(command);
            self.state.result = CuSessionResult::Rejected;
            self.state.rejection = CuSessionRejection::NotInControl;
            self.state.detail = CuSessionDetail::default();
            return CuSessionDecision::Refused;
        }
        self.outgoing = Some(command);
        CuSessionDecision::Send
    }

    /// The queued command, removed from the queue. Pass it to
    /// [`mark_sent`](Self::mark_sent) once written.
    pub fn take_outgoing(&mut self) -> Option<CuSessionCommand> {
        let command = self.outgoing.take();
        self.in_transit |= command.is_some();
        command
    }

    /// Record that `command` went out with protocol sequence `sequence`.
    pub fn mark_sent(&mut self, command: CuSessionCommand, sequence: u32) {
        self.in_transit = false;
        if command == CuSessionCommand::AcquireControl {
            self.held_before_acquire = self.state.control_acquired;
        }
        self.state.command = Some(command);
        self.state.sequence = sequence;
        self.state.result = CuSessionResult::Pending;
        self.state.rejection = CuSessionRejection::None;
        self.state.detail = CuSessionDetail::default();
        self.changed = true;
    }

    /// The controller accepted command `sequence`. Other sequences are ignored.
    pub fn on_acknowledged(&mut self, sequence: u32) {
        if !self.pending(sequence) {
            return;
        }
        self.state.result = CuSessionResult::Acknowledged;
        match self.state.command {
            Some(CuSessionCommand::AcquireControl) => {
                self.state.control_acquired = true;
                self.acquired_here = !self.held_before_acquire;
            }
            Some(CuSessionCommand::ReleaseControl) => self.lose_control(),
            Some(CuSessionCommand::EnableDrives) => self.state.drives_enabled = true,
            Some(CuSessionCommand::DisableDrives) => self.disable(),
            Some(CuSessionCommand::ClearFaults) | None => {}
        }
        self.changed = true;
    }

    /// The controller refused command `sequence` with `detail`.
    pub fn on_rejected(&mut self, sequence: u32, detail: &str) {
        if !self.pending(sequence) {
            return;
        }
        self.state.result = CuSessionResult::Rejected;
        self.state.rejection = CuSessionRejection::Controller;
        self.state.detail = CuSessionDetail::new(detail);
        self.changed = true;
    }

    /// The controller reported whether this client is in control. On
    /// controllers that answer control requests this way instead of with an
    /// acknowledgement, it also completes a pending acquire or release.
    pub fn on_control_report(&mut self, in_control: bool) {
        if self.state.result == CuSessionResult::Pending {
            match (self.state.command, in_control) {
                (Some(CuSessionCommand::AcquireControl), true) => {
                    self.state.result = CuSessionResult::Acknowledged;
                    self.acquired_here = !self.held_before_acquire;
                }
                (Some(CuSessionCommand::ReleaseControl), false) => {
                    self.state.result = CuSessionResult::Acknowledged;
                }
                _ => {}
            }
        }
        if in_control {
            self.state.control_acquired = true;
        } else {
            self.lose_control();
        }
        self.changed = true;
    }

    /// The controller reported drive readiness and whether it would accept motion.
    pub fn on_drive_report(&mut self, drives_ready: bool, motion_allowed: bool) {
        if (drives_ready, motion_allowed) != (self.state.drives_ready, self.state.motion_allowed) {
            self.state.drives_ready = drives_ready;
            self.state.motion_allowed = motion_allowed;
            self.changed = true;
        }
    }

    /// The transport was lost: nothing is pending and nothing is held.
    pub fn on_disconnected(&mut self) {
        self.outgoing = None;
        self.in_transit = false;
        if self.state.result == CuSessionResult::Pending {
            self.state.result = CuSessionResult::Idle;
        }
        self.lose_control();
        self.changed = true;
    }

    /// Next command to wind the session down, or `None` once this session
    /// holds nothing: disable drives this session enabled, then release
    /// control this session acquired. Queue it with [`request`](Self::request)
    /// when the tracker is not busy.
    pub fn shutdown_step(&self) -> Option<CuSessionCommand> {
        if !self.state.control_acquired {
            None
        } else if self.state.drives_enabled {
            Some(CuSessionCommand::DisableDrives)
        } else if self.acquired_here {
            Some(CuSessionCommand::ReleaseControl)
        } else {
            None
        }
    }

    fn pending(&self, sequence: u32) -> bool {
        self.state.result == CuSessionResult::Pending && self.state.sequence == sequence
    }

    fn disable(&mut self) {
        self.state.drives_enabled = false;
        self.state.drives_ready = false;
        self.state.motion_allowed = false;
    }

    /// Control is gone: drop queued commands that need it, and fail a
    /// pending one so a late answer cannot mark drives enabled.
    fn lose_control(&mut self) {
        self.state.control_acquired = false;
        self.acquired_here = false;
        self.held_before_acquire = false;
        self.disable();
        if self
            .outgoing
            .is_some_and(|command| command != CuSessionCommand::AcquireControl)
        {
            self.outgoing = None;
        }
        let needs_control = !matches!(
            self.state.command,
            Some(CuSessionCommand::AcquireControl | CuSessionCommand::ReleaseControl)
        );
        if self.state.result == CuSessionResult::Pending && needs_control {
            self.state.result = CuSessionResult::Rejected;
            self.state.rejection = CuSessionRejection::NotInControl;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn send(tracker: &mut CuSessionTracker, command: CuSessionCommand, sequence: u32) {
        assert_eq!(tracker.request(command), CuSessionDecision::Send);
        let out = tracker.take_outgoing().unwrap();
        assert_eq!(out, command);
        tracker.mark_sent(out, sequence);
    }

    #[test]
    fn test_refuses_without_control_and_while_busy() {
        let mut tracker = CuSessionTracker::default();
        assert_eq!(
            tracker.request(CuSessionCommand::EnableDrives),
            CuSessionDecision::Refused
        );
        assert_eq!(tracker.state().rejection, CuSessionRejection::NotInControl);
        assert!(tracker.take_outgoing().is_none());
        send(&mut tracker, CuSessionCommand::AcquireControl, 1);
        assert_eq!(
            tracker.request(CuSessionCommand::ClearFaults),
            CuSessionDecision::Refused
        );
        assert_eq!(tracker.state().refused, 1);
        // A busy refusal leaves the pending command intact.
        assert_eq!(tracker.state().result, CuSessionResult::Pending);
        tracker.on_control_report(true);
        assert_eq!(tracker.state().result, CuSessionResult::Acknowledged);
    }

    #[test]
    fn test_answers_match_sequence() {
        let mut tracker = CuSessionTracker::default();
        tracker.on_control_report(true);
        send(&mut tracker, CuSessionCommand::EnableDrives, 5);
        tracker.on_acknowledged(4);
        assert_eq!(tracker.state().result, CuSessionResult::Pending);
        tracker.on_acknowledged(5);
        assert!(tracker.state().drives_enabled);
        send(&mut tracker, CuSessionCommand::DisableDrives, 6);
        tracker.on_rejected(6, "drives busy and this text is longer than thirty-two");
        assert_eq!(tracker.state().rejection, CuSessionRejection::Controller);
        assert_eq!(
            tracker.state().detail.as_str(),
            "drives busy and this text is lon"
        );
        assert!(tracker.state().drives_enabled);
    }

    #[test]
    fn test_shutdown_releases_only_what_it_took() {
        let mut tracker = CuSessionTracker::default();
        tracker.on_control_report(true);
        // In control from the start, not acquired here: nothing to release.
        assert_eq!(tracker.shutdown_step(), None);
        send(&mut tracker, CuSessionCommand::EnableDrives, 1);
        tracker.on_acknowledged(1);
        tracker.on_drive_report(true, true);
        assert_eq!(
            tracker.shutdown_step(),
            Some(CuSessionCommand::DisableDrives)
        );
        send(&mut tracker, CuSessionCommand::DisableDrives, 2);
        tracker.on_acknowledged(2);
        assert!(!tracker.state().drives_ready && !tracker.state().motion_allowed);
        assert_eq!(tracker.shutdown_step(), None);

        let mut tracker = CuSessionTracker::default();
        tracker.on_control_report(false);
        send(&mut tracker, CuSessionCommand::AcquireControl, 1);
        tracker.on_control_report(true);
        assert_eq!(
            tracker.shutdown_step(),
            Some(CuSessionCommand::ReleaseControl)
        );
        send(&mut tracker, CuSessionCommand::ReleaseControl, 2);
        tracker.on_control_report(false);
        assert_eq!(tracker.state().result, CuSessionResult::Acknowledged);
        assert_eq!(tracker.shutdown_step(), None);
    }

    #[test]
    fn test_losing_control_cancels_privileged_commands() {
        let mut tracker = CuSessionTracker::default();
        tracker.on_control_report(true);
        assert_eq!(
            tracker.request(CuSessionCommand::EnableDrives),
            CuSessionDecision::Send
        );
        tracker.on_control_report(false);
        assert_eq!(tracker.take_outgoing(), None);

        tracker.on_control_report(true);
        send(&mut tracker, CuSessionCommand::EnableDrives, 3);
        tracker.on_control_report(false);
        assert_eq!(tracker.state().result, CuSessionResult::Rejected);
        assert_eq!(tracker.state().rejection, CuSessionRejection::NotInControl);
        tracker.on_acknowledged(3); // late answer is ignored
        assert!(!tracker.state().drives_enabled);
    }

    #[test]
    fn test_busy_between_take_and_mark_sent() {
        let mut tracker = CuSessionTracker::default();
        tracker.on_control_report(true);
        tracker.request(CuSessionCommand::ClearFaults);
        let command = tracker.take_outgoing().unwrap();
        assert!(tracker.is_busy());
        assert_eq!(
            tracker.request(CuSessionCommand::EnableDrives),
            CuSessionDecision::Refused
        );
        tracker.mark_sent(command, 1);
        assert_eq!(tracker.state().command, Some(CuSessionCommand::ClearFaults));
    }

    #[test]
    fn test_acquire_while_already_held_is_not_released() {
        for via_report in [false, true] {
            let mut tracker = CuSessionTracker::default();
            tracker.on_control_report(true);
            send(&mut tracker, CuSessionCommand::AcquireControl, 1);
            if via_report {
                tracker.on_control_report(true);
            } else {
                tracker.on_acknowledged(1);
            }
            assert_eq!(tracker.state().result, CuSessionResult::Acknowledged);
            assert_eq!(tracker.shutdown_step(), None);
        }
    }

    #[test]
    fn test_disconnect_clears_session() {
        let mut tracker = CuSessionTracker::default();
        tracker.on_control_report(true);
        send(&mut tracker, CuSessionCommand::EnableDrives, 1);
        tracker.on_disconnected();
        assert!(!tracker.is_busy());
        let state = tracker.state();
        assert_eq!(state.result, CuSessionResult::Idle);
        assert!(!state.control_acquired && !state.drives_enabled);
        assert_eq!(tracker.shutdown_step(), None);
    }

    #[test]
    fn test_command_ron_spelling() {
        let command: CuSessionCommand = ron::from_str("enable_drives").unwrap();
        assert_eq!(command, CuSessionCommand::EnableDrives);
    }
}
