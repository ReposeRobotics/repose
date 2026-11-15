//! [`CuGripperActionTask`]: one element of a raw policy action in, an
//! open/close gripper command out.

use crate::payload::CuPolicyAction;
use crate::payload::CuPolicyStatus;
use bincode::Decode;
use bincode::Encode;
use cu29::prelude::*;
use repose_io::CuGripperAction;
use repose_io::CuGripperCommandPayload;
use serde::Deserialize;
use serde::Serialize;

/// Which side of the threshold means "close".
#[derive(
    Default, Debug, Clone, Copy, PartialEq, Eq, Encode, Decode, Serialize, Deserialize, Reflect,
)]
#[serde(rename_all = "snake_case")]
pub enum CuGripperCloseWhen {
    /// A value above the threshold closes the gripper (a "closedness" action,
    /// as in RT-1).
    #[default]
    Above,
    /// A value below the threshold closes the gripper (an "openness" action,
    /// as in Bridge and Open X-Embodiment data, where `1` is open).
    Below,
}

/// Parameters of [`CuGripperActionTask`]; the node `config:` block.
///
/// ```ron
/// config: {
///     "offset": 6,           // action element this task reads
///     "threshold": 0.5,      // dimensionless, like the action
///     "close_when": "below", // above | below
///     "hysteresis": 0.1,     // optional; dead band on each side of the threshold
/// }
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Reflect)]
#[serde(default, deny_unknown_fields)]
pub struct CuGripperActionParams {
    /// Index of the action element this task reads; defaults to `0`.
    pub offset: usize,
    /// Switching threshold (dimensionless); defaults to `0.0`.
    pub threshold: f64,
    /// Which side of the threshold closes; defaults to `above`.
    pub close_when: CuGripperCloseWhen,
    /// Dead band (dimensionless, `>= 0`) on each side of the threshold. A
    /// value within `threshold ± hysteresis` keeps the last command, so a
    /// model output hovering at the threshold does not chatter the valve.
    /// Defaults to `0.0`.
    pub hysteresis: f64,
}

impl Default for CuGripperActionParams {
    fn default() -> Self {
        Self {
            offset: 0,
            threshold: 0.0,
            close_when: CuGripperCloseWhen::Above,
            hysteresis: 0.0,
        }
    }
}

impl CuGripperActionParams {
    /// Checks that the threshold is finite and the hysteresis finite and
    /// non-negative.
    pub fn validate(&self) -> CuResult<()> {
        if !self.threshold.is_finite() {
            return Err(CuError::from(
                "repose-policy-rl: gripper 'threshold' must be finite",
            ));
        }
        if !self.hysteresis.is_finite() || self.hysteresis < 0.0 {
            return Err(CuError::from(
                "repose-policy-rl: gripper 'hysteresis' must be finite and non-negative",
            ));
        }
        Ok(())
    }
}

/// Turns one element of a raw policy action into an open/close command for
/// `repose_io::CuGripperTask`.
///
/// Input is a [`CuPolicyAction<ACT>`]; the task reads element `offset`
/// (`offset < ACT` is checked at construction), so it shares one action with
/// a [`crate::CuCartesianActionTask`] or [`crate::CuJointActionTask`] reading
/// other elements. Output is a [`CuGripperCommandPayload`] carrying the
/// action's `Tov`.
///
/// A `Fresh` or `Held` action with a finite value beyond the dead band
/// commands `close` or `open` per `close_when`. Everything else publishes no
/// payload, which the gripper task treats as "keep the last action": no
/// action yet, `Idle`, `Fault`, a non-finite value, or a value inside the
/// dead band. A faulted policy therefore never opens a closed gripper and
/// drops what it holds.
///
/// The task is stateless: the gripper task holds the last command, so a value
/// inside the dead band needs no memory here.
#[derive(Reflect)]
#[reflect(no_field_bounds, from_reflect = false)]
pub struct CuGripperActionTask<const ACT: usize> {
    params: CuGripperActionParams,
}

impl<const ACT: usize> CuGripperActionTask<ACT> {
    /// Creates the task from validated params; rejects `offset >= ACT`.
    pub fn from_params(params: CuGripperActionParams) -> CuResult<Self> {
        params.validate()?;
        if params.offset >= ACT {
            return Err(CuError::from(
                "repose-policy-rl: gripper 'offset' must be below the policy action width ACT",
            ));
        }
        Ok(Self { params })
    }

    /// Params this task runs with.
    pub fn params(&self) -> &CuGripperActionParams {
        &self.params
    }

    /// The command for one action value, or `None` inside the dead band.
    fn decide(&self, value: f64) -> Option<CuGripperAction> {
        let p = &self.params;
        let (high, low) = (p.threshold + p.hysteresis, p.threshold - p.hysteresis);
        let (above, below) = match p.close_when {
            CuGripperCloseWhen::Above => (CuGripperAction::Close, CuGripperAction::Open),
            CuGripperCloseWhen::Below => (CuGripperAction::Open, CuGripperAction::Close),
        };
        if value > high {
            Some(above)
        } else if value < low {
            Some(below)
        } else {
            None
        }
    }
}

impl<const ACT: usize> Freezable for CuGripperActionTask<ACT> {}

impl<const ACT: usize> CuTask for CuGripperActionTask<ACT> {
    type Resources<'r> = ();
    type Input<'m> = input_msg!(CuPolicyAction<ACT>);
    type Output<'m> = output_msg!(CuGripperCommandPayload);

    fn new(cfg: Option<&ComponentConfig>, _res: Self::Resources<'_>) -> CuResult<Self>
    where
        Self: Sized,
    {
        let params = cfg
            .map(ComponentConfig::deserialize_into::<CuGripperActionParams>)
            .transpose()?
            .unwrap_or_default();
        Self::from_params(params)
    }

    fn process<'i, 'o>(
        &mut self,
        _ctx: &CuContext,
        input: &Self::Input<'i>,
        output: &mut Self::Output<'o>,
    ) -> CuResult<()> {
        output.tov = input.tov;
        let value = input
            .payload()
            .filter(|a| matches!(a.status, CuPolicyStatus::Fresh | CuPolicyStatus::Held))
            .and_then(|a| a.values.get(self.params.offset).copied())
            .map(f64::from)
            .filter(|v| v.is_finite());
        match value.and_then(|v| self.decide(v)) {
            Some(action) => output.set_payload(CuGripperCommandPayload { action }),
            None => output.clear_payload(),
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run<const ACT: usize>(
        task: &mut CuGripperActionTask<ACT>,
        act: Option<CuPolicyAction<ACT>>,
    ) -> Option<CuGripperAction> {
        let (ctx, _clock) = CuContext::new_mock_clock();
        let mut input = CuMsg::new(act);
        input.tov = Tov::Time(CuTime::from_secs(3));
        let mut output = CuMsg::<CuGripperCommandPayload>::default();
        task.process(&ctx, &input, &mut output).unwrap();
        assert_eq!(output.tov, Tov::Time(CuTime::from_secs(3)));
        output.payload().map(|p| p.action)
    }

    fn act(value: f32, status: CuPolicyStatus) -> Option<CuPolicyAction<7>> {
        let mut values = [0.0; 7];
        values[6] = value;
        Some(CuPolicyAction { values, status })
    }

    #[test]
    fn openness_convention_with_a_dead_band() {
        let mut task = CuGripperActionTask::<7>::from_params(CuGripperActionParams {
            offset: 6,
            threshold: 0.5,
            close_when: CuGripperCloseWhen::Below,
            hysteresis: 0.1,
        })
        .unwrap();
        use CuGripperAction::{Close, Open};
        use CuPolicyStatus::{Fault, Fresh, Held, Idle};
        assert_eq!(run(&mut task, act(1.0, Fresh)), Some(Open));
        assert_eq!(run(&mut task, act(0.0, Held)), Some(Close));
        // Inside the dead band, or not a usable value: keep the last command.
        for (value, status) in [
            (0.55, Fresh),
            (0.45, Fresh),
            (1.0, Idle),
            (1.0, Fault),
            (f32::NAN, Fresh),
            (f32::INFINITY, Fresh),
        ] {
            assert_eq!(
                run(&mut task, act(value, status)),
                None,
                "{value} {status:?}"
            );
        }
        assert_eq!(run(&mut task, None), None);
    }

    #[test]
    fn closedness_convention() {
        let mut task =
            CuGripperActionTask::<1>::from_params(CuGripperActionParams::default()).unwrap();
        let one = |v: f32| {
            Some(CuPolicyAction {
                values: [v],
                status: CuPolicyStatus::Fresh,
            })
        };
        assert_eq!(run(&mut task, one(0.3)), Some(CuGripperAction::Close));
        assert_eq!(run(&mut task, one(-0.3)), Some(CuGripperAction::Open));
        assert_eq!(run(&mut task, one(0.0)), None);
    }

    #[test]
    fn invalid_params_are_rejected() {
        let base = CuGripperActionParams::default();
        for p in [
            CuGripperActionParams { offset: 1, ..base },
            CuGripperActionParams {
                threshold: f64::NAN,
                ..base
            },
            CuGripperActionParams {
                hysteresis: -0.1,
                ..base
            },
        ] {
            assert!(CuGripperActionTask::<1>::from_params(p).is_err(), "{p:?}");
        }
    }

    #[cfg(feature = "std")]
    #[test]
    fn params_read_from_ron_and_reject_typos() {
        let cfg: ComponentConfig = ron::from_str(
            r#"#![enable(unwrap_newtypes)] {"offset": 6, "threshold": 0.5, "close_when": "below", "hysteresis": 0.1}"#,
        )
        .unwrap();
        let task = CuGripperActionTask::<7>::new(Some(&cfg), ()).unwrap();
        assert_eq!(task.params().close_when, CuGripperCloseWhen::Below);
        let cfg: ComponentConfig =
            ron::from_str(r#"#![enable(unwrap_newtypes)] {"treshold": 0.5}"#).unwrap();
        assert!(CuGripperActionTask::<7>::new(Some(&cfg), ()).is_err());
    }
}
