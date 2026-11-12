//! Wire payloads of the policy task: the observation exactly as it was fed
//! to the model, and the raw model output before any scaling or limiting.
//!
//! Both vectors are dimensionless `f32`: they are model-space numbers, not
//! physical quantities. `Tov` stays on the `CuMsg` envelope and is propagated
//! from the joint state input by [`crate::CuPolicyTask`].

use bincode::Decode;
use bincode::Encode;
use cu29::prelude::*;
use serde::Deserialize;
use serde::Serialize;

/// Lifecycle of one [`CuPolicyAction`].
#[derive(
    Default, Debug, Clone, Copy, PartialEq, Eq, Encode, Decode, Serialize, Deserialize, Reflect,
)]
#[serde(rename_all = "snake_case")]
pub enum CuPolicyStatus {
    /// No usable joint state or goal yet; `values` are zero.
    #[default]
    Idle,
    /// Inferred this cycle.
    Fresh,
    /// Decimation: repeats the last inferred action.
    Held,
    /// Inference failed or produced a non-finite value; `values` are zero.
    Fault,
}

/// Observation vector exactly as it was fed to the model (dimensionless).
#[derive(Debug, Clone, PartialEq, Encode, Decode, Serialize, Deserialize, Reflect)]
#[reflect(no_field_bounds, from_reflect = false)]
pub struct CuPolicyObservation<const OBS: usize> {
    #[serde(with = "repose_kinematics::serde_arrays::f32_array")]
    pub values: [f32; OBS],
}

impl<const OBS: usize> Default for CuPolicyObservation<OBS> {
    fn default() -> Self {
        Self { values: [0.0; OBS] }
    }
}

/// Raw model output, before any scaling or limiting (dimensionless).
#[derive(Debug, Clone, PartialEq, Encode, Decode, Serialize, Deserialize, Reflect)]
#[reflect(no_field_bounds, from_reflect = false)]
pub struct CuPolicyAction<const ACT: usize> {
    #[serde(with = "repose_kinematics::serde_arrays::f32_array")]
    pub values: [f32; ACT],
    pub status: CuPolicyStatus,
}

impl<const ACT: usize> Default for CuPolicyAction<ACT> {
    fn default() -> Self {
        Self {
            values: [0.0; ACT],
            status: CuPolicyStatus::Idle,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wide_payloads_round_trip_serde_and_bincode() {
        // OBS/ACT above 32: the fixed-array serde helper must carry them.
        let observation = CuPolicyObservation::<48> {
            values: core::array::from_fn(|i| i as f32 * 0.5),
        };
        let text = ron::to_string(&observation).unwrap();
        let back: CuPolicyObservation<48> = ron::from_str(&text).unwrap();
        assert_eq!(back, observation);

        let action = CuPolicyAction::<40> {
            values: core::array::from_fn(|i| -(i as f32)),
            status: CuPolicyStatus::Held,
        };
        let mut bytes = [0u8; 512];
        let len =
            bincode::encode_into_slice(&action, &mut bytes, bincode::config::standard()).unwrap();
        let (decoded, used): (CuPolicyAction<40>, usize) =
            bincode::decode_from_slice(&bytes[..len], bincode::config::standard()).unwrap();
        assert_eq!(used, len);
        assert_eq!(decoded, action);
    }

    #[test]
    fn defaults_are_zero_and_idle() {
        assert_eq!(CuPolicyObservation::<4>::default().values, [0.0; 4]);
        let action = CuPolicyAction::<4>::default();
        assert_eq!(action.values, [0.0; 4]);
        assert_eq!(action.status, CuPolicyStatus::Idle);
    }
}
