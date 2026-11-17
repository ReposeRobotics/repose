//! Wire payloads of the policy tasks: the observation (the feature vector an
//! observation task feeds the policy, and the model input the policy
//! publishes each cycle), the raw single-step model output, and the raw
//! `H`-step action chunk before temporal ensembling or scaling.
//!
//! All vectors are dimensionless `f32`: they are model-space numbers, not
//! physical quantities. `Tov` stays on the `CuMsg` envelope and is propagated
//! from input to output by the observation, policy, and chunk tasks.

use bincode::Decode;
use bincode::Encode;
use cu29::prelude::*;
use serde::Deserialize;
use serde::Serialize;

/// Serde `with` module for `[[f32; ACT]; H]` arrays of any `H` and `ACT`.
pub(crate) mod chunk_array {
    use serde::Deserialize;
    use serde::Deserializer;
    use serde::Serialize;
    use serde::Serializer;
    use serde::de::Error as DeError;
    use serde::de::SeqAccess;
    use serde::de::Visitor;
    use serde::ser::SerializeTuple;

    struct RowRef<'a, const ACT: usize>(&'a [f32; ACT]);

    impl<const ACT: usize> Serialize for RowRef<'_, ACT> {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            repose_kinematics::serde_arrays::f32_array::serialize(self.0, serializer)
        }
    }

    struct Row<const ACT: usize>([f32; ACT]);

    impl<'de, const ACT: usize> Deserialize<'de> for Row<ACT> {
        fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            repose_kinematics::serde_arrays::f32_array::deserialize(deserializer).map(Self)
        }
    }

    pub fn serialize<S, const H: usize, const ACT: usize>(
        values: &[[f32; ACT]; H],
        serializer: S,
    ) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut seq = serializer.serialize_tuple(H)?;
        for row in values {
            seq.serialize_element(&RowRef(row))?;
        }
        seq.end()
    }

    pub fn deserialize<'de, D, const H: usize, const ACT: usize>(
        deserializer: D,
    ) -> Result<[[f32; ACT]; H], D::Error>
    where
        D: Deserializer<'de>,
    {
        struct ChunkVisitor<const M: usize, const W: usize>;

        impl<'de, const M: usize, const W: usize> Visitor<'de> for ChunkVisitor<M, W> {
            type Value = [[f32; W]; M];

            fn expecting(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                write!(formatter, "an array of {M} action steps of width {W}")
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let mut out = [[0.0f32; W]; M];
                let mut i = 0;
                while i < M {
                    out[i] = seq
                        .next_element::<Row<W>>()?
                        .ok_or_else(|| DeError::invalid_length(i, &self))?
                        .0;
                    i += 1;
                }
                if seq.next_element::<serde::de::IgnoredAny>()?.is_some() {
                    return Err(DeError::invalid_length(M + 1, &self));
                }
                Ok(out)
            }
        }

        deserializer.deserialize_tuple(H, ChunkVisitor::<H, ACT>)
    }
}

/// Lifecycle of one [`CuPolicyAction`] or [`CuPolicyActionChunk`].
#[derive(
    Default, Debug, Clone, Copy, PartialEq, Eq, Encode, Decode, Serialize, Deserialize, Reflect,
)]
#[serde(rename_all = "snake_case")]
pub enum CuPolicyStatus {
    /// No input observation (for example no usable joint state or goal
    /// upstream), or an exhausted chunk horizon; `values` are zero.
    #[default]
    Idle,
    /// Newly inferred or stepped this cycle.
    Fresh,
    /// Decimation: repeats the last inferred action (or signals no new chunk
    /// between chunk-policy queries while upstream remains healthy).
    Held,
    /// Non-finite model input, failed inference, or a non-finite model
    /// output; `values` are zero.
    Fault,
}

/// Observation vector (dimensionless). As the input of
/// [`crate::CuPolicyTask`] or [`crate::CuChunkPolicyTask`] it is the flat
/// feature vector an observation task built; a non-finite value there signals
/// a fault. As the output of [`crate::CuPolicyTask`] it is the model input
/// built this cycle: what the model was fed when the action of the same cycle
/// is [`CuPolicyStatus::Fresh`]; `Held` and `Idle` cycles run no inference on
/// it, and it is zero on `Idle`.
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

/// Raw single-step model output, before any scaling or limiting (dimensionless).
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

/// Raw `H`-step action chunk, each step of width `ACT`, before temporal
/// ensembling, scaling, or limiting (dimensionless).
///
/// `values[0]` is predicted for the cycle the chunk arrives on, `values[1]` is
/// one chunk step ahead, and `values[H - 1]` is `H - 1` chunk steps ahead.
///
/// Upstream contract for [`crate::CuActionChunkTask`]:
///
/// - **No payload** on the message, or [`CuPolicyStatus::Held`]: no new chunk
///   arrived this cycle; keep stepping the buffered horizon.
/// - [`CuPolicyStatus::Fresh`]: ingest `values` into the horizon and emit the
///   current step. Any non-finite element faults and clears the buffer.
/// - [`CuPolicyStatus::Idle`]: clear the buffered horizon immediately and
///   report `Idle`.
/// - [`CuPolicyStatus::Fault`]: clear the buffered horizon immediately and
///   latch `Fault` until the next `Idle` or valid `Fresh` chunk.
#[derive(Debug, Clone, PartialEq, Encode, Decode, Serialize, Deserialize, Reflect)]
#[reflect(no_field_bounds, from_reflect = false)]
pub struct CuPolicyActionChunk<const H: usize, const ACT: usize> {
    #[serde(with = "chunk_array")]
    pub values: [[f32; ACT]; H],
    pub status: CuPolicyStatus,
}

impl<const H: usize, const ACT: usize> Default for CuPolicyActionChunk<H, ACT> {
    fn default() -> Self {
        Self {
            values: [[0.0; ACT]; H],
            status: CuPolicyStatus::Idle,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wide_payloads_round_trip_serde_and_bincode() {
        // OBS/ACT/H above 32: the fixed-array serde helpers must carry them.
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

        let chunk = CuPolicyActionChunk::<50, 40> {
            values: core::array::from_fn(|s| core::array::from_fn(|j| (s * 40 + j) as f32 * 0.25)),
            status: CuPolicyStatus::Fresh,
        };
        let chunk_text = ron::to_string(&chunk).unwrap();
        let chunk_back: CuPolicyActionChunk<50, 40> = ron::from_str(&chunk_text).unwrap();
        assert_eq!(chunk_back, chunk);

        let mut chunk_bytes = vec![0u8; 16_384];
        let chunk_len =
            bincode::encode_into_slice(&chunk, &mut chunk_bytes, bincode::config::standard())
                .unwrap();
        let (chunk_decoded, chunk_used): (CuPolicyActionChunk<50, 40>, usize) =
            bincode::decode_from_slice(&chunk_bytes[..chunk_len], bincode::config::standard())
                .unwrap();
        assert_eq!(chunk_used, chunk_len);
        assert_eq!(chunk_decoded, chunk);
    }

    #[test]
    fn chunk_serde_rejects_mismatched_horizon_or_width() {
        let two_by_three = CuPolicyActionChunk::<2, 3> {
            values: [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            status: CuPolicyStatus::Fresh,
        };
        let text = ron::to_string(&two_by_three).unwrap();
        assert!(ron::from_str::<CuPolicyActionChunk<3, 3>>(&text).is_err());
        assert!(ron::from_str::<CuPolicyActionChunk<2, 2>>(&text).is_err());
        assert!(ron::from_str::<CuPolicyActionChunk<1, 3>>(&text).is_err());
    }

    #[test]
    fn defaults_are_zero_and_idle() {
        assert_eq!(CuPolicyObservation::<4>::default().values, [0.0; 4]);
        let action = CuPolicyAction::<4>::default();
        assert_eq!(action.values, [0.0; 4]);
        assert_eq!(action.status, CuPolicyStatus::Idle);
        let chunk = CuPolicyActionChunk::<3, 4>::default();
        assert_eq!(chunk.values, [[0.0; 4]; 3]);
        assert_eq!(chunk.status, CuPolicyStatus::Idle);
    }
}
