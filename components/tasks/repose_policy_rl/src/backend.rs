//! Private ONNX backend: model loading and inference with `tract-onnx`.
//!
//! There is deliberately no public backend trait: one is added when a second
//! backend exists, not before. Loading runs once at startup and may allocate
//! freely; [`CuPolicyBackend::infer`] is the hot path and allocates only what
//! `tract` allocates per run (see `tests/alloc.rs`).

use cu29::prelude::*;
use std::io::Cursor;
use std::path::Path;
use std::sync::Arc;
use tract_onnx::prelude::*;
use tract_onnx::tract_hir::infer::Factoid;
use tract_onnx::tract_hir::internal::DimLike;

/// FNV-1a 64-bit hash of the model file bytes.
///
/// Stable across runs and platforms, so a hash pinned in RON
/// (`model_hash`) identifies the exact model file a graph was recorded with.
pub(crate) fn hash_model_bytes(bytes: &[u8]) -> u64 {
    const OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut hash = OFFSET_BASIS;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(PRIME);
    }
    hash
}

/// Wraps a `tract` failure (an `anyhow` error, not a `std::error::Error`) in
/// a [`CuError`] with the rendered cause attached.
fn backend_error(message: &'static str, cause: impl core::fmt::Display) -> CuError {
    CuError::from(message).add_cause(&cause.to_string())
}

/// An optimized, runnable stateless feed-forward model.
pub(crate) struct CuPolicyBackend {
    model: Arc<TypedRunnableModel>,
}

fn check_fact(fact: &TypedFact, expected_shape: &[usize], message: &'static str) -> CuResult<()> {
    if fact.datum_type != f32::datum_type() || fact.shape.as_concrete() != Some(expected_shape) {
        return Err(CuError::from(message));
    }
    Ok(())
}

impl CuPolicyBackend {
    /// Reads, hash-checks, and optimizes a single-step `[1, obs] -> [1, act]`
    /// `f32` model at `path`.
    pub(crate) fn load(
        path: &Path,
        obs: usize,
        act: usize,
        expected_hash: Option<u64>,
    ) -> CuResult<Self> {
        Self::load_with_output_shape(
            path,
            obs,
            &[1, act],
            expected_hash,
            "repose-policy-rl: model output must be f32 with shape [1, ACT]",
        )
    }

    /// Reads, hash-checks, and optimizes a chunked `[1, obs] -> [1, h, act]`
    /// `f32` model at `path`.
    pub(crate) fn load_chunk(
        path: &Path,
        obs: usize,
        h: usize,
        act: usize,
        expected_hash: Option<u64>,
    ) -> CuResult<Self> {
        Self::load_with_output_shape(
            path,
            obs,
            &[1, h, act],
            expected_hash,
            "repose-policy-rl: model output must be f32 with shape [1, H, ACT]",
        )
    }

    fn load_with_output_shape(
        path: &Path,
        obs: usize,
        expected_output: &[usize],
        expected_hash: Option<u64>,
        output_shape_error: &'static str,
    ) -> CuResult<Self> {
        let bytes = std::fs::read(path)
            .map_err(|e| CuError::new_with_cause("repose-policy-rl: cannot read model file", e))?;
        if let Some(expected) = expected_hash {
            let actual = hash_model_bytes(&bytes);
            if actual != expected {
                return Err(CuError::from(
                    "repose-policy-rl: model file hash does not match 'model_hash'",
                ));
            }
        }
        let mut reader = Cursor::new(bytes);
        let model = tract_onnx::onnx()
            .model_for_read(&mut reader)
            .map_err(|e| backend_error("repose-policy-rl: cannot parse ONNX model", e))?;
        let inputs = model
            .input_outlets()
            .map_err(|e| backend_error("repose-policy-rl: model inputs", e))?;
        if inputs.len() != 1 {
            return Err(CuError::from(
                "repose-policy-rl: model must have exactly one input",
            ));
        }
        let outputs = model
            .output_outlets()
            .map_err(|e| backend_error("repose-policy-rl: model outputs", e))?;
        if outputs.len() != 1 {
            return Err(CuError::from(
                "repose-policy-rl: model must have exactly one output",
            ));
        }
        // Pinning the input fact below overwrites the declared element type,
        // so a non-f32 model has to be rejected before that.
        // The same goes for the declared shape: a concrete rank or dimension
        // that differs from [1, OBS] is a different model, even when its
        // graph would run at the pinned width. Open (symbolic) dimensions are
        // accepted; pinning specializes them.
        let declared = model
            .input_fact(0)
            .map_err(|e| backend_error("repose-policy-rl: model input fact", e))?;
        let wrong_type = declared
            .datum_type
            .concretize()
            .is_some_and(|datum_type| datum_type != f32::datum_type());
        // Each dimension is checked on its own, so an unknown batch
        // dimension does not hide a wrong width.
        let wrong_rank = !declared.shape.is_open() && declared.shape.dims().count() != 2;
        let wrong_dim = declared.shape.dims().zip([1usize, obs]).any(|(dim, want)| {
            dim.concretize()
                .and_then(|dim| dim.to_usize().ok())
                .is_some_and(|have| have != want)
        });
        let wrong_shape = wrong_rank || wrong_dim;
        if wrong_type || wrong_shape {
            return Err(CuError::from(
                "repose-policy-rl: model input must be f32 with shape [1, OBS]",
            ));
        }
        let model = model
            .with_input_fact(0, InferenceFact::dt_shape(f32::datum_type(), [1usize, obs]))
            .map_err(|e| backend_error("repose-policy-rl: cannot pin model input shape", e))?
            .into_optimized()
            .map_err(|e| backend_error("repose-policy-rl: cannot optimize model", e))?;
        let input_fact = model
            .input_fact(0)
            .map_err(|e| backend_error("repose-policy-rl: model input fact", e))?;
        check_fact(
            input_fact,
            &[1, obs],
            "repose-policy-rl: model input must be f32 with shape [1, OBS]",
        )?;
        let output_outlets = model
            .output_outlets()
            .map_err(|e| backend_error("repose-policy-rl: model outputs", e))?;
        let output_fact = model
            .outlet_fact(output_outlets[0])
            .map_err(|e| backend_error("repose-policy-rl: model output fact", e))?;
        check_fact(output_fact, expected_output, output_shape_error)?;
        let model = model
            .into_runnable()
            .map_err(|e| backend_error("repose-policy-rl: cannot build runnable model", e))?;
        Ok(Self { model })
    }

    /// Runs a single-step model on `observation`, writing `[1, ACT]` into `out`.
    pub(crate) fn infer<const OBS: usize, const ACT: usize>(
        &self,
        observation: &[f32; OBS],
        out: &mut [f32; ACT],
    ) -> CuResult<()> {
        let input = Tensor::from_shape(&[1, OBS], observation.as_slice())
            .map_err(|e| backend_error("repose-policy-rl: cannot build input tensor", e))?;
        let outputs = self
            .model
            .run(tvec!(input.into()))
            .map_err(|e| backend_error("repose-policy-rl: inference failed", e))?;
        if outputs.len() != 1 {
            return Err(CuError::from(
                "repose-policy-rl: model must produce exactly one output",
            ));
        }
        let output = &outputs[0];
        if output.datum_type() != f32::datum_type() || output.shape() != [1, ACT] {
            return Err(CuError::from(
                "repose-policy-rl: model output shape or element type changed at run time",
            ));
        }
        let view = output
            .to_plain_array_view::<f32>()
            .map_err(|e| backend_error("repose-policy-rl: cannot read output tensor", e))?;
        for (dst, src) in out.iter_mut().zip(view.iter()) {
            *dst = *src;
        }
        Ok(())
    }

    /// Runs a chunked model on `observation`, writing `[1, H, ACT]` into `out`.
    pub(crate) fn infer_chunk<const OBS: usize, const H: usize, const ACT: usize>(
        &self,
        observation: &[f32; OBS],
        out: &mut [[f32; ACT]; H],
    ) -> CuResult<()> {
        let input = Tensor::from_shape(&[1, OBS], observation.as_slice())
            .map_err(|e| backend_error("repose-policy-rl: cannot build input tensor", e))?;
        let outputs = self
            .model
            .run(tvec!(input.into()))
            .map_err(|e| backend_error("repose-policy-rl: inference failed", e))?;
        if outputs.len() != 1 {
            return Err(CuError::from(
                "repose-policy-rl: model must produce exactly one output",
            ));
        }
        let output = &outputs[0];
        if output.datum_type() != f32::datum_type() || output.shape() != [1, H, ACT] {
            return Err(CuError::from(
                "repose-policy-rl: model output shape or element type changed at run time",
            ));
        }
        let view = output
            .to_plain_array_view::<f32>()
            .map_err(|e| backend_error("repose-policy-rl: cannot read output tensor", e))?;
        for (dst, src) in out.iter_mut().flatten().zip(view.iter()) {
            *dst = *src;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_is_stable_and_content_sensitive() {
        assert_eq!(hash_model_bytes(b"model"), hash_model_bytes(b"model"));
        assert_ne!(hash_model_bytes(b"model"), hash_model_bytes(b"modeM"));
        // FNV-1a of the empty input is the offset basis.
        assert_eq!(hash_model_bytes(b""), 0xcbf2_9ce4_8422_2325);
    }
}
