//! Counterfactual checkpoint comparison scorecard for Flightdeck and `ci.getrepose.dev`.

use crate::detector::CuIncidentSummary;
use cu29::prelude::*;
use cu29::units::si::f64::Length;
use cu29::units::si::length::meter;
use serde::Deserialize;
use serde::Serialize;

const DEFAULT_CLEARANCE_THRESHOLD_M: f64 = 0.02;

/// Counterfactual comparison between a baseline run and a candidate checkpoint
/// replayed over the same scenario or incident window.
///
/// # Example
/// ```rust
/// use repose_incident::{CuCheckpointDiff, CuIncidentSummary};
///
/// let baseline = CuIncidentSummary {
///     model_hash: Some(0x1111),
///     collision_episodes: 1,
///     min_clearance_m: Some(-0.012),
///     ..CuIncidentSummary::default()
/// };
/// let candidate = CuIncidentSummary {
///     model_hash: Some(0x2222),
///     collision_episodes: 0,
///     min_clearance_m: Some(0.034),
///     ..CuIncidentSummary::default()
/// };
///
/// let diff = CuCheckpointDiff::compare(&baseline, &candidate);
/// assert_eq!(diff.collision_delta, -1);
/// assert!(!diff.regression);
/// assert!(diff.passed_gate());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Reflect)]
pub struct CuCheckpointDiff {
    /// Model hash of the baseline run.
    pub baseline_model_hash: Option<u64>,
    /// Model hash of the candidate checkpoint.
    pub candidate_model_hash: Option<u64>,
    /// Change in collision episodes (`candidate - baseline`; negative is better).
    pub collision_delta: i64,
    /// Change in near-miss episodes (`candidate - baseline`; negative is better).
    pub near_miss_delta: i64,
    /// Change in policy fault count (`candidate - baseline`; negative is better).
    pub policy_fault_delta: i64,
    /// Change in controller fault count (`candidate - baseline`; negative is better).
    pub controller_fault_delta: i64,
    /// Change in clamped cycles (`candidate - baseline`; negative is better).
    pub clamped_cycles_delta: i64,
    /// Change in minimum clearance in meters (`candidate - baseline`; positive is better).
    pub min_clearance_delta_m: Option<f64>,
    /// `true` when the candidate introduced more collisions, more policy or
    /// controller faults, or worse clearance below the safety threshold.
    pub regression: bool,
    /// Candidate summary collision episodes (used by [`Self::passed_gate`]).
    pub candidate_collision_episodes: usize,
    /// Candidate summary policy faults (used by [`Self::passed_gate`]).
    pub candidate_policy_faults: usize,
    /// Candidate summary controller faults (used by [`Self::passed_gate`]).
    pub candidate_controller_faults: usize,
}

impl CuCheckpointDiff {
    /// Compares `candidate` against `baseline` using the default `0.02` m
    /// clearance threshold.
    pub fn compare(baseline: &CuIncidentSummary, candidate: &CuIncidentSummary) -> Self {
        Self::compare_with_threshold_m(baseline, candidate, DEFAULT_CLEARANCE_THRESHOLD_M)
    }

    /// Compares `candidate` against `baseline` using a custom minimum clearance
    /// threshold (must be finite and `>= 0`).
    pub fn compare_with_clearance_threshold(
        baseline: &CuIncidentSummary,
        candidate: &CuIncidentSummary,
        clearance_threshold: Length,
    ) -> CuResult<Self> {
        let threshold_m = clearance_threshold.get::<meter>();
        if !threshold_m.is_finite() || threshold_m < 0.0 {
            return Err(CuError::from(
                "repose-incident: 'clearance_threshold' must be finite and >= 0",
            ));
        }
        Ok(Self::compare_with_threshold_m(
            baseline,
            candidate,
            threshold_m,
        ))
    }

    fn compare_with_threshold_m(
        baseline: &CuIncidentSummary,
        candidate: &CuIncidentSummary,
        threshold_m: f64,
    ) -> Self {
        let collision_delta =
            delta_usize(baseline.collision_episodes, candidate.collision_episodes);
        let near_miss_delta =
            delta_usize(baseline.near_miss_episodes, candidate.near_miss_episodes);
        let policy_fault_delta = delta_usize(baseline.policy_faults, candidate.policy_faults);
        let controller_fault_delta =
            delta_usize(baseline.controller_faults, candidate.controller_faults);
        let clamped_cycles_delta = delta_u64(baseline.clamped_cycles, candidate.clamped_cycles);

        let min_clearance_delta_m = match (candidate.min_clearance_m, baseline.min_clearance_m) {
            (Some(c), Some(b)) if c.is_finite() && b.is_finite() => Some(c - b),
            _ => None,
        };

        let clearance_regressed = candidate
            .min_clearance_m
            .filter(|c| c.is_finite())
            .is_some_and(|c| {
                c < threshold_m
                    && baseline
                        .min_clearance_m
                        .filter(|b| b.is_finite())
                        .is_none_or(|b| c < b)
            });

        let regression = collision_delta > 0
            || policy_fault_delta > 0
            || controller_fault_delta > 0
            || clearance_regressed;

        Self {
            baseline_model_hash: baseline.model_hash,
            candidate_model_hash: candidate.model_hash,
            collision_delta,
            near_miss_delta,
            policy_fault_delta,
            controller_fault_delta,
            clamped_cycles_delta,
            min_clearance_delta_m,
            regression,
            candidate_collision_episodes: candidate.collision_episodes,
            candidate_policy_faults: candidate.policy_faults,
            candidate_controller_faults: candidate.controller_faults,
        }
    }

    /// Returns `true` when there is no regression and the candidate run has
    /// zero collisions, zero policy faults, and zero controller faults.
    pub const fn passed_gate(&self) -> bool {
        !self.regression
            && self.candidate_collision_episodes == 0
            && self.candidate_policy_faults == 0
            && self.candidate_controller_faults == 0
    }

    /// Serializes the scorecard diff as pretty-printed JSON.
    pub fn to_json_pretty(&self) -> CuResult<String> {
        serde_json::to_string_pretty(self).map_err(|e| {
            CuError::new_with_cause("repose-incident: failed to serialize checkpoint diff", e)
        })
    }
}

fn delta_usize(baseline: usize, candidate: usize) -> i64 {
    let c = i64::try_from(candidate).unwrap_or(i64::MAX);
    let b = i64::try_from(baseline).unwrap_or(i64::MAX);
    c.saturating_sub(b)
}

fn delta_u64(baseline: u64, candidate: u64) -> i64 {
    let c = i64::try_from(candidate).unwrap_or(i64::MAX);
    let b = i64::try_from(baseline).unwrap_or(i64::MAX);
    c.saturating_sub(b)
}
