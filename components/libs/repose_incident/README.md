# repose-incident

Host-side black-box incident detector, timeline summarizer, and counterfactual
checkpoint scorecard engine for `copper-rs`.

## Overview

`repose-incident` turns any sequence of recorded robot cycles
(`CuCycleSample<N>`) from a `.copper` log reader, deterministic replay, or
ring buffer into:

- **Coalesced incident episodes (`CuIncidentRecord`)**: edge-detects collisions
  (`CuCollisionMonitor`), near-misses, policy faults, clamp bursts, controller
  faults, robot bridge faults, and single-cycle setpoint discontinuities,
  merging cascading triggers within a configurable cooldown into `[T - pre_window, T + post_window]`
  keyframe extraction windows.
- **Deterministic root-cause attribution (`CuIncidentBlame`)**: prioritizes
  `LearnedPolicy` > `EnvironmentChange` > `TrajectoryOrCollision` >
  `HardwareOrBridge` > `ControllerOrTracking`.
- **Run-wide summaries (`CuIncidentSummary`)**: aggregate safety metrics and
  one-call `to_json_pretty()` export (`incident_report.json`).
- **Counterfactual scorecards (`CuCheckpointDiff`)**: compares a candidate
  checkpoint against a baseline summary, flags regressions, and evaluates
  `passed_gate()`.
