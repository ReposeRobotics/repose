# Safety requirements and traceability (stub)

Scope, assumptions, and the requirement → design → verification map for the
candidate safety-related slice (the v1 certification boundary). Companion to
`docs/regulatory-positioning.md` (§5, "Certification path", and roadmap item 5:
"Slice hardening"). This file is a **stub**: the requirements below are the
starting frame for the safety lifecycle, not a completed IEC 61508-3
work product. Refine with the assessor before treating any row as evidence.

> **Status:** draft, 2026-11-08. **Target:** IEC 61508 SIL 2 / ISO 13849-1 PL d.
> **Normative language:** SHALL (binding requirement), SHOULD (strong
> recommendation), MAY (permitted option).

## 1. Scope and boundary

In scope — the `repose_controller` slice and the payload contracts it shares
with a robot bridge:

| # | Component | Role |
|---|---|---|
| C1 | Watchdog | Faults and holds on stale or invalid measured joint state |
| C2 | Command rate limiting | Bounds commanded velocity and acceleration |
| C3 | Joint-limit enforcement | Rejects commands outside joint limits |
| C4 | Collision-triggered abort | Stops motion on collision detection (planned) |
| C5 | Payload contracts | The one command type and status a bridge carries |

Out of scope — Copper runtime (`cu29`), planners, trajectory timing, robot
bridges, Studio. Copper stays outside the certification boundary (see
`regulatory-positioning.md`, §5 "Boundary").

## 2. Requirements

| ID | Requirement | Maps to | Verified by |
|---|---|---|---|
| S-1 | On stale or invalid measured joint state, the slice SHALL enter Fault and hold the last safe command. | C1 (`repose_controller`: stale-input fault path) | Unit tests, stale/stale-signature cases; fault-injection fuzzing later |
| S-2 | Commanded velocity and acceleration SHALL be bounded to configured limits. | C2 (rate limiting in `repose_controller`) | Unit tests on limit clamping; property test: output within limits for any input |
| S-3 | Commands outside joint limits SHALL be rejected before reaching the bridge. | C3 (joint-limit enforcement) | Unit tests; property test: no out-of-limit command emitted |
| S-4 | On collision detection during execution, the slice SHALL trigger abort. | C4 (planned) | Unit tests once implemented; HIL smoke test later |
| S-5 | The slice's command and status payloads SHALL be the only interface a bridge depends on for safety-relevant state. | C5 (`CuJointCommandPayload`, `CuRobotStatus`) | API review; doc test asserting the contract surface |
| S-6 | The slice SHALL build with no operating system: `core` and `alloc` only, no heap allocation on the real-time path. | All of C1–C5 | `just bare-metal` (CI, `aarch64-unknown-none` and `thumbv7em-none-eabihf`); Ferrocene CI lane when the CriticalUp token is available |
| S-7 | Behaviour SHALL be deterministic and replayable: fixed task graph, seeded randomness, no input-dependent scheduling. | All of C1–C5 | `just determinism` (CI, Linux and macOS) |

## 3. Traceability

| Requirement | Design unit | Verification artifact |
|---|---|---|
| S-1 | `repose_controller` stale/invalid-state fault path | Unit tests, `#[test]` coverage in `repose_controller` |
| S-2 | Rate limiter in `repose_controller` | Unit tests; property tests |
| S-3 | Joint-limit check in `repose_controller` | Unit tests; property tests |
| S-4 | Collision-triggered abort (planned) | To be specified with C4 |
| S-5 | Payload contracts in `repose_kinematics` / `repose_controller` | API review; doc tests |
| S-6 | Crate feature config, `no_std` gating | `just bare-metal`; `ferrocene` CI job |
| S-7 | Copper task graph, seeded planners | `just determinism` |

Every requirement maps to at least one design unit and one verification
artifact; no requirement is verified by review alone. The reverse direction
(every design unit traces to a requirement) is the part to complete with the
assessor before the gap assessment.

## 4. Open items

- S-4 has no design yet; write the requirement together with the abort
  implementation, not before it.
- Fuzzing and coverage targets on the slice (`llvm-cov`, fuzz harnesses) are
  roadmap item 5's tooling, not yet wired.
- The safety manual (`docs/safety-manual.md`, planned) states the assumptions
  of use this stub only gestures at.
