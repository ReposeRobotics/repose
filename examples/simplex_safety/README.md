# Simplex Safety Demo

End-to-end closed-loop demonstration of the Repose Physical AI Simplex Safety
Architecture on a 6-DOF igus ReBeL arm:

```text
plant/joint_state ─► scenario ─► planner (RRT*) ─► timing ─► follower ──┐
        │             ▲                                                 ├─► mux (CuAuditedSetpointMuxTask)
        ├─────────────┼─► policy (v1 rogue / v2 safe) ─► joint_action ──┘         │
        │             │                                                           ▼
        │             └─ scene_pub (CuPlanningScene w/ obstacle) ────────► shield (CuSafetyShieldTask)
        │                                                                         │
        └────────────────────────────◄── controller (CuControllerTask) ◄──────────┘
```

It runs two learned-policy checkpoints against the **exact same** obstacle,
workspace fence, braking lookahead, and RRT* planner, recording cycle-exact
`.copper` logs:

1. **Checkpoint `v1` (Rogue Policy)** — `logs/simplex_v1.copper`:
   - RRT* transits home -> pre-grasp; `CuAuditedSetpointMuxTask` bumplessly
     hands control to the policy.
   - The rogue policy drives toward the workspace obstacle sphere;
     `CuSafetyShieldTask` intercepts before physical contact and holds the arm
     safely (`136` shield-hold cycles) until the policy faults, after which
     RRT* plans a safe classical retract back to home.
2. **Checkpoint `v2` (Retrained Safe Policy)** — `logs/simplex_v2.copper`:
   - Identical transit and obstacle; the retrained policy steers smoothly
     around the obstacle with positive clearance throughout (`0` shield holds,
     `0` incidents) and retracts cleanly home.

## Run

From `examples/simplex_safety`:

```bash
# 1. Record both checkpoints (logs/simplex_v1.copper and logs/simplex_v2.copper)
cargo run --bin simplex_safety_demo

# 2. Offline black-box incident triage + counterfactual checkpoint gate
cargo run --bin simplex_safety_triage

# 3. Deterministic replay (verifies bit-identical mux, shield, and controller outputs)
cargo run --bin simplex_safety_resim -- --log-base logs/simplex_v1.copper
cargo run --bin simplex_safety_resim -- --log-base logs/simplex_v2.copper --replay-log-base logs/simplex_v2_resim.copper
```

`simplex_safety_triage` ingests both `.copper` logs via `CuCycleSample::with_*`
into `CuIncidentDetector<6>`, compares them with `CuCheckpointDiff::compare`,
and writes:

- `logs/incident_v1.json` & `logs/incident_v2.json` — windowed `CuIncidentSummary` manifests
- `logs/scorecard_diff.json` — counterfactual safety gate result (`passed_gate() == true`)
- `logs/telemetry_v1.json` & `logs/telemetry_v2.json` — per-cycle 3D FK link frames & live telemetry

## Render Side-by-Side 1080p Video

```bash
uv run --with matplotlib --with numpy --with pillow --with imageio-ffmpeg \
    python3 render_video.py
```

Writes `simplex_safety_demo.mp4` (`1920x1080`, 30 fps) showing `v1` vs. `v2`
side-by-side with live signed clearance curves, shield intervention vectors,
and the `CuCheckpointDiff` counterfactual gate scorecard.
