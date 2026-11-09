# Benchmarks

Benchmarks run continuously on every pull request and every push to `master`
with [CodSpeed](https://app.codspeed.io/ReposeRobotics/repose) (Divan through
`codspeed-divan-compat`, CPU simulation instrument on `ubuntu-latest`).
Simulation numbers are stable across runners and are used for regression
detection; they are not wall-clock latencies. Live results, history, and
per-commit comparisons are on the
[CodSpeed dashboard](https://app.codspeed.io/ReposeRobotics/repose).

All scenarios use the 6-DOF igus ReBeL preset. The collision scene contains
11 obstacles; planners run one full anytime job around a sphere placed on the
TCP path of the straight joint line; trajectory parameterization uses a
5-waypoint path.

The table below is a local wall-clock snapshot (Divan median of 100 samples)
for orientation. Reproduce it with `just bench` and
`cargo bench -p repose-kinematics --bench ik`.

| Benchmark | Scenario | Median |
|---|---|---|
| `forward_kinematics` | FK, 6-DOF ReBeL | 312 ns |
| `geometric_jacobian` | Space Jacobian | 325 ns |
| `body_jacobian` | Body Jacobian | 375 ns |
| `differential_ik` | 6D twist → joint velocities | 682 ns |
| `inverse_kinematics_dls` | Seed → reachable pose (DLS) | 10.8 µs |
| `inverse_kinematics_pseudoinverse` | Seed → reachable pose (undamped) | 4.29 µs |
| `check_configuration_self` | Self-collision, one configuration | 427 ns |
| `check_configuration_scene` | Configuration vs 11-obstacle scene | 1.12 µs |
| `is_segment_collision_free_scene` | Swept segment vs scene | 51.6 µs |
| `plan` (RRT-Connect) | Full anytime planning job | 121 µs |
| `plan` (RRT*) | Full anytime planning job | 2.77 ms |
| `plan` (PRM) | Full anytime planning job | 37.3 ms |
| `parameterize` (S-curve) | 5-waypoint trajectory timing | 170 µs |
| `sample_quintic_1khz` | Quintic sampling at 1 kHz | 608 µs |

Snapshot machine: Apple M4 (arm64), rustc 1.99.0, measured 2026-11-04.
Treat these as indicative only; use the CodSpeed dashboard for
release-to-release comparisons.

IK is additionally fuzzed continuously with ClusterFuzzLite (OSS-Fuzz);
see the badges in the [README](../README.md).

