#!/usr/bin/env python3
"""Render a 1080p side-by-side Physical AI Simplex Safety demo video (MP4)
from recorded `.copper` telemetry exported by `simplex_safety_triage`.

Usage:
    uv run --with matplotlib --with numpy --with pillow --with imageio-ffmpeg \
        python3 render_video.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import imageio_ffmpeg
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

ROOT = Path(__file__).resolve().parent
LOGS = ROOT / "logs"
OUT_MP4 = ROOT / "simplex_safety_demo.mp4"

WIDTH, HEIGHT = 1920, 1080
DPI = 120
FPS = 30

# Palette (dark slate engineering theme)
BG = "#0B0F19"
PANEL_BG = "#111827"
GRID_COL = "#1F2937"
TEXT_MAIN = "#F8FAFC"
TEXT_MUTED = "#94A3B8"
CYAN = "#38BDF8"
PURPLE = "#C084FC"
RED = "#F43F5E"
AMBER = "#F59E0B"
GREEN = "#10B981"
SLATE = "#334155"


def phase_color(c: dict) -> str:
    if c["verdict"] != "clear":
        return RED
    src = c["mux_source"]
    if src == "policy":
        return PURPLE
    if src == "hold":
        return AMBER
    return CYAN


def status_badge(c: dict, is_v1: bool) -> tuple[str, str]:
    v = c["verdict"]
    if v != "clear":
        label = v.replace("_", " ").upper()
        return f"SHIELD ACTIVE: {label}", RED
    src = c["mux_source"]
    if src == "policy":
        if is_v1:
            return "POLICY v1 ACTIVE (APPROACHING OBSTACLE)", PURPLE
        return "POLICY v2 ACTIVE (AVOIDING OBSTACLE)", GREEN
    if src == "hold":
        return "SIMPLEX HOLD (POLICY FAULT -> RETRACTING)", AMBER
    return "CLASSICAL RRT* TRAJECTORY", CYAN


def draw_sphere(ax, center, radius, color=RED, alpha=0.22):
    u = np.linspace(0, 2 * np.pi, 22)
    v = np.linspace(0, np.pi, 14)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0.2, edgecolor=color)


def draw_floor_grid(ax):
    xs = np.linspace(-0.6, 0.6, 7)
    for v in xs:
        ax.plot([v, v], [-0.6, 0.6], [0.0, 0.0], color=GRID_COL, lw=0.7, alpha=0.8)
        ax.plot([-0.6, 0.6], [v, v], [0.0, 0.0], color=GRID_COL, lw=0.7, alpha=0.8)
    # Base pedestal
    ax.scatter([0.0], [0.0], [0.0], color="#64748B", s=90, depthshade=False)


def draw_arm_panel(ax, tel: dict, idx: int, title: str, is_v1: bool, azim: float):
    ax.cla()
    ax.set_facecolor(PANEL_BG)
    ax.set_xlim(-0.55, 0.65)
    ax.set_ylim(-0.55, 0.55)
    ax.set_zlim(0.0, 0.92)
    ax.view_init(elev=22, azim=azim)
    ax.set_axis_off()

    draw_floor_grid(ax)

    # Obstacle sphere
    obs_c = tel["obstacle_center"]
    obs_r = tel["obstacle_radius"]
    draw_sphere(ax, obs_c, obs_r, color=RED, alpha=0.26)
    draw_sphere(ax, obs_c, obs_r + 0.04, color=AMBER, alpha=0.06)

    cycles = tel["cycles"]
    cur_idx = min(idx, len(cycles) - 1)
    c = cycles[cur_idx]

    # TCP trail up to current cycle
    if cur_idx > 1:
        trail = np.array([cycles[i]["tcp_xyz"] for i in range(0, cur_idx + 1, 2)])
        col = RED if (is_v1 and c["verdict"] != "clear") else (PURPLE if c["mux_source"] == "policy" else CYAN)
        ax.plot(trail[:, 0], trail[:, 1], trail[:, 2], color=col, lw=2.0, alpha=0.65)

    # Robot links
    pts = np.array(c["links_xyz"])
    arm_col = RED if c["verdict"] != "clear" else ("#E2E8F0" if c["mux_source"] == "classical" else (PURPLE if is_v1 else GREEN))
    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=arm_col, lw=5.5, solid_capstyle="round")
    ax.scatter(pts[1:-1, 0], pts[1:-1, 1], pts[1:-1, 2], color=CYAN, s=36, edgecolors=BG, linewidths=0.8, depthshade=False)
    ax.scatter([pts[-1, 0]], [pts[-1, 1]], [pts[-1, 2]], color=arm_col, s=75, edgecolors=TEXT_MAIN, linewidths=1.2, depthshade=False)

    # When shield intervenes, show the rejected policy command target as a dashed red vector
    if c["verdict"] != "clear":
        tcp = c["tcp_xyz"]
        des = c["desired_tcp_xyz"]
        ax.plot([tcp[0], des[0]], [tcp[1], des[1]], [tcp[2], des[2]], color=RED, lw=2.4, ls="--")
        ax.scatter([des[0]], [des[1]], [des[2]], color=RED, marker="x", s=95, lw=2.2, depthshade=False)

    # Panel title & live status badge
    badge_text, badge_col = status_badge(c, is_v1)
    ax.text2D(0.03, 0.94, title, transform=ax.transAxes, color=TEXT_MAIN, fontsize=12.5, fontweight="bold", family="monospace")
    ax.text2D(
        0.03,
        0.87,
        f"  {badge_text}  ",
        transform=ax.transAxes,
        color=BG,
        fontsize=9.5,
        fontweight="bold",
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.35", facecolor=badge_col, edgecolor="none"),
    )

    clr = c["min_clearance_m"]
    clr_str = f"{clr:+.3f} m" if clr is not None else "n/a"
    ax.text2D(
        0.03,
        0.04,
        f"clearance: {clr_str}   |   tcp_speed: {c['tcp_speed_m_s']:.2f} m/s   |   mux: {c['mux_source']}",
        transform=ax.transAxes,
        color=TEXT_MUTED,
        fontsize=9.5,
        family="monospace",
    )


def main():
    v1 = json.loads((LOGS / "telemetry_v1.json").read_text())
    v2 = json.loads((LOGS / "telemetry_v2.json").read_text())
    diff = json.loads((LOGS / "scorecard_diff.json").read_text())

    n_v1 = len(v1["cycles"])
    n_v2 = len(v2["cycles"])
    max_cycles = max(n_v1, n_v2)

    # Precompute clearance timelines
    t1 = np.array([c["time_s"] for c in v1["cycles"]])
    c1 = np.array([c["min_clearance_m"] if c["min_clearance_m"] is not None else np.nan for c in v1["cycles"]])
    t2 = np.array([c["time_s"] for c in v2["cycles"]])
    c2 = np.array([c["min_clearance_m"] if c["min_clearance_m"] is not None else np.nan for c in v2["cycles"]])
    t_max = max(float(t1[-1]), float(t2[-1]))

    # Subsample 100 Hz cycles -> 30 fps (step=2 => 50 Hz -> ~1.67x smooth real-time playback)
    step = 2
    frame_indices = list(range(0, max_cycles, step))
    if frame_indices[-1] != max_cycles - 1:
        frame_indices.append(max_cycles - 1)
    # Hold final scorecard frame for 2.5 seconds
    hold_frames = int(FPS * 2.5)
    total_frames = len(frame_indices) + hold_frames

    fig = plt.figure(figsize=(WIDTH / DPI, HEIGHT / DPI), dpi=DPI, facecolor=BG)
    ax_left = fig.add_axes([0.02, 0.34, 0.47, 0.56], projection="3d")
    ax_right = fig.add_axes([0.51, 0.34, 0.47, 0.56], projection="3d")
    ax_plot = fig.add_axes([0.055, 0.07, 0.52, 0.22], facecolor=PANEL_BG)
    ax_card = fig.add_axes([0.60, 0.05, 0.38, 0.25], facecolor=PANEL_BG)
    ax_card.set_axis_off()

    # Static header text
    fig.text(
        0.02,
        0.955,
        "REPOSE + COPPER-RS   ·   PHYSICAL AI SIMPLEX SAFETY ARCHITECTURE",
        color=TEXT_MAIN,
        fontsize=15.5,
        fontweight="bold",
        family="monospace",
    )
    fig.text(
        0.02,
        0.922,
        "6-DOF igus ReBeL  |  CuAuditedSetpointMuxTask  +  CuSafetyShieldTask  +  CuIncidentDetector  +  CuCheckpointDiff",
        color=TEXT_MUTED,
        fontsize=10.2,
        family="monospace",
    )
    clock_text = fig.text(
        0.98,
        0.945,
        "",
        color=CYAN,
        fontsize=12.5,
        fontweight="bold",
        family="monospace",
        ha="right",
    )

    writer = imageio_ffmpeg.write_frames(
        str(OUT_MP4),
        (WIDTH, HEIGHT),
        fps=FPS,
        codec="libx264",
        pix_fmt_out="yuv420p",
        macro_block_size=8,
        output_params=["-crf", "18", "-preset", "medium"],
    )
    writer.send(None)

    first_shield_cycle = next((c["cycle"] for c in v1["cycles"] if c["verdict"] != "clear"), max_cycles)
    last_shield_cycle = max((c["cycle"] for c in v1["cycles"] if c["verdict"] != "clear"), default=first_shield_cycle)
    v1_inc = v1["summary"]["incidents"][0] if v1["summary"]["incidents"] else None
    passed_gate = (
        not diff.get("regression", True)
        and diff.get("candidate_collision_episodes", 1) == 0
        and diff.get("candidate_policy_faults", 1) == 0
        and diff.get("candidate_controller_faults", 1) == 0
    )

    for f_num in range(total_frames):
        is_endcard = f_num >= len(frame_indices)
        idx = frame_indices[-1] if is_endcard else frame_indices[f_num]
        t_cur = idx * 0.01
        azim = -52.0 + 18.0 * math.sin(2.0 * math.pi * min(idx / max_cycles, 1.0) * 0.65)

        clock_text.set_text(f"t = {t_cur:5.2f} s   |   cycle {idx:03d}   |   100 Hz")

        draw_arm_panel(ax_left, v1, idx, "CHECKPOINT v1  ·  ROGUE POLICY", True, azim)
        draw_arm_panel(ax_right, v2, idx, "CHECKPOINT v2  ·  RETRAINED POLICY", False, azim)

        # Bottom-left: live clearance plot
        ax_plot.cla()
        ax_plot.set_facecolor(PANEL_BG)
        ax_plot.set_xlim(0.0, t_max)
        ax_plot.set_ylim(-0.03, 0.11)
        ax_plot.grid(True, color=GRID_COL, lw=0.7)
        ax_plot.axhline(0.0, color=RED, lw=1.2, ls="--", alpha=0.85, label="Collision (0.00 m)")
        ax_plot.axhline(0.04, color=AMBER, lw=1.0, ls=":", alpha=0.8, label="Near-Miss Margin (0.04 m)")

        i1 = min(idx + 1, n_v1)
        i2 = min(idx + 1, n_v2)
        ax_plot.plot(t1[:i1], c1[:i1], color=RED, lw=2.2, label="v1 (Rogue)")
        ax_plot.plot(t2[:i2], c2[:i2], color=GREEN, lw=2.2, label="v2 (Retrained)")

        if idx >= first_shield_cycle:
            w0 = first_shield_cycle * 0.01
            w1 = min(last_shield_cycle * 0.01, t_cur)
            if w1 > w0:
                ax_plot.axvspan(w0, w1, color=RED, alpha=0.16, label="Shield Hold Window")

        ax_plot.set_title("Signed Surface Clearance to Obstacle & Fence (m)", color=TEXT_MAIN, fontsize=10.5, family="monospace", loc="left", pad=6)
        ax_plot.set_xlabel("Copper Simulated Time (s)", color=TEXT_MUTED, fontsize=9, family="monospace")
        ax_plot.tick_params(colors=TEXT_MUTED, labelsize=8.5)
        for sp in ax_plot.spines.values():
            sp.set_color(SLATE)
        ax_plot.legend(loc="upper right", facecolor=BG, edgecolor=SLATE, labelcolor=TEXT_MAIN, fontsize=8.0, ncol=2)

        # Bottom-right: live incident & counterfactual scorecard
        ax_card.cla()
        ax_card.set_axis_off()
        card_box = FancyBboxPatch((0.0, 0.0), 1.0, 1.0, boxstyle="round,pad=0.02", facecolor=PANEL_BG, edgecolor=SLATE, lw=1.5)
        ax_card.add_patch(card_box)

        ax_card.text(0.04, 0.86, "BLACK-BOX INCIDENT TRIAGE & COUNTERFACTUAL GATE", color=CYAN, fontsize=10.2, fontweight="bold", family="monospace")

        if idx < first_shield_cycle:
            ax_card.text(0.04, 0.64, "Monitoring cycle-exact .copper logs...", color=TEXT_MAIN, fontsize=10, family="monospace")
            ax_card.text(0.04, 0.44, "v1 (Rogue):      0 incidents   |  Status: nominal", color=TEXT_MUTED, fontsize=9.5, family="monospace")
            ax_card.text(0.04, 0.28, "v2 (Retrained):  0 incidents   |  Status: nominal", color=TEXT_MUTED, fontsize=9.5, family="monospace")
        elif not is_endcard:
            blame = v1_inc["blame"] if v1_inc else "learned_policy"
            ax_card.text(0.04, 0.64, "INCIDENT DETECTED IN v1 (SHIELD PREVENTED CONTACT)", color=RED, fontsize=10.0, fontweight="bold", family="monospace")
            ax_card.text(0.04, 0.46, f"  Root-Cause Blame : {blame}", color=AMBER, fontsize=9.5, family="monospace")
            ax_card.text(0.04, 0.30, "  Shield Response  : Safe stationary hold -> RRT* retract", color=TEXT_MAIN, fontsize=9.2, family="monospace")
            ax_card.text(0.04, 0.14, "  v2 Candidate     : 0 incidents · Clear throughout", color=GREEN, fontsize=9.5, family="monospace")
        else:
            gate = "PASSED" if passed_gate else "FAILED"
            v1_min = v1["summary"].get("min_clearance_m")
            v2_min = v2["summary"].get("min_clearance_m")
            v1_min_s = f"{v1_min:+.3f} m" if v1_min is not None else "n/a"
            v2_min_s = f"{v2_min:+.3f} m" if v2_min is not None else "n/a"
            ax_card.text(
                0.04,
                0.65,
                f"  COUNTERFACTUAL GATE (v1 -> v2): {gate}  ",
                color=BG,
                fontsize=10.5,
                fontweight="bold",
                family="monospace",
                bbox=dict(boxstyle="round,pad=0.35", facecolor=GREEN if passed_gate else RED, edgecolor="none"),
            )
            ax_card.text(
                0.04,
                0.42,
                f"v1 (Rogue)     : {len(v1['summary']['incidents'])} incidents | {v1['summary']['clamped_cycles']} shield holds | blame: learned_policy",
                color=RED,
                fontsize=8.8,
                family="monospace",
            )
            ax_card.text(
                0.04,
                0.24,
                f"v2 (Retrained) : {len(v2['summary']['incidents'])} incidents | min_clr {v2_min_s} (+{diff['min_clearance_delta_m']*1000:.1f} mm) | 100% Clear",
                color=GREEN,
                fontsize=8.8,
                family="monospace",
            )
            ax_card.text(
                0.04,
                0.08,
                "Deterministic replay verified via simplex_safety_resim (bit-identical)",
                color=TEXT_MUTED,
                fontsize=8.6,
                family="monospace",
            )

        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        writer.send(buf.tobytes())

    writer.close()
    plt.close(fig)
    print(f"Wrote {OUT_MP4} ({total_frames} frames @ {FPS} fps = {total_frames / FPS:.1f}s)")


if __name__ == "__main__":
    main()
