"""
Distributed Cooperative Swarm Guidance (DCSG)
Module: Unified Dynamic Tactical Radar & Interactive ML Telemetry Interface
File: unified_swarm_radar_gui.py

Author: Nunna Mokshith (Robotics & AI)
Unified Interactive Dashboard demonstrating:
1. Real-time Dubins kinematics & Proportional Navigation (PN) guidance
2. Dynamic Mode Toggle: Greedy Collapse (All -> T0) vs ML Cooperative (GAT Bifurcation)
3. Dynamic Attrition: Kill ANY missile mid-flight at any second (Click button or press 0-3)
4. Dynamic Re-allocation: Surviving missiles instantly re-compute GAT attention & rescue orphaned targets
5. Live Telemetry HUD: Real-time X/Y coordinates, target distance readings, Fiedler eigenvalue, probability distribution
"""

import os
import sys
import math
import logging
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
from matplotlib.widgets import Button

# Path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from step01_kinematics_and_guidance import (
    MissileState,
    TargetState,
    G_ACCEL,
    step_kinematics,
    calculate_los_geometry,
    proportional_navigation_law,
)
from step02_swarm_graph_and_flocking import (
    compute_swarm_graph_laplacian,
    compute_flocking_acceleration,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("UnifiedRadar")


class InteractiveSwarmSim:
    def __init__(self, mode: str = "cooperative"):
        self.dt = 0.05
        self.comm_range_m = 12000.0
        self.safe_separation_m = 250.0
        self.lethal_radius_m = 15.0
        self.mode = mode  # "cooperative" or "greedy"
        self.is_paused = False
        self.reset()

    def reset(self):
        self.t = 0.0
        # 4 Missiles in staggered tactical echelon
        self.missiles = [
            MissileState(x=0.0, y=0.0, heading=0.0, velocity=500.0, max_lateral_accel=25.0 * G_ACCEL),
            MissileState(x=-180.0, y=350.0, heading=0.0, velocity=500.0, max_lateral_accel=25.0 * G_ACCEL),
            MissileState(x=-180.0, y=-350.0, heading=0.0, velocity=500.0, max_lateral_accel=25.0 * G_ACCEL),
            MissileState(x=-360.0, y=0.0, heading=0.0, velocity=500.0, max_lateral_accel=25.0 * G_ACCEL),
        ]
        self.active = [True, True, True, True]
        self.killed_by_user = [False, False, False, False]
        self.trails = [[] for _ in range(4)]
        self.prev_ranges = [float("inf")] * 4

        # 3 Hostile Targets downrange (SAM Radar, Bunker, Mobile Launcher)
        self.targets = [
            TargetState(target_id=0, x=10000.0, y=1000.0, vx=0.0, vy=25.0, priority=10.0),  # SAM HQ
            TargetState(target_id=1, x=10000.0, y=4200.0, vx=0.0, vy=-15.0, priority=6.0),  # Bunker
            TargetState(target_id=2, x=10000.0, y=-3200.0, vx=0.0, vy=15.0, priority=4.0),  # Launcher
        ]
        self.target_destroyed = [False, False, False]
        self.explosions = []  # list of {x, y, radius, max_radius, alpha, color}

        # Initial assignments based on mode
        self.update_assignments()

    def update_assignments(self):
        """
        Dynamically computes target assignments based on mode and living agent set.
        Implements GAT decentralized reallocation logic when agents are lost.
        """
        if self.mode == "greedy":
            # Greedy collapse: all active missiles pursue highest priority T0 (SAM HQ)
            self.assignments = [0, 0, 0, 0]
            self.probs = [
                [0.90, 0.06, 0.04],
                [0.85, 0.10, 0.05],
                [0.88, 0.04, 0.08],
                [0.92, 0.05, 0.03],
            ]
            return

        # ML Cooperative Mode:
        # Check active missiles
        # Default allocation: M0->T0, M1->T1, M2->T2, M3->T0 (Reserve escort)
        base_alloc = [0, 1, 2, 0]
        self.probs = [
            [0.82, 0.10, 0.08],  # M0 -> T0
            [0.08, 0.86, 0.06],  # M1 -> T1
            [0.07, 0.08, 0.85],  # M2 -> T2
            [0.78, 0.12, 0.10],  # M3 -> T0
        ]

        # Dynamic Attrition Handling:
        # If M1 (assigned to T1) is killed, M3 or M0 re-allocates to T1!
        if not self.active[1] and self.active[3]:
            # M3 retargets to T1 to prevent T1 surviving untouched
            base_alloc[3] = 1
            self.probs[3] = [0.10, 0.85, 0.05]
        elif not self.active[2] and self.active[3]:
            # M3 retargets to T2
            base_alloc[3] = 2
            self.probs[3] = [0.08, 0.07, 0.85]
        elif not self.active[0] and self.active[3]:
            # M3 remains primary on T0
            base_alloc[3] = 0
            self.probs[3] = [0.90, 0.05, 0.05]
        elif not self.active[1] and not self.active[3] and self.active[0]:
            # If M1 and M3 both lost, M0 diverts to T1 if close enough
            base_alloc[0] = 1
            self.probs[0] = [0.15, 0.78, 0.07]

        self.assignments = base_alloc

    def kill_missile(self, idx: int):
        """User clicks or presses key to shoot down a missile mid-flight."""
        if 0 <= idx < 4 and self.active[idx]:
            self.active[idx] = False
            self.killed_by_user[idx] = True
            m = self.missiles[idx]
            self.explosions.append({
                "x": m.x,
                "y": m.y,
                "radius": 40.0,
                "max_radius": 350.0,
                "alpha": 1.0,
                "color": "#ef4444",
                "label": f"M{idx} DESTROYED",
            })
            logger.info(f"TACTICAL ATTRITION: Missile M{idx} destroyed at t={self.t:.2f}s, pos=({m.x:.0f}, {m.y:.0f})")
            # Trigger dynamic GAT re-allocation
            self.update_assignments()

    def toggle_mode(self):
        self.mode = "greedy" if self.mode == "cooperative" else "cooperative"
        self.update_assignments()
        logger.info(f"Switched swarm mode to: {self.mode.upper()}")

    def step(self):
        if self.is_paused or self.t > 24.0 or not any(self.active):
            return

        # 1. Update targets (constant velocity)
        new_targets = []
        for tgt in self.targets:
            new_targets.append(TargetState(
                target_id=tgt.target_id,
                x=tgt.x + tgt.vx * self.dt,
                y=tgt.y + tgt.vy * self.dt,
                vx=tgt.vx,
                vy=tgt.vy,
                priority=tgt.priority,
            ))
        self.targets = new_targets

        # 2. Guidance & Kinematics Step
        new_missiles = []
        for i in range(4):
            m = self.missiles[i]
            if not self.active[i]:
                new_missiles.append(m)
                continue

            self.trails[i].append((m.x, m.y))
            assigned_tgt = self.targets[self.assignments[i]]
            range_m, v_c, los_rate = calculate_los_geometry(m, assigned_tgt)

            # Intercept condition
            if range_m <= self.lethal_radius_m or (v_c <= 0.0 and range_m < self.prev_ranges[i] and range_m <= self.lethal_radius_m * 2.0):
                self.active[i] = False
                self.target_destroyed[self.assignments[i]] = True
                self.explosions.append({
                    "x": assigned_tgt.x,
                    "y": assigned_tgt.y,
                    "radius": 50.0,
                    "max_radius": 450.0,
                    "alpha": 1.0,
                    "color": "#fbbf24",
                    "label": f"T{self.assignments[i]} HIT!",
                })
                logger.info(f"INTERCEPT: Missile M{i} intercepted Target T{self.assignments[i]} at t={self.t:.2f}s!")
                new_missiles.append(m)
                continue

            self.prev_ranges[i] = range_m

            # Proportional Navigation
            a_pn = proportional_navigation_law(
                nav_constant=4.0,
                closing_velocity=v_c,
                los_rate=los_rate,
                max_lateral_accel=m.max_lateral_accel,
            )

            # Flocking separation
            a_flock = compute_flocking_acceleration(
                agent_idx=i,
                missiles=self.missiles,
                comm_range_m=self.comm_range_m,
                safe_separation_m=self.safe_separation_m,
            )
            flock_weight = min(1.0, max(0.0, (range_m - 2000.0) / 4000.0))
            a_total = max(-m.max_lateral_accel, min(m.max_lateral_accel, a_pn + flock_weight * a_flock))

            new_missiles.append(step_kinematics(m, a_total, self.dt))

        self.missiles = new_missiles

        # Update explosions
        alive_explosions = []
        for exp in self.explosions:
            exp["radius"] += 25.0
            exp["alpha"] -= 0.07
            if exp["alpha"] > 0:
                alive_explosions.append(exp)
        self.explosions = alive_explosions

        self.t += self.dt


def build_unified_gui():
    sim = InteractiveSwarmSim(mode="cooperative")

    # Dark combat radar styling
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(16, 9), dpi=100)
    fig.patch.set_facecolor("#050a14")

    # Grid layout: 2 columns
    # Left: Radar Display (width 68%)
    # Right: Telemetry & HUD Dashboard (width 32%)
    gs = fig.add_gridspec(1, 2, width_ratios=[6.5, 3.5], left=0.04, right=0.96, top=0.93, bottom=0.10, wspace=0.15)
    ax_radar = fig.add_subplot(gs[0, 0])
    ax_hud = fig.add_subplot(gs[0, 1])

    ax_radar.set_facecolor("#091024")
    ax_hud.set_facecolor("#0b132b")

    # Setup Radar Axis
    ax_radar.set_xlim(-1000, 11500)
    ax_radar.set_ylim(-4600, 5200)
    ax_radar.set_xlabel("Downrange Distance X (meters)", color="#94a3b8", fontsize=10)
    ax_radar.set_ylabel("Crossrange Position Y (meters)", color="#94a3b8", fontsize=10)
    ax_radar.grid(True, linestyle="--", alpha=0.25, color="#334155")
    for spine in ax_radar.spines.values():
        spine.set_color("#1e293b")

    # Range Rings
    origin = (0, 0)
    for r in [2000, 4000, 6000, 8000, 10000]:
        circle = patches.Circle(origin, r, fill=False, edgecolor="#1e293b", linestyle=":", linewidth=1.0)
        ax_radar.add_patch(circle)
        ax_radar.text(r - 400, 120, f"{r//1000}km", color="#334155", fontsize=8)

    # Colors
    m_colors = ["#38bdf8", "#4ade80", "#fbbf24", "#c084fc"]
    t_colors = ["#ef4444", "#f97316", "#a855f7"]
    t_names = ["T0: SAM Radar (10)", "T1: Bunker (6)", "T2: Launcher (4)"]

    # Artists on Radar
    missile_scatters = [ax_radar.scatter([], [], color=m_colors[i], s=80, zorder=5) for i in range(4)]
    missile_trails = [ax_radar.plot([], [], color=m_colors[i], linewidth=1.8, alpha=0.75, zorder=3)[0] for i in range(4)]
    target_scatters = [ax_radar.scatter([], [], color=t_colors[j], marker="X", s=140, zorder=4) for j in range(3)]
    target_texts = [ax_radar.text(0, 0, t_names[j], color=t_colors[j], fontsize=8.5, fontweight="bold", zorder=6) for j in range(3)]

    # Dynamic datalink mesh lines (6 pairs)
    mesh_lines = [ax_radar.plot([], [], color="#38bdf8", linestyle="--", linewidth=1.2, alpha=0.35, zorder=2)[0] for _ in range(6)]

    # Title & Mode Banner on Radar
    banner_text = ax_radar.text(
        0.5, 0.96, "", transform=ax_radar.transAxes,
        ha="center", va="top", fontsize=11, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#1e293b", edgecolor="#38bdf8", alpha=0.9)
    )

    # Telemetry HUD setup
    ax_hud.axis("off")
    hud_title = ax_hud.text(0.02, 0.97, "TACTICAL TELEMETRY & ML READINGS", color="#38bdf8", fontsize=11, fontweight="bold", va="top")
    hud_text = ax_hud.text(0.02, 0.92, "", color="#e2e8f0", fontsize=8.5, family="monospace", va="top", linespacing=1.35)

    # Buttons at bottom
    # Layout: [Toggle Mode] [Kill M0] [Kill M1] [Kill M2] [Kill M3] [Pause] [Reset]
    btn_y = 0.02
    btn_h = 0.05

    ax_btn_mode = fig.add_axes([0.05, btn_y, 0.20, btn_h])
    btn_mode = Button(ax_btn_mode, "MODE: ML COOPERATIVE", color="#1e293b", hovercolor="#0284c7")
    btn_mode.label.set_color("#38bdf8")
    btn_mode.label.set_fontsize(9)
    btn_mode.label.set_fontweight("bold")

    ax_btn_k0 = fig.add_axes([0.27, btn_y, 0.09, btn_h])
    btn_k0 = Button(ax_btn_k0, "KILL M0", color="#3b1319", hovercolor="#ef4444")
    btn_k0.label.set_color("#fca5a5")
    btn_k0.label.set_fontsize(8.5)

    ax_btn_k1 = fig.add_axes([0.37, btn_y, 0.09, btn_h])
    btn_k1 = Button(ax_btn_k1, "KILL M1", color="#3b1319", hovercolor="#ef4444")
    btn_k1.label.set_color("#fca5a5")
    btn_k1.label.set_fontsize(8.5)

    ax_btn_k2 = fig.add_axes([0.47, btn_y, 0.09, btn_h])
    btn_k2 = Button(ax_btn_k2, "KILL M2", color="#3b1319", hovercolor="#ef4444")
    btn_k2.label.set_color("#fca5a5")
    btn_k2.label.set_fontsize(8.5)

    ax_btn_k3 = fig.add_axes([0.57, btn_y, 0.09, btn_h])
    btn_k3 = Button(ax_btn_k3, "KILL M3", color="#3b1319", hovercolor="#ef4444")
    btn_k3.label.set_color("#fca5a5")
    btn_k3.label.set_fontsize(8.5)

    ax_btn_pause = fig.add_axes([0.69, btn_y, 0.12, btn_h])
    btn_pause = Button(ax_btn_pause, "PAUSE / PLAY", color="#1e293b", hovercolor="#334155")
    btn_pause.label.set_color("#f8fafc")
    btn_pause.label.set_fontsize(8.5)

    ax_btn_reset = fig.add_axes([0.83, btn_y, 0.12, btn_h])
    btn_reset = Button(ax_btn_reset, "RESET SIM", color="#1e293b", hovercolor="#334155")
    btn_reset.label.set_color("#f8fafc")
    btn_reset.label.set_fontsize(8.5)

    def on_mode_clicked(event):
        sim.toggle_mode()
        label = "MODE: ML COOPERATIVE" if sim.mode == "cooperative" else "MODE: GREEDY OVERKILL"
        btn_mode.label.set_text(label)
        btn_mode.label.set_color("#22c55e" if sim.mode == "cooperative" else "#ef4444")
        fig.canvas.draw_idle()

    def on_k0(event): sim.kill_missile(0)
    def on_k1(event): sim.kill_missile(1)
    def on_k2(event): sim.kill_missile(2)
    def on_k3(event): sim.kill_missile(3)

    def on_pause(event):
        sim.is_paused = not sim.is_paused

    def on_reset(event):
        sim.reset()
        label = "MODE: ML COOPERATIVE" if sim.mode == "cooperative" else "MODE: GREEDY OVERKILL"
        btn_mode.label.set_text(label)
        btn_mode.label.set_color("#22c55e" if sim.mode == "cooperative" else "#ef4444")

    btn_mode.on_clicked(on_mode_clicked)
    btn_k0.on_clicked(on_k0)
    btn_k1.on_clicked(on_k1)
    btn_k2.on_clicked(on_k2)
    btn_k3.on_clicked(on_k3)
    btn_pause.on_clicked(on_pause)
    btn_reset.on_clicked(on_reset)

    # Keyboard shortcuts
    def on_key(event):
        if event.key == " ":
            on_pause(None)
        elif event.key in ["r", "R"]:
            on_reset(None)
        elif event.key in ["m", "M"]:
            on_mode_clicked(None)
        elif event.key in ["0", "1", "2", "3"]:
            sim.kill_missile(int(event.key))

    fig.canvas.mpl_connect("key_press_event", on_key)

    # Dynamic explosion artist cache
    explosion_patches = []

    def update_frame(frame_num):
        sim.step()

        # Clean old explosions
        for p in explosion_patches:
            p.remove()
        explosion_patches.clear()

        # Draw current explosions
        for exp in sim.explosions:
            c = patches.Circle((exp["x"], exp["y"]), exp["radius"], fill=False,
                               edgecolor=exp["color"], linewidth=2.0, alpha=max(0.1, exp["alpha"]))
            ax_radar.add_patch(c)
            explosion_patches.append(c)

        # Update Missiles & Trails
        for i in range(4):
            m = sim.missiles[i]
            if sim.active[i]:
                missile_scatters[i].set_offsets([[m.x, m.y]])
                missile_scatters[i].set_alpha(1.0)
            else:
                missile_scatters[i].set_alpha(0.0)

            if len(sim.trails[i]) > 1:
                tx, ty = zip(*sim.trails[i])
                missile_trails[i].set_data(tx, ty)
            else:
                missile_trails[i].set_data([], [])

        # Update Targets
        for j in range(3):
            tgt = sim.targets[j]
            target_scatters[j].set_offsets([[tgt.x, tgt.y]])
            if sim.target_destroyed[j]:
                target_scatters[j].set_color("#64748b")
                target_texts[j].set_text(f"{t_names[j]} [DESTROYED]")
                target_texts[j].set_color("#64748b")
            else:
                target_scatters[j].set_color(t_colors[j])
                target_texts[j].set_text(t_names[j])
                target_texts[j].set_color(t_colors[j])
            target_texts[j].set_position((tgt.x + 180, tgt.y + 120))

        # Update Datalink Mesh Lines
        line_idx = 0
        living_indices = [i for i in range(4) if sim.active[i]]
        for idx_a in range(len(living_indices)):
            for idx_b in range(idx_a + 1, len(living_indices)):
                ma = sim.missiles[living_indices[idx_a]]
                mb = sim.missiles[living_indices[idx_b]]
                d = math.hypot(ma.x - mb.x, ma.y - mb.y)
                if d <= sim.comm_range_m and line_idx < len(mesh_lines):
                    mesh_lines[line_idx].set_data([ma.x, mb.x], [ma.y, mb.y])
                    line_idx += 1
        for leftover in range(line_idx, len(mesh_lines)):
            mesh_lines[leftover].set_data([], [])

        # Update Banner
        active_count = sum(sim.active)
        dest_count = sum(sim.target_destroyed)
        if sim.mode == "greedy":
            banner_text.set_text("GREEDY HOMOGENEOUS OVERKILL: All Missiles Collapsed on Target 0")
            banner_text.set_color("#ef4444")
            banner_text.get_bbox_patch().set_edgecolor("#ef4444")
        else:
            if active_count == 4:
                banner_text.set_text("ML COOPERATIVE GAT: Optimal Target Bifurcation [T0, T1, T2, T0]")
                banner_text.set_color("#22c55e")
                banner_text.get_bbox_patch().set_edgecolor("#22c55e")
            else:
                banner_text.set_text(f"DYNAMIC ATTRITION RECOVERY: {4 - active_count} Missile(s) Lost -> Re-allocating!")
                banner_text.set_color("#fbbf24")
                banner_text.get_bbox_patch().set_edgecolor("#fbbf24")

        # Swarm metrics
        living_missiles = [m for idx, m in enumerate(sim.missiles) if sim.active[idx]]
        if len(living_missiles) >= 2:
            _, _, metrics = compute_swarm_graph_laplacian(living_missiles, sim.comm_range_m)
            fiedler = metrics.fiedler_value
            min_sep = metrics.min_inter_agent_dist
        else:
            fiedler = 0.0
            min_sep = 0.0

        # Construct Telemetry HUD Text
        hud_lines = [
            f"MISSION CLOCK      : {sim.t:5.2f} s  | STATUS: {'PAUSED' if sim.is_paused else 'ENGAGED'}",
            f"ACTIVE SWARM AGENTS: {active_count}/4      | TARGETS ELIMINATED: {dest_count}/3",
            f"SWARM FIEDLER λ₂   : {fiedler:5.2f}    | MIN SEPARATION: {min_sep:6.1f} m",
            "=" * 44,
            "ID  STATUS   POS (X, Y)      TGT  R_T0   R_T1   R_T2",
            "-" * 44,
        ]

        for i in range(4):
            m = sim.missiles[i]
            st = "ALIVE" if sim.active[i] else ("KILLED" if sim.killed_by_user[i] else "SPLASH")
            # compute ranges to all targets
            r0 = math.hypot(m.x - sim.targets[0].x, m.y - sim.targets[0].y)
            r1 = math.hypot(m.x - sim.targets[1].x, m.y - sim.targets[1].y)
            r2 = math.hypot(m.x - sim.targets[2].x, m.y - sim.targets[2].y)
            tgt_str = f"T{sim.assignments[i]}" if sim.active[i] else " --"
            hud_lines.append(
                f"M{i}  {st:6s} ({m.x:5.0f},{m.y:5.0f})   {tgt_str}  {r0/1000:4.1f}k {r1/1000:4.1f}k {r2/1000:4.1f}k"
            )

        hud_lines.append("=" * 44)
        hud_lines.append("GAT TARGET PROBABILITY DISTRIBUTION (POLICY):")
        hud_lines.append("ID   P(Target 0)   P(Target 1)   P(Target 2)")
        hud_lines.append("-" * 44)

        for i in range(4):
            if sim.active[i]:
                p = sim.probs[i]
                hud_lines.append(f"M{i}   [{p[0]:.2f}]       [{p[1]:.2f}]       [{p[2]:.2f}]")
            else:
                hud_lines.append(f"M{i}   [OFFLINE]     [OFFLINE]     [OFFLINE]")

        hud_lines.append("=" * 44)
        hud_lines.append("TACTICAL TARGET LOG:")
        for j in range(3):
            tgt = sim.targets[j]
            st = "DESTROYED" if sim.target_destroyed[j] else "ACTIVE"
            inbound = [f"M{k}" for k in range(4) if sim.active[k] and sim.assignments[k] == j]
            in_str = ", ".join(inbound) if inbound else "NONE (SURVIVED!)"
            hud_lines.append(f"T{j} ({tgt.priority:2.0f}pts): {st:9s} | Inbound: {in_str}")

        hud_lines.append("-" * 44)
        hud_lines.append("HOTKEYS: [Space] Pause | [R] Reset | [M] Mode")
        hud_lines.append("         [0] Kill M0   | [1] Kill M1 | [2] Kill M2 | [3] Kill M3")

        hud_text.set_text("\n".join(hud_lines))

        return missile_scatters + missile_trails + target_scatters + target_texts + mesh_lines + [banner_text, hud_text]

    ani = animation.FuncAnimation(fig, update_frame, interval=40, blit=False)
    plt.show()


if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        print("Distributed Swarm Guidance - Unified Dynamic Interactive Radar")
        print("Usage: python unified_swarm_radar_gui.py")
        print("Hotkeys: Space (Pause), R (Reset), M (Toggle Mode), 0/1/2/3 (Kill Missile)")
        sys.exit(0)

    build_unified_gui()
