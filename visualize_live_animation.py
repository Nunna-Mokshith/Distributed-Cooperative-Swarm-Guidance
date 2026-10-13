"""
Distributed Cooperative Swarm Guidance (DCSG)
Module: Live 2D Combat Radar Animation & GIF Generator
File: visualize_live_animation.py

Author: Nunna Mokshith (Robotics & AI)
Renders a real-time animated tactical radar display showing:
- 4 Missiles in Mach 1.5 formation with Dubins kinematics & dynamic trails
- 3 Hostile moving targets (SAM, Bunker, Mobile Launcher)
- Dynamic radio datalink mesh edges connecting missiles within R_comm (12 km)
- Dynamic bifurcation as missiles split into sub-swarms
- Real-time HUD gauges: Clock, Fiedler Value lambda_2, Min Separation, G-Loads
- Terminal detonations on intercept
"""

import os
import sys
import math
import logging
from typing import List, Dict, Any, Tuple
import numpy as np
import matplotlib
if "--no-gui" in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches

# Ensure imports resolve cleanly
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
logger = logging.getLogger("LiveAnimation")


def precompute_salvo_trajectory(
    dt: float = 0.05,
    max_time_s: float = 23.0,
    comm_range_m: float = 12000.0,
    safe_separation_m: float = 200.0,
    lethal_radius_m: float = 10.0,
) -> Dict[str, Any]:
    """
    Simulates the engagement tick-by-tick and records exact states for animation rendering.
    """
    # 4 Missiles in staggered echelon
    missiles = [
        MissileState(x=0.0, y=0.0, heading=0.0, velocity=500.0, max_lateral_accel=25.0 * G_ACCEL),
        MissileState(x=-150.0, y=300.0, heading=0.0, velocity=500.0, max_lateral_accel=25.0 * G_ACCEL),
        MissileState(x=-150.0, y=-300.0, heading=0.0, velocity=500.0, max_lateral_accel=25.0 * G_ACCEL),
        MissileState(x=-300.0, y=0.0, heading=0.0, velocity=500.0, max_lateral_accel=25.0 * G_ACCEL),
    ]

    # 3 Hostile Targets downrange
    targets = [
        TargetState(target_id=0, x=10000.0, y=1000.0, vx=0.0, vy=30.0, priority=10.0),  # SAM Radar
        TargetState(target_id=1, x=10000.0, y=4000.0, vx=0.0, vy=-20.0, priority=6.0),  # Bunker
        TargetState(target_id=2, x=10000.0, y=-3000.0, vx=0.0, vy=15.0, priority=4.0),  # Launcher
    ]

    assignments = [0, 1, 2, 0]  # M0->T0, M1->T1, M2->T2, M3->T0
    n_missiles = len(missiles)
    active = [True] * n_missiles

    frames_data: List[Dict[str, Any]] = []
    t = 0.0
    prev_ranges = [float("inf")] * n_missiles
    explosions: List[Dict[str, Any]] = []

    while t < max_time_s and any(active):
        # 1. Update targets (constant velocity)
        targets = [
            TargetState(
                target_id=tgt.target_id,
                x=tgt.x + tgt.vx * dt,
                y=tgt.y + tgt.vy * dt,
                vx=tgt.vx,
                vy=tgt.vy,
                priority=tgt.priority,
            )
            for tgt in targets
        ]

        # 2. Graph topology
        living = [m for idx, m in enumerate(missiles) if active[idx]]
        if len(living) >= 2:
            adj, _, metrics = compute_swarm_graph_laplacian(living, comm_range_m)
            fiedler = metrics.fiedler_value
            min_dist = metrics.min_inter_agent_dist
        else:
            adj = np.zeros((len(living), len(living)))
            fiedler = 0.0
            min_dist = 0.0

        # Record snapshot
        snapshot = {
            "time": t,
            "missiles": [(m.x, m.y, m.heading, active[i]) for i, m in enumerate(missiles)],
            "targets": [(tgt.x, tgt.y) for tgt in targets],
            "fiedler": fiedler,
            "min_dist": min_dist,
            "active": list(active),
            "explosions": list(explosions),
        }
        frames_data.append(snapshot)

        # 3. Guidance and physics step
        new_missiles = []
        for i in range(n_missiles):
            if not active[i]:
                new_missiles.append(missiles[i])
                continue

            ego = missiles[i]
            assigned_tgt = targets[assignments[i]]
            range_m, v_c, los_rate = calculate_los_geometry(ego, assigned_tgt)

            # Intercept detection
            if range_m <= lethal_radius_m or (v_c <= 0.0 and range_m < prev_ranges[i] and range_m <= lethal_radius_m * 1.5):
                active[i] = False
                explosions.append({"x": assigned_tgt.x, "y": assigned_tgt.y, "birth_time": t})
                new_missiles.append(ego)
                continue

            prev_ranges[i] = range_m

            # Proportional Navigation
            a_pn = proportional_navigation_law(
                nav_constant=4.0,
                closing_velocity=v_c,
                los_rate=los_rate,
                max_lateral_accel=ego.max_lateral_accel,
            )

            # Flocking force
            a_flock = compute_flocking_acceleration(
                agent_idx=i,
                missiles=missiles,
                comm_range_m=comm_range_m,
                safe_separation_m=safe_separation_m,
            )
            flock_weight = min(1.0, max(0.0, (range_m - 2000.0) / 4000.0))
            a_total = max(-ego.max_lateral_accel, min(ego.max_lateral_accel, a_pn + flock_weight * a_flock))

            new_missiles.append(step_kinematics(ego, a_total, dt))

        missiles = new_missiles
        t += dt

    return {"frames": frames_data, "targets": targets}


def create_swarm_animation(
    data: Dict[str, Any],
    save_gif_path: str = "figures/swarm_salvo_animation.gif",
    subsample_step: int = 2,
) -> animation.FuncAnimation:
    """
    Constructs the Matplotlib animation.
    """
    frames = data["frames"][::subsample_step]
    n_frames = len(frames)

    fig, ax = plt.subplots(figsize=(11, 7), dpi=120)
    fig.patch.set_facecolor("#070d19")
    ax.set_facecolor("#0b1329")

    ax.set_xlim(-800, 11500)
    ax.set_ylim(-4500, 5200)
    ax.grid(True, linestyle="--", alpha=0.25, color="#334155")
    for spine in ax.spines.values():
        spine.set_color("#1e293b")
    ax.tick_params(colors="#64748b", labelsize=9)
    ax.set_xlabel("Downrange Distance X (meters)", color="#94a3b8", fontsize=10)
    ax.set_ylabel("Crossrange Position Y (meters)", color="#94a3b8", fontsize=10)
    ax.set_title("DCSG Live Tactical Radar: Distributed Swarm Salvo & Dynamic Bifurcation",
                 color="#38bdf8", fontsize=12, fontweight="bold", pad=12)

    # Range rings
    origin = (0, 0)
    for r in [2000, 4000, 6000, 8000, 10000]:
        circle = patches.Circle(origin, r, fill=False, edgecolor="#1e293b", linestyle=":", linewidth=1.0)
        ax.add_patch(circle)
        ax.text(r - 300, 100, f"{r//1000}km", color="#334155", fontsize=8)

    colors = ["#38bdf8", "#4ade80", "#f43f5e", "#fbbf24"]
    target_colors = ["#ef4444", "#f97316", "#a855f7"]
    target_names = ["T0: SAM HQ", "T1: Bunker", "T2: Launcher"]

    # Artists
    missile_scatters = [ax.scatter([], [], color=colors[i], s=70, zorder=5) for i in range(4)]
    missile_trails = [ax.plot([], [], color=colors[i], linewidth=1.8, alpha=0.7, zorder=3)[0] for i in range(4)]
    target_scatters = [ax.scatter([], [], color=target_colors[j], marker="X", s=130, zorder=4) for j in range(3)]
    target_labels = [ax.text(0, 0, target_names[j], color=target_colors[j], fontsize=8.5, fontweight="bold") for j in range(3)]

    # Datalink mesh lines (6 possible pairs)
    mesh_lines = [ax.plot([], [], color="#38bdf8", linestyle="--", linewidth=1.2, alpha=0.4, zorder=2)[0] for _ in range(6)]

    # Explosion markers
    explosion_scatter = ax.scatter([], [], color="#f43f5e", marker="o", s=150, edgecolors="#ffffff", linewidths=1.5, zorder=6)

    # HUD text overlay
    hud_box = ax.text(
        0.02, 0.95, "", transform=ax.transAxes, verticalalignment="top",
        color="#f8fafc", fontsize=9.5, fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#0f172a", edgecolor="#38bdf8", alpha=0.85)
    )

    trail_history: List[List[Tuple[float, float]]] = [[] for _ in range(4)]

    def update(frame_idx: int):
        f = frames[frame_idx]
        cur_time = f["time"]

        # Update targets
        for j, (tx, ty) in enumerate(f["targets"]):
            target_scatters[j].set_offsets([[tx, ty]])
            target_labels[j].set_position((tx + 150, ty + 120))

        # Update missiles
        active_coords = []
        for i, (mx, my, heading, is_act) in enumerate(f["missiles"]):
            if is_act:
                missile_scatters[i].set_offsets([[mx, my]])
                trail_history[i].append((mx, my))
                if len(trail_history[i]) > 60:
                    trail_history[i].pop(0)
                active_coords.append((i, mx, my))
            else:
                missile_scatters[i].set_offsets(np.empty((0, 2)))

            # Draw trails
            if trail_history[i]:
                txs, tys = zip(*trail_history[i])
                missile_trails[i].set_data(txs, tys)

        # Update communication mesh links
        line_idx = 0
        for idx_a in range(len(active_coords)):
            for idx_b in range(idx_a + 1, len(active_coords)):
                ia, xa, ya = active_coords[idx_a]
                ib, xb, yb = active_coords[idx_b]
                dist = math.hypot(xa - xb, ya - yb)
                if dist <= 12000.0:
                    mesh_lines[line_idx].set_data([xa, xb], [ya, yb])
                    line_idx += 1
        # Clear unused lines
        for k in range(line_idx, len(mesh_lines)):
            mesh_lines[k].set_data([], [])

        # Update HUD text
        fiedler = f["fiedler"]
        min_d = f["min_dist"]
        conn_str = f"CONNECTED (lambda_2={fiedler:.2f})" if fiedler > 0 else "TERMINAL DIVE"
        warn_str = " [SAFE > 200m]" if min_d >= 200.0 else " [! PROXIMITY ALERT]"

        hud_content = (
            f"TIME: {cur_time:05.2f}s | MACH 1.5 (500 m/s)\n"
            f"MESH: {conn_str}\n"
            f"SEPARATION: {min_d:4.0f}m{warn_str}\n"
            f"TARGETS: T0(SAM) | T1(Bunker) | T2(Launcher)\n"
            f"ALLOCATION: [M0->T0, M1->T1, M2->T2, M3->T0]"
        )
        hud_box.set_text(hud_content)

        return missile_scatters + missile_trails + target_scatters + mesh_lines + [hud_box]

    anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=35, blit=False)

    if save_gif_path:
        os.makedirs(os.path.dirname(save_gif_path), exist_ok=True)
        logger.info(f"Rendering and saving animated GIF to: {save_gif_path} ...")
        anim.save(save_gif_path, writer="pillow", fps=25)
        logger.info(f"Animation GIF successfully saved to: {save_gif_path}")

    return anim


if __name__ == "__main__":
    show_gui = "--no-gui" not in sys.argv

    logger.info("Precomputing salvo flight data...")
    salvo_data = precompute_salvo_trajectory(dt=0.04)

    gif_out = os.path.join(current_dir, "figures", "swarm_salvo_animation.gif")
    anim = create_swarm_animation(salvo_data, save_gif_path=gif_out)

    if show_gui:
        logger.info("Opening live interactive animation window on screen...")
        plt.show()
    else:
        logger.info("Animation GIF saved. Exiting without opening GUI window.")
