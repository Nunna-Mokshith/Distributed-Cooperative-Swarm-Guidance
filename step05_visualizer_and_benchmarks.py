"""
Distributed Cooperative Swarm Guidance (DCSG)
Module 5: Mission Telemetry, Comparative Benchmarks & Trajectory Visualizer
File: 01_applied_ml/distributed_swarm_guidance/step05_visualizer_and_benchmarks.py

Candidate: Nunna Mokshith (Robotics & AI)
Core Concept:
- High-fidelity trajectory telemetry logger
- Multi-panel visual analysis (Flight Trajectories, Inter-Agent Spacing, Algebraic Connectivity lambda_2, G-Load)
- Side-by-side benchmark: Uncoordinated Saturation Attack vs. Coordinated Distributed Swarm
"""

from typing import List, Dict, Any, Tuple
import math
import logging
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless file generation
import matplotlib.pyplot as plt

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
logger = logging.getLogger("TelemetryVisualizer")


def run_logged_salvo_simulation(
    initial_swarm: List[MissileState],
    initial_targets: List[TargetState],
    assignments: List[int],
    enable_flocking: bool = True,
    comm_range_m: float = 12000.0,
    safe_separation_m: float = 250.0,
    dt: float = 0.05,
    max_time_s: float = 25.0,
    lethal_radius_m: float = 10.0,
) -> Dict[str, Any]:
    """
    Simulates a salvo while recording fine-grained telemetry at each timestep.
    """
    missiles = list(initial_swarm)
    targets = list(initial_targets)
    n_missiles = len(missiles)
    active = [True] * n_missiles

    # Telemetry buffers
    timestamps: List[float] = []
    missile_x_hist: List[List[float]] = [[] for _ in range(n_missiles)]
    missile_y_hist: List[List[float]] = [[] for _ in range(n_missiles)]
    target_x_hist: List[List[float]] = [[] for _ in range(len(targets))]
    target_y_hist: List[List[float]] = [[] for _ in range(len(targets))]
    min_dist_hist: List[float] = []
    fiedler_hist: List[float] = []
    g_load_hist: List[List[float]] = [[] for _ in range(n_missiles)]

    t = 0.0
    prev_ranges = [float("inf")] * n_missiles

    while t < max_time_s and any(active):
        timestamps.append(t)

        # Log target positions and step kinematics
        for j, tgt in enumerate(targets):
            target_x_hist[j].append(tgt.x)
            target_y_hist[j].append(tgt.y)
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

        # Log living missile topology
        living = [m for idx, m in enumerate(missiles) if active[idx]]
        if len(living) >= 2:
            _, _, metrics = compute_swarm_graph_laplacian(living, comm_range_m)
            min_dist_hist.append(metrics.min_inter_agent_dist)
            fiedler_hist.append(metrics.fiedler_value)
        else:
            min_dist_hist.append(0.0)
            fiedler_hist.append(0.0)

        new_missiles = []
        for i in range(n_missiles):
            ego = missiles[i]
            missile_x_hist[i].append(ego.x)
            missile_y_hist[i].append(ego.y)

            if not active[i]:
                g_load_hist[i].append(0.0)
                new_missiles.append(ego)
                continue

            target_idx = assignments[i]
            tgt = targets[target_idx]

            # LOS geometry & PN guidance
            range_m, v_c, los_rate = calculate_los_geometry(ego, tgt)

            # Intercept detection
            if range_m <= lethal_radius_m or (v_c <= 0.0 and range_m < prev_ranges[i] and range_m <= lethal_radius_m * 1.5):
                active[i] = False
                g_load_hist[i].append(0.0)
                new_missiles.append(ego)
                continue

            prev_ranges[i] = range_m

            a_pn = proportional_navigation_law(
                nav_constant=4.0,
                closing_velocity=v_c,
                los_rate=los_rate,
                max_lateral_accel=ego.max_lateral_accel,
            )

            a_flock = 0.0
            if enable_flocking:
                a_flock = compute_flocking_acceleration(
                    agent_idx=i,
                    missiles=missiles,
                    comm_range_m=comm_range_m,
                    safe_separation_m=safe_separation_m,
                )
                flock_weight = min(1.0, max(0.0, (range_m - 2000.0) / 4000.0))
                a_flock *= flock_weight

            a_total_raw = a_pn + a_flock
            a_total = max(-ego.max_lateral_accel, min(ego.max_lateral_accel, a_total_raw))
            g_load_hist[i].append(abs(a_total) / G_ACCEL)

            new_missiles.append(step_kinematics(ego, a_total, dt))

        missiles = new_missiles
        t += dt

    return {
        "timestamps": timestamps,
        "missile_x": missile_x_hist,
        "missile_y": missile_y_hist,
        "target_x": target_x_hist,
        "target_y": target_y_hist,
        "min_dist": min_dist_hist,
        "fiedler": fiedler_hist,
        "g_loads": g_load_hist,
    }


def generate_comparative_telemetry_plot(
    uncoord_telemetry: Dict[str, Any],
    coord_telemetry: Dict[str, Any],
    output_filepath: str = "01_applied_ml/distributed_swarm_guidance/swarm_engagement_telemetry.png",
) -> None:
    """
    Renders a 4-panel publication-ready engineering dashboard comparing both salvos.
    """
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    fig.patch.set_facecolor("#0f172a")

    for ax in axs.flat:
        ax.set_facecolor("#1e293b")
        ax.grid(True, linestyle="--", alpha=0.3, color="#94a3b8")
        ax.tick_params(colors="#cbd5e1")
        for spine in ax.spines.values():
            spine.set_color("#475569")

    # -------------------------------------------------------------
    # Panel 1: Trajectory Paths (Top-Left)
    # -------------------------------------------------------------
    ax1 = axs[0, 0]
    ax1.set_title("Coordinated Swarm Trajectories (Dynamic Bifurcation)", fontsize=13, fontweight="bold", color="#38bdf8")
    ax1.set_xlabel("X Position (m)", color="#cbd5e1")
    ax1.set_ylabel("Y Position (m)", color="#cbd5e1")

    colors = ["#38bdf8", "#4ade80", "#f43f5e", "#fbbf24"]
    for i in range(len(coord_telemetry["missile_x"])):
        ax1.plot(
            coord_telemetry["missile_x"][i],
            coord_telemetry["missile_y"][i],
            color=colors[i % len(colors)],
            linewidth=2.0,
            label=f"Missile M{i}",
        )
        # Starting point
        ax1.scatter(coord_telemetry["missile_x"][i][0], coord_telemetry["missile_y"][i][0], color=colors[i % len(colors)], s=40)

    # Plot targets
    target_colors = ["#ef4444", "#f97316", "#a855f7"]
    for j in range(len(coord_telemetry["target_x"])):
        ax1.plot(
            coord_telemetry["target_x"][j],
            coord_telemetry["target_y"][j],
            color=target_colors[j % len(target_colors)],
            linestyle=":",
            linewidth=2.5,
            label=f"Target T{j}",
        )
        ax1.scatter(
            coord_telemetry["target_x"][j][-1],
            coord_telemetry["target_y"][j][-1],
            color=target_colors[j % len(target_colors)],
            marker="X",
            s=120,
        )

    ax1.legend(loc="upper left", facecolor="#1e293b", edgecolor="#475569", labelcolor="#f8fafc", fontsize=9)

    # -------------------------------------------------------------
    # Panel 2: Inter-Agent Spacing & Collision Avoidance (Top-Right)
    # -------------------------------------------------------------
    ax2 = axs[0, 1]
    ax2.set_title("Minimum Pairwise Distance vs Safety Threshold", fontsize=13, fontweight="bold", color="#38bdf8")
    ax2.set_xlabel("Time (seconds)", color="#cbd5e1")
    ax2.set_ylabel("Min Inter-Agent Distance (m)", color="#cbd5e1")

    ax2.plot(
        coord_telemetry["timestamps"][:len(coord_telemetry["min_dist"])],
        coord_telemetry["min_dist"],
        color="#4ade80",
        linewidth=2.0,
        label="Coordinated Swarm",
    )
    ax2.plot(
        uncoord_telemetry["timestamps"][:len(uncoord_telemetry["min_dist"])],
        uncoord_telemetry["min_dist"],
        color="#f43f5e",
        linestyle="--",
        linewidth=2.0,
        label="Uncoordinated (Saturation)",
    )
    ax2.axhline(200.0, color="#fbbf24", linestyle="-.", linewidth=1.5, label="Safety Bubble (200m)")
    ax2.legend(loc="upper right", facecolor="#1e293b", edgecolor="#475569", labelcolor="#f8fafc", fontsize=9)

    # -------------------------------------------------------------
    # Panel 3: Swarm Algebraic Connectivity (Bottom-Left)
    # -------------------------------------------------------------
    ax3 = axs[1, 0]
    ax3.set_title("Network Algebraic Connectivity (Fiedler Value lambda_2)", fontsize=13, fontweight="bold", color="#38bdf8")
    ax3.set_xlabel("Time (seconds)", color="#cbd5e1")
    ax3.set_ylabel("Fiedler Eigenvalue lambda_2(L)", color="#cbd5e1")

    ax3.plot(
        coord_telemetry["timestamps"][:len(coord_telemetry["fiedler"])],
        coord_telemetry["fiedler"],
        color="#38bdf8",
        linewidth=2.2,
        label="Swarm Mesh Topology",
    )
    ax3.axhline(0.0, color="#ef4444", linestyle="-", linewidth=1.0, alpha=0.7)
    ax3.fill_between(
        coord_telemetry["timestamps"][:len(coord_telemetry["fiedler"])],
        coord_telemetry["fiedler"],
        color="#38bdf8",
        alpha=0.15,
    )
    ax3.legend(loc="upper right", facecolor="#1e293b", edgecolor="#475569", labelcolor="#f8fafc", fontsize=9)

    # -------------------------------------------------------------
    # Panel 4: Autopilot G-Load Demands (Bottom-Right)
    # -------------------------------------------------------------
    ax4 = axs[1, 1]
    ax4.set_title("Missile Autopilot G-Load (Lateral Acceleration)", fontsize=13, fontweight="bold", color="#38bdf8")
    ax4.set_xlabel("Time (seconds)", color="#cbd5e1")
    ax4.set_ylabel("G-Force (G)", color="#cbd5e1")

    for i in range(len(coord_telemetry["g_loads"])):
        t_slice = coord_telemetry["timestamps"][:len(coord_telemetry["g_loads"][i])]
        ax4.plot(t_slice, coord_telemetry["g_loads"][i], color=colors[i % len(colors)], linewidth=1.8, label=f"M{i} G-Load")

    ax4.axhline(25.0, color="#ef4444", linestyle="--", linewidth=1.5, label="Max Fin Limit (25G)")
    ax4.legend(loc="upper right", facecolor="#1e293b", edgecolor="#475569", labelcolor="#f8fafc", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_filepath, dpi=200, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    logger.info(f"Engineering telemetry visualization saved to: {output_filepath}")


if __name__ == "__main__":
    logger.info("Running comparative mission telemetry benchmarks...")

    # Standard 4-Missile Swarm
    base_swarm = [
        MissileState(x=0.0, y=0.0, heading=0.0, velocity=500.0, max_lateral_accel=25.0 * G_ACCEL),
        MissileState(x=-150.0, y=300.0, heading=0.0, velocity=500.0, max_lateral_accel=25.0 * G_ACCEL),
        MissileState(x=-150.0, y=-300.0, heading=0.0, velocity=500.0, max_lateral_accel=25.0 * G_ACCEL),
        MissileState(x=-300.0, y=0.0, heading=0.0, velocity=500.0, max_lateral_accel=25.0 * G_ACCEL),
    ]

    base_targets = [
        TargetState(target_id=0, x=10000.0, y=1000.0, vx=0.0, vy=30.0, priority=10.0),
        TargetState(target_id=1, x=10000.0, y=4000.0, vx=0.0, vy=-20.0, priority=6.0),
        TargetState(target_id=2, x=10000.0, y=-3000.0, vx=0.0, vy=15.0, priority=4.0),
    ]

    # Run 1: Uncoordinated Salvo (All missiles chase Target 0 - Overkill, no flocking)
    logger.info("Simulating Uncoordinated Saturation Salvo...")
    uncoord_data = run_logged_salvo_simulation(
        initial_swarm=base_swarm,
        initial_targets=base_targets,
        assignments=[0, 0, 0, 0],
        enable_flocking=False,
    )

    # Run 2: Coordinated Swarm Salvo (Multi-target allocation [0, 1, 2, 0] + Flocking)
    logger.info("Simulating Coordinated Distributed Swarm Salvo...")
    coord_data = run_logged_salvo_simulation(
        initial_swarm=base_swarm,
        initial_targets=base_targets,
        assignments=[0, 1, 2, 0],
        enable_flocking=True,
    )

    # Generate telemetry chart
    output_png = "01_applied_ml/distributed_swarm_guidance/swarm_engagement_telemetry.png"
    generate_comparative_telemetry_plot(uncoord_data, coord_data, output_png)
    print(f"\nTelemetry analysis complete. Plot generated at: {output_png}\n")
