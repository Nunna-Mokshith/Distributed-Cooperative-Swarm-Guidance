"""
Generates publication-quality technical figures for the comprehensive project report.
Files generated:
1. figures/fig1_guidance_geometry.png
2. figures/fig2_graph_topology_laplacian.png
3. figures/fig3_neural_architecture.png
"""

import os
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# FIGURE 1: Guidance Geometry & Proportional Navigation (PN Law)
# -----------------------------------------------------------------------------
def generate_fig1_guidance_geometry():
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=200)
    fig.patch.set_facecolor("#0b1120")
    ax.set_facecolor("#0f172a")

    # Grid lines
    ax.grid(True, linestyle="--", alpha=0.25, color="#64748b")
    ax.set_xlim(-1, 11)
    ax.set_ylim(-1, 8)
    for spine in ax.spines.values():
        spine.set_color("#334155")
    ax.tick_params(colors="#94a3b8", labelsize=9)

    # Coordinates
    xm, ym = 1.5, 1.5   # Missile
    xt, yt = 8.5, 6.0   # Target

    # Range / Line-of-sight vector
    ax.plot([xm, xt], [ym, yt], color="#38bdf8", linestyle="--", linewidth=2.0, label="Line-of-Sight (LOS) Range R", zorder=2)

    # Missile velocity vector
    theta_m = math.radians(45.0)
    vm_len = 2.4
    vm_x = xm + vm_len * math.cos(theta_m)
    vm_y = ym + vm_len * math.sin(theta_m)
    ax.annotate("", xy=(vm_x, vm_y), xytext=(xm, ym),
                arrowprops=dict(arrowstyle="->", color="#22c55e", lw=3.0), zorder=3)
    ax.text(vm_x + 0.15, vm_y, r"$\mathbf{v}_m$ (Speed $v$, Heading $\theta$)", color="#4ade80", fontsize=11, fontweight="bold")

    # Missile normal acceleration vector (a_n perpendicular to v_m)
    an_angle = theta_m + math.pi / 2.0
    an_len = 1.3
    an_x = xm + an_len * math.cos(an_angle)
    an_y = ym + an_len * math.sin(an_angle)
    ax.annotate("", xy=(an_x, an_y), xytext=(xm, ym),
                arrowprops=dict(arrowstyle="->", color="#f43f5e", lw=2.5), zorder=3)
    ax.text(an_x - 1.8, an_y + 0.2, r"$a_{\text{cmd}} = N' V_c \dot{\lambda}$", color="#f43f5e", fontsize=11, fontweight="bold")

    # Target velocity vector
    theta_t = math.radians(160.0)
    vt_len = 1.8
    vt_x = xt + vt_len * math.cos(theta_t)
    vt_y = yt + vt_len * math.sin(theta_t)
    ax.annotate("", xy=(vt_x, vt_y), xytext=(xt, yt),
                arrowprops=dict(arrowstyle="->", color="#f97316", lw=2.5), zorder=3)
    ax.text(vt_x - 1.6, vt_y + 0.3, r"$\mathbf{v}_t$ (Target Speed)", color="#fb923c", fontsize=11, fontweight="bold")

    # LOS angle lambda reference line
    ax.plot([xm, xm + 3.0], [ym, ym], color="#94a3b8", linestyle=":", linewidth=1.5)
    lambda_angle = math.degrees(math.atan2(yt - ym, xt - xm))
    arc_lambda = patches.Arc((xm, ym), 2.2, 2.2, angle=0, theta1=0, theta2=lambda_angle, color="#38bdf8", linewidth=1.8)
    ax.add_patch(arc_lambda)
    ax.text(xm + 1.4, ym + 0.4, r"LOS Angle $\lambda$", color="#38bdf8", fontsize=11, fontweight="bold")

    # Draw Heading angle theta arc
    arc_theta = patches.Arc((xm, ym), 1.4, 1.4, angle=0, theta1=0, theta2=math.degrees(theta_m), color="#4ade80", linewidth=1.8)
    ax.add_patch(arc_theta)
    ax.text(xm + 0.8, ym + 0.1, r"$\theta$", color="#4ade80", fontsize=11)

    # Plot Missile and Target Points
    ax.scatter([xm], [ym], color="#38bdf8", s=140, zorder=4, label="Missile M(x_m, y_m)")
    ax.scatter([xt], [yt], color="#ef4444", marker="X", s=220, zorder=4, label="Target T(x_t, y_t)")

    # Formula Box
    formula_box = (
        r"$\bf{Line-of-Sight\ Geometry:}$" + "\n"
        r"$R = \sqrt{(x_t - x_m)^2 + (y_t - y_m)^2}$" + "\n"
        r"$V_c = -\dot{R} = -\frac{\Delta x \Delta v_x + \Delta y \Delta v_y}{R}$" + "\n"
        r"$\dot{\lambda} = \frac{\Delta x \Delta v_y - \Delta y \Delta v_x}{R^2}$" + "\n"
        r"$\bf{Guidance\ Law\ (PPN):}$" + "\n"
        r"$a_{\mathrm{cmd}} = \mathrm{clamp}(N' \cdot V_c \cdot \dot{\lambda},\ -a_{\max},\ +a_{\max})$"
    )
    ax.text(5.5, 0.4, formula_box, color="#f8fafc", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.8", facecolor="#1e293b", edgecolor="#38bdf8", alpha=0.9))

    ax.set_title("Classical Guidance Geometry: 2D Line-of-Sight (LOS) & Proportional Navigation",
                 fontsize=13, fontweight="bold", color="#38bdf8", pad=15)
    ax.set_xlabel("Inertial X Axis (meters)", color="#94a3b8", fontsize=10)
    ax.set_ylabel("Inertial Y Axis (meters)", color="#94a3b8", fontsize=10)
    ax.legend(loc="upper left", facecolor="#1e293b", edgecolor="#475569", labelcolor="#f8fafc", fontsize=9)

    fig.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "fig1_guidance_geometry.png")
    fig.savefig(out_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"Generated: {out_path}")

# -----------------------------------------------------------------------------
# FIGURE 2: Swarm Graph Topology & Dynamic Formation Bifurcation
# -----------------------------------------------------------------------------
def generate_fig2_graph_topology():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=200)
    fig.patch.set_facecolor("#0b1120")

    for ax in (ax1, ax2):
        ax.set_facecolor("#0f172a")
        ax.grid(True, linestyle="--", alpha=0.2, color="#64748b")
        for spine in ax.spines.values():
            spine.set_color("#334155")
        ax.tick_params(colors="#94a3b8", labelsize=8)

    # --- Subplot 1: Swarm Graph & Laplacian ---
    ax1.set_xlim(-1, 5)
    ax1.set_ylim(-1, 5)
    ax1.set_title("Swarm Communication Mesh & Spectral Laplacian", fontsize=12, fontweight="bold", color="#38bdf8", pad=12)

    nodes = {
        "M0": (2.0, 4.0),
        "M1": (0.8, 2.0),
        "M2": (3.2, 2.0),
        "M3": (2.0, 0.4)
    }

    # Draw edges (Ad-hoc radio links)
    edges = [("M0", "M1"), ("M0", "M2"), ("M1", "M2"), ("M1", "M3"), ("M2", "M3"), ("M0", "M3")]
    for u, v in edges:
        p1, p2 = nodes[u], nodes[v]
        ax1.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#38bdf8", linestyle="-", linewidth=1.8, alpha=0.6, zorder=1)

    # Draw nodes
    colors = ["#38bdf8", "#4ade80", "#f43f5e", "#fbbf24"]
    for i, (name, (x, y)) in enumerate(nodes.items()):
        ax1.scatter([x], [y], color=colors[i], s=260, zorder=3, edgecolors="#ffffff", linewidths=1.5)
        ax1.text(x, y + 0.3, name, color="#ffffff", fontweight="bold", fontsize=10, ha="center")

    laplacian_text = (
        "Graph Laplacian: L = D - A\n"
        "L = [[ 3, -1, -1, -1],\n"
        "     [-1,  3, -1, -1],\n"
        "     [-1, -1,  3, -1],\n"
        "     [-1, -1, -1,  3]]\n\n"
        "Spectral Connectivity Metric:\n"
        r"$\lambda_1 = 0 \leq \lambda_2 \leq \dots \leq \lambda_N$" + "\n"
        r"$\lambda_2(L) = 4.00 > 0 \rightarrow \mathrm{Connected}$"
    )
    ax1.text(0.1, -0.7, laplacian_text, color="#e2e8f0", fontsize=9.5,
             bbox=dict(boxstyle="round,pad=0.6", facecolor="#1e293b", edgecolor="#38bdf8", alpha=0.9))

    # --- Subplot 2: Dynamic Formation Bifurcation ---
    ax2.set_xlim(-100, 11000)
    ax2.set_ylim(-4500, 4500)
    ax2.set_title("Tactical Bifurcation: Cohesion to Terminal Decoupling", fontsize=12, fontweight="bold", color="#38bdf8", pad=12)

    # Trajectories
    t_vals = np.linspace(0, 1, 100)
    # M1 to T1 (North)
    x_m1 = t_vals * 10000.0
    y_m1 = 300.0 + (4000.0 - 300.0) * (t_vals ** 1.3)
    ax2.plot(x_m1, y_m1, color="#4ade80", linewidth=2.2, label="Sub-cell 1 (M1 -> T1)")

    # M2 to T2 (South)
    x_m2 = t_vals * 10000.0
    y_m2 = -300.0 + (-3000.0 + 300.0) * (t_vals ** 1.3)
    ax2.plot(x_m2, y_m2, color="#f43f5e", linewidth=2.2, label="Sub-cell 2 (M2 -> T2)")

    # M0 & M3 to T0 (Center)
    x_m0 = t_vals * 10000.0
    y_m0 = 0.0 + (1000.0) * (t_vals ** 1.1)
    ax2.plot(x_m0, y_m0, color="#38bdf8", linewidth=2.2, label="Sub-cell 0 (M0, M3 -> T0)")

    # Phase vertical lines
    ax2.axvline(3000, color="#64748b", linestyle=":", alpha=0.7)
    ax2.axvline(7500, color="#64748b", linestyle=":", alpha=0.7)
    ax2.text(1500, 3800, "Phase 1: Cruise\n(High Cohesion)", color="#94a3b8", fontsize=8.5, ha="center")
    ax2.text(5250, 3800, "Phase 2: Split\n(Dynamic GNN)", color="#38bdf8", fontsize=8.5, ha="center")
    ax2.text(9000, 3800, "Phase 3: Terminal\n(Decoupled PN)", color="#fbbf24", fontsize=8.5, ha="center")

    # Plot targets
    ax2.scatter([10000], [1000], color="#ef4444", marker="X", s=180, label="Target T0 (SAM)")
    ax2.scatter([10000], [4000], color="#f97316", marker="X", s=180, label="Target T1 (Bunker)")
    ax2.scatter([10000], [-3000], color="#a855f7", marker="X", s=180, label="Target T2 (Launcher)")

    ax2.set_xlabel("Downrange Distance X (meters)", color="#94a3b8", fontsize=9)
    ax2.set_ylabel("Crossrange Y (meters)", color="#94a3b8", fontsize=9)
    ax2.legend(loc="lower left", facecolor="#1e293b", edgecolor="#475569", labelcolor="#f8fafc", fontsize=8)

    fig.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "fig2_graph_topology_laplacian.png")
    fig.savefig(out_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"Generated: {out_path}")

# -----------------------------------------------------------------------------
# FIGURE 3: GAT Neural Architecture & CTDE Information Flow
# -----------------------------------------------------------------------------
def generate_fig3_neural_architecture():
    fig, ax = plt.subplots(figsize=(12, 6.8), dpi=200)
    fig.patch.set_facecolor("#0b1120")
    ax.set_facecolor("#0f172a")
    ax.axis("off")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    # Title
    ax.text(50, 95, "Graph Attention Network (GAT) Actor-Critic Architecture",
            color="#38bdf8", fontsize=14, fontweight="bold", ha="center")
    ax.text(50, 90, "Centralized Training with Decentralized Execution (CTDE) Pipeline",
            color="#94a3b8", fontsize=10, ha="center")

    def draw_box(x, y, w, h, title, subtitle, color, text_color="#ffffff"):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.8",
                                      facecolor="#1e293b", edgecolor=color, linewidth=2.0)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h * 0.65, title, color=color, fontsize=10, fontweight="bold", ha="center")
        ax.text(x + w / 2, y + h * 0.28, subtitle, color=text_color, fontsize=8, ha="center")

    def draw_arrow(x1, y1, x2, y2, color="#38bdf8"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=color, lw=2.0))

    # Column 1: Inputs
    draw_box(4, 58, 22, 22, "Agent State Matrix", "[N x 4]: x, y, vx, vy", "#38bdf8")
    draw_box(4, 28, 22, 22, "Target Feature Tensor", "[N x M x 5]: dx, dy, R, P, v_rel", "#f97316")
    draw_box(4, 4, 22, 18, "Adjacency Matrix A", "[N x N]: 1-Hop Datalinks", "#4ade80")

    # Column 2: Encoders
    draw_box(32, 58, 18, 22, "Agent MLP Encoder", "Linear -> ReLU -> Linear\nh_local: (N, 64)", "#38bdf8")
    draw_box(32, 28, 18, 22, "Target MLP Encoder", "Linear -> ReLU -> Linear\nh_target: (N, M, 64)", "#f97316")

    draw_arrow(26, 69, 32, 69)
    draw_arrow(26, 39, 32, 39)

    # Column 3: Graph Attention Layer
    draw_box(56, 50, 20, 30, "Graph Attention (GAT)", "Message Passing over A\nAttention α_ij = Softmax(e_ij)\nh_swarm: (N, 64)", "#4ade80")

    draw_arrow(50, 69, 56, 69)
    draw_arrow(26, 13, 66, 50, color="#4ade80") # Adjacency connects to GAT

    # Column 4: Utility Head & Value Head
    draw_box(80, 56, 16, 24, "Actor Policy Head", "Cross-Attention MLP\nLogits -> Softmax\nP(Target_k | Agent_i)", "#38bdf8")
    draw_box(80, 20, 16, 24, "Centralized Critic", "Mean-Pool Swarm\nValue Head MLP\nV(s): State Value", "#a855f7")

    draw_arrow(76, 68, 80, 68)
    draw_arrow(50, 39, 80, 64, color="#f97316") # Target connects to policy
    draw_arrow(76, 58, 80, 32, color="#a855f7") # GAT connects to Critic

    # Output annotations
    ax.text(88, 85, "DECISION:\nTarget Selection\n(< 2.4 ms)", color="#38bdf8", fontsize=9, fontweight="bold", ha="center")
    ax.text(88, 10, "TRAINING ONLY:\nPPO Value Loss\nBaseline for GAE", color="#a855f7", fontsize=9, fontweight="bold", ha="center")

    fig.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "fig3_neural_architecture.png")
    fig.savefig(out_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"Generated: {out_path}")


if __name__ == "__main__":
    generate_fig1_guidance_geometry()
    generate_fig2_graph_topology()
    generate_fig3_neural_architecture()
    print("All report figures generated successfully!")
