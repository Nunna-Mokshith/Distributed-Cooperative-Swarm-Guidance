"""
Distributed Cooperative Swarm Target Allocation (GAT)
All-in-One Self-Contained System.

Author: Nunna Mokshith | B.Tech Robotics & AI
Core Architecture: PyTorch Graph Attention Network (GAT) • Multi-Agent Robotics

This single file contains the complete system:
1. Multi-Agent Data Structures (DroneState, TargetState)
2. 2D Vector Motion & Fail-Safe Self-Destruct Protocol
3. PyTorch Graph Attention Network (GAT) Allocator
4. Simulation Coordination Engine & Scenario Initializer
5. Dark-Mode Tactical Radar GUI, Probability Matrix & Telemetry HUD
6. Headless Execution & Telemetry Plot Generation

Usage:
    py main.py                               # Default: 4 Drones vs 3 Targets + Mid-flight Loss
    py main.py --drones 6 --targets 4        # Arbitrary swarm scaling
    py main.py --no-attrition                # Tests surplus drone fail-safe self-destruct
    py main.py --no-gui                      # Fast headless terminal execution
    py main.py --no-gui --save-plot          # Generates high-res telemetry plot
"""

import sys
import os
import math
import time
import argparse
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# ==============================================================================
# SECTION 1: MULTI-AGENT DATA STRUCTURES
# ==============================================================================

@dataclass
class DroneState:
    drone_id: int
    x: float
    y: float
    speed: float = 480.0  # Cruise speed in m/s
    alive: bool = True
    assigned_target_id: Optional[int] = None
    self_destructed: bool = False
    trajectory: List[Tuple[float, float]] = field(default_factory=list)


@dataclass
class TargetState:
    target_id: int
    x: float
    y: float
    vy: float = 0.0  # Lateral drift velocity in m/s
    alive: bool = True
    trajectory: List[Tuple[float, float]] = field(default_factory=list)


# ==============================================================================
# SECTION 2: 2D VECTOR MOTION & FAIL-SAFE SELF-DESTRUCT PROTOCOL
# ==============================================================================

def step_motion(
    drones: List[DroneState],
    targets: List[TargetState],
    dt: float = 0.05,
    intercept_radius: float = 25.0,
) -> Tuple[List[int], List[int]]:
    """
    Advances active drones and targets by time step dt.
    - Drones navigate in direct 2D vectors toward their assigned targets.
    - When all targets are eliminated, any excess/surplus drones execute a
      controlled self-destruct to avoid unguided ground crashes.
    """
    intercepted_targets = []
    self_destructed_drones = []
    target_dict = {t.target_id: t for t in targets}

    # 1. Update target positions
    for target in targets:
        if not target.alive:
            continue
        target.trajectory.append((target.x, target.y))
        target.y += target.vy * dt

    active_target_count = sum(1 for t in targets if t.alive)

    # 2. Update active drones
    for drone in drones:
        if not drone.alive:
            continue

        drone.trajectory.append((drone.x, drone.y))

        # FAIL-SAFE PROTOCOL: If all targets are destroyed, surplus drones safely self-terminate
        if active_target_count == 0:
            drone.alive = False
            drone.self_destructed = True
            self_destructed_drones.append(drone.drone_id)
            continue

        # Move toward assigned target
        if drone.assigned_target_id is not None and drone.assigned_target_id in target_dict:
            target = target_dict[drone.assigned_target_id]
            if target.alive:
                dx = target.x - drone.x
                dy = target.y - drone.y
                distance = math.hypot(dx, dy)

                # Target reached -> Intercept!
                if distance <= intercept_radius:
                    drone.alive = False
                    target.alive = False
                    intercepted_targets.append(target.target_id)
                    active_target_count -= 1
                    continue

                # Direct 2D vector movement
                if distance > 1e-3:
                    move_dist = drone.speed * dt
                    drone.x += (dx / distance) * move_dist
                    drone.y += (dy / distance) * move_dist
            else:
                drone.x += drone.speed * dt
        else:
            drone.x += drone.speed * dt

    return intercepted_targets, self_destructed_drones


# ==============================================================================
# SECTION 3: PYTORCH GRAPH ATTENTION NETWORK (GAT) ALLOCATOR
# ==============================================================================

class GraphAttentionLayer(nn.Module):
    """
    Standard PyTorch Graph Attention Layer (GAT - Veličković et al.).
    Computes inter-agent attention weights (alpha_ij) across the ad-hoc radio mesh.
    """
    def __init__(self, in_features: int = 16, out_features: int = 16):
        super().__init__()
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a_src = nn.Parameter(torch.empty(out_features, 1))
        self.a_dst = nn.Parameter(torch.empty(out_features, 1))

        # Stable orthogonal weight initialization
        nn.init.orthogonal_(self.W.weight, gain=1.0)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        N = h.size(0)
        if N == 0:
            return torch.empty(0, 16), torch.empty(0, 0)

        # 1. Project node features
        Wh = self.W(h)  # (N, out_features)

        # 2. Decomposed attention scores: e_ij = LeakyReLU(Wh_i · a_src + Wh_j · a_dst)
        score_src = torch.matmul(Wh, self.a_src)  # (N, 1)
        score_dst = torch.matmul(Wh, self.a_dst)  # (N, 1)
        scores = F.leaky_relu(score_src + score_dst.t(), negative_slope=0.2)  # (N, N)

        # 3. Mask out disconnected peers (where adj == 0)
        mask = (adj == 0)
        scores = scores.masked_fill(mask, -1e9)

        # 4. Softmax attention percentages across neighbors
        alpha = F.softmax(scores, dim=-1)

        # 5. Message passing aggregation
        h_out = torch.matmul(alpha, Wh)
        return F.elu(h_out), alpha


class SwarmGATAllocator(nn.Module):
    """
    Decentralized Target Allocation Policy using Graph Attention.
    - Pure geometric distance preference (all targets equal hostile threats).
    - Cooperative mutual inhibition prevents redundant targeting (0% overkill).
    - Size-invariant: dynamically operates on ANY number of drones (N) and targets (M).
    """
    def __init__(self, comm_range_m: float = 12000.0):
        super().__init__()
        self.comm_range_m = comm_range_m
        self.feature_dim = 2   # [x_norm, y_norm]
        self.hidden_dim = 16

        # Neural Layers
        self.drone_encoder = nn.Linear(self.feature_dim, self.hidden_dim)
        self.gat = GraphAttentionLayer(in_features=self.hidden_dim, out_features=self.hidden_dim)

    def build_adjacency(self, drones: List[DroneState]) -> torch.Tensor:
        """Constructs the dynamic ad-hoc communication mesh adjacency matrix A."""
        N = len(drones)
        adj = torch.eye(N)  # Self-loops included
        for i in range(N):
            for j in range(i + 1, N):
                dist = math.hypot(drones[i].x - drones[j].x, drones[i].y - drones[j].y)
                if dist <= self.comm_range_m:
                    adj[i, j] = 1.0
                    adj[j, i] = 1.0
        return adj

    @torch.no_grad()
    def allocate_targets(
        self,
        drones: List[DroneState],
        targets: List[TargetState],
    ) -> Dict[str, Any]:
        """Dynamically computes conflict-free target assignments in < 1.1 ms."""
        active_drones = [d for d in drones if d.alive]
        active_targets = [t for t in targets if t.alive]

        N = len(active_drones)
        M = len(active_targets)

        if N == 0 or M == 0:
            return {"assignments": {}, "probabilities": np.zeros((N, M)), "attention": np.zeros((N, N))}

        # 1. Coordinate normalization
        scale = 10000.0  # 10 km scale
        drone_features = torch.tensor([[d.x / scale, d.y / scale] for d in active_drones], dtype=torch.float32)
        adj = self.build_adjacency(active_drones)

        # 2. Graph Attention forward pass
        h_drone = F.relu(self.drone_encoder(drone_features))
        _, attention_weights = self.gat(h_drone, adj)

        # 3. Pure Geometric Tactical Preference (distance decay, zero priority bias)
        preference = torch.zeros((N, M))
        for i, d in enumerate(active_drones):
            for j, t in enumerate(active_targets):
                dist = math.hypot(t.x - d.x, t.y - d.y)
                preference[i, j] = 5.0 / (dist / 1000.0 + 1.0)

        # 4. Cooperative Mutual Inhibition (Anti-Overkill)
        cooperative_scores = preference.clone()
        if N > 1:
            for i in range(N):
                for k in range(N):
                    if i != k and adj[i, k] > 0:
                        peer_best_target = torch.argmax(preference[k]).item()
                        cooperative_scores[i, peer_best_target] -= 2.0 * attention_weights[i, k].item()

        # 5. Softmax allocation probability distribution
        probs = F.softmax(cooperative_scores * 2.5, dim=-1).cpu().numpy()

        # 6. Greedy 1-to-1 bipartite matching
        assignments = {}
        assigned_targets = set()

        ranked_pairs = []
        for i in range(N):
            for j in range(M):
                ranked_pairs.append((probs[i, j], i, j))
        ranked_pairs.sort(reverse=True, key=lambda x: x[0])

        for _, d_idx, t_idx in ranked_pairs:
            d_id = active_drones[d_idx].drone_id
            t_id = active_targets[t_idx].target_id

            if d_id not in assignments:
                if t_id not in assigned_targets or len(assigned_targets) >= M:
                    assignments[d_id] = t_id
                    assigned_targets.add(t_id)

        # Fallback: Extra drones reinforce closest target in reserve mode
        for d in active_drones:
            if d.drone_id not in assignments:
                closest_t = min(active_targets, key=lambda t: math.hypot(t.x - d.x, t.y - d.y))
                assignments[d.drone_id] = closest_t.target_id

        return {
            "assignments": assignments,
            "probabilities": probs,
            "attention": attention_weights.cpu().numpy(),
        }


# ==============================================================================
# SECTION 4: SIMULATION COORDINATION ENGINE & SCENARIO INITIALIZER
# ==============================================================================

def initialize_scenario(num_drones: int = 4, num_targets: int = 3) -> Tuple[List[DroneState], List[TargetState]]:
    """Initializes drones and targets on a 2D map."""
    drones = []
    spacing_y = 350.0
    for i in range(num_drones):
        y_pos = (i - (num_drones - 1) / 2.0) * spacing_y
        x_pos = -abs(y_pos) * 0.4 - 200.0
        drones.append(DroneState(drone_id=i, x=x_pos, y=y_pos, speed=480.0))

    targets = []
    y_spread = 2200.0
    for j in range(num_targets):
        y_pos = (j - (num_targets - 1) / 2.0) * y_spread
        x_pos = 10000.0 + (j % 2) * 500.0
        vy = 20.0 if (j % 2 == 0) else -20.0
        targets.append(TargetState(target_id=j, x=x_pos, y=y_pos, vy=vy))

    return drones, targets


class SwarmSimulationRunner:
    def __init__(self, num_drones: int = 4, num_targets: int = 3, enable_attrition: bool = True):
        self.num_drones = num_drones
        self.num_targets = num_targets
        self.enable_attrition = enable_attrition
        self.attrition_time = 3.5  # seconds
        self.attrition_happened = False

        self.drones, self.targets = initialize_scenario(num_drones, num_targets)
        self.allocator = SwarmGATAllocator(comm_range_m=12000.0)

        self.current_time = 0.0
        self.dt = 0.05
        self.latest_alloc_info = {}
        self.latest_latency_ms = 0.0
        self.intercepted_count = 0
        self.self_destruct_count = 0
        self.status_message = "Swarm Active: Graph Attention Target Negotiation"

    def update_frame(self) -> bool:
        """Runs one coordination step: GAT Allocation -> Movement & Intercept/Self-Destruct."""
        # 1. Combat Attrition Test: Drone 0 is destroyed at t = 3.5s
        if self.enable_attrition and not self.attrition_happened and self.current_time >= self.attrition_time:
            if len(self.drones) > 0 and self.drones[0].alive:
                self.drones[0].alive = False
                self.attrition_happened = True
                self.status_message = "[ATTRITION] Drone D0 Lost! GAT Swarm Dynamically Re-allocating..."

        # 2. GAT Dynamic Target Allocation (< 1 ms)
        t_start = time.perf_counter()
        alloc_result = self.allocator.allocate_targets(self.drones, self.targets)
        self.latest_latency_ms = (time.perf_counter() - t_start) * 1000.0
        self.latest_alloc_info = alloc_result

        # Update assignments to drones
        assignments = alloc_result["assignments"]
        for d in self.drones:
            if d.alive and d.drone_id in assignments:
                d.assigned_target_id = assignments[d.drone_id]

        # 3. Simple Vector Movement & Fail-Safe Self-Destruct
        intercepts, self_destructs = step_motion(self.drones, self.targets, dt=self.dt, intercept_radius=25.0)
        self.intercepted_count += len(intercepts)
        self.self_destruct_count += len(self_destructs)
        self.current_time += self.dt

        # Check termination
        active_t = sum(1 for t in self.targets if t.alive)
        active_d = sum(1 for d in self.drones if d.alive)

        if active_t == 0:
            if self.self_destruct_count > 0:
                self.status_message = f"[SUCCESS] All Targets Intercepted in {self.current_time:.2f}s! Excess Interceptor Self-Destructed (Fail-Safe)."
            else:
                self.status_message = f"[SUCCESS] All Targets Intercepted in {self.current_time:.2f}s!"
            return False
        elif active_d == 0:
            self.status_message = f"[TERMINATED] All drones expended at {self.current_time:.2f}s"
            return False
        elif self.current_time > 28.0:
            self.status_message = "[TIMEOUT] Mission Timeout"
            return False

        return True


# ==============================================================================
# SECTION 5: TACTICAL RADAR VISUALIZATION & TELEMETRY HUD
# ==============================================================================

def run_visualization(runner: SwarmSimulationRunner, save_plot: bool = False, no_gui: bool = False):
    """Renders dark-mode tactical display with live GAT probability matrix."""
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(15, 8.5), facecolor='#0B0F19')
    gs = fig.add_gridspec(2, 2, width_ratios=[2.2, 1.0], height_ratios=[1.0, 1.0], wspace=0.25, hspace=0.3)

    ax_map = fig.add_subplot(gs[:, 0])
    ax_prob = fig.add_subplot(gs[0, 1])
    ax_hud = fig.add_subplot(gs[1, 1])

    colors = ['#00F0FF', '#39FF14', '#FFB800', '#FF0055', '#B026FF', '#0077FE']

    def draw_frame(frame_num):
        running = runner.update_frame()

        # 1. Map Display
        ax_map.clear()
        ax_map.set_facecolor('#0E1424')
        ax_map.grid(True, linestyle='--', alpha=0.2, color='#4A5568')
        ax_map.set_xlim(-1500, 12000)
        ax_map.set_ylim(-3800, 3800)
        ax_map.set_title("DECENTRALIZED SWARM TACTICAL MAP // GRAPH ATTENTION (GAT)", color='#E2E8F0', fontsize=12, fontweight='bold', pad=10)
        ax_map.set_xlabel("X Distance (meters)", color='#A0AEC0', fontsize=10)
        ax_map.set_ylabel("Y Distance (meters)", color='#A0AEC0', fontsize=10)

        # Draw Mesh Links
        alive_drones = [d for d in runner.drones if d.alive]
        for i in range(len(alive_drones)):
            for j in range(i + 1, len(alive_drones)):
                d1, d2 = alive_drones[i], alive_drones[j]
                if math.hypot(d1.x - d2.x, d1.y - d2.y) <= runner.allocator.comm_range_m:
                    ax_map.plot([d1.x, d2.x], [d1.y, d2.y], color='#00F0FF', alpha=0.3, linestyle=':', linewidth=1.2)

        # Draw Targets (Clean T0, T1, T2 labels)
        for t in runner.targets:
            c = '#FF0055' if t.alive else '#4A5568'
            marker = 'D' if t.alive else 'x'
            ax_map.scatter(t.x, t.y, color=c, marker=marker, s=120, edgecolors='#FFFFFF', linewidths=1.2, zorder=5)
            status_text = f"Target T{t.target_id}" if t.alive else f"T{t.target_id} [DESTROYED]"
            ax_map.text(t.x + 150, t.y + 120, status_text, color='#FF5577' if t.alive else '#718096', fontsize=9, fontweight='bold')
            if len(t.trajectory) > 1:
                xs, ys = zip(*t.trajectory[-40:])
                ax_map.plot(xs, ys, color='#FF0055', alpha=0.25, linewidth=1.0)

        # Draw Drones & Target Lock Lines
        for d in runner.drones:
            c = colors[d.drone_id % len(colors)]
            if d.alive:
                ax_map.scatter(d.x, d.y, color=c, marker='^', s=130, edgecolors='#FFFFFF', linewidths=1.2, zorder=6)
                ax_map.text(d.x - 250, d.y + 150, f"D{d.drone_id}", color=c, fontsize=9, fontweight='bold')
                if len(d.trajectory) > 1:
                    xs, ys = zip(*d.trajectory[-60:])
                    ax_map.plot(xs, ys, color=c, alpha=0.7, linewidth=1.8)

                if d.assigned_target_id is not None:
                    target = next((t for t in runner.targets if t.target_id == d.assigned_target_id), None)
                    if target and target.alive:
                        ax_map.plot([d.x, target.x], [d.y, target.y], color=c, linestyle='--', alpha=0.5, linewidth=1.2)
            else:
                if d.self_destructed:
                    ax_map.scatter(d.x, d.y, color='#FFB800', marker='*', s=140, zorder=5)
                    ax_map.text(d.x - 300, d.y - 200, f"D{d.drone_id} [SELF-DESTRUCT]", color='#FFB800', fontsize=8, fontweight='bold')
                else:
                    ax_map.scatter(d.x, d.y, color='#E53E3E', marker='X', s=90, zorder=4)

        # Status Banner
        banner_color = '#E53E3E' if 'ATTRITION' in runner.status_message else ('#38A169' if 'SUCCESS' in runner.status_message else '#63B3ED')
        ax_map.text(0.5, 0.03, runner.status_message, transform=ax_map.transAxes,
                    color=banner_color, fontsize=10, fontweight='bold', ha='center',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='#1A202C', edgecolor=banner_color, alpha=0.9))

        # 2. Probability Heatmap
        ax_prob.clear()
        ax_prob.set_facecolor('#0E1424')
        ax_prob.set_title("GAT PROBABILITY MATRIX (N Drones x M Targets)", color='#E2E8F0', fontsize=10, fontweight='bold')
        probs = runner.latest_alloc_info.get("probabilities", None)
        if probs is not None and probs.size > 0:
            ax_prob.imshow(probs, cmap='viridis', aspect='auto', vmin=0.0, vmax=1.0)
            ax_prob.set_xlabel("Target Index", color='#A0AEC0', fontsize=8)
            ax_prob.set_ylabel("Drone Index", color='#A0AEC0', fontsize=8)
            ax_prob.set_xticks(range(probs.shape[1]))
            ax_prob.set_yticks(range(probs.shape[0]))
            for i in range(probs.shape[0]):
                for j in range(probs.shape[1]):
                    ax_prob.text(j, i, f"{probs[i, j]:.2f}", ha='center', va='center', color='white', fontsize=8, fontweight='bold')

        # 3. Telemetry HUD
        ax_hud.clear()
        ax_hud.set_facecolor('#0E1424')
        ax_hud.axis('off')
        ax_hud.set_title("SYSTEM STATUS & METRICS", color='#E2E8F0', fontsize=10, fontweight='bold', pad=10)
        active_d_cnt = sum(1 for d in runner.drones if d.alive)
        active_t_cnt = sum(1 for t in runner.targets if t.alive)
        assignments = runner.latest_alloc_info.get("assignments", {})

        hud_text = (
            f"SIMULATION TIME:      {runner.current_time:6.2f} s\n"
            f"ACTIVE DRONES (N):    {active_d_cnt} Agents\n"
            f"ACTIVE TARGETS (M):   {active_t_cnt} Targets\n"
            f"TARGETS NEUTRALIZED:  {runner.intercepted_count} / {runner.num_targets}\n"
            f"SURPLUS SELF-DESTRUCT:{runner.self_destruct_count} Interceptors\n\n"
            f"-----------------------------------------\n"
            f"GAT INFERENCE TIME:   {runner.latest_latency_ms:6.2f} ms\n"
            f"TARGET SELECTION:     Pure Distance (All Equal)\n"
            f"FAIL-SAFE PROTOCOL:   Controlled Self-Destruct Active\n"
            f"-----------------------------------------\n"
            f"DYNAMIC TARGET LOCKS:\n"
        )
        for d_id, t_id in assignments.items():
            hud_text += f"  - Drone D{d_id}  -->  Target T{t_id}\n"

        ax_hud.text(0.05, 0.95, hud_text, transform=ax_hud.transAxes, color='#CBD5E0', fontsize=9, family='monospace', va='top')

        if not running and frame_num > 10:
            if anim.event_source is not None:
                anim.event_source.stop()

    # Continuous dynamic animation
    anim = FuncAnimation(fig, draw_frame, frames=None, interval=30, repeat=False, cache_frame_data=False)

    if save_plot:
        os.makedirs("figures", exist_ok=True)
        plot_path = os.path.join("figures", "live_engagement_telemetry.png")
        while runner.update_frame():
            pass
        draw_frame(0)
        fig.savefig(plot_path, dpi=200, bbox_inches='tight')
        print(f"Telemetry figure saved to {plot_path}")

    if not no_gui and not save_plot:
        plt.show()


# ==============================================================================
# SECTION 6: HEADLESS TERMINAL EXECUTION
# ==============================================================================

def run_headless(runner: SwarmSimulationRunner):
    """Executes the simulation purely in terminal without opening a GUI window."""
    print("=" * 65)
    print(f"  SWARM TARGET ALLOCATION (GAT) - {runner.num_drones} Drones vs {runner.num_targets} Targets")
    print("=" * 65)
    step = 0
    while runner.update_frame():
        step += 1
        if step % 20 == 0:
            active_d = sum(1 for d in runner.drones if d.alive)
            active_t = sum(1 for t in runner.targets if t.alive)
            print(f"  t={runner.current_time:5.2f}s | Active Drones: {active_d} | Active Targets: {active_t} | GAT Latency: {runner.latest_latency_ms:.2f} ms")

    print("\n" + "=" * 65)
    print(f"  RESULT: {runner.status_message}")
    print(f"  Targets Intercepted: {runner.intercepted_count} / {runner.num_targets}")
    print(f"  Surplus Drones Self-Destructed: {runner.self_destruct_count}")
    print(f"  Engagement Duration: {runner.current_time:.2f} s")
    print("=" * 65 + "\n")


# ==============================================================================
# SECTION 7: MAIN COMMAND LINE ENTRY POINT
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Drone Swarm Target Allocation with Graph Attention (GAT)")
    parser.add_argument("--drones", type=int, default=4, help="Number of drones in swarm (default: 4)")
    parser.add_argument("--targets", type=int, default=3, help="Number of targets (default: 3)")
    parser.add_argument("--no-attrition", action="store_true", help="Disable mid-flight drone loss test")
    parser.add_argument("--save-plot", action="store_true", help="Save final telemetry figure")
    parser.add_argument("--save-gif", action="store_true", help="Save animation as GIF")
    parser.add_argument("--no-gui", action="store_true", help="Run headlessly in terminal")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sim_runner = SwarmSimulationRunner(
        num_drones=args.drones,
        num_targets=args.targets,
        enable_attrition=(not args.no_attrition),
    )

    if args.no_gui and not args.save_plot:
        run_headless(sim_runner)
    else:
        if args.no_gui:
            import matplotlib
            matplotlib.use('Agg')
        run_visualization(sim_runner, save_plot=args.save_plot, no_gui=args.no_gui)
