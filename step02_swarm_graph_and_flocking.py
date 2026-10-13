"""
Distributed Cooperative Swarm Guidance (DCSG)
Module 2: Swarm Graph Topology, Laplacian Connectivity & Flocking Consensus
File: 01_applied_ml/distributed_swarm_guidance/step02_swarm_graph_and_flocking.py

Candidate: Nunna Mokshith (Robotics & AI)
Core Concept:
- Graph Laplacian L = D - A
- Algebraic Connectivity (Fiedler eigenvalue lambda_2(L))
- Distributed Reynolds Flocking (Separation, Cohesion, Alignment) converted to Dubins lateral acceleration
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
import math
import numpy as np
import numpy.typing as npt
import logging

from step01_kinematics_and_guidance import (
    MissileState,
    TargetState,
    G_ACCEL,
    wrap_to_pi,
    step_kinematics,
    calculate_los_geometry,
    proportional_navigation_law,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SwarmGraphEngine")


@dataclass(frozen=True)
class SwarmTopologyMetrics:
    """Holds graph-theoretic properties of the swarm communication network."""
    num_nodes: int
    is_connected: bool
    fiedler_value: float          # lambda_2 of Graph Laplacian L
    min_inter_agent_dist: float   # Minimum pairwise Euclidean distance (m)
    avg_neighbors_per_agent: float


def compute_pairwise_distances(missiles: List[MissileState]) -> npt.NDArray[np.float64]:
    """
    Computes Euclidean distance matrix D_dist between all agents.
    Returns:
        N x N symmetric matrix where element (i, j) is distance in meters.
    """
    n = len(missiles)
    coords = np.array([[m.x, m.y] for m in missiles], dtype=np.float64)
    # Broadcast subtraction: (N, 1, 2) - (1, N, 2) -> (N, N, 2)
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff ** 2, axis=-1))
    return dist_matrix


def compute_swarm_graph_laplacian(
    missiles: List[MissileState],
    comm_range_m: float = 12000.0,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], SwarmTopologyMetrics]:
    """
    Builds the Adjacency (A) and Laplacian (L = D - A) matrices for the swarm.

    The Fiedler value (lambda_2 of L) measures algebraic connectivity:
    - lambda_2 > 0: Graph is globally connected (information can propagate).
    - lambda_2 = 0: Graph is disconnected (one or more sub-swarms are isolated).

    Args:
        missiles: List of N missile states.
        comm_range_m: Max communication radio range (meters).

    Returns:
        Tuple:
            1. Adjacency matrix A (N x N)
            2. Laplacian matrix L (N x N)
            3. SwarmTopologyMetrics
    """
    n = len(missiles)
    if n < 2:
        empty_mat = np.zeros((n, n), dtype=np.float64)
        return empty_mat, empty_mat, SwarmTopologyMetrics(
            num_nodes=n,
            is_connected=True,
            fiedler_value=0.0,
            min_inter_agent_dist=0.0,
            avg_neighbors_per_agent=0.0,
        )

    dist_matrix = compute_pairwise_distances(missiles)

    # Adjacency: 1 if distance <= comm_range_m and i != j, else 0
    adj_matrix = ((dist_matrix <= comm_range_m) & (dist_matrix > 1e-4)).astype(np.float64)

    # Degree matrix D: diagonal sum of row connections
    degrees = np.sum(adj_matrix, axis=1)
    degree_matrix = np.diag(degrees)

    # Laplacian: L = D - A
    laplacian = degree_matrix - adj_matrix

    # Compute eigenvalues (Laplacian is symmetric positive semi-definite)
    eigenvalues = np.linalg.eigvalsh(laplacian)
    sorted_eigs = np.sort(eigenvalues)

    # Fiedler eigenvalue is the second smallest (index 1)
    fiedler_value = float(max(0.0, sorted_eigs[1]))
    is_connected = fiedler_value > 1e-4

    # Calculate min non-zero pairwise distance
    np.fill_diagonal(dist_matrix, np.inf)
    min_dist = float(np.min(dist_matrix))

    metrics = SwarmTopologyMetrics(
        num_nodes=n,
        is_connected=is_connected,
        fiedler_value=fiedler_value,
        min_inter_agent_dist=min_dist,
        avg_neighbors_per_agent=float(np.mean(degrees)),
    )

    return adj_matrix, laplacian, metrics


def compute_flocking_acceleration(
    agent_idx: int,
    missiles: List[MissileState],
    comm_range_m: float = 12000.0,
    safe_separation_m: float = 250.0,
    w_separation: float = 2.5,
    w_cohesion: float = 0.5,
    w_alignment: float = 1.0,
) -> float:
    """
    Computes the decentralized Reynolds Flocking acceleration for an agent.

    Because a Dubins missile only has lateral fin acceleration (cannot instantly change speed),
    the desired 2D force vector F_des is projected onto the body-normal axis:
        n_body = [-sin(theta), cos(theta)]
        a_flock = F_des . n_body

    Rules:
    1. Separation: Repel neighbors closer than safe_separation_m (collision avoidance).
    2. Cohesion: Steer toward the center of mass of 1-hop network neighbors.
    3. Alignment: Steer to match velocity heading with 1-hop network neighbors.
    """
    ego = missiles[agent_idx]
    neighbors = [
        (idx, m) for idx, m in enumerate(missiles)
        if idx != agent_idx and math.hypot(m.x - ego.x, m.y - ego.y) <= comm_range_m
    ]

    if not neighbors:
        return 0.0

    force_sep_x, force_sep_y = 0.0, 0.0
    center_x, center_y = 0.0, 0.0
    avg_vx, avg_vy = 0.0, 0.0

    for _, n_state in neighbors:
        dx = n_state.x - ego.x
        dy = n_state.y - ego.y
        dist = max(math.hypot(dx, dy), 1.0)

        # 1. Separation force (inverse square repulsion if within safety bubble)
        if dist < safe_separation_m:
            repel_mag = (safe_separation_m - dist) / dist
            force_sep_x -= (dx / dist) * repel_mag * 100.0
            force_sep_y -= (dy / dist) * repel_mag * 100.0

        # Accumulate for cohesion and alignment
        center_x += n_state.x
        center_y += n_state.y
        avg_vx += n_state.vx
        avg_vy += n_state.vy

    num_k = len(neighbors)
    center_x /= num_k
    center_y /= num_k
    avg_vx /= num_k
    avg_vy /= num_k

    # 2. Cohesion vector: toward neighbor center of mass
    vec_coh_x = (center_x - ego.x) / max(comm_range_m, 1.0)
    vec_coh_y = (center_y - ego.y) / max(comm_range_m, 1.0)

    # 3. Alignment vector: velocity mismatch
    vec_ali_x = (avg_vx - ego.vx) / max(ego.velocity, 1.0)
    vec_ali_y = (avg_vy - ego.vy) / max(ego.velocity, 1.0)

    # Composite desired lateral force vector
    f_des_x = (w_separation * force_sep_x) + (w_cohesion * vec_coh_x * 50.0) + (w_alignment * vec_ali_x * 20.0)
    f_des_y = (w_separation * force_sep_y) + (w_cohesion * vec_coh_y * 50.0) + (w_alignment * vec_ali_y * 20.0)

    # Project onto normal unit vector: n = [-sin(theta), cos(theta)]
    n_x = -math.sin(ego.heading)
    n_y = math.cos(ego.heading)

    a_flock = (f_des_x * n_x) + (f_des_y * n_y)
    return a_flock


def simulate_cooperative_swarm_salvo(
    initial_swarm: List[MissileState],
    targets: List[TargetState],
    target_assignments: List[int],  # e.g. [0, 0, 1, 2] -> which target index each missile tracks
    nav_constant: float = 3.5,
    comm_range_m: float = 12000.0,
    safe_separation_m: float = 250.0,
    dt: float = 0.05,
    max_time_s: float = 30.0,
    lethal_radius_m: float = 10.0,
) -> Dict[str, Any]:
    """
    Simulates a coordinated multi-missile salvo where agents fly toward assigned targets
    while simultaneously executing decentralized flocking to maintain mesh connectivity
    and avoid inter-agent collisions.
    """
    current_missiles = list(initial_swarm)
    current_targets = list(targets)
    num_missiles = len(current_missiles)

    active_missiles = [True] * num_missiles
    intercept_times = [None] * num_missiles

    time_elapsed = 0.0
    min_observed_dist = float("inf")
    fiedler_history: List[float] = []
    prev_ranges: List[float] = [float("inf")] * num_missiles

    while time_elapsed < max_time_s and any(active_missiles):
        # 1. Update targets (constant velocity motion)
        current_targets = [
            TargetState(
                target_id=t.target_id,
                x=t.x + t.vx * dt,
                y=t.y + t.vy * dt,
                vx=t.vx,
                vy=t.vy,
                priority=t.priority,
            )
            for t in current_targets
        ]

        # 2. Check swarm graph topology & connectivity
        living_missiles = [m for idx, m in enumerate(current_missiles) if active_missiles[idx]]
        if len(living_missiles) >= 2:
            _, _, metrics = compute_swarm_graph_laplacian(living_missiles, comm_range_m)
            fiedler_history.append(metrics.fiedler_value)
            if metrics.min_inter_agent_dist < min_observed_dist:
                min_observed_dist = metrics.min_inter_agent_dist
        else:
            fiedler_history.append(0.0)

        # 3. Compute hybrid guidance commands for each active missile
        new_states: List[MissileState] = []
        for i in range(num_missiles):
            if not active_missiles[i]:
                new_states.append(current_missiles[i])
                continue

            ego = current_missiles[i]
            assigned_target = current_targets[target_assignments[i]]

            # A. Compute target intercept guidance (Classical PN)
            range_m, v_c, los_rate = calculate_los_geometry(ego, assigned_target)

            if range_m <= lethal_radius_m or (v_c <= 0.0 and range_m < prev_ranges[i] and range_m <= lethal_radius_m * 1.5):
                active_missiles[i] = False
                intercept_times[i] = time_elapsed
                logger.info(
                    f"Missile M{i} intercepted Target T{assigned_target.target_id} "
                    f"at t={time_elapsed:.2f}s! (Range: {range_m:.2f}m)"
                )
                new_states.append(ego)
                continue

            prev_ranges[i] = range_m

            a_pn = proportional_navigation_law(
                nav_constant=nav_constant,
                closing_velocity=v_c,
                los_rate=los_rate,
                max_lateral_accel=ego.max_lateral_accel,
            )

            # B. Compute decentralized flocking acceleration
            a_flock = compute_flocking_acceleration(
                agent_idx=i,
                missiles=current_missiles,
                comm_range_m=comm_range_m,
                safe_separation_m=safe_separation_m,
            )

            # Weighting: As range closes (< 3km), guidance dominates and flocking diminishes
            flock_weight = min(1.0, max(0.0, (range_m - 2000.0) / 4000.0))
            a_total_raw = a_pn + (flock_weight * a_flock)

            # Clamp total demand to physical fin acceleration
            a_total = max(-ego.max_lateral_accel, min(ego.max_lateral_accel, a_total_raw))

            # Step kinematics
            updated_state = step_kinematics(ego, a_total, dt)
            new_states.append(updated_state)

        current_missiles = new_states
        time_elapsed += dt

    all_intercepted = all(t_int is not None for t_int in intercept_times)
    mean_fiedler = float(np.mean(fiedler_history)) if fiedler_history else 0.0

    return {
        "all_intercepted": all_intercepted,
        "intercept_times": intercept_times,
        "min_separation_m": min_observed_dist,
        "mean_fiedler_connectivity": mean_fiedler,
        "total_sim_time_s": time_elapsed,
    }


if __name__ == "__main__":
    logger.info("Initializing 4-Missile Swarm Salvo Simulation...")

    # Swarm of 4 missiles launched in staggered formation
    # Flying East at 500 m/s (~Mach 1.5)
    swarm = [
        MissileState(x=0.0, y=0.0, heading=0.0, velocity=500.0, max_lateral_accel=25.0 * G_ACCEL),
        MissileState(x=-150.0, y=300.0, heading=0.0, velocity=500.0, max_lateral_accel=25.0 * G_ACCEL),
        MissileState(x=-150.0, y=-300.0, heading=0.0, velocity=500.0, max_lateral_accel=25.0 * G_ACCEL),
        MissileState(x=-300.0, y=0.0, heading=0.0, velocity=500.0, max_lateral_accel=25.0 * G_ACCEL),
    ]

    # 3 Separate Targets 10 km downrange
    # Target 0: (10 km East, 1 km North) - High Priority SAM
    # Target 1: (10 km East, 4 km North) - Command Vehicle
    # Target 2: (10 km East, -3 km South) - Mobile Launcher
    target_list = [
        TargetState(target_id=0, x=10000.0, y=1000.0, vx=0.0, vy=30.0, priority=10.0),
        TargetState(target_id=1, x=10000.0, y=4000.0, vx=0.0, vy=-20.0, priority=6.0),
        TargetState(target_id=2, x=10000.0, y=-3000.0, vx=0.0, vy=15.0, priority=4.0),
    ]

    # Target assignment:
    # M0 -> T0 (SAM)
    # M1 -> T1 (Command vehicle)
    # M2 -> T2 (Mobile launcher)
    # M3 -> T0 (Reinforces SAM for guaranteed kill)
    assignments = [0, 1, 2, 0]

    results = simulate_cooperative_swarm_salvo(
        initial_swarm=swarm,
        targets=target_list,
        target_assignments=assignments,
        comm_range_m=12000.0,
        safe_separation_m=200.0,
    )

    print("\n================ SALVO SIMULATION AUDIT ================")
    print(f"All Targets Intercepted:      {results['all_intercepted']}")
    print(f"Individual Intercept Times:   {[f'{t:.2f}s' if t else 'Miss' for t in results['intercept_times']]}")
    print(f"Minimum Inter-Missile Dist:   {results['min_separation_m']:.2f} m (Safety threshold: 200 m)")
    print(f"Mean Graph Fiedler Value lambda_2: {results['mean_fiedler_connectivity']:.3f} (Connected: > 0)")
    print(f"Total Engagement Duration:    {results['total_sim_time_s']:.2f} s")
    print("========================================================\n")
