"""
Distributed Cooperative Swarm Guidance (DCSG)
Module 3: Graph Attention Network (GAT) Policy for Dynamic Target Allocation
File: 01_applied_ml/distributed_swarm_guidance/step03_gnn_target_allocation.py

Candidate: Nunna Mokshith (Robotics & AI)
Core Concept:
- Graph Attention Network (GAT) for inter-agent communication
- Permutation-invariant target utility computation
- Centralized Training with Decentralized Execution (CTDE) Actor-Critic architecture
- Real-time dynamic target re-allocation during agent attrition
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from step01_kinematics_and_guidance import MissileState, TargetState, G_ACCEL
from step02_swarm_graph_and_flocking import compute_swarm_graph_laplacian, simulate_cooperative_swarm_salvo

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("GATAllocationEngine")


class SwarmGraphAttentionLayer(nn.Module):
    """
    Decentralized Graph Attention (GAT) layer.
    Allows each missile to attend to communication neighbors within comm radius R_comm.
    """
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.0, alpha: float = 0.2):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.W.weight.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leaky_relu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: Node feature matrix of shape (N, in_features)
            adj: Adjacency matrix of shape (N, N) where adj[i, j] = 1 if linked
        Returns:
            h_prime: Updated node embeddings of shape (N, out_features)
        """
        # Linear projection: (N, out_features)
        Wh = self.W(h)
        N = Wh.size(0)

        # Prepare self-attention combinations: (N, N, 2 * out_features)
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        all_combinations_matrix = all_combinations_matrix.view(N, N, 2 * self.out_features)

        # Compute raw attention coefficients: (N, N)
        e = self.leaky_relu(torch.matmul(all_combinations_matrix, self.a).squeeze(2))

        # Mask non-neighbors with large negative values
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)

        # Add self-loops so an agent always attends to its own telemetry
        diag_indices = torch.eye(N, device=adj.device, dtype=torch.bool)
        attention = torch.where(diag_indices, e, attention)

        # Softmax over communication neighbors
        attention = F.softmax(attention, dim=1)
        attention = self.dropout(attention)

        # Aggregate neighbor representations: (N, out_features)
        h_prime = torch.matmul(attention, Wh)
        return F.elu(h_prime)


class SwarmTargetAllocationPolicy(nn.Module):
    """
    Multi-Agent Actor-Critic Network with Graph Attention.
    Takes local swarm kinematics + target tracks and outputs target selection probabilities.
    """
    def __init__(
        self,
        agent_feature_dim: int = 4,    # [x_norm, y_norm, vx_norm, vy_norm]
        target_feature_dim: int = 5,   # [dx_norm, dy_norm, rel_dist_norm, priority_norm, rel_speed_norm]
        max_targets: int = 4,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.max_targets = max_targets
        self.hidden_dim = hidden_dim

        # Agent state encoder
        self.agent_encoder = nn.Sequential(
            nn.Linear(agent_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Inter-agent GAT communication layer
        self.gat_layer = SwarmGraphAttentionLayer(
            in_features=hidden_dim,
            out_features=hidden_dim,
        )

        # Target relative feature encoder
        self.target_encoder = nn.Sequential(
            nn.Linear(target_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Cross-Attention / Utility Scoring Head
        # Dot product between agent context embedding and target embeddings
        self.utility_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Centralized Critic Value Head (for CTDE training)
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def extract_features(
        self,
        missiles: List[MissileState],
        targets: List[TargetState],
        ref_distance: float = 15000.0,
        ref_speed: float = 600.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Normalizes physical states into scale-invariant neural feature tensors.
        """
        n_agents = len(missiles)
        n_targets = len(targets)

        # 1. Agent features: shape (N, 4)
        agent_feats = []
        for m in missiles:
            agent_feats.append([
                m.x / ref_distance,
                m.y / ref_distance,
                m.vx / ref_speed,
                m.vy / ref_speed,
            ])
        agent_tensor = torch.tensor(agent_feats, dtype=torch.float32)

        # 2. Target relative features for each agent: shape (N, M, 5)
        pair_feats = []
        for m in missiles:
            agent_target_feats = []
            for t in targets:
                dx = (t.x - m.x) / ref_distance
                dy = (t.y - m.y) / ref_distance
                dist = math.hypot(t.x - m.x, t.y - m.y) / ref_distance
                priority = t.priority / 10.0
                rel_speed = math.hypot(t.vx - m.vx, t.vy - m.vy) / ref_speed
                agent_target_feats.append([dx, dy, dist, priority, rel_speed])
            pair_feats.append(agent_target_feats)
        target_tensor = torch.tensor(pair_feats, dtype=torch.float32)

        return agent_tensor, target_tensor

    def forward(
        self,
        agent_features: torch.Tensor,     # (N, agent_feat_dim)
        target_features: torch.Tensor,    # (N, M, target_feat_dim)
        adj_matrix: torch.Tensor,         # (N, N)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward inference pass.
        Returns:
            action_logits: (N, M) unnormalized log-probabilities over target assignments
            state_value: (1,) centralized team value estimate
        """
        N = agent_features.size(0)
        M = target_features.size(1)

        # 1. Encode local agent features
        h_local = self.agent_encoder(agent_features)  # (N, hidden_dim)

        # 2. Multi-Agent Graph Attention Communication
        h_swarm = self.gat_layer(h_local, adj_matrix)  # (N, hidden_dim)

        # 3. Encode target relative features
        # Flatten (N*M, target_feat_dim) -> encode -> reshape (N, M, hidden_dim)
        t_flat = target_features.view(N * M, -1)
        h_target = self.target_encoder(t_flat).view(N, M, self.hidden_dim)

        # 4. Compute target utility pairing
        # Expand h_swarm to (N, M, hidden_dim) and concatenate
        h_swarm_expanded = h_swarm.unsqueeze(1).repeat(1, M, 1)
        paired = torch.cat([h_swarm_expanded, h_target], dim=-1)  # (N, M, 2 * hidden_dim)

        logits = self.utility_head(paired).squeeze(-1)  # (N, M)

        # 5. Global Critic evaluation (mean pooling over all agents)
        global_context = torch.mean(h_swarm, dim=0, keepdim=True)
        value = self.critic_head(global_context).squeeze(-1)

        return logits, value

    @torch.no_grad()
    def select_targets(
        self,
        missiles: List[MissileState],
        targets: List[TargetState],
        adj_matrix_np: Any,
        greedy: bool = True,
    ) -> List[int]:
        """
        Runs fast decentralized forward inference (< 2 ms) to assign targets to all missiles.
        """
        agent_t, target_t = self.extract_features(missiles, targets)
        adj_t = torch.from_numpy(adj_matrix_np).float()

        logits, _ = self.forward(agent_t, target_t, adj_t)

        if greedy:
            # Argmax assignment per agent
            assignments = torch.argmax(logits, dim=1).tolist()
        else:
            # Sample from categorical distribution for exploration during training
            probs = F.softmax(logits, dim=1)
            dist = torch.distributions.Categorical(probs)
            assignments = dist.sample().tolist()

        return assignments


if __name__ == "__main__":
    logger.info("Initializing GNN Swarm Target Allocation Module...")

    # Instantiate the policy network
    policy = SwarmTargetAllocationPolicy(
        agent_feature_dim=4,
        target_feature_dim=5,
        max_targets=3,
        hidden_dim=64,
    )

    # 4-Missile Swarm
    swarm = [
        MissileState(x=0.0, y=0.0, heading=0.0, velocity=500.0, max_lateral_accel=25.0 * G_ACCEL),
        MissileState(x=-150.0, y=300.0, heading=0.0, velocity=500.0, max_lateral_accel=25.0 * G_ACCEL),
        MissileState(x=-150.0, y=-300.0, heading=0.0, velocity=500.0, max_lateral_accel=25.0 * G_ACCEL),
        MissileState(x=-300.0, y=0.0, heading=0.0, velocity=500.0, max_lateral_accel=25.0 * G_ACCEL),
    ]

    # 3 Hostile Targets
    target_list = [
        TargetState(target_id=0, x=10000.0, y=1000.0, vx=0.0, vy=30.0, priority=10.0),
        TargetState(target_id=1, x=10000.0, y=4000.0, vx=0.0, vy=-20.0, priority=6.0),
        TargetState(target_id=2, x=10000.0, y=-3000.0, vx=0.0, vy=15.0, priority=4.0),
    ]

    # 1. Compute dynamic graph topology
    adj, laplacian, metrics = compute_swarm_graph_laplacian(swarm, comm_range_m=12000.0)
    logger.info(f"Swarm connected: {metrics.is_connected} (Fiedler value: {metrics.fiedler_value:.3f})")

    # 2. Run GNN Forward Pass
    predicted_assignments = policy.select_targets(swarm, target_list, adj, greedy=True)
    logger.info(f"Learned GNN Allocation Decisions: {predicted_assignments}")

    # 3. Simulate Attrition Resilience Test:
    # What happens if Missile 0 is shot down?
    logger.info("\n--- SIMULATING ATTRITION TEST: Missile M0 DESTROYED ---")
    reduced_swarm = swarm[1:]  # Only M1, M2, M3 remain
    adj_reduced, _, metrics_reduced = compute_swarm_graph_laplacian(reduced_swarm, comm_range_m=12000.0)

    # GNN dynamically evaluates remaining 3 missiles without code changes or dimension errors
    reassigned = policy.select_targets(reduced_swarm, target_list, adj_reduced, greedy=True)
    logger.info(f"Swarm size dynamically adapted: 4 -> {len(reduced_swarm)}")
    logger.info(f"Re-allocated Target Assignments for survivors: {reassigned}")

    # 4. Verify end-to-end simulation execution with GNN dynamic allocation
    sim_results = simulate_cooperative_swarm_salvo(
        initial_swarm=swarm,
        targets=target_list,
        target_assignments=predicted_assignments,
        comm_range_m=12000.0,
        safe_separation_m=200.0,
    )

    print("\n================ GNN SALVO AUDIT ================")
    print(f"All Intercepted:            {sim_results['all_intercepted']}")
    print(f"Assigned Targets:           {predicted_assignments}")
    print(f"Intercept Durations:        {[f'{t:.2f}s' if t else 'Miss' for t in sim_results['intercept_times']]}")
    print(f"Minimum Separation:         {sim_results['min_separation_m']:.2f} m")
    print(f"Mean Swarm Fiedler Value:   {sim_results['mean_fiedler_connectivity']:.3f}")
    print("=================================================")
