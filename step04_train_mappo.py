"""
Distributed Cooperative Swarm Guidance (DCSG)
Module 4: Multi-Agent Reinforcement Learning (PPO) Training Engine
File: 01_applied_ml/distributed_swarm_guidance/step04_train_mappo.py

Candidate: Nunna Mokshith (Robotics & AI)
Core Concept:
- Multi-Agent PPO with Centralized Training and Decentralized Execution (CTDE)
- Multi-objective reward shaping: (Target coverage + Priority weighting - Overkill penalty - Collision risk)
- Trains the Graph Attention Network (GAT) to dynamically allocate targets across the swarm
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
import math
import random
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from step01_kinematics_and_guidance import MissileState, TargetState, G_ACCEL
from step02_swarm_graph_and_flocking import compute_swarm_graph_laplacian, simulate_cooperative_swarm_salvo
from step03_gnn_target_allocation import SwarmTargetAllocationPolicy

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("MAPPOTrainer")


class SwarmEngagementEnvironment:
    """
    Simulation environment for training the GNN tactical allocation policy.
    Generates varied tactical scenarios with moving targets and evaluates team performance.
    """
    def __init__(self, num_missiles: int = 4, num_targets: int = 3):
        self.num_missiles = num_missiles
        self.num_targets = num_targets

    def reset(self) -> Tuple[List[MissileState], List[TargetState]]:
        """Generates a randomized tactical engagement scenario."""
        # 1. Swarm launch with slight dispersion
        missiles = []
        for i in range(self.num_missiles):
            offset_y = (i - (self.num_missiles - 1) / 2.0) * 300.0 + random.uniform(-50.0, 50.0)
            offset_x = -abs(offset_y) * 0.5 + random.uniform(-20.0, 20.0)
            missiles.append(
                MissileState(
                    x=offset_x,
                    y=offset_y,
                    heading=random.uniform(-0.05, 0.05),
                    velocity=500.0,
                    max_lateral_accel=25.0 * G_ACCEL,
                )
            )

        # 2. Hostile Targets 9-12 km downrange
        targets = []
        priorities = [10.0, 6.0, 4.0]
        y_positions = [1000.0, 4000.0, -3500.0]

        for j in range(self.num_targets):
            t_x = random.uniform(9500.0, 11000.0)
            t_y = y_positions[j] + random.uniform(-500.0, 500.0)
            vy = random.uniform(-25.0, 25.0)
            targets.append(
                TargetState(
                    target_id=j,
                    x=t_x,
                    y=t_y,
                    vx=0.0,
                    vy=vy,
                    priority=priorities[j],
                )
            )

        return missiles, targets

    def evaluate_allocation(
        self,
        missiles: List[MissileState],
        targets: List[TargetState],
        assignments: List[int],
    ) -> Tuple[float, Dict[str, float]]:
        """
        Computes the tactical reward for a given assignment across the swarm.

        Reward Components:
        - Target Coverage: Reward each unique target assigned (+30 per unique target)
        - Priority Weighting: Bonus for assigning missiles to high-priority targets
        - Overkill Penalty: Severe penalty if any target receives 3+ missiles while another target has 0
        - Physical Intercept Simulation: Reward confirmed intercepts from flight physics
        """
        # Count allocations per target
        target_counts = {t.target_id: 0 for t in targets}
        for a in assignments:
            if a in target_counts:
                target_counts[a] += 1

        # 1. Coverage reward
        num_covered = sum(1 for count in target_counts.values() if count > 0)
        r_coverage = (num_covered / self.num_targets) * 60.0

        # 2. Priority alignment reward
        r_priority = 0.0
        for t in targets:
            count = target_counts[t.target_id]
            if count == 1:
                r_priority += t.priority * 3.0
            elif count == 2 and t.priority >= 8.0:
                # High priority SAM justifies a 2-missile salvo
                r_priority += t.priority * 4.0

        # 3. Overkill penalty: penalize leaving targets uncovered
        uncovered = sum(1 for count in target_counts.values() if count == 0)
        r_overkill = -uncovered * 40.0

        # 4. Fast physics check
        sim = simulate_cooperative_swarm_salvo(
            initial_swarm=missiles,
            targets=targets,
            target_assignments=assignments,
            comm_range_m=12000.0,
            safe_separation_m=200.0,
            lethal_radius_m=10.0,
        )
        r_intercept = 50.0 if sim["all_intercepted"] else -20.0
        r_connectivity = min(sim["mean_fiedler_connectivity"], 3.0) * 5.0

        total_reward = r_coverage + r_priority + r_overkill + r_intercept + r_connectivity

        breakdown = {
            "coverage": r_coverage,
            "priority": r_priority,
            "overkill_penalty": r_overkill,
            "intercept": r_intercept,
            "total": total_reward,
        }
        return total_reward, breakdown


def train_swarm_policy(
    num_epochs: int = 50,
    episodes_per_epoch: int = 16,
    lr: float = 3e-4,
    clip_epsilon: float = 0.2,
    entropy_coeff: float = 0.02,
) -> SwarmTargetAllocationPolicy:
    """
    Trains the SwarmTargetAllocationPolicy using Policy Gradients (REINFORCE / PPO).
    """
    env = SwarmEngagementEnvironment(num_missiles=4, num_targets=3)
    policy = SwarmTargetAllocationPolicy(
        agent_feature_dim=4,
        target_feature_dim=5,
        max_targets=3,
        hidden_dim=64,
    )
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    logger.info("Beginning Multi-Agent Policy Gradient Training...")

    for epoch in range(1, num_epochs + 1):
        batch_log_probs = []
        batch_rewards = []
        batch_values = []
        epoch_rewards = []
        overkill_penalties = []

        for ep in range(episodes_per_epoch):
            missiles, targets = env.reset()
            adj, _, _ = compute_swarm_graph_laplacian(missiles, comm_range_m=12000.0)

            # Feature extraction
            agent_t, target_t = policy.extract_features(missiles, targets)
            adj_t = torch.from_numpy(adj).float()

            # Forward pass
            logits, value = policy(agent_t, target_t, adj_t)
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)

            # Sample action for all 4 missiles
            actions = dist.sample()
            action_log_prob = dist.log_prob(actions).sum()  # Joint swarm action log-prob

            # Step environment
            assignments = actions.tolist()
            reward, breakdown = env.evaluate_allocation(missiles, targets, assignments)

            batch_log_probs.append(action_log_prob)
            batch_rewards.append(reward)
            batch_values.append(value)
            epoch_rewards.append(reward)
            overkill_penalties.append(breakdown["overkill_penalty"])

        # Compute Advantage
        rewards_t = torch.tensor(batch_rewards, dtype=torch.float32)
        values_t = torch.cat(batch_values).squeeze(-1)
        # Normalize returns for stable gradients
        baseline = values_t.detach()
        advantages = rewards_t - baseline
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Policy & Value Loss
        log_probs_t = torch.stack(batch_log_probs)
        policy_loss = -(log_probs_t * advantages).mean()
        value_loss = F.mse_loss(values_t, rewards_t)
        total_loss = policy_loss + 0.5 * value_loss

        optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        optimizer.step()

        if epoch % 10 == 0 or epoch == 1:
            mean_r = sum(epoch_rewards) / len(epoch_rewards)
            mean_overkill = sum(overkill_penalties) / len(overkill_penalties)
            logger.info(
                f"Epoch {epoch:02d}/{num_epochs:02d} | "
                f"Mean Reward: {mean_r:6.1f} | "
                f"Overkill Penalty: {mean_overkill:5.1f} | "
                f"Loss: {total_loss.item():.4f}"
            )

    logger.info("Training complete! Saving trained policy weights...")
    torch.save(policy.state_dict(), "01_applied_ml/distributed_swarm_guidance/trained_swarm_policy.pt")
    return policy


if __name__ == "__main__":
    # Train the policy over 40 epochs
    trained_policy = train_swarm_policy(num_epochs=40, episodes_per_epoch=12, lr=5e-4)

    # Test the trained policy on a fresh scenario
    env = SwarmEngagementEnvironment(num_missiles=4, num_targets=3)
    test_missiles, test_targets = env.reset()
    adj, _, _ = compute_swarm_graph_laplacian(test_missiles, comm_range_m=12000.0)

    trained_assignments = trained_policy.select_targets(test_missiles, test_targets, adj, greedy=True)
    reward, breakdown = env.evaluate_allocation(test_missiles, test_targets, trained_assignments)

    print("\n================ TRAINED POLICY TEST EVALUATION ================")
    print(f"Trained Allocation Decisions: {trained_assignments}")
    print(f"Target Coverage Score:        {breakdown['coverage']:.1f}")
    print(f"Priority Score:               {breakdown['priority']:.1f}")
    print(f"Overkill Penalty:             {breakdown['overkill_penalty']:.1f} (0.0 means perfect distribution)")
    print(f"Total Tactical Reward:        {reward:.1f}")
    print("=================================================================\n")
