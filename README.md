# Distributed Cooperative Swarm Guidance (DCSG)
### Learning-Based Multi-Target Allocation with Graph Topology Preservation

**Author:** Nunna Mokshith | B.Tech Robotics & AI  
**Domain:** Multi-Agent Robotics, Graph Neural Networks (GNN), Guidance, Navigation & Control (GNC), Reinforcement Learning (MARL)

> **Base System (`base` branch):**  
> This branch hosts the standalone single-file core implementation ([`main.py`](./main.py) / [`run_simulation.py`](./run_simulation.py)) featuring real-time decentralized GAT target allocation, fail-safe surplus self-destruction, and tactical telemetry. Run with `py main.py` or `py main.py --no-gui`.

---

## 🚀 Project Overview

Traditional missile salvos and drone swarms suffer from **coordination collapse**:
1. Without a centralized command link, independent seekers gravitate toward the highest-signature radar target, causing massive **overkill** while leaving secondary threats alive.
2. Centralized datalinks represent a single point of failure vulnerable to Electronic Warfare (EW) jamming and packet loss.

This project implements a **Hierarchical Hybrid Autonomous Swarm**:
- **Tactical Layer (Machine Learning):** A decentralized **Graph Attention Network (GAT)** trained via **Multi-Agent Reinforcement Learning (MAPPO)** that negotiates target allocations over an ad-hoc radio mesh in $< 2\text{ ms}$.
- **Consensus Layer (Graph Theory):** Monitors the **Graph Laplacian ($L = D - A$)** and tracks algebraic connectivity ($\lambda_2(L) > 0$) to preserve communication mesh integrity while computing **Reynolds flocking forces** for collision avoidance.
- **Guidance Layer (Robotics & Control):** Integrates non-holonomic **2D Dubins unicycle kinematics** and **Pure Proportional Navigation (PPN)** with physical fin acceleration constraints ($\pm 25G$).

---

## 🏛️ System Architecture

```
                                  TACTICAL LAYER (AI)
                        ┌─────────────────────────────────────┐
                        │   Graph Attention Network (GAT)     │
                        │   - 1-hop message passing           │
                        │   - Permutation-invariant utility   │
                        │   - CTDE Actor-Critic Policy        │
                        └──────────────────┬──────────────────┘
                                           │ Target Assignments [T_k]
                                           ▼
                                 CONSENSUS LAYER (GRAPH)
                        ┌─────────────────────────────────────┐
                        │   Dynamic Communication Graph       │
                        │   - Adjacency A(t), Laplacian L(t)  │
                        │   - Fiedler Eigenvalue λ_2(L) > 0   │
                        │   - Reynolds Flocking (Sep/Coh/Ali) │
                        └──────────────────┬──────────────────┘
                                           │ a_flock demand
                                           ▼
                                 GUIDANCE LAYER (PHYSICS)
                        ┌─────────────────────────────────────┐
                        │   Proportional Navigation (PN Law)  │
                        │   - a_cmd = N' * V_c * d_lambda/dt  │
                        │   - Clamped to fin limits (±25G)    │
                        │   - Dubins Kinematic Integration    │
                        └─────────────────────────────────────┘
```

---

## 📂 Codebase Modules

| Module | File | Core Concept / Technology |
| :--- | :--- | :--- |
| **Module 1** | [`step01_kinematics_and_guidance.py`](./step01_kinematics_and_guidance.py) | Non-holonomic Dubins physics, Line-of-Sight rate $\dot{\lambda}$, Proportional Navigation (PN) law. |
| **Module 2** | [`step02_swarm_graph_and_flocking.py`](./step02_swarm_graph_and_flocking.py) | Dynamic ad-hoc mesh, Graph Laplacian, Fiedler eigenvalue $\lambda_2$, Reynolds flocking forces. |
| **Module 3** | [`step03_gnn_target_allocation.py`](./step03_gnn_target_allocation.py) | PyTorch Graph Attention Network (GAT), dynamic agent resizing during mid-air attrition ($N=4 \to 3$). |
| **Module 4** | [`step04_train_mappo.py`](./step04_train_mappo.py) | Multi-Agent PPO training loop, multi-objective reward function (coverage + priority - overkill). |
| **Module 5** | [`step05_visualizer_and_benchmarks.py`](./step05_visualizer_and_benchmarks.py) | 4-panel telemetry dashboard comparing uncoordinated vs. coordinated salvo dynamics. |

---

## 🔬 Mathematical Formulations

### 1. Dubins Vehicle Kinematics
$$\dot{x} = v \cos(\theta), \quad \dot{y} = v \sin(\theta), \quad \dot{\theta} = \frac{a_n}{v}$$

### 2. Line-of-Sight Geometry & Pure Proportional Navigation (PPN)
Closing velocity:
$$V_c = -\frac{\Delta x \Delta v_x + \Delta y \Delta v_y}{R}$$
Line-of-Sight angular rate:
$$\dot{\lambda} = \frac{\Delta x \Delta v_y - \Delta y \Delta v_x}{R^2}$$
Autopilot fin command:
$$a_{\text{cmd}} = \text{clamp}\left(N' \cdot V_c \cdot \dot{\lambda}, -a_{\max}, +a_{\max}\right)$$

### 3. Graph Laplacian & Algebraic Connectivity
$$A_{ij} = \begin{cases} 1 & \text{if } 0 < \|\mathbf{p}_i - \mathbf{p}_j\| \le R_{\text{comm}} \\ 0 & \text{otherwise} \end{cases}$$
$$L = D - A, \quad \lambda_2(L) = \min_{\mathbf{x} \perp \mathbf{1}, \|\mathbf{x}\|=1} \mathbf{x}^T L \mathbf{x}$$
Connectivity holds if and only if $\lambda_2(L) > 0$.

### 4. Graph Attention (GAT) Node Aggregation
$$\alpha_{ij} = \frac{\exp\left(\text{LeakyReLU}\left(\mathbf{a}^T [\mathbf{W}h_i \parallel \mathbf{W}h_j]\right)\right)}{\sum_{k \in \mathcal{N}_i \cup \{i\}} \exp\left(\text{LeakyReLU}\left(\mathbf{a}^T [\mathbf{W}h_i \parallel \mathbf{W}h_k]\right)\right)}$$
$$h_i' = \sigma\left(\sum_{j \in \mathcal{N}_i \cup \{i\}} \alpha_{ij} \mathbf{W} h_j\right)$$

---

## 📊 Telemetry & Benchmark Results

Run the telemetry visualization engine:
```powershell
python step05_visualizer_and_benchmarks.py
```

The output figure is saved to:
`figures/fig4_telemetry_dashboard.png` (and `swarm_engagement_telemetry.png`)

### Key Results:
- **Simultaneous Impact Synchronization:** All 4 missiles intercept their respective targets within **$1.25\text{ seconds}$** of each other across an engagement distance of $10\text{ km}$.
- **Collision Safety:** Minimum inter-agent separation remained at **$295.36\text{ m}$**, never breaching the $200\text{ m}$ safety boundary.
- **Mesh Integrity:** The Fiedler eigenvalue remained strictly positive ($\lambda_2 \approx 3.86$) until the intentional terminal dive phase.
- **Fin Actuation Feasibility:** Peak lateral acceleration remained under **$3.5G$**, well below the physical airframe limit of $25G$.

---

## 💼 Resume & Interview Talking Points

- **Robotics & Control:** Implemented non-holonomic Dubins unicycle dynamics and classical Proportional Navigation (PN) guidance with aerodynamic fin saturation constraints.
- **Distributed Systems:** Applied spectral graph theory (Graph Laplacian, Fiedler value $\lambda_2$) and Reynolds flocking rules to maintain multi-agent mesh connectivity during dynamic formation bifurcation.
- **Deep Learning / MARL:** Built a permutation-invariant Graph Attention Network (GAT) actor-critic architecture in PyTorch, handling mid-air agent attrition ($N \to N-1$) without dimensionality errors.
- **Performance:** Formulated target allocation as an amortized neural policy achieving sub-$2\text{ ms}$ inference on embedded edge platforms, outperforming traditional Mixed-Integer Linear Programming (MILP) solvers under strict real-time deadlines.
