"""
Generates a dark-themed syntax-highlighted code snippet image for the PowerPoint deck.
File: generate_code_card.py
"""
import os
import matplotlib.pyplot as plt

def generate_code_image():
    fig, ax = plt.subplots(figsize=(10, 6.2), dpi=200)
    fig.patch.set_facecolor("#030712")
    ax.set_facecolor("#0a0f1d")

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#1e293b")

    # Header bar
    header_rect = plt.Rectangle((0, 0.92), 1, 0.08, transform=ax.transAxes, color="#0f172a", zorder=2)
    ax.add_patch(header_rect)
    ax.scatter([0.025, 0.045, 0.065], [0.96, 0.96, 0.96], color=["#ef4444", "#f59e0b", "#10b981"], s=45, transform=ax.transAxes, zorder=3)
    ax.text(0.5, 0.96, "PyTorch Core: SwarmGATLayer & Policy Gradient Loss", color="#94a3b8", fontsize=10,
            family="monospace", ha="center", va="center", transform=ax.transAxes, zorder=3)

    code_text = """
# 1. GRAPH ATTENTION LAYER WITH RESIDUAL CONNECTION (Prevents Over-Smoothing)
class SwarmGATLayer(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.W_self = nn.Linear(in_dim, hidden_dim)          # Local identity
        self.W_neigh = nn.Linear(in_dim, hidden_dim, bias=False)  # Neighbor projection
        self.attn = nn.Linear(2 * hidden_dim, 1, bias=False) # Attention scoring

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h_self = self.W_self(x)
        h_neigh = self.W_neigh(x)
        
        # Pairwise attention scoring between connected agents
        h_i = h_neigh.unsqueeze(1).repeat(1, N, 1)
        h_j = h_neigh.unsqueeze(0).repeat(N, 1, 1)
        scores = F.leaky_relu(self.attn(torch.cat([h_i, h_j], dim=-1)).squeeze(-1))
        scores = scores.masked_fill(adj == 0, -1e9)          # Datalink masking
        alpha = F.softmax(scores, dim=-1)                    # Attention weights α_ij
        
        h_agg = torch.matmul(alpha, h_neigh)                 # Context aggregation
        return torch.cat([h_self, h_agg], dim=-1)            # Residual fusion

# 2. REINFORCE POLICY GRADIENT UPDATE WITH RUNNING BASELINE
reward = compute_reward(actions, priorities, distances, overkill_weight=30.0)
baseline = 0.9 * baseline + 0.1 * reward                    # Moving baseline
advantage = reward - baseline                                # Advantage A = R - b
loss = -dist.log_prob(actions).sum() * advantage             # Policy gradient loss
loss.backward(); optimizer.step()
"""

    ax.text(0.03, 0.88, code_text.strip(), color="#4ade80", fontsize=8.2, family="monospace",
            va="top", transform=ax.transAxes, linespacing=1.28)

    os.makedirs("figures", exist_ok=True)
    out_file = os.path.join("figures", "code_gat_architecture.png")
    plt.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Generated code image: {out_file}")

if __name__ == "__main__":
    generate_code_image()
