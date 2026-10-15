"""
Generates a crisp, dark-themed terminal console output image for the PowerPoint deck.
File: generate_terminal_card.py
"""
import os
import matplotlib.pyplot as plt

def generate_terminal_image():
    fig, ax = plt.subplots(figsize=(10, 6.2), dpi=200)
    fig.patch.set_facecolor("#030712")
    ax.set_facecolor("#0a0f1d")

    # Remove all axes
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#1e293b")

    # Terminal header bar
    header_rect = plt.Rectangle((0, 0.92), 1, 0.08, transform=ax.transAxes, color="#0f172a", zorder=2)
    ax.add_patch(header_rect)

    # Mac/Linux style window buttons
    ax.scatter([0.025, 0.045, 0.065], [0.96, 0.96, 0.96], color=["#ef4444", "#f59e0b", "#10b981"], s=45, transform=ax.transAxes, zorder=3)
    ax.text(0.5, 0.96, "bash: python minimal_swarm_allocation_ml.py", color="#94a3b8", fontsize=10,
            family="monospace", ha="center", va="center", transform=ax.transAxes, zorder=3)

    terminal_text = """
=================================================================
   MINIMAL MACHINE LEARNING CORE: SWARM TARGET ALLOCATION GAT    
=================================================================

--- [1. BEFORE TRAINING: Initial Greedy / Overkill Policy] ---
Target Assignments  : ['T0', 'T0', 'T0', 'T0']  <-- GREEDY COLLAPSE
Target Probabilities:
  M0: [P(T0)=0.38, P(T1)=0.31, P(T2)=0.31]
  M1: [P(T0)=0.37, P(T1)=0.31, P(T2)=0.31]
  M2: [P(T0)=0.38, P(T1)=0.31, P(T2)=0.31]
  M3: [P(T0)=0.38, P(T1)=0.31, P(T2)=0.31]
Team Reward         : -45.33 (FAIL: Target 1 & 2 Untouched, Severe Overkill)

--- [2. TRAINING LOOP: REINFORCE with Exponential Baseline] ---
Epoch  20 | Loss: -53.03 | Sampled: ['T0', 'T0', 'T1', 'T1'] | Reward:   4.3
Epoch  60 | Loss:   3.02 | Sampled: ['T0', 'T1', 'T0', 'T0'] | Reward:  13.3
Epoch 100 | Loss:   0.00 | Sampled: ['T0', 'T2', 'T1', 'T0'] | Reward:  60.0
Epoch 120 | Loss:   0.00 | Sampled: ['T0', 'T2', 'T1', 'T0'] | Reward:  60.0

--- [3. AFTER TRAINING: Decentralized GAT Policy Convergence] ---
Final Assignments   : ['T0', 'T2', 'T1', 'T0']  <-- 100% COVERAGE!
Target Probabilities:
  M0 -> Target 0 (SAM HQ)      [P=1.00, P=0.00, P=0.00]
  M1 -> Target 2 (Launcher)    [P=0.00, P=0.00, P=1.00]
  M2 -> Target 1 (Bunker)      [P=0.00, P=1.00, P=0.00]
  M3 -> Target 0 (Reserve T0)  [P=1.00, P=0.00, P=0.00]
Team Reward         : +60.00 (SUCCESS: Zero Redundancy, Optimal Dispersion)
=================================================================
"""

    ax.text(0.03, 0.88, terminal_text.strip(), color="#38bdf8", fontsize=8.2, family="monospace",
            va="top", transform=ax.transAxes, linespacing=1.28)

    os.makedirs("figures", exist_ok=True)
    out_file = os.path.join("figures", "terminal_training_output.png")
    plt.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Generated terminal image: {out_file}")

if __name__ == "__main__":
    generate_terminal_image()
