"""Generate LinkedIn chart for GRPO project results."""

import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.use("Agg")

# Dark theme matching MTP chart style
plt.style.use("dark_background")
fig = plt.figure(figsize=(14, 7))

# Load training logs
with open("/home/akshay/grpo-workspace/grpo-output/checkpoint-15900/trainer_state.json") as f:
    state = json.load(f)
logs = state["log_history"]

steps = [l["step"] for l in logs]
rewards = [l["reward"] for l in logs]
math_rewards = [l["rewards/math_reward/mean"] for l in logs]
kl = [l["kl"] for l in logs]

# Smooth rewards with moving average
def smooth(values, window=20):
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")

smooth_steps = steps[len(steps) - len(smooth(rewards)):]

# --- Left plot: Training Progress ---
ax1 = fig.add_axes([0.06, 0.12, 0.42, 0.72])

# Raw rewards (faded)
ax1.plot(steps, rewards, alpha=0.15, color="#4FC3F7", linewidth=0.5)
# Smoothed total reward
ax1.plot(smooth_steps, smooth(rewards), color="#4FC3F7", linewidth=2, label="Total Reward")
# Smoothed math reward
ax1.plot(steps, math_rewards, alpha=0.15, color="#81C784", linewidth=0.5)
ax1.plot(smooth_steps, smooth(math_rewards), color="#81C784", linewidth=2, label="Math Reward")

# Annotations
ax1.annotate("Start: 0.33", xy=(10, 0.33), fontsize=9, color="#FF9800",
             fontweight="bold", ha="left",
             xytext=(200, 0.15), arrowprops=dict(arrowstyle="->", color="#FF9800", lw=1.5))
ax1.annotate("Peak: 0.87", xy=(3500, 0.87), fontsize=9, color="#FF9800",
             fontweight="bold", ha="left",
             xytext=(4500, 0.95), arrowprops=dict(arrowstyle="->", color="#FF9800", lw=1.5))
ax1.annotate("Final: 0.77", xy=(15900, 0.77), fontsize=9, color="#4FC3F7",
             fontweight="bold", ha="right",
             xytext=(13500, 0.88), arrowprops=dict(arrowstyle="->", color="#4FC3F7", lw=1.5))

# Epoch markers
epoch_steps = [7473, 14946]  # approximate epoch boundaries
for i, es in enumerate(epoch_steps):
    ax1.axvline(x=es, color="#666666", linestyle="--", alpha=0.5)
    ax1.text(es + 100, 0.05, f"Epoch {i+1}", fontsize=8, color="#999999", rotation=90, va="bottom")

ax1.set_xlabel("Training Steps", fontsize=11)
ax1.set_ylabel("Reward", fontsize=11)
ax1.set_title("Training Progress — 15,900 Steps", fontsize=13, fontweight="bold", pad=10)
ax1.legend(loc="lower right", fontsize=9, framealpha=0.3)
ax1.set_ylim(-0.05, 1.05)
ax1.set_xlim(0, 16500)
ax1.grid(alpha=0.15)

# --- Right plot: Benchmark Results ---
ax2 = fig.add_axes([0.58, 0.12, 0.38, 0.72])

labels = ["Baseline\n8-shot", "Baseline\n0-shot", "GRPO\n8-shot", "GRPO\n0-shot"]
values = [53.5, 52.1, 50.4, 58.0]
colors = ["#78909C", "#78909C", "#EF5350", "#4CAF50"]

bars = ax2.bar(labels, values, color=colors, width=0.6, edgecolor="none", alpha=0.85)

# Value labels on bars
for bar, val in zip(bars, values):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
             f"{val}%", ha="center", va="bottom", fontsize=12, fontweight="bold", color="white")

# Delta annotations with arrows
ax2.annotate("-3.1pp", xy=(2, 50.4), fontsize=10, color="#EF5350",
             fontweight="bold", ha="center",
             xytext=(2, 44), arrowprops=dict(arrowstyle="->", color="#EF5350", lw=1.5))
ax2.annotate("+5.9pp", xy=(3, 58.0), fontsize=11, color="#66BB6A",
             fontweight="bold", ha="center",
             xytext=(3, 64), arrowprops=dict(arrowstyle="->", color="#66BB6A", lw=1.5))

ax2.set_ylabel("GSM8K Accuracy (%)", fontsize=11)
ax2.set_title("Benchmark Results", fontsize=13, fontweight="bold", pad=10)
ax2.set_ylim(0, 72)
ax2.grid(axis="y", alpha=0.15)

# Remove top/right spines
for ax in [ax1, ax2]:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# --- Main title ---
fig.text(0.5, 0.96,
         "GRPO Reasoning Training: Qwen3.5-0.8B on Single RTX 5090",
         fontsize=16, fontweight="bold", ha="center", color="white")
fig.text(0.5, 0.925,
         "SFT warmup (3.5K examples) → GRPO with math rewards  |  77 hours training  |  Zero-shot GSM8K: 52.1% → 58.0%",
         fontsize=10, ha="center", color="#AAAAAA")

out = "/home/akshay/gpu-lab/projects/05-grpo-reasoning/linkedin_chart.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved to {out}")
plt.close()
