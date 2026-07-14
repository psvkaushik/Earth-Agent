import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

data = [
    ("GPT-5",               8.2,  "other"),
    ("DeepSeek-V3.1",       8.5,  "other"),
    ("Qwen3-Max",           8.8,  "other"),
    ("Kimi-k2",             9.1,  "other"),
    ("EVE",                 15.9, "eve"),
    ("GPT-4o",              16.7, "other"),
    ("Gemini-2.5",          18.1, "other"),
    ("Mistral",             24.2, "mistral"),
    ("SEED-1.6",            33.1, "other"),
    ("Qwen-Plus",           37.0, "other"),
    ("GLM-4.5v",            51.0, "other"),
    ("InternVL3.5",         53.1, "other"),
    ("Qwen3-32B",           73.4, "other"),
    ("LLaMA-4",             76.7, "other"),
]

# Sort ascending by error rate
data.sort(key=lambda x: x[1])

labels = [d[0] for d in data]
values = [d[1] for d in data]
tags   = [d[2] for d in data]

color_map = {"eve": "#2a78d6", "mistral": "#e34948", "other": "#c3c2b7"}
colors = [color_map[t] for t in tags]

fig, ax = plt.subplots(figsize=(9, 6))
fig.patch.set_facecolor("#f9f9f7")
ax.set_facecolor("#f9f9f7")

bars = ax.barh(labels, values, color=colors, height=0.6, zorder=3)

# Value labels
for bar, val in zip(bars, values):
    ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%", va='center', ha='left', fontsize=9, color="#0b0b0b")

ax.set_xlabel("Tool Call Error Rate (%)", fontsize=11, color="#52514e")
ax.set_title("Tool Call Error Rate — AP Models", fontsize=13, fontweight='bold',
             color="#0b0b0b", pad=12)
ax.set_xlim(0, 85)
ax.tick_params(colors="#52514e", labelsize=9)
ax.spines[['top', 'right', 'left']].set_visible(False)
ax.spines['bottom'].set_color("#c3c2b7")
ax.xaxis.grid(True, color="#e1e0d9", linestyle='--', linewidth=0.6, zorder=0)
ax.set_axisbelow(True)

legend_patches = [
    mpatches.Patch(color="#2a78d6", label="EVE"),
    mpatches.Patch(color="#e34948", label="Mistral"),
    mpatches.Patch(color="#c3c2b7", label="Other models"),
]
ax.legend(handles=legend_patches, loc="lower right", fontsize=9,
          framealpha=0.8, edgecolor="#e1e0d9")

plt.tight_layout()
output_path = os.path.join(OUTPUT_DIR, "tool_error_rate_AP.png")
plt.savefig(output_path, dpi=150, bbox_inches='tight',
            facecolor="#f9f9f7")
print(f"Saved {output_path}")
