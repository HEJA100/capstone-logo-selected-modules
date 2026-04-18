import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path("/home/users/nus/e1538285/scratch/LOGO/02_LOGO_Promoter")
CSV = ROOT / "promoter_final_comparison_table.csv"
OUT = ROOT / "figures_logicA"
OUT.mkdir(parents=True, exist_ok=True)

TASKS = ["BOTH", "TATA_BOX", "NO_TATA_BOX"]
MODELS = [
    "LOGO_sequence_only",
    "LOGO_knowledge_enabled",
    "knowledge_structural",
    "knowledge_regulatory",
    "knowledge_shuffled",
]

LABELS = {
    "LOGO_sequence_only": "Sequence-only",
    "LOGO_knowledge_enabled": "Knowledge-enabled",
    "knowledge_structural": "Structural",
    "knowledge_regulatory": "Regulatory",
    "knowledge_shuffled": "Shuffled",
}

# ---------- load csv ----------
rows = []
with open(CSV, newline="") as f:
    reader = csv.DictReader(f)
    for r in reader:
        if r["task"] in TASKS and r["model"] in MODELS:
            rows.append(r)

# build lookup
lookup = {(r["task"], r["model"]): r for r in rows}

# ---------- Figure 1: grouped bar chart of F1 ----------
x = np.arange(len(TASKS))
width = 0.15

fig, ax = plt.subplots(figsize=(10, 6))

for i, model in enumerate(MODELS):
    vals = []
    for task in TASKS:
        vals.append(float(lookup[(task, model)]["f1_mean"]))
    ax.bar(x + (i - 2) * width, vals, width, label=LABELS[model])

ax.set_xticks(x)
ax.set_xticklabels(TASKS)
ax.set_ylabel("F1 mean")
ax.set_title("Logic Line A: Knowledge Ablation on Promoter Tasks")
ax.legend(frameon=False)
ax.set_ylim(0.84, 0.98)
ax.grid(axis="y", alpha=0.25)

fig.tight_layout()
fig.savefig(OUT / "logicA_knowledge_ablation_grouped_f1.png", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "logicA_knowledge_ablation_grouped_f1.pdf", bbox_inches="tight")
plt.close(fig)

# ---------- Figure 2: delta vs sequence-only heatmap ----------
delta_models = [
    "LOGO_knowledge_enabled",
    "knowledge_structural",
    "knowledge_regulatory",
    "knowledge_shuffled",
]

mat = np.zeros((len(delta_models), len(TASKS)), dtype=float)

for i, model in enumerate(delta_models):
    for j, task in enumerate(TASKS):
        mat[i, j] = float(lookup[(task, model)]["delta_vs_sequence_only_f1"])

fig, ax = plt.subplots(figsize=(8, 4.8))
im = ax.imshow(mat, aspect="auto")

ax.set_xticks(np.arange(len(TASKS)))
ax.set_xticklabels(TASKS)
ax.set_yticks(np.arange(len(delta_models)))
ax.set_yticklabels([LABELS[m] for m in delta_models])
ax.set_title("Delta F1 vs Sequence-only")

for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        ax.text(j, i, f"{mat[i,j]:+.3f}", ha="center", va="center")

cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Δ F1")

fig.tight_layout()
fig.savefig(OUT / "logicA_knowledge_ablation_delta_heatmap.png", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "logicA_knowledge_ablation_delta_heatmap.pdf", bbox_inches="tight")
plt.close(fig)

# ---------- Figure 3: task-wise ranking (optional but useful) ----------
for task in TASKS:
    vals = []
    for model in MODELS:
        vals.append((LABELS[model], float(lookup[(task, model)]["f1_mean"])))
    vals = sorted(vals, key=lambda x: x[1], reverse=True)

    labels = [v[0] for v in vals]
    numbers = [v[1] for v in vals]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    y = np.arange(len(labels))
    ax.barh(y, numbers)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("F1 mean")
    ax.set_title(f"Knowledge Ablation Ranking - {task}")
    ax.set_xlim(0.84, 0.98)
    ax.grid(axis="x", alpha=0.25)

    for yi, num in enumerate(numbers):
        ax.text(num + 0.001, yi, f"{num:.3f}", va="center")

    fig.tight_layout()
    fig.savefig(OUT / f"logicA_ranking_{task}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"logicA_ranking_{task}.pdf", bbox_inches="tight")
    plt.close(fig)

print("saved figures to:", OUT)
for p in sorted(OUT.glob("*")):
    print(p.name)
