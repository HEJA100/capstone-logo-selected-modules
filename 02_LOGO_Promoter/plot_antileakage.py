import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path("/home/users/nus/e1538285/scratch/LOGO/02_LOGO_Promoter")
CSV = ROOT / "anti_leakage_comparison_table.csv"
OUT = ROOT / "figures_antileakage"
OUT.mkdir(parents=True, exist_ok=True)

TASKS = ["BOTH", "TATA_BOX", "NO_TATA_BOX"]
MODELS = [
    "LOGO_sequence_only",
    "LOGO_knowledge_enabled",
    "knowledge_coarsebin25",
    "knowledge_distmask_tsspm100",
    "knowledge_shuffled",
]
LABELS = {
    "LOGO_sequence_only": "Sequence-only",
    "LOGO_knowledge_enabled": "Full knowledge",
    "knowledge_coarsebin25": "Coarse-bin",
    "knowledge_distmask_tsspm100": "Distmask",
    "knowledge_shuffled": "Shuffled",
}

rows = []
with open(CSV, newline="") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

lookup = {(r["task"], r["model"]): r for r in rows}

# ---------- Fig 1: grouped F1 ----------
x = np.arange(len(TASKS))
width = 0.15

fig, ax = plt.subplots(figsize=(10, 6))
for i, model in enumerate(MODELS):
    vals = [float(lookup[(task, model)]["f1_mean"]) for task in TASKS]
    ax.bar(x + (i - 2) * width, vals, width, label=LABELS[model])

ax.set_xticks(x)
ax.set_xticklabels(TASKS)
ax.set_ylabel("F1 mean")
ax.set_title("Anti-leakage comparison on promoter tasks")
ax.legend(frameon=False)
ax.set_ylim(0.84, 0.98)
ax.grid(axis="y", alpha=0.25)

fig.tight_layout()
fig.savefig(OUT / "antileakage_grouped_f1.png", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "antileakage_grouped_f1.pdf", bbox_inches="tight")
plt.close(fig)

# ---------- Fig 2: retention vs full knowledge ----------
ret_models = [
    "LOGO_sequence_only",
    "knowledge_coarsebin25",
    "knowledge_distmask_tsspm100",
    "knowledge_shuffled",
]

fig, ax = plt.subplots(figsize=(10, 6))
for i, model in enumerate(ret_models):
    vals = [float(lookup[(task, model)]["retention_vs_full_knowledge_pct"]) for task in TASKS]
    ax.bar(x + (i - 1.5) * 0.18, vals, 0.18, label=LABELS[model])

ax.axhline(100.0, linestyle="--", linewidth=1)
ax.set_xticks(x)
ax.set_xticklabels(TASKS)
ax.set_ylabel("Retention vs full knowledge (%)")
ax.set_title("How much performance is retained after anti-leakage modifications?")
ax.legend(frameon=False)
ax.grid(axis="y", alpha=0.25)

fig.tight_layout()
fig.savefig(OUT / "antileakage_retention_vs_full.png", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "antileakage_retention_vs_full.pdf", bbox_inches="tight")
plt.close(fig)

print("saved to:", OUT)
for p in sorted(OUT.glob("*")):
    print(p.name)
