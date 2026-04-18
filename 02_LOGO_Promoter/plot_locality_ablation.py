import os
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/users/nus/e1538285/scratch/LOGO/02_LOGO_Promoter"
RANK_CSV = os.path.join(ROOT, "locality_ablation_ranked.csv")
TOP_CSV = os.path.join(ROOT, "locality_ablation_top_per_branch_task.csv")
OUTDIR = os.path.join(ROOT, "locality_figures")
os.makedirs(OUTDIR, exist_ok=True)

task_order = ["BOTH", "TATA_BOX", "NO_TATA_BOX"]
mode_order = ["none", "single", "multi", "depthwise"]

task_label = {
    "BOTH": "BOTH",
    "TATA_BOX": "TATA",
    "NO_TATA_BOX": "NO_TATA",
}
mode_label = {
    "none": "No Conv",
    "single": "Single-kernel",
    "multi": "Multi-scale",
    "depthwise": "Depthwise",
}
branch_label = {
    "sequence_only": "Sequence-only",
    "structural_knowledge": "Structural knowledge",
}

if not os.path.exists(RANK_CSV):
    raise FileNotFoundError(f"Missing file: {RANK_CSV}")

df = pd.read_csv(RANK_CSV)
df.columns = [c.strip() for c in df.columns]

for c in df.columns:
    if df[c].dtype == object:
        df[c] = df[c].astype(str).str.strip()

num_cols = [
    "n_folds", "acc_mean", "acc_sd", "precision_mean", "precision_sd",
    "recall_mean", "recall_sd", "f1_mean", "f1_sd",
    "delta_vs_seq_base_f1", "delta_vs_struct_base_f1"
]
for c in num_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

def plot_branch_bar(branch, delta_col, out_prefix):
    sub = df[df["branch"] == branch].copy()
    sub = sub[sub["task"].isin(task_order) & sub["locality_mode"].isin(mode_order)].copy()

    sub["task"] = pd.Categorical(sub["task"], categories=task_order, ordered=True)
    sub["locality_mode"] = pd.Categorical(sub["locality_mode"], categories=mode_order, ordered=True)
    sub = sub.sort_values(["task", "locality_mode"])

    x = np.arange(len(task_order))
    width = 0.18
    offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * width

    fig, ax = plt.subplots(figsize=(11, 6.5))

    ymin = sub["f1_mean"].min()
    ymax = sub["f1_mean"].max()
    pad_low = 0.015
    pad_high = 0.02

    for i, mode in enumerate(mode_order):
        part = sub[sub["locality_mode"] == mode].set_index("task").reindex(task_order)
        y = part["f1_mean"].to_numpy(dtype=float)
        e = part["f1_sd"].fillna(0).to_numpy(dtype=float)

        valid = ~np.isnan(y)
        ax.bar(
            x[valid] + offsets[i],
            y[valid],
            width=width,
            yerr=e[valid],
            capsize=3,
            label=mode_label[mode]
        )

    ax.set_xticks(x)
    ax.set_xticklabels([task_label[t] for t in task_order], fontsize=11)
    ax.set_ylabel("F1 score", fontsize=12)
    ax.set_title(f"{branch_label[branch]} locality ablation", fontsize=14)
    ax.legend(frameon=False, ncol=2)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    ax.set_ylim(max(0.85, ymin - pad_low), min(1.0, ymax + pad_high))

    plt.tight_layout()
    png = os.path.join(OUTDIR, f"{out_prefix}.png")
    pdf = os.path.join(OUTDIR, f"{out_prefix}.pdf")
    plt.savefig(png, dpi=300, bbox_inches="tight")
    plt.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print("written", png)
    print("written", pdf)

def plot_delta_heatmaps():
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)

    branch_specs = [
        ("sequence_only", "delta_vs_seq_base_f1"),
        ("structural_knowledge", "delta_vs_struct_base_f1"),
    ]

    all_vals = []
    pivots = []

    for branch, delta_col in branch_specs:
        sub = df[df["branch"] == branch].copy()
        sub = sub[sub["task"].isin(task_order) & sub["locality_mode"].isin(mode_order)].copy()
        pivot = (
            sub.pivot_table(index="task", columns="locality_mode", values=delta_col, aggfunc="first")
               .reindex(index=task_order, columns=mode_order)
        )
        pivots.append((branch, delta_col, pivot))
        vals = pivot.to_numpy(dtype=float)
        if np.isfinite(vals).any():
            all_vals.append(np.nanmax(np.abs(vals)))

    vmax = max(all_vals) if all_vals else 0.05
    if vmax < 0.01:
        vmax = 0.01

    for ax, (branch, delta_col, pivot) in zip(axes, pivots):
        vals = pivot.to_numpy(dtype=float)
        im = ax.imshow(vals, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)

        ax.set_xticks(np.arange(len(mode_order)))
        ax.set_xticklabels([mode_label[m] for m in mode_order], rotation=25, ha="right")
        ax.set_yticks(np.arange(len(task_order)))
        ax.set_yticklabels([task_label[t] for t in task_order])
        ax.set_title(f"{branch_label[branch]}: ΔF1 vs baseline", fontsize=13)

        for i in range(vals.shape[0]):
            for j in range(vals.shape[1]):
                v = vals[i, j]
                txt = "" if np.isnan(v) else f"{v:+.3f}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=10)

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.9)
    cbar.set_label("ΔF1")

    png = os.path.join(OUTDIR, "figB3_locality_delta_heatmaps.png")
    pdf = os.path.join(OUTDIR, "figB3_locality_delta_heatmaps.pdf")
    plt.savefig(png, dpi=300, bbox_inches="tight")
    plt.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print("written", png)
    print("written", pdf)

def plot_top_summary():
    if not os.path.exists(TOP_CSV):
        print("skip top summary: missing", TOP_CSV)
        return

    top = pd.read_csv(TOP_CSV)
    top.columns = [c.strip() for c in top.columns]
    for c in top.columns:
        if top[c].dtype == object:
            top[c] = top[c].astype(str).str.strip()
    for c in ["f1_mean", "f1_sd", "delta_vs_seq_base_f1", "delta_vs_struct_base_f1"]:
        if c in top.columns:
            top[c] = pd.to_numeric(top[c], errors="coerce")

    top = top[top["task"].isin(task_order)].copy()

    order = [
        ("sequence_only", "BOTH"),
        ("sequence_only", "TATA_BOX"),
        ("sequence_only", "NO_TATA_BOX"),
        ("structural_knowledge", "BOTH"),
        ("structural_knowledge", "TATA_BOX"),
        ("structural_knowledge", "NO_TATA_BOX"),
    ]

    rows = []
    for b, t in order:
        hit = top[(top["branch"] == b) & (top["task"] == t)]
        if len(hit):
            rows.append(hit.iloc[0])

    if not rows:
        print("skip top summary: no rows")
        return

    plot_df = pd.DataFrame(rows)
    ylabels = [
        f"{branch_label[r['branch']]} | {task_label[r['task']]} | {mode_label[r['locality_mode']]}"
        for _, r in plot_df.iterrows()
    ]
    y = np.arange(len(plot_df))

    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    ax.barh(y, plot_df["f1_mean"].to_numpy(dtype=float), xerr=plot_df["f1_sd"].fillna(0).to_numpy(dtype=float), capsize=3)
    ax.set_yticks(y)
    ax.set_yticklabels(ylabels, fontsize=10)
    ax.set_xlabel("F1 score")
    ax.set_title("Best locality mode per branch-task")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)

    xmin = max(0.85, plot_df["f1_mean"].min() - 0.02)
    xmax = min(1.0, plot_df["f1_mean"].max() + 0.02)
    ax.set_xlim(xmin, xmax)

    plt.tight_layout()
    png = os.path.join(OUTDIR, "figB4_locality_top_modes.png")
    pdf = os.path.join(OUTDIR, "figB4_locality_top_modes.pdf")
    plt.savefig(png, dpi=300, bbox_inches="tight")
    plt.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print("written", png)
    print("written", pdf)

plot_branch_bar(
    branch="sequence_only",
    delta_col="delta_vs_seq_base_f1",
    out_prefix="figB1_sequence_locality_f1"
)

plot_branch_bar(
    branch="structural_knowledge",
    delta_col="delta_vs_struct_base_f1",
    out_prefix="figB2_structural_locality_f1"
)

plot_delta_heatmaps()
plot_top_summary()

print("\nDone. Output directory:")
print(OUTDIR)
