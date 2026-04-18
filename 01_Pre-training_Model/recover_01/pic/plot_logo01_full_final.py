from pathlib import Path
import re
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# Absolute paths on NSCC
# =========================
LOG_3 = Path("/home/users/nus/e1538285/scratch/LOGO/01_Pre-training_Model/recover_01/logs/genebert_3_gram_full.log")
LOG_4 = Path("/home/users/nus/e1538285/scratch/LOGO/01_Pre-training_Model/recover_01/logs/genebert_4_gram_full.log")
LOG_5 = Path("/home/users/nus/e1538285/scratch/LOGO/01_Pre-training_Model/recover_01/logs/logo01_tr5g_full_hg19.$PBS_JOBID.log")
LOG_6 = Path("/home/users/nus/e1538285/scratch/LOGO/01_Pre-training_Model/recover_01/logs/genebert_6_gram_full.log")

OUTDIR = Path("/home/users/nus/e1538285/scratch/LOGO/01_Pre-training_Model/recover_01/pic")
OUTDIR.mkdir(parents=True, exist_ok=True)

LOGS = {
    "3-mer": LOG_3,
    "4-mer": LOG_4,
    "5-mer": LOG_5,
    "6-mer": LOG_6,
}

ORDER = ["3-mer", "4-mer", "5-mer", "6-mer"]

COLORS = {
    "3-mer": "#4C78A8",  # blue
    "4-mer": "#F58518",  # orange
    "5-mer": "#54A24B",  # green
    "6-mer": "#E45756",  # red
}


def time_to_seconds(s: str):
    parts = s.strip().split(":")
    if len(parts) == 2:
        m, sec = parts
        return float(m) * 60 + float(sec)
    elif len(parts) == 3:
        h, m, sec = parts
        return float(h) * 3600 + float(m) * 60 + float(sec)
    return None


def parse_log(log_path: Path, kmer: str):
    if not log_path.exists():
        raise FileNotFoundError(f"Log not found: {log_path}")

    text = log_path.read_text(errors="ignore")

    # Parse accuracy from checkpoint save lines
    epoch_matches = re.findall(
        r"Epoch\s+0*(\d+): saving model to .*?-(\d+\.\d+)\.hdf5",
        text
    )

    epoch_rows = []
    for ep, acc in epoch_matches:
        epoch_rows.append({
            "kmer": kmer,
            "epoch": int(ep),
            "accuracy": float(acc),
            "log_path": str(log_path),
        })

    # Parse total wall time
    m = re.search(
        r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s*([0-9:.]+)",
        text
    )
    elapsed_str = m.group(1) if m else None
    elapsed_s = time_to_seconds(elapsed_str) if elapsed_str else None

    summary_row = {
        "kmer": kmer,
        "log_path": str(log_path),
        "epochs_completed": len(epoch_rows),
        "final_accuracy": epoch_rows[-1]["accuracy"] if epoch_rows else None,
        "best_accuracy": max([r["accuracy"] for r in epoch_rows]) if epoch_rows else None,
        "elapsed_wall_time_str": elapsed_str,
        "elapsed_wall_time_s": elapsed_s,
        "avg_time_per_epoch_s": (elapsed_s / len(epoch_rows)) if (elapsed_s and epoch_rows) else None,
    }

    return epoch_rows, summary_row


def style_axes(ax):
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.25)
    ax.tick_params(axis="both", labelsize=10)


# -------------------------
# Parse all logs
# -------------------------
epoch_rows = []
summary_rows = []

for kmer in ORDER:
    erows, srow = parse_log(LOGS[kmer], kmer)
    epoch_rows.extend(erows)
    summary_rows.append(srow)

epoch_df = pd.DataFrame(epoch_rows).sort_values(["kmer", "epoch"]).reset_index(drop=True)
summary_df = pd.DataFrame(summary_rows)
summary_df["kmer"] = pd.Categorical(summary_df["kmer"], categories=ORDER, ordered=True)
summary_df = summary_df.sort_values("kmer").reset_index(drop=True)

# Save tables
epoch_csv = OUTDIR / "logo01_full_accuracy_long.csv"
summary_csv = OUTDIR / "logo01_full_summary.csv"
epoch_df.to_csv(epoch_csv, index=False)
summary_df.to_csv(summary_csv, index=False)

# -------------------------
# Figure 1: Accuracy by epoch
# -------------------------
fig, ax = plt.subplots(figsize=(8.4, 5.2))

for kmer in ORDER:
    sub = epoch_df[epoch_df["kmer"] == kmer]
    ax.plot(
        sub["epoch"],
        sub["accuracy"],
        color=COLORS[kmer],
        linewidth=1.8,
        marker="o",
        markersize=3.2,
        label=kmer
    )

style_axes(ax)
ax.set_xlabel("Epoch", fontsize=11)
ax.set_ylabel("Accuracy", fontsize=11)
ax.set_title("Pre-training accuracy across k-mer settings", fontsize=14, pad=10)

# tighter y-range
ymin = float(epoch_df["accuracy"].min()) - 0.0015
ymax = float(epoch_df["accuracy"].max()) + 0.0012
ax.set_ylim(ymin, ymax)
ax.set_xlim(1, int(epoch_df["epoch"].max()))

ax.legend(
    frameon=False,
    fontsize=10,
    loc="lower right",
    ncol=2
)

fig.tight_layout()
fig1_png = OUTDIR / "logo01_full_accuracy_by_epoch_final.png"
fig1_pdf = OUTDIR / "logo01_full_accuracy_by_epoch_final.pdf"
fig.savefig(fig1_png, dpi=300, bbox_inches="tight")
fig.savefig(fig1_pdf, bbox_inches="tight")
plt.close(fig)

# -------------------------
# Figure 2: Average time per epoch
# -------------------------
fig, ax = plt.subplots(figsize=(7.2, 4.8))

bars = ax.bar(
    summary_df["kmer"],
    summary_df["avg_time_per_epoch_s"],
    color=[COLORS[k] for k in summary_df["kmer"]],
    width=0.6
)

style_axes(ax)
ax.set_xlabel("k-mer", fontsize=11)
ax.set_ylabel("Average wall time per epoch (s)", fontsize=11)
ax.set_title("Average epoch wall time in our reproduction", fontsize=14, pad=10)

ymax = float(summary_df["avg_time_per_epoch_s"].max())
for bar, value in zip(bars, summary_df["avg_time_per_epoch_s"]):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + ymax * 0.015,
        f"{value:.1f}",
        ha="center",
        va="bottom",
        fontsize=9
    )

fig.text(
    0.5,
    -0.035,
    "Runtime is measured on our NSCC reproduction setup and is not directly equivalent to the paper's hardware setting.",
    ha="center",
    fontsize=8.8
)

fig.tight_layout()
fig2_png = OUTDIR / "logo01_full_avg_time_per_epoch_final.png"
fig2_pdf = OUTDIR / "logo01_full_avg_time_per_epoch_final.pdf"
fig.savefig(fig2_png, dpi=300, bbox_inches="tight")
fig.savefig(fig2_pdf, bbox_inches="tight")
plt.close(fig)

print("Saved files:")
print(epoch_csv)
print(summary_csv)
print(fig1_png)
print(fig1_pdf)
print(fig2_png)
print(fig2_pdf)

print("\nSummary:")
print(summary_df.to_string(index=False))