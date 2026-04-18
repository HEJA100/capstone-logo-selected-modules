import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

csv_path = "05_LOGO_Variant_Prioritization/my_repro/final_eval/c2p_metrics_summary.csv"
out_file = "05_LOGO_Variant_Prioritization/my_repro/final_eval/c2p_metric_bars_v2.png"

df = pd.read_csv(csv_path)

dataset_order = ["GWAS", "ClinVar"]
series_order = [
    ("919", "original"),
    ("919", "reproduced"),
    ("2002", "original"),
    ("2002", "reproduced"),
]
series_labels = ["919 orig", "919 repro", "2002 orig", "2002 repro"]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharex=True)
x = np.arange(len(dataset_order))
width = 0.17

for ax, metric in zip(axes, ["AUROC", "AUPRC"]):
    for i, ((mark, source), label) in enumerate(zip(series_order, series_labels)):
        vals = []
        for ds in dataset_order:
            row = df[(df["dataset"] == ds) & (df["mark"].astype(str) == mark) & (df["source"] == source)]
            vals.append(float(row[metric].iloc[0]))
        ax.bar(x + (i - 1.5) * width, vals, width, label=label)

    ax.set_xticks(x)
    ax.set_xticklabels(dataset_order)
    ax.set_ylim(0.78, 0.96 if metric == "AUROC" else 0.94)
    ax.set_ylabel(metric)
    ax.set_title(metric)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

handles, labels = axes[1].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.02))
fig.suptitle("LOGO-C2P reproduction on GWAS and ClinVar datasets", y=1.08, fontsize=17)
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(out_file, dpi=300, bbox_inches="tight")
print("Saved:", out_file)
