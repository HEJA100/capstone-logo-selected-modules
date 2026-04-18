import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score

BASE = "05_LOGO_Variant_Prioritization"
OUT_DIR = os.path.join(BASE, "my_repro", "benchmark_eval")
os.makedirs(OUT_DIR, exist_ok=True)

FILES = {
    ("GWAS", "919", "original"): f"{BASE}/1. script/05_LOGO-C2P/GWAS_C2P/1000G_GWAS_1408.vcf_output/1000G_GWAS_1408_XGboost_shuffle8_919mark_Trible.predict",
    ("GWAS", "919", "reproduced"): f"{BASE}/my_repro/GWAS_C2P_919_2002/1000G_GWAS_1408_XGboost_shuffle8_919mark_Trible.reproduced.predict",
    ("GWAS", "2002", "original"): f"{BASE}/1. script/05_LOGO-C2P/GWAS_C2P/1000G_GWAS_1408.vcf_output/1000G_GWAS_1408_XGboost_shuffle8_2002mark_Trible.predict",
    ("GWAS", "2002", "reproduced"): f"{BASE}/my_repro/GWAS_C2P_919_2002/1000G_GWAS_1408_XGboost_shuffle8_2002mark_Trible.reproduced.predict",
    ("ClinVar", "919", "original"): f"{BASE}/1. script/05_LOGO-C2P/Clinvar_C2P/GBERT-C2P/Clinvar_nc_snv_pathogenic_354_XGboost_shuffle8_919mark_Trible.predict",
    ("ClinVar", "919", "reproduced"): f"{BASE}/my_repro/Clinvar_C2P_919_2002/Clinvar_nc_snv_pathogenic_354_XGboost_shuffle8_919mark_Trible.reproduced.predict",
    ("ClinVar", "2002", "original"): f"{BASE}/1. script/05_LOGO-C2P/Clinvar_C2P/GBERT-C2P/Clinvar_nc_snv_pathogenic_354_XGboost_shuffle8_2002mark_Trible.predict",
    ("ClinVar", "2002", "reproduced"): f"{BASE}/my_repro/Clinvar_C2P_919_2002/Clinvar_nc_snv_pathogenic_354_XGboost_shuffle8_2002mark_Trible.reproduced.predict",
}

def read_predict(path):
    return pd.read_csv(path, sep=r"\s+", header=None, names=["chr","pos","id","ref","alt","score"])

def infer_label(v):
    v = str(v)
    if v.startswith("GWAS") or v.startswith("Clinvar"):
        return 1
    if v.startswith("1000G"):
        return 0
    raise ValueError(f"Unknown ID prefix: {v}")

def bootstrap_eval(df, dataset, n_boot=10, seed=42):
    rng = np.random.default_rng(seed)
    y = df["id"].map(infer_label).to_numpy()
    s = df["score"].astype(float).to_numpy()

    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]

    rows = []
    for b in range(n_boot):
        if dataset == "GWAS":
            # mimic positive-side bootstrapping idea from paper
            take_pos = rng.choice(pos_idx, size=len(neg_idx), replace=True)
            take_neg = neg_idx
        elif dataset == "ClinVar":
            # mimic negative-side bootstrapping idea from paper
            take_pos = pos_idx
            take_neg = rng.choice(neg_idx, size=len(pos_idx), replace=True)
        else:
            raise ValueError(dataset)

        take = np.concatenate([take_pos, take_neg])
        yb = y[take]
        sb = s[take]

        rows.append({
            "bootstrap": b,
            "n_pos": int((yb == 1).sum()),
            "n_neg": int((yb == 0).sum()),
            "AUROC": roc_auc_score(yb, sb),
            "AUPRC": average_precision_score(yb, sb),
        })
    return pd.DataFrame(rows)

all_boot = []
summary_rows = []

for (dataset, mark, source), path in FILES.items():
    df = read_predict(path)
    boot_df = bootstrap_eval(df, dataset=dataset, n_boot=10, seed=42)
    boot_df["dataset"] = dataset
    boot_df["mark"] = mark
    boot_df["source"] = source
    all_boot.append(boot_df)

    summary_rows.append({
        "dataset": dataset,
        "mark": mark,
        "source": source,
        "AUROC_mean": boot_df["AUROC"].mean(),
        "AUROC_sd": boot_df["AUROC"].std(ddof=1),
        "AUPRC_mean": boot_df["AUPRC"].mean(),
        "AUPRC_sd": boot_df["AUPRC"].std(ddof=1),
        "bootstrap_n": len(boot_df),
        "avg_n_pos": boot_df["n_pos"].mean(),
        "avg_n_neg": boot_df["n_neg"].mean(),
    })

boot_all_df = pd.concat(all_boot, ignore_index=True)
summary_df = pd.DataFrame(summary_rows)

boot_csv = os.path.join(OUT_DIR, "c2p_bootstrap_all.csv")
summary_csv = os.path.join(OUT_DIR, "c2p_bootstrap_summary.csv")
boot_all_df.to_csv(boot_csv, index=False)
summary_df.to_csv(summary_csv, index=False)

# plot mean ± sd bars
dataset_order = ["GWAS", "ClinVar"]
series_order = [("919","original"), ("919","reproduced"), ("2002","original"), ("2002","reproduced")]
series_labels = ["919 orig", "919 repro", "2002 orig", "2002 repro"]

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
x = np.arange(len(dataset_order))
width = 0.17

for ax, metric_mean, metric_sd, ylim in [
    (axes[0], "AUROC_mean", "AUROC_sd", (0.78, 0.96)),
    (axes[1], "AUPRC_mean", "AUPRC_sd", (0.78, 0.94)),
]:
    for i, ((mark, source), label) in enumerate(zip(series_order, series_labels)):
        vals, errs = [], []
        for ds in dataset_order:
            row = summary_df[
                (summary_df["dataset"] == ds) &
                (summary_df["mark"].astype(str) == mark) &
                (summary_df["source"] == source)
            ].iloc[0]
            vals.append(float(row[metric_mean]))
            errs.append(float(row[metric_sd]))
        ax.bar(x + (i - 1.5) * width, vals, width, yerr=errs, capsize=3, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(dataset_order)
    ax.set_ylim(*ylim)
    ax.set_title(metric_mean.replace("_mean",""))
    ax.grid(axis="y", linestyle="--", alpha=0.35)

axes[0].set_ylabel("Mean AUROC ± SD")
axes[1].set_ylabel("Mean AUPRC ± SD")
handles, labels = axes[1].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.02))
fig.suptitle("Bootstrap benchmark of LOGO-C2P reproduction", y=1.08, fontsize=17)
plt.tight_layout(rect=[0, 0, 1, 0.93])
plot_file = os.path.join(OUT_DIR, "c2p_bootstrap_metric_bars.png")
plt.savefig(plot_file, dpi=300, bbox_inches="tight")

print("=== C2P BOOTSTRAP SUMMARY CSV ===")
print(summary_df.to_csv(index=False).strip())
print("\nSaved:")
print(boot_csv)
print(summary_csv)
print(plot_file)
