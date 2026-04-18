import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve

BASE = "05_LOGO_Variant_Prioritization"

FILES = {
    "GWAS_919_original": os.path.join(BASE, "1. script/05_LOGO-C2P/GWAS_C2P/1000G_GWAS_1408.vcf_output/1000G_GWAS_1408_XGboost_shuffle8_919mark_Trible.predict"),
    "GWAS_919_reproduced": os.path.join(BASE, "my_repro/GWAS_C2P_919_2002/1000G_GWAS_1408_XGboost_shuffle8_919mark_Trible.reproduced.predict"),
    "GWAS_2002_original": os.path.join(BASE, "1. script/05_LOGO-C2P/GWAS_C2P/1000G_GWAS_1408.vcf_output/1000G_GWAS_1408_XGboost_shuffle8_2002mark_Trible.predict"),
    "GWAS_2002_reproduced": os.path.join(BASE, "my_repro/GWAS_C2P_919_2002/1000G_GWAS_1408_XGboost_shuffle8_2002mark_Trible.reproduced.predict"),

    "ClinVar_919_original": os.path.join(BASE, "1. script/05_LOGO-C2P/Clinvar_C2P/GBERT-C2P/Clinvar_nc_snv_pathogenic_354_XGboost_shuffle8_919mark_Trible.predict"),
    "ClinVar_919_reproduced": os.path.join(BASE, "my_repro/Clinvar_C2P_919_2002/Clinvar_nc_snv_pathogenic_354_XGboost_shuffle8_919mark_Trible.reproduced.predict"),
    "ClinVar_2002_original": os.path.join(BASE, "1. script/05_LOGO-C2P/Clinvar_C2P/GBERT-C2P/Clinvar_nc_snv_pathogenic_354_XGboost_shuffle8_2002mark_Trible.predict"),
    "ClinVar_2002_reproduced": os.path.join(BASE, "my_repro/Clinvar_C2P_919_2002/Clinvar_nc_snv_pathogenic_354_XGboost_shuffle8_2002mark_Trible.reproduced.predict"),
}

OUT_DIR = os.path.join(BASE, "my_repro", "final_eval")
os.makedirs(OUT_DIR, exist_ok=True)

def read_predict(path):
    df = pd.read_csv(path, sep=r"\s+", header=None, names=["chr", "pos", "id", "ref", "alt", "score"])
    return df

def infer_label(variant_id):
    s = str(variant_id)
    if s.startswith("GWAS") or s.startswith("Clinvar"):
        return 1
    if s.startswith("1000G"):
        return 0
    raise ValueError(f"Unknown ID prefix: {variant_id}")

summary_rows = []
compare_rows = []

loaded = {k: read_predict(v) for k, v in FILES.items()}

# metric summary
for name, df in loaded.items():
    y_true = df["id"].map(infer_label).astype(int).values
    y_score = df["score"].astype(float).values
    auroc = roc_auc_score(y_true, y_score)
    auprc = average_precision_score(y_true, y_score)

    if name.startswith("GWAS"):
        dataset = "GWAS"
    else:
        dataset = "ClinVar"

    mark = "919" if "_919_" in name else "2002"
    source = "original" if name.endswith("original") else "reproduced"

    summary_rows.append([dataset, mark, source, len(df), int(y_true.sum()), int((1-y_true).sum()), auroc, auprc])

# original vs reproduced score comparison
pairs = [
    ("GWAS", "919"),
    ("GWAS", "2002"),
    ("ClinVar", "919"),
    ("ClinVar", "2002"),
]

for dataset, mark in pairs:
    a = loaded[f"{dataset}_{mark}_original"]
    b = loaded[f"{dataset}_{mark}_reproduced"]

    same_shape = a.shape == b.shape
    same_ids = a["id"].tolist() == b["id"].tolist()
    diff = np.abs(a["score"].astype(float).values - b["score"].astype(float).values)
    compare_rows.append([
        dataset, mark, same_shape, same_ids,
        diff.max(), diff.mean()
    ])

summary_df = pd.DataFrame(summary_rows, columns=[
    "dataset", "mark", "source", "n_rows", "n_pos", "n_neg", "AUROC", "AUPRC"
])
compare_df = pd.DataFrame(compare_rows, columns=[
    "dataset", "mark", "same_shape", "same_id_order", "max_abs_diff", "mean_abs_diff"
])

summary_csv = os.path.join(OUT_DIR, "c2p_metrics_summary.csv")
compare_csv = os.path.join(OUT_DIR, "c2p_original_vs_reproduced_compare.csv")
summary_df.to_csv(summary_csv, index=False)
compare_df.to_csv(compare_csv, index=False)

# ROC plots
for dataset in ["GWAS", "ClinVar"]:
    fig, ax = plt.subplots(figsize=(6, 5))
    for mark, source, label in [
        ("919", "original", "919 original"),
        ("919", "reproduced", "919 reproduced"),
        ("2002", "original", "2002 original"),
        ("2002", "reproduced", "2002 reproduced"),
    ]:
        df = loaded[f"{dataset}_{mark}_{source}"]
        y_true = df["id"].map(infer_label).astype(int).values
        y_score = df["score"].astype(float).values
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)
        ax.plot(fpr, tpr, linewidth=2, label=f"{label} (AUROC={auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{dataset} C2P ROC")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{dataset.lower()}_c2p_roc.png"), dpi=300, bbox_inches="tight")
    plt.close()

# PR plots
for dataset in ["GWAS", "ClinVar"]:
    fig, ax = plt.subplots(figsize=(6, 5))
    for mark, source, label in [
        ("919", "original", "919 original"),
        ("919", "reproduced", "919 reproduced"),
        ("2002", "original", "2002 original"),
        ("2002", "reproduced", "2002 reproduced"),
    ]:
        df = loaded[f"{dataset}_{mark}_{source}"]
        y_true = df["id"].map(infer_label).astype(int).values
        y_score = df["score"].astype(float).values
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        ap = average_precision_score(y_true, y_score)
        ax.plot(recall, precision, linewidth=2, label=f"{label} (AUPRC={ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"{dataset} C2P Precision-Recall")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{dataset.lower()}_c2p_pr.png"), dpi=300, bbox_inches="tight")
    plt.close()

print("=== METRICS SUMMARY CSV ===")
print(summary_df.to_csv(index=False).strip())

print("\n=== ORIGINAL VS REPRODUCED COMPARE CSV ===")
print(compare_df.to_csv(index=False).strip())

print("\nSaved files:")
print(summary_csv)
print(compare_csv)
print(os.path.join(OUT_DIR, "gwas_c2p_roc.png"))
print(os.path.join(OUT_DIR, "gwas_c2p_pr.png"))
print(os.path.join(OUT_DIR, "clinvar_c2p_roc.png"))
print(os.path.join(OUT_DIR, "clinvar_c2p_pr.png"))
