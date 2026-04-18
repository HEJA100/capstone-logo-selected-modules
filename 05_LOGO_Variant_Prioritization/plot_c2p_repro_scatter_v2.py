import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

BASE = "05_LOGO_Variant_Prioritization"

pairs = {
    "GWAS 919": (
        f"{BASE}/1. script/05_LOGO-C2P/GWAS_C2P/1000G_GWAS_1408.vcf_output/1000G_GWAS_1408_XGboost_shuffle8_919mark_Trible.predict",
        f"{BASE}/my_repro/GWAS_C2P_919_2002/1000G_GWAS_1408_XGboost_shuffle8_919mark_Trible.reproduced.predict",
    ),
    "GWAS 2002": (
        f"{BASE}/1. script/05_LOGO-C2P/GWAS_C2P/1000G_GWAS_1408.vcf_output/1000G_GWAS_1408_XGboost_shuffle8_2002mark_Trible.predict",
        f"{BASE}/my_repro/GWAS_C2P_919_2002/1000G_GWAS_1408_XGboost_shuffle8_2002mark_Trible.reproduced.predict",
    ),
    "ClinVar 919": (
        f"{BASE}/1. script/05_LOGO-C2P/Clinvar_C2P/GBERT-C2P/Clinvar_nc_snv_pathogenic_354_XGboost_shuffle8_919mark_Trible.predict",
        f"{BASE}/my_repro/Clinvar_C2P_919_2002/Clinvar_nc_snv_pathogenic_354_XGboost_shuffle8_919mark_Trible.reproduced.predict",
    ),
    "ClinVar 2002": (
        f"{BASE}/1. script/05_LOGO-C2P/Clinvar_C2P/GBERT-C2P/Clinvar_nc_snv_pathogenic_354_XGboost_shuffle8_2002mark_Trible.predict",
        f"{BASE}/my_repro/Clinvar_C2P_919_2002/Clinvar_nc_snv_pathogenic_354_XGboost_shuffle8_2002mark_Trible.reproduced.predict",
    ),
}

def read_score(path):
    return pd.read_csv(path, sep=r"\s+", header=None)[5].astype(float).values

fig, axes = plt.subplots(2, 2, figsize=(9.5, 8.5))
axes = axes.ravel()

for ax, (title, (orig_path, repro_path)) in zip(axes, pairs.items()):
    x = read_score(orig_path)
    y = read_score(repro_path)

    mn = min(x.min(), y.min())
    mx = max(x.max(), y.max())
    r = np.corrcoef(x, y)[0, 1]
    mean_abs = np.mean(np.abs(x - y))

    ax.scatter(x, y, s=14, alpha=0.7)
    ax.plot([mn, mx], [mn, mx], linestyle="--", linewidth=1)
    ax.set_title(f"{title}\nr={r:.6f}, mean|Δ|={mean_abs:.6g}", fontsize=11)
    ax.set_xlabel("Original score")
    ax.set_ylabel("Reproduced score")
    ax.grid(alpha=0.3, linestyle="--")

fig.suptitle("Agreement between original and reproduced LOGO-C2P scores", y=0.98, fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
out_file = f"{BASE}/my_repro/final_eval/c2p_reproduced_vs_original_scatter_v2.png"
plt.savefig(out_file, dpi=300, bbox_inches="tight")
print("Saved:", out_file)
