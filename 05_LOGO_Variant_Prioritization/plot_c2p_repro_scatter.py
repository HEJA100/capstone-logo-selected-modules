import pandas as pd
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
    return pd.read_csv(path, sep=r"\s+", header=None)[5].astype(float)

fig, axes = plt.subplots(2, 2, figsize=(10, 9))
axes = axes.ravel()

for ax, (title, (orig_path, repro_path)) in zip(axes, pairs.items()):
    x = read_score(orig_path)
    y = read_score(repro_path)

    ax.scatter(x, y, s=10, alpha=0.6)
    mn = min(x.min(), y.min())
    mx = max(x.max(), y.max())
    ax.plot([mn, mx], [mn, mx], linestyle="--", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Original score")
    ax.set_ylabel("Reproduced score")
    ax.grid(alpha=0.3, linestyle="--")

plt.tight_layout()
out_file = f"{BASE}/my_repro/final_eval/c2p_reproduced_vs_original_scatter.png"
plt.savefig(out_file, dpi=300, bbox_inches="tight")
print("Saved:", out_file)
