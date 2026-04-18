import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score

BASE = os.path.expanduser("~/scratch/LOGO/05_LOGO_Variant_Prioritization/1. script/05_LOGO-C2P")
CLINVAR_DIR = os.path.join(BASE, "Clinvar_C2P", "GBERT-C2P")
VCF_PATH = os.path.join(CLINVAR_DIR, "Clinvar_nc_snv_pathogenic_354.vcf")
EVO_PATH = os.path.join(CLINVAR_DIR, "Clinvar_nc_snv_pathogenic_354.vcf_getevo_output", "infile.vcf.evoall.wholerow")
MODEL_DIR = os.path.join(BASE, "xgboost_model_weight")

def get_DMatrix_concat_useevo(chrom_df, diff_df, evo_df):
    chrom_df = chrom_df.copy()
    diff_df = diff_df.copy()
    evo_df = evo_df.copy()

    print("Chromatin feature df shape:", chrom_df.shape)
    print("Chromatin diff df shape   :", diff_df.shape)
    print("Chromatin evo df shape    :", evo_df.shape)

    chrom_df["label"] = np.where(chrom_df["name"].astype(str).str.startswith("1000G"), 0, 1)
    diff_df["label"] = np.where(diff_df["name"].astype(str).str.startswith("1000G"), 0, 1)

    sub_df = chrom_df.iloc[:, 5:]
    sub_diff_df = diff_df.iloc[:, 5:]

    y = np.asarray(sub_df["label"])
    X = np.asarray(sub_df.drop(["label"], axis=1))
    X_diff = np.asarray(sub_diff_df.drop(["label"], axis=1))
    sub_evo_df = evo_df.iloc[:, 5:]

    X_all = np.hstack((X, X_diff))
    X_all = np.hstack((X_all, sub_evo_df))
    print("Final feature matrix shape:", X_all.shape)

    dmat = xgb.DMatrix(X_all, y)
    return dmat, y, chrom_df

def run_one(mark_type):
    print(f"\n===== RUNNING {mark_type} =====")

    chrom_file = os.path.join(
        CLINVAR_DIR,
        f"{os.path.basename(VCF_PATH)}_128bs_5gram_{mark_type}feature.out.logfoldchange.csv"
    )
    diff_file = os.path.join(
        CLINVAR_DIR,
        f"{os.path.basename(VCF_PATH)}_128bs_5gram_{mark_type}feature.out.diff.csv"
    )
    model_file = os.path.join(
        MODEL_DIR,
        f"1000G_HGMD_posstrand_8softwares_5_test_shuffle8_XGboost_{mark_type}mark_Trible.model"
    )

    print("chrom_file:", chrom_file)
    print("diff_file :", diff_file)
    print("evo_file  :", EVO_PATH)
    print("model_file:", model_file)

    chrom_df = pd.read_csv(chrom_file)
    diff_df = pd.read_csv(diff_file)
    evo_df = pd.read_csv(EVO_PATH)

    dmat, y_true, df_ori = get_DMatrix_concat_useevo(chrom_df, diff_df, evo_df)

    bst = xgb.Booster({'nthread': 8})
    bst.load_model(model_file)

    pred_raw = bst.predict(dmat)
    pred_bin = (pred_raw > 0.5).astype(int)

    acc = accuracy_score(y_true, pred_bin)
    auroc = roc_auc_score(y_true, pred_raw)

    print(f"{mark_type} ACC   = {acc:.6f}")
    print(f"{mark_type} AUROC = {auroc:.6f}")

    out_df = df_ori.iloc[:, :5].copy()
    out_df["value"] = pred_raw

    out_file = os.path.join(
        CLINVAR_DIR,
        os.path.basename(VCF_PATH).replace(
            ".vcf", f"_XGboost_shuffle8_{mark_type}mark_Trible.reproduced.predict"
        )
    )
    out_df.to_csv(out_file, sep="\t", index=False, header=False)
    print("Saved:", out_file)

for mark in [919, 2002]:
    run_one(mark)

print("\n=== Clinvar_C2P local reproduction finished ===")
