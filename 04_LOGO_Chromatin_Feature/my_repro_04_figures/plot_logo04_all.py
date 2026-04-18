from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path("/home/users/nus/e1538285/scratch/LOGO/04_LOGO_Chromatin_Feature")
OUTDIR = ROOT / "my_repro_04_figures" / "plots"
MAIN_DIR = OUTDIR / "main_figures"
SUPP_DIR = OUTDIR / "supplementary_figures"

OUTDIR.mkdir(parents=True, exist_ok=True)
MAIN_DIR.mkdir(parents=True, exist_ok=True)
SUPP_DIR.mkdir(parents=True, exist_ok=True)

FIGURE_REGISTRY = {"main": [], "supplementary": []}

# =========================
# Paths
# =========================
train_log = ROOT / "my_repro_04_rerun919" / "logs" / "logo919_rerun.log"
test_logs_dir = ROOT / "my_repro_04_rerun919" / "metrics" / "test_logs"
summary_csv = ROOT / "my_repro_04_rerun919" / "metrics" / "logo919_train_summary.csv"

predict_dir = ROOT / "1. script" / "04_LOGO_Chrom_predict"
f2002_diff = predict_dir / "demo.vcf_128bs_5gram_2002feature.out.diff.csv"
f2002_logfc = predict_dir / "demo.vcf_128bs_5gram_2002feature.out.logfoldchange.csv"
f3357_diff = predict_dir / "demo.vcf_128bs_5gram_3357feature.out.diff.csv"
f3357_logfc = predict_dir / "demo.vcf_128bs_5gram_3357feature.out.logfoldchange.csv"

# Paper values from Figure 3A
paper_919 = {"TF": 0.965, "DHS": 0.930, "HM": 0.864}
# Best reproduced result from your tested checkpoints
best_919 = {"TF": 0.941, "DHS": 0.901, "HM": 0.832}

RNG = np.random.default_rng(42)


# =========================
# Generic helpers
# =========================
def savefig(name: str, section: str = "supplementary"):
    target_dir = MAIN_DIR if section == "main" else SUPP_DIR
    plt.tight_layout()
    out = target_dir / name
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    FIGURE_REGISTRY[section].append(out.name)
    print(f"Saved: {out}")


def write_manifest():
    manifest = OUTDIR / "figure_manifest.txt"
    lines = []
    lines.append("Main figures:\n")
    for x in FIGURE_REGISTRY["main"]:
        lines.append(f"- {x}\n")
    lines.append("\nSupplementary figures:\n")
    for x in FIGURE_REGISTRY["supplementary"]:
        lines.append(f"- {x}\n")
    manifest.write_text("".join(lines))
    print(f"Saved: {manifest}")


def style_axes(ax, ygrid=True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if ygrid:
        ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.set_axisbelow(True)


def annotate_bars(ax, bars, fmt="{:.3f}", dy=0.003, fontsize=10):
    for b in bars:
        h = b.get_height()
        ax.text(
            b.get_x() + b.get_width() / 2,
            h + dy,
            fmt.format(h),
            ha="center",
            va="bottom",
            fontsize=fontsize,
        )


# =========================
# 919 helpers
# =========================
def parse_train_log(log_path: Path) -> pd.DataFrame:
    if not log_path.exists():
        return pd.DataFrame()

    text = log_path.read_text(errors="ignore")
    save_pat = re.compile(r"weights_(\d+)-([0-9.]+)-([0-9.]+)\.hdf5")
    metric_pat = re.compile(
        r"loss:\s*([0-9.]+).*?"
        r"accuracy:\s*([0-9.]+).*?"
        r"auc:\s*([0-9.]+).*?"
        r"val_loss:\s*([0-9.]+).*?"
        r"val_accuracy:\s*([0-9.]+).*?"
        r"val_auc:\s*([0-9.]+)",
        re.S,
    )

    rows = []
    pending = None

    for line in text.splitlines():
        m = save_pat.search(line)
        if m:
            pending = {
                "epoch": int(m.group(1)),
                "train_accuracy_file": float(m.group(2)),
                "val_accuracy_file": float(m.group(3)),
            }
            continue

        if pending and ("val_auc:" in line and "val_accuracy:" in line):
            mm = metric_pat.search(line)
            if mm:
                pending.update(
                    {
                        "loss": float(mm.group(1)),
                        "accuracy": float(mm.group(2)),
                        "auc": float(mm.group(3)),
                        "val_loss": float(mm.group(4)),
                        "val_accuracy": float(mm.group(5)),
                        "val_auc": float(mm.group(6)),
                    }
                )
                rows.append(pending)
                pending = None

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values("epoch")


def parse_test_logs(logdir: Path) -> pd.DataFrame:
    pat_tf = re.compile(r"Transcription factors:\s*([0-9.]+)")
    pat_dhs = re.compile(r"DNase I-hypersensitive sites:\s*([0-9.]+)")
    pat_hm = re.compile(r"Histone marks:\s*([0-9.]+)")
    pat_epoch = re.compile(r"weights_(\d+)-")

    rows = []
    for f in sorted(logdir.glob("*.test.log")):
        text = f.read_text(errors="ignore")
        mtf = pat_tf.search(text)
        mdhs = pat_dhs.search(text)
        mhm = pat_hm.search(text)
        me = pat_epoch.search(f.name)

        if mtf and mdhs and mhm:
            tf = float(mtf.group(1))
            dhs = float(mdhs.group(1))
            hm = float(mhm.group(1))
            avg = (tf + dhs + hm) / 3.0
            epoch = int(me.group(1)) if me else None
            rows.append(
                {
                    "file": f.name,
                    "epoch": epoch,
                    "TF": tf,
                    "DHS": dhs,
                    "HM": hm,
                    "AVG": avg,
                }
            )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


# =========================
# 2002 / 3357 robust readers
# =========================
def _safe_numeric_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def read_effect_csv_clean(path: Path) -> pd.DataFrame:
    """
    Read effect CSV robustly and drop likely non-feature columns.

    Strategy:
    1. Prefer header + first column as index.
    2. Fall back to other common separators.
    3. Remove columns whose names look like metadata.
    4. Convert all remaining columns to numeric.
    5. Drop empty / constant / monotonic integer-like columns.
    """
    if not path.exists():
        raise FileNotFoundError(path)

    tries = [
        dict(sep=",", index_col=0),
        dict(sep="\t", index_col=0),
        dict(sep=None, engine="python", index_col=0),
        dict(sep=","),
        dict(sep="\t"),
        dict(sep=None, engine="python"),
    ]

    df = None
    for kw in tries:
        try:
            tmp = pd.read_csv(path, **kw)
            if tmp.shape[1] >= 2:
                df = tmp.copy()
                break
        except Exception:
            continue

    if df is None:
        raise ValueError(f"Cannot read usable table from {path}")

    bad_name_patterns = [
        "unnamed",
        "index",
        "idx",
        "id",
        "chr",
        "chrom",
        "pos",
        "start",
        "end",
        "ref",
        "alt",
        "rs",
        "variant",
    ]
    keep_cols = []
    for c in df.columns:
        cname = str(c).strip().lower()
        if any(p in cname for p in bad_name_patterns):
            continue
        keep_cols.append(c)
    df = df[keep_cols]

    for c in df.columns:
        df[c] = _safe_numeric_series(df[c])

    if df.shape[1] > 0:
        df = df.loc[:, df.notna().mean() > 0.95]

    drop_cols = []
    for c in df.columns:
        s = df[c].dropna()
        if len(s) == 0:
            drop_cols.append(c)
            continue

        vals = s.to_numpy()

        if np.nanstd(vals) == 0:
            drop_cols.append(c)
            continue

        if s.is_monotonic_increasing and np.allclose(vals, np.round(vals)):
            drop_cols.append(c)
            continue

    df = df.drop(columns=drop_cols, errors="ignore")

    if df.shape[1] == 0:
        raise ValueError(f"No usable numeric feature columns found in {path}")

    return df


def robust_feature_scores(df: pd.DataFrame):
    arr = df.to_numpy(dtype=float)
    abs_arr = np.abs(arr)

    # Robust scores
    feature_score = np.nanmedian(abs_arr, axis=0)
    variant_score = np.nanmax(abs_arr, axis=1)
    return arr, abs_arr, feature_score, variant_score


def standardized_variant_max(abs_arr: np.ndarray) -> np.ndarray:
    """
    Standardize per feature using robust z-score based on MAD,
    then compute per-variant max standardized absolute effect.
    """
    med = np.nanmedian(abs_arr, axis=0)
    mad = np.nanmedian(np.abs(abs_arr - med), axis=0)
    mad = np.where(mad == 0, 1e-8, mad)
    z = (abs_arr - med) / mad
    return np.nanmax(np.abs(z), axis=1)


def effect_plots_v2(diff_path: Path, logfc_path: Path, prefix: str):
    """
    Prefer diff; fallback to log fold change.
    Produces qualitative effect summary figures.
    """
    src = None
    df = None

    for p in [diff_path, logfc_path]:
        if p.exists():
            try:
                df = read_effect_csv_clean(p)
                src = p
                break
            except Exception as e:
                print(f"[{prefix}] failed reading {p}: {e}")

    if df is None:
        print(f"[{prefix}] no usable data")
        return None

    arr, abs_arr, feature_score, variant_score = robust_feature_scores(df)

    # 1) Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(variant_score, bins=40)
    plt.xlabel("Per-variant max |effect|")
    plt.ylabel("Count")
    plt.title(f"{prefix}: distribution of per-variant max absolute effect")
    savefig(f"{prefix}_effect_distribution_v2.png", section="supplementary")

    # 2) Top features bar chart
    topk = min(20, len(feature_score))
    idx = np.argsort(feature_score)[-topk:][::-1]
    feat_names = [str(df.columns[i]) for i in idx]

    plt.figure(figsize=(10, 7))
    plt.barh(range(topk), feature_score[idx][::-1])
    plt.yticks(range(topk), feat_names[::-1])
    plt.xlabel("Median |effect|")
    plt.title(f"{prefix}: top {topk} features by median absolute effect")
    savefig(f"{prefix}_top_features_v2.png", section="supplementary")

    # 3) Heatmap
    top_var_n = min(50, arr.shape[0])
    top_feat_n = min(30, arr.shape[1])

    var_idx = np.argsort(variant_score)[-top_var_n:][::-1]
    feat_idx = np.argsort(feature_score)[-top_feat_n:][::-1]
    sub = arr[np.ix_(var_idx, feat_idx)]

    lo, hi = np.nanpercentile(sub, [2, 98])
    if np.isclose(lo, hi):
        lo, hi = np.nanmin(sub), np.nanmax(sub)

    plt.figure(figsize=(12, 8))
    plt.imshow(sub, aspect="auto", cmap="coolwarm", vmin=lo, vmax=hi)
    plt.colorbar(label="Effect value")
    plt.xticks(
        range(top_feat_n),
        [str(df.columns[i]) for i in feat_idx],
        rotation=90,
        fontsize=7,
    )
    plt.ylabel("Top variants")
    plt.xlabel("Top features")
    plt.title(f"{prefix}: heatmap of top variants and top features")
    savefig(f"{prefix}_heatmap_top_variants_v2.png", section="supplementary")

    return {
        "df": df,
        "arr": arr,
        "variant_score": variant_score,
        "feature_score": feature_score,
        "std_variant_max": standardized_variant_max(abs_arr),
        "source": str(src),
    }


def top_feature_boxplots(res, prefix: str, topn: int = 10):
    if res is None:
        return

    df = res["df"]
    score = res["feature_score"]
    idx = np.argsort(score)[-topn:][::-1]
    cols = [df.columns[i] for i in idx]

    data = [np.abs(df[c].to_numpy(dtype=float)) for c in cols]

    plt.figure(figsize=(12, 6))
    plt.boxplot(data, labels=[str(c) for c in cols], showfliers=False)
    plt.xticks(rotation=90)
    plt.ylabel("|effect|")
    plt.title(f"{prefix}: top {topn} features effect distribution")
    savefig(f"{prefix}_top{topn}_features_boxplot.png", section="supplementary")

def compare_2002_3357(res2002, res3357):
    """
    Main figure:
      - cleaner log1p boxplot for 2002 vs 3357

    Supplementary:
      - standardized boxplot
    """
    if res2002 is None or res3357 is None:
        print("Skip comparison: one side missing")
        return

    raw2002 = np.log1p(res2002["variant_score"])
    raw3357 = np.log1p(res3357["variant_score"])

    # ---- Main figure ----
    fig, ax = plt.subplots(figsize=(9, 6.2))

    bp = ax.boxplot(
        [raw2002, raw3357],
        labels=[f"2002\n(n={len(raw2002)})", f"3357\n(n={len(raw3357)})"],
        showfliers=False,
        patch_artist=True,
        widths=0.5,
        medianprops=dict(color="#FF7F0E", linewidth=2.6),
        whiskerprops=dict(linewidth=1.8),
        capprops=dict(linewidth=1.8),
        boxprops=dict(linewidth=1.8),
    )

    colors = ["#6BAED6", "#FDAE6B"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.72)

    med2002 = np.nanmedian(raw2002)
    med3357 = np.nanmedian(raw3357)

    iqr2002 = np.nanpercentile(raw2002, 75) - np.nanpercentile(raw2002, 25)
    iqr3357 = np.nanpercentile(raw3357, 75) - np.nanpercentile(raw3357, 25)

    # 把文字放到箱体右侧，并加白底框
    ax.text(
        1.18, med2002 + 0.002, f"median={med2002:.3f}",
        va="center", ha="left", fontsize=10,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1.5)
    )
    ax.text(
        2.18, med3357 + 0.002, f"median={med3357:.3f}",
        va="center", ha="left", fontsize=10,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1.5)
    )

    ax.text(
        1.0, np.nanpercentile(raw2002, 75) + 0.008, f"IQR≈{iqr2002:.3f}",
        ha="center", va="bottom", fontsize=9, color="dimgray",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1.0)
    )
    ax.text(
        2.0, np.nanpercentile(raw3357, 75) + 0.008, f"IQR≈{iqr3357:.3f}",
        ha="center", va="bottom", fontsize=9, color="dimgray",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1.0)
    )

    ax.set_ylabel("log1p(per-variant max |effect|)")
    ax.set_title("Qualitative effect comparison between LOGO-2002 and LOGO-3357")
    style_axes(ax)
    savefig("logo2002_vs_3357_boxplot_log1p_main.png", section="main")

    # ---- Supplementary standardized comparison ----
    std2002 = res2002["std_variant_max"]
    std3357 = res3357["std_variant_max"]

    c2002 = np.clip(std2002, None, np.nanpercentile(std2002, 99.5))
    c3357 = np.clip(std3357, None, np.nanpercentile(std3357, 99.5))

    fig, ax = plt.subplots(figsize=(9, 6))
    bp = ax.boxplot(
        [c2002, c3357],
        labels=["2002", "3357"],
        showfliers=False,
        patch_artist=True,
        widths=0.5,
        medianprops=dict(color="#FF7F0E", linewidth=2.2),
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)

    ax.set_ylabel("Per-variant max standardized |effect|")
    ax.set_title("2002 vs 3357 effect comparison (standardized, clipped)")
    style_axes(ax)
    savefig("logo2002_vs_3357_boxplot_standardized.png", section="supplementary")

# =========================
# 1) LOGO-919 fine-tuning curves
# =========================
if summary_csv.exists():
    df_train = pd.read_csv(summary_csv)
else:
    df_train = parse_train_log(train_log)

if not df_train.empty:
    # supplementary: loss
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df_train["epoch"], df_train["loss"], label="train loss", linewidth=2)
    ax.plot(df_train["epoch"], df_train["val_loss"], label="val loss", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("LOGO-919 fine-tuning loss")
    style_axes(ax)
    ax.legend(frameon=True)
    savefig("logo919_training_loss.png", section="supplementary")

    # supplementary: auc
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df_train["epoch"], df_train["auc"], label="train auc", linewidth=2)
    ax.plot(df_train["epoch"], df_train["val_auc"], label="val auc", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("AUC")
    ax.set_title("LOGO-919 fine-tuning AUC")
    style_axes(ax)
    ax.legend(frameon=True)
    savefig("logo919_training_auc.png", section="supplementary")

    # supplementary: accuracy
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df_train["epoch"], df_train["accuracy"], label="train accuracy", linewidth=2)
    ax.plot(df_train["epoch"], df_train["val_accuracy"], label="val accuracy", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("LOGO-919 fine-tuning accuracy")
    style_axes(ax)
    ax.legend(frameon=True)
    savefig("logo919_training_accuracy.png", section="supplementary")


# =========================
# 2) LOGO-919 checkpoint comparison
# =========================
df_test = parse_test_logs(test_logs_dir)
if not df_test.empty:
    df_test_plot = df_test.sort_values("epoch").reset_index(drop=True)
    x = np.arange(len(df_test_plot))
    w = 0.22

    fig, ax = plt.subplots(figsize=(12, 6))

    tf_colors = ["#4C78A8"] * len(df_test_plot)
    dhs_colors = ["#F58518"] * len(df_test_plot)
    hm_colors = ["#54A24B"] * len(df_test_plot)

    best_epoch = 150
    if best_epoch in df_test_plot["epoch"].tolist():
        best_idx = df_test_plot.index[df_test_plot["epoch"] == best_epoch][0]
        tf_colors[best_idx] = "#2F5597"
        dhs_colors[best_idx] = "#C55A11"
        hm_colors[best_idx] = "#2E7D32"

    bars1 = ax.bar(x - w, df_test_plot["TF"], width=w, label="TF", color=tf_colors)
    bars2 = ax.bar(x, df_test_plot["DHS"], width=w, label="DHS", color=dhs_colors)
    bars3 = ax.bar(x + w, df_test_plot["HM"], width=w, label="HM", color=hm_colors)

    ax.set_xticks(x)
    ax.set_xticklabels([str(e) for e in df_test_plot["epoch"]])
    ax.set_ylim(0.75, 1.0)
    ax.set_xlabel("Checkpoint epoch")
    ax.set_ylabel("Median AUROC")
    ax.set_title("LOGO-919 checkpoint comparison on held-out test")
    style_axes(ax)

    if best_epoch in df_test_plot["epoch"].tolist():
        ax.text(
            best_idx,
            0.997,
            "best",
            ha="center",
            va="top",
            fontsize=12,
            fontweight="bold",
            color="black",
        )
        
    ax.legend(frameon=True)
    savefig("logo919_checkpoint_comparison_main.png", section="main")


# =========================
# 3) Paper vs reproduced LOGO-919
# =========================
cats = ["TF", "DHS", "HM"]
paper_vals = [paper_919[c] for c in cats]
repro_vals = [best_919[c] for c in cats]
x = np.arange(len(cats))
w = 0.34

fig, ax = plt.subplots(figsize=(8.5, 6))

bars1 = ax.bar(
    x - w / 2,
    paper_vals,
    width=w,
    label="Paper LOGO-919",
    color="#4C78A8",
)
bars2 = ax.bar(
    x + w / 2,
    repro_vals,
    width=w,
    label="Your LOGO-919",
    color="#F58518",
)

ax.set_xticks(x)
ax.set_xticklabels(cats)
ax.set_ylim(0.75, 1.0)
ax.set_ylabel("Median AUROC")
ax.set_title("Paper vs reproduced LOGO-919")
style_axes(ax)

for i, (p, r) in enumerate(zip(paper_vals, repro_vals)):
    delta = r - p
    ax.plot(
        [i - w / 2, i + w / 2],
        [max(p, r) + 0.006, max(p, r) + 0.006],
        color="gray",
        lw=1,
    )
    ax.text(
        i,
        max(p, r) + 0.008,
        f"Δ={delta:.3f}",
        ha="center",
        va="bottom",
        fontsize=10,
        color="dimgray",
    )

ax.legend(frameon=True)
savefig("logo919_paper_vs_repro_main.png", section="main")


# =========================
# 4) 2002 / 3357 qualitative effect plots
# =========================
res2002 = effect_plots_v2(f2002_diff, f2002_logfc, "logo2002")
res3357 = effect_plots_v2(f3357_diff, f3357_logfc, "logo3357")
top_feature_boxplots(res2002, "logo2002", topn=10)
top_feature_boxplots(res3357, "logo3357", topn=10)
compare_2002_3357(res2002, res3357)

if res2002 is not None:
    print("2002 source:", res2002["source"], "shape:", res2002["df"].shape)
    print("2002 first columns:", list(res2002["df"].columns[:10]))

if res3357 is not None:
    print("3357 source:", res3357["source"], "shape:", res3357["df"].shape)
    print("3357 first columns:", list(res3357["df"].columns[:10]))

write_manifest()
print(f"\nAll figures saved under: {OUTDIR}")
print(f"Main figures: {MAIN_DIR}")
print(f"Supplementary figures: {SUPP_DIR}")