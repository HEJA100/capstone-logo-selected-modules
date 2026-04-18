#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


def parse_args():
    p = argparse.ArgumentParser(description="Extract tB/tb EPI evaluation results and generate CSV + plot-ready summary.")
    p.add_argument("--root", default="~/scratch/LOGO/03_LOGO_EPI", help="Root directory of 03_LOGO_EPI")
    p.add_argument("--outdir", default="~/scratch/LOGO/03_LOGO_EPI/my_repro_03_figures", help="Output directory")
    return p.parse_args()


def infer_tag(relpath: str, context: str) -> str:
    s = (relpath + "\n" + context).lower()
    if any(k in s for k in ["knowledge", "know", "kb", "prior"]):
        return "knowledge_enabled"
    if any(k in s for k in ["extension", "extend", "enhance"]):
        return "extension_other"
    return "baseline_or_other"


def infer_cell(line: str) -> str | None:
    # Most common case: Eval: [..] CELLTYPE
    m = re.search(r"Eval:\s*\[[^\]]+\]\s*([A-Za-z0-9_\-]+)", line)
    if m:
        return m.group(1)

    candidates = [
        r"\btb-?6\b",
        r"\btb\b",
        r"\btB\b",
        r"\bTB\b",
        r"\bBcell\b",
        r"\bB-cell\b",
    ]
    for pat in candidates:
        m = re.search(pat, line, flags=re.I)
        if m:
            return m.group(0)
    return None


def is_tb_like(cell: str) -> bool:
    c = cell.lower()
    return c in {"tb", "tb-6", "tb6", "bcell", "b-cell"} or c.startswith("tb")


def parse_eval_line(line: str):
    m = re.search(r"Eval:\s*\[([^\]]+)\]", line)
    if not m:
        return None
    nums = [float(x) for x in re.findall(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+", m.group(1))]
    if not nums:
        return None
    return nums


def safe_sd(xs):
    return stdev(xs) if len(xs) >= 2 else 0.0


def main():
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    exts = {".log", ".out", ".txt"}
    raw_rows = []

    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in exts:
            continue
        try:
            lines = path.read_text(errors="ignore").splitlines()
        except Exception:
            continue
        rel = str(path.relative_to(root))
        for i, line in enumerate(lines):
            if "Eval:" not in line:
                continue
            nums = parse_eval_line(line)
            if not nums:
                continue
            cell = infer_cell(line)
            context = "\n".join(lines[max(0, i - 5): min(len(lines), i + 6)])
            if cell is None:
                cell = infer_cell(context)
            if cell is None or not is_tb_like(cell):
                continue
            tag = infer_tag(rel, context)
            row = {
                "file": rel,
                "tag": tag,
                "cell_type": cell,
                "eval_vector": " ".join(str(x) for x in nums),
            }
            for j, x in enumerate(nums, start=1):
                row[f"score{j}"] = x
            raw_rows.append(row)

    raw_csv = outdir / "tb_knowledge_eval_raw.csv"
    if not raw_rows:
        with raw_csv.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["message"])
            writer.writerow(["No tb/tB Eval lines found. Run the diagnostic grep commands from the chat response."])
        print(f"No tb/tB Eval lines found. Wrote placeholder: {raw_csv}")
        return

    max_scores = max(max(int(k.replace('score', '')) for k in r.keys() if k.startswith('score')) for r in raw_rows)
    raw_fields = ["file", "tag", "cell_type", "eval_vector"] + [f"score{i}" for i in range(1, max_scores + 1)]
    with raw_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=raw_fields)
        writer.writeheader()
        for r in raw_rows:
            writer.writerow(r)

    grouped = defaultdict(list)
    for r in raw_rows:
        grouped[r["tag"]].append(r)

    summary_rows = []
    for tag, rows in grouped.items():
        rec = {"tag": tag, "n": len(rows)}
        for i in range(1, max_scores + 1):
            xs = [float(r.get(f"score{i}", math.nan)) for r in rows if r.get(f"score{i}") is not None]
            xs = [x for x in xs if not math.isnan(x)]
            if xs:
                rec[f"score{i}_mean"] = mean(xs)
                rec[f"score{i}_sd"] = safe_sd(xs)
            else:
                rec[f"score{i}_mean"] = ""
                rec[f"score{i}_sd"] = ""
        summary_rows.append(rec)

    summary_rows.sort(key=lambda x: (0 if x["tag"] == "knowledge_enabled" else 1, x["tag"]))
    summary_csv = outdir / "tb_knowledge_eval_summary.csv"
    fields = ["tag", "n"] + [z for i in range(1, max_scores + 1) for z in (f"score{i}_mean", f"score{i}_sd")]
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in summary_rows:
            writer.writerow(r)

    # Plot score2 and score3 if present, else fallback to last two scores or one score.
    try:
        import matplotlib.pyplot as plt
        plot_scores = []
        if max_scores >= 3:
            plot_scores = [2, 3]
        elif max_scores == 2:
            plot_scores = [1, 2]
        else:
            plot_scores = [1]

        tags = [r["tag"] for r in summary_rows]
        width = 0.35
        x = list(range(len(tags)))

        fig, axes = plt.subplots(1, len(plot_scores), figsize=(6 * len(plot_scores), 5), squeeze=False)
        axes = axes[0]
        for ax, si in zip(axes, plot_scores):
            ys = [float(r.get(f"score{si}_mean") or 0) for r in summary_rows]
            es = [float(r.get(f"score{si}_sd") or 0) for r in summary_rows]
            ax.bar(x, ys, yerr=es, capsize=4)
            ax.set_xticks(x)
            ax.set_xticklabels(tags, rotation=15, ha="right")
            ax.set_ylabel(f"score{si}")
            ax.set_title(f"tB summary: score{si}")
            for xi, yi in zip(x, ys):
                ax.text(xi, yi, f"{yi:.4f}", ha="center", va="bottom", fontsize=9)
            ax.grid(axis="y", linestyle="--", alpha=0.3)
        fig.suptitle("03_LOGO_EPI tB knowledge-enabled summary", fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        fig.savefig(outdir / "tb_knowledge_eval_summary.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print(f"Plot generation skipped: {e}")

    print(f"Wrote raw CSV: {raw_csv}")
    print(f"Wrote summary CSV: {summary_csv}")
    print(f"If matplotlib was available, wrote PNG: {outdir / 'tb_knowledge_eval_summary.png'}")
    print("Done.")


if __name__ == "__main__":
    main()
