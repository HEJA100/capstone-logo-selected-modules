
import csv
import math
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path("/home/users/nus/e1538285/scratch/LOGO")
PROM = ROOT / "02_LOGO_Promoter"
LOGDIR = PROM / "locality_logs"

def infer_task(line: str):
    u = line.upper()
    if "NONTATA" in u or "NO_TATA" in u or "NO-TATA" in u:
        return "NO_TATA_BOX"
    if "TATA_BOX" in u or ("TATA" in u and "NONTATA" not in u and "NO_TATA" not in u):
        return "TATA_BOX"
    if "EPDNEW_BOTH" in u or "BOTH" in u:
        return "BOTH"
    return None

def parse_eval(line: str):
    m = re.search(r"Eval:\s*\[([^\]]+)\]", line)
    if not m:
        return None
    parts = [x.strip() for x in m.group(1).split(",")]
    if len(parts) < 5:
        return None
    vals = [float(x) for x in parts[:5]]
    return {
        "loss": vals[0],
        "acc": vals[1],
        "precision": vals[2],
        "recall": vals[3],
        "f1": vals[4],
    }

def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")

def sd(xs):
    if len(xs) <= 1:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))

rows = []
current_task = {}

for path in sorted(LOGDIR.glob("*.log")):
    stem = path.stem
    if "_" not in stem:
        continue
    branch_short, mode = stem.split("_", 1)
    branch = "sequence_only" if branch_short == "seq" else "structural_knowledge"
    current_task[path.name] = None

    with path.open() as f:
        for line in f:
            task = infer_task(line)
            if task is not None:
                current_task[path.name] = task

            ev = parse_eval(line)
            if ev is None:
                continue

            task = current_task[path.name]
            if task is None:
                continue

            rows.append({
                "task": task,
                "branch": branch,
                "locality_mode": mode,
                **ev,
                "source": path.name,
            })

# add existing corners if present
final_table = PROM / "promoter_final_comparison_table.csv"
if final_table.exists():
    with final_table.open() as f:
        for r in csv.DictReader(f):
            if r.get("model") == "LOGO_sequence_only":
                rows.append({
                    "task": r["task"],
                    "branch": "sequence_only",
                    "locality_mode": "none",
                    "loss": float("nan"),
                    "acc": float(r["acc_mean"]),
                    "precision": float(r["precision_mean"]),
                    "recall": float(r["recall_mean"]),
                    "f1": float(r["f1_mean"]),
                    "source": "existing_promoter_final_comparison_table.csv",
                })

knowledge_table = PROM / "knowledge_ablation_results.csv"
if knowledge_table.exists():
    with knowledge_table.open() as f:
        for r in csv.DictReader(f):
            if r.get("knowledge_type") == "structural":
                rows.append({
                    "task": r["task"],
                    "branch": "structural_knowledge",
                    "locality_mode": "multi",
                    "loss": float("nan"),
                    "acc": float(r["acc_mean"]),
                    "precision": float(r["precision_mean"]),
                    "recall": float(r["recall_mean"]),
                    "f1": float(r["f1_mean"]),
                    "source": "existing_knowledge_ablation_results.csv",
                })

grouped = defaultdict(list)
for r in rows:
    grouped[(r["task"], r["branch"], r["locality_mode"])].append(r)

out_csv = PROM / "locality_ablation_results.csv"
with out_csv.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow([
        "task", "branch", "locality_mode", "n_folds",
        "acc_mean", "acc_sd",
        "precision_mean", "precision_sd",
        "recall_mean", "recall_sd",
        "f1_mean", "f1_sd",
        "sources"
    ])
    for key in sorted(grouped):
        xs = grouped[key]
        w.writerow([
            key[0], key[1], key[2], len(xs),
            mean([x["acc"] for x in xs]), sd([x["acc"] for x in xs]),
            mean([x["precision"] for x in xs]), sd([x["precision"] for x in xs]),
            mean([x["recall"] for x in xs]), sd([x["recall"] for x in xs]),
            mean([x["f1"] for x in xs]), sd([x["f1"] for x in xs]),
            ";".join(sorted({x["source"] for x in xs})),
        ])

print(f"[OK] wrote {out_csv}")
