import csv
import ast
import math
from pathlib import Path
from statistics import mean, stdev

ROOT = Path("/home/users/nus/e1538285/scratch/LOGO/02_LOGO_Promoter")

LOGS = {
    "coarsebin25": ROOT / "K_coarsebin25.log",
    "distmask_tsspm100": ROOT / "K_distmask_tsspm100.log",
}

TASK_KEYS = {
    "epdnew_BOTH": "BOTH",
    "epdnew_TATA_BOX": "TATA_BOX",
    "epdnew_NO_TATA_BOX": "NO_TATA_BOX",
}

METRIC_NAMES = ["loss", "acc", "precision", "recall", "f1"]

def parse_log(path: Path):
    task_to_rows = {"BOTH": [], "TATA_BOX": [], "NO_TATA_BOX": []}
    current_task = None

    with path.open() as f:
        for line in f:
            for key, task in TASK_KEYS.items():
                if key in line:
                    current_task = task
                    break

            if "Eval:" in line and current_task is not None:
                payload = line.split("Eval:", 1)[1].strip()
                vals = ast.literal_eval(payload)
                if len(vals) != 5:
                    continue
                task_to_rows[current_task].append(vals)

    return task_to_rows

def msd(xs):
    if not xs:
        return ("", "")
    if len(xs) == 1:
        return (xs[0], 0.0)
    return (mean(xs), stdev(xs))

all_rows = []
for knowledge_type, log_path in LOGS.items():
    if not log_path.exists():
        raise FileNotFoundError(f"missing log: {log_path}")

    parsed = parse_log(log_path)

    for task in ["BOTH", "TATA_BOX", "NO_TATA_BOX"]:
        rows = parsed[task]
        if len(rows) != 10:
            raise RuntimeError(f"{knowledge_type} / {task}: expected 10 Eval rows, got {len(rows)}")

        cols = list(zip(*rows))
        stats = {}
        for name, values in zip(METRIC_NAMES, cols):
            m, s = msd(list(values))
            stats[f"{name}_mean"] = m
            stats[f"{name}_sd"] = s

        all_rows.append({
            "knowledge_type": knowledge_type,
            "task": task,
            "n_folds": len(rows),
            **stats,
        })

out1 = ROOT / "anti_leakage_results.csv"
with out1.open("w", newline="") as f:
    fieldnames = [
        "knowledge_type", "task", "n_folds",
        "loss_mean", "acc_mean", "precision_mean", "recall_mean", "f1_mean",
        "loss_sd", "acc_sd", "precision_sd", "recall_sd", "f1_sd",
    ]
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for r in all_rows:
        w.writerow(r)

# ---------- merge with existing comparison table ----------
base_csv = ROOT / "promoter_final_comparison_table.csv"
base_rows = []
with base_csv.open() as f:
    reader = csv.DictReader(f)
    for r in reader:
        base_rows.append(r)

base_lookup = {(r["task"], r["model"]): r for r in base_rows}
anti_lookup = {(r["task"], r["knowledge_type"]): r for r in all_rows}

final_rows = []
tasks = ["BOTH", "TATA_BOX", "NO_TATA_BOX"]

for task in tasks:
    seq = float(base_lookup[(task, "LOGO_sequence_only")]["f1_mean"])
    full = float(base_lookup[(task, "LOGO_knowledge_enabled")]["f1_mean"])
    shuf = float(base_lookup[(task, "knowledge_shuffled")]["f1_mean"])

    # existing baseline rows
    keep_models = [
        "LOGO_sequence_only",
        "LOGO_knowledge_enabled",
        "knowledge_shuffled",
    ]
    for model in keep_models:
        r = dict(base_lookup[(task, model)])
        f1 = float(r["f1_mean"])
        r["delta_vs_sequence_only_f1"] = f1 - seq
        r["delta_vs_full_knowledge_f1"] = f1 - full
        r["delta_vs_shuffled_f1"] = f1 - shuf
        r["retention_vs_full_knowledge_pct"] = 100.0 * f1 / full if full else ""
        final_rows.append(r)

    # anti-leakage rows
    for ktype, model_name in [
        ("coarsebin25", "knowledge_coarsebin25"),
        ("distmask_tsspm100", "knowledge_distmask_tsspm100"),
    ]:
        r = anti_lookup[(task, ktype)]
        f1 = float(r["f1_mean"])
        row = {
            "task": task,
            "model": model_name,
            "acc_mean": r["acc_mean"],
            "precision_mean": r["precision_mean"],
            "recall_mean": r["recall_mean"],
            "f1_mean": r["f1_mean"],
            "acc_sd": r["acc_sd"],
            "precision_sd": r["precision_sd"],
            "recall_sd": r["recall_sd"],
            "f1_sd": r["f1_sd"],
            "delta_vs_sequence_only_f1": f1 - seq,
            "delta_vs_full_knowledge_f1": f1 - full,
            "delta_vs_shuffled_f1": f1 - shuf,
            "retention_vs_full_knowledge_pct": 100.0 * f1 / full if full else "",
        }
        final_rows.append(row)

model_order = {
    "LOGO_sequence_only": 0,
    "LOGO_knowledge_enabled": 1,
    "knowledge_coarsebin25": 2,
    "knowledge_distmask_tsspm100": 3,
    "knowledge_shuffled": 4,
}
task_order = {"BOTH": 0, "TATA_BOX": 1, "NO_TATA_BOX": 2}

final_rows.sort(key=lambda r: (task_order[r["task"]], model_order[r["model"]]))

out2 = ROOT / "anti_leakage_comparison_table.csv"
with out2.open("w", newline="") as f:
    fieldnames = [
        "task", "model",
        "acc_mean", "precision_mean", "recall_mean", "f1_mean",
        "acc_sd", "precision_sd", "recall_sd", "f1_sd",
        "delta_vs_sequence_only_f1",
        "delta_vs_full_knowledge_f1",
        "delta_vs_shuffled_f1",
        "retention_vs_full_knowledge_pct",
    ]
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for r in final_rows:
        w.writerow(r)

print(out1)
print(out2)
