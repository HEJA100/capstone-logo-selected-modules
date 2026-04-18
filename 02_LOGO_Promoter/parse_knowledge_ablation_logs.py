from pathlib import Path
import re
import csv
from statistics import mean, pstdev

ROOT = Path("/home/users/nus/e1538285/scratch/LOGO/02_LOGO_Promoter")

LOGS = {
    "structural": ROOT / "K_structural.log",
    "regulatory": ROOT / "K_regulatory.log",
    "shuffled": ROOT / "K_shuffled.log",
}

TASK_PATTERNS = {
    "BOTH": re.compile(r"epdnew_BOTH_Knowledge_5_gram\.npz"),
    "TATA_BOX": re.compile(r"epdnew_TATA_BOX_Knowledge_5_gram\.npz"),
    "NO_TATA_BOX": re.compile(r"epdnew_NO_TATA_BOX_Knowledge_5_gram\.npz"),
}

EVAL_RE = re.compile(
    r"Eval:\s*\[\s*([0-9eE\+\-\.]+)\s*,\s*([0-9eE\+\-\.]+)\s*,\s*([0-9eE\+\-\.]+)\s*,\s*([0-9eE\+\-\.]+)\s*,\s*([0-9eE\+\-\.]+)\s*\]"
)

rows = []

for tag, log_path in LOGS.items():
    if not log_path.exists():
        print(f"missing log: {log_path}")
        continue

    current_task = None
    task_to_metrics = {"BOTH": [], "TATA_BOX": [], "NO_TATA_BOX": []}

    for line in log_path.read_text(errors="ignore").splitlines():
        for task, pat in TASK_PATTERNS.items():
            if pat.search(line):
                current_task = task

        m = EVAL_RE.search(line)
        if m and current_task is not None:
            vals = list(map(float, m.groups()))
            # loss, acc, precision, recall, f1
            task_to_metrics[current_task].append(vals)

    for task, arr in task_to_metrics.items():
        if not arr:
            rows.append([tag, task, 0, "", "", "", "", "", "", "", "", ""])
            continue

        loss = [x[0] for x in arr]
        acc = [x[1] for x in arr]
        prec = [x[2] for x in arr]
        rec = [x[3] for x in arr]
        f1 = [x[4] for x in arr]

        rows.append([
            tag, task, len(arr),
            mean(loss), mean(acc), mean(prec), mean(rec), mean(f1),
            pstdev(acc), pstdev(prec), pstdev(rec), pstdev(f1)
        ])

out_csv = ROOT / "knowledge_ablation_results.csv"
with out_csv.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow([
        "knowledge_type", "task", "n_folds",
        "loss_mean", "acc_mean", "precision_mean", "recall_mean", "f1_mean",
        "acc_sd", "precision_sd", "recall_sd", "f1_sd"
    ])
    w.writerows(rows)

print(out_csv)
for r in rows:
    print(r)
