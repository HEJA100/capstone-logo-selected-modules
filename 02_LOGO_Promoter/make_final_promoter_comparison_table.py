from pathlib import Path
import csv

root = Path("/home/users/nus/e1538285/scratch/LOGO/02_LOGO_Promoter")
base_csv = root / "promoter_results_all_models.csv"
abl_csv = root / "knowledge_ablation_results.csv"
out_csv = root / "promoter_final_comparison_table.csv"

# 统一输出顺序
task_order = ["BOTH", "TATA_BOX", "NO_TATA_BOX"]
model_order = [
    "LOGO_sequence_only",
    "LOGO_knowledge_enabled",
    "knowledge_structural",
    "knowledge_regulatory",
    "knowledge_shuffled",
]

rows = []

# 读取原总表
with base_csv.open() as f:
    r = csv.DictReader(f)
    for row in r:
        model = row["model"]
        if model not in {"LOGO_sequence_only", "LOGO_knowledge_enabled"}:
            continue
        rows.append({
            "task": row["task"],
            "model": model,
            "acc_mean": float(row["acc_mean"]),
            "precision_mean": float(row["precision_mean"]),
            "recall_mean": float(row["recall_mean"]),
            "f1_mean": float(row["f1_mean"]),
            "acc_sd": "",
            "precision_sd": "",
            "recall_sd": "",
            "f1_sd": "",
        })

# 读取 ablation 表
name_map = {
    "structural": "knowledge_structural",
    "regulatory": "knowledge_regulatory",
    "shuffled": "knowledge_shuffled",
}

with abl_csv.open() as f:
    r = csv.DictReader(f)
    for row in r:
        rows.append({
            "task": row["knowledge_type"] and row["task"],
            "model": name_map[row["knowledge_type"]],
            "acc_mean": float(row["acc_mean"]),
            "precision_mean": float(row["precision_mean"]),
            "recall_mean": float(row["recall_mean"]),
            "f1_mean": float(row["f1_mean"]),
            "acc_sd": row["acc_sd"],
            "precision_sd": row["precision_sd"],
            "recall_sd": row["recall_sd"],
            "f1_sd": row["f1_sd"],
        })

# 排序
task_rank = {t: i for i, t in enumerate(task_order)}
model_rank = {m: i for i, m in enumerate(model_order)}
rows.sort(key=lambda x: (task_rank[x["task"]], model_rank[x["model"]]))

# 计算相对 sequence-only / shuffled 的 F1 增益
seq_f1 = {}
shuf_f1 = {}
for r in rows:
    if r["model"] == "LOGO_sequence_only":
        seq_f1[r["task"]] = r["f1_mean"]
    if r["model"] == "knowledge_shuffled":
        shuf_f1[r["task"]] = r["f1_mean"]

for r in rows:
    t = r["task"]
    r["delta_vs_sequence_only_f1"] = round(r["f1_mean"] - seq_f1.get(t, 0), 6)
    r["delta_vs_shuffled_f1"] = round(r["f1_mean"] - shuf_f1.get(t, 0), 6) if t in shuf_f1 else ""

# 写出
fields = [
    "task", "model",
    "acc_mean", "precision_mean", "recall_mean", "f1_mean",
    "acc_sd", "precision_sd", "recall_sd", "f1_sd",
    "delta_vs_sequence_only_f1", "delta_vs_shuffled_f1"
]

with out_csv.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)

print(out_csv)
for r in rows:
    print(
        r["task"], r["model"],
        f'acc={r["acc_mean"]:.6f}',
        f'prec={r["precision_mean"]:.6f}',
        f'rec={r["recall_mean"]:.6f}',
        f'f1={r["f1_mean"]:.6f}',
        f'delta_vs_seq={r["delta_vs_sequence_only_f1"]}',
        f'delta_vs_shuf={r["delta_vs_shuffled_f1"]}',
    )
