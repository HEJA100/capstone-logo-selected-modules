import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

csv_path = "02_LOGO_Promoter/promoter_results_all_models.csv"
out_file = "02_LOGO_Promoter/promoter_f1_gain_bar.png"

task_order = ["BOTH", "TATA_BOX", "NO_TATA_BOX"]

df = pd.read_csv(csv_path)

def get_f1(task, model):
    row = df[(df["task"] == task) & (df["model"] == model)]
    return float(row["f1_mean"].iloc[0])

gain_rows = []
for task in task_order:
    cnn = get_f1(task, "CNN")
    bilstm = get_f1(task, "BiLSTM")
    logo_seq = get_f1(task, "LOGO_sequence_only")
    logo_k = get_f1(task, "LOGO_knowledge_enabled")

    gain_rows.append([
        task,
        bilstm - cnn,
        logo_seq - bilstm,
        logo_k - logo_seq
    ])

gain_df = pd.DataFrame(
    gain_rows,
    columns=["task", "BiLSTM_minus_CNN", "LOGOseq_minus_BiLSTM", "LOGOK_minus_LOGOseq"]
)

print("=== CSV_GAINS ===")
print("task,BiLSTM_minus_CNN,LOGOseq_minus_BiLSTM,LOGOK_minus_LOGOseq")
for _, r in gain_df.iterrows():
    print(f"{r['task']},{r['BiLSTM_minus_CNN']:.6f},{r['LOGOseq_minus_BiLSTM']:.6f},{r['LOGOK_minus_LOGOseq']:.6f}")

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(task_order))
width = 0.22

ax.bar(x - width, gain_df["BiLSTM_minus_CNN"], width, label="BiLSTM - CNN")
ax.bar(x, gain_df["LOGOseq_minus_BiLSTM"], width, label="LOGO_seq - BiLSTM")
ax.bar(x + width, gain_df["LOGOK_minus_LOGOseq"], width, label="LOGO_K - LOGO_seq")

ax.set_xticks(x)
ax.set_xticklabels(task_order)
ax.set_ylabel("ΔF1")
ax.set_title("Incremental F1 gains across model upgrades")
ax.axhline(0, linewidth=1)
ax.legend(frameon=False)
ax.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig(out_file, dpi=300, bbox_inches="tight")
print("Saved:", out_file)
