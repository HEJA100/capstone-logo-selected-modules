import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

csv_path = "02_LOGO_Promoter/promoter_results_all_models.csv"
out_file = "02_LOGO_Promoter/promoter_f1_grouped_bar.png"

task_order = ["BOTH", "TATA_BOX", "NO_TATA_BOX"]
model_order = ["CNN", "BiLSTM", "LOGO_sequence_only", "LOGO_knowledge_enabled"]

df = pd.read_csv(csv_path)
df["task"] = pd.Categorical(df["task"], categories=task_order, ordered=True)
df["model"] = pd.Categorical(df["model"], categories=model_order, ordered=True)
df = df.sort_values(["task", "model"])

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(task_order))
width = 0.18

for i, model in enumerate(model_order):
    vals = []
    for task in task_order:
        row = df[(df["task"] == task) & (df["model"] == model)]
        vals.append(float(row["f1_mean"].iloc[0]))
    ax.bar(x + (i - 1.5) * width, vals, width, label=model)

ax.set_xticks(x)
ax.set_xticklabels(task_order)
ax.set_ylabel("F1-score")
ax.set_title("Promoter prediction performance across models (F1-score)")
ax.set_ylim(0.7, 1.0)
ax.legend(frameon=False)
ax.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig(out_file, dpi=300, bbox_inches="tight")
print("Saved:", out_file)
