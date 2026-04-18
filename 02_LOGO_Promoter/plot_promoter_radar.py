import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

csv_path = "02_LOGO_Promoter/promoter_results_all_models.csv"
out_file = "02_LOGO_Promoter/promoter_radar_3tasks.png"

df = pd.read_csv(csv_path)

task_order = ["BOTH", "TATA_BOX", "NO_TATA_BOX"]
model_order = ["CNN", "BiLSTM", "LOGO_sequence_only", "LOGO_knowledge_enabled"]
metrics = ["acc_mean", "precision_mean", "recall_mean", "f1_mean"]
metric_labels = ["Accuracy", "Precision", "Recall", "F1"]

angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
angles += angles[:1]

fig, axes = plt.subplots(1, 3, figsize=(15, 5), subplot_kw=dict(polar=True))

for ax, task in zip(axes, task_order):
    sub = df[df["task"] == task].copy()
    sub["model"] = pd.Categorical(sub["model"], categories=model_order, ordered=True)
    sub = sub.sort_values("model")

    for _, row in sub.iterrows():
        values = [row[m] for m in metrics]
        values += values[:1]
        ax.plot(angles, values, linewidth=2, label=row["model"])
        ax.fill(angles, values, alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0.7, 1.0)
    ax.set_title(task, pad=18)

handles, labels = axes[-1].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
plt.tight_layout(rect=[0, 0, 1, 0.90])
plt.savefig(out_file, dpi=300, bbox_inches="tight")
print("Saved:", out_file)
