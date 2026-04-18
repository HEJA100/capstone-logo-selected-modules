import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

csv_path = "02_LOGO_Promoter/promoter_results_all_models_with_sd.csv"
out_dir = "02_LOGO_Promoter"

task_order = ["BOTH", "TATA_BOX", "NO_TATA_BOX"]
model_order = ["CNN", "BiLSTM", "LOGO_sequence_only", "LOGO_knowledge_enabled"]

df = pd.read_csv(csv_path)
df["task"] = pd.Categorical(df["task"], categories=task_order, ordered=True)
df["model"] = pd.Categorical(df["model"], categories=model_order, ordered=True)
df = df.sort_values(["task", "model"])

def vals(metric):
    out = {}
    for model in model_order:
        out[model] = []
        for task in task_order:
            row = df[(df["task"] == task) & (df["model"] == model)]
            out[model].append(float(row[metric].iloc[0]))
    return out

# S1: Accuracy grouped bar
acc = vals("acc_mean")
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(task_order))
width = 0.18
for i, model in enumerate(model_order):
    ax.bar(x + (i - 1.5) * width, acc[model], width, label=model)
ax.set_xticks(x)
ax.set_xticklabels(task_order)
ax.set_ylabel("Accuracy")
ax.set_title("Promoter prediction performance across models (Accuracy)")
ax.set_ylim(0.7, 1.0)
ax.legend(frameon=False)
ax.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(f"{out_dir}/promoter_accuracy_grouped_bar.png", dpi=300, bbox_inches="tight")
plt.close()

# S2: Precision and Recall dual-panel
prec = vals("precision_mean")
rec = vals("recall_mean")
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

for i, model in enumerate(model_order):
    axes[0].bar(x + (i - 1.5) * width, prec[model], width, label=model)
axes[0].set_xticks(x)
axes[0].set_xticklabels(task_order)
axes[0].set_ylabel("Precision")
axes[0].set_title("Precision comparison")
axes[0].set_ylim(0.7, 1.0)
axes[0].grid(axis="y", linestyle="--", alpha=0.4)

for i, model in enumerate(model_order):
    axes[1].bar(x + (i - 1.5) * width, rec[model], width, label=model)
axes[1].set_xticks(x)
axes[1].set_xticklabels(task_order)
axes[1].set_ylabel("Recall")
axes[1].set_title("Recall comparison")
axes[1].set_ylim(0.7, 1.0)
axes[1].grid(axis="y", linestyle="--", alpha=0.4)

handles, labels = axes[1].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig(f"{out_dir}/promoter_precision_recall_grouped_bar.png", dpi=300, bbox_inches="tight")
plt.close()

# S4: F1 mean ± SD error-bar plot
fig, ax = plt.subplots(figsize=(10, 6))
markers = ["o", "s", "^", "D"]

for model, marker in zip(model_order, markers):
    y = []
    yerr = []
    for task in task_order:
        row = df[(df["task"] == task) & (df["model"] == model)]
        y.append(float(row["f1_mean"].iloc[0]))
        yerr.append(float(row["f1_sd"].iloc[0]))
    ax.errorbar(x, y, yerr=yerr, marker=marker, linewidth=2, capsize=4, label=model)

ax.set_xticks(x)
ax.set_xticklabels(task_order)
ax.set_ylabel("F1-score (mean ± SD)")
ax.set_title("Promoter prediction F1-score with 10-fold variability")
ax.set_ylim(0.7, 1.0)
ax.legend(frameon=False)
ax.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(f"{out_dir}/promoter_f1_errorbar.png", dpi=300, bbox_inches="tight")
plt.close()

print("Saved:")
print(f"{out_dir}/promoter_accuracy_grouped_bar.png")
print(f"{out_dir}/promoter_precision_recall_grouped_bar.png")
print(f"{out_dir}/promoter_f1_errorbar.png")
