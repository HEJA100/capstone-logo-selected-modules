import re
from pathlib import Path
import matplotlib.pyplot as plt

files = {
    "CNN": Path("02_LOGO_Promoter/cnn_both.log"),
    "BiLSTM": Path("02_LOGO_Promoter/bilstm_BOTH.log"),
}

pattern = re.compile(
    r"Epoch\s+(\d+)/\d+.*?\n.*?loss:\s*([0-9.]+)\s*-\s*accuracy:\s*([0-9.]+)\s*-\s*val_loss:\s*([0-9.]+)\s*-\s*val_accuracy:\s*([0-9.]+)",
    re.S
)

parsed = {}

for name, path in files.items():
    if not path.exists():
        print(f"Missing log: {path}")
        continue
    text = path.read_text(errors="ignore")
    matches = pattern.findall(text)
    if not matches:
        print(f"No epoch records found in: {path}")
        continue

    epochs = [int(m[0]) for m in matches]
    loss = [float(m[1]) for m in matches]
    acc = [float(m[2]) for m in matches]
    val_loss = [float(m[3]) for m in matches]
    val_acc = [float(m[4]) for m in matches]

    parsed[name] = {
        "epochs": epochs,
        "loss": loss,
        "acc": acc,
        "val_loss": val_loss,
        "val_acc": val_acc,
    }

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for name, d in parsed.items():
    axes[0].plot(d["epochs"], d["val_acc"], marker="o", linewidth=2, label=name)
    axes[1].plot(d["epochs"], d["val_loss"], marker="o", linewidth=2, label=name)

axes[0].set_title("Validation accuracy curves (BOTH)")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Validation accuracy")
axes[0].grid(True, linestyle="--", alpha=0.4)
axes[0].legend(frameon=False)

axes[1].set_title("Validation loss curves (BOTH)")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Validation loss")
axes[1].grid(True, linestyle="--", alpha=0.4)
axes[1].legend(frameon=False)

plt.tight_layout()
out_file = "02_LOGO_Promoter/promoter_training_curves_both.png"
plt.savefig(out_file, dpi=300, bbox_inches="tight")
print("Saved:", out_file)
