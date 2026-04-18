from pathlib import Path
import numpy as np

ROOT = Path("/home/users/nus/e1538285/scratch/LOGO/02_LOGO_Promoter")
SRC_DIR = ROOT / "data/5_gram_11_knowledge"
OUT_DIR = ROOT / "data/5_gram_11_knowledge_coarsebin25"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BIN_SIZE = 25
SEQ_LEN = 600
TASKS = ["BOTH", "TATA_BOX", "NO_TATA_BOX"]

if SEQ_LEN % BIN_SIZE != 0:
    raise ValueError(f"SEQ_LEN={SEQ_LEN} must be divisible by BIN_SIZE={BIN_SIZE}")

def coarse_bin_annotation(annotation, bin_size):
    # annotation: (N, C, L)
    n, c, l = annotation.shape
    n_bins = l // bin_size
    x = annotation.reshape(n, c, n_bins, bin_size)
    pooled = x.max(axis=3, keepdims=True).astype(annotation.dtype)   # (N, C, n_bins, 1)
    expanded = np.repeat(pooled, bin_size, axis=3).reshape(n, c, l)  # back to (N, C, L)
    return expanded

for task in TASKS:
    src = SRC_DIR / f"epdnew_{task}_Knowledge_5_gram.npz"
    out = OUT_DIR / f"epdnew_{task}_Knowledge_5_gram.npz"

    data = np.load(src, allow_pickle=True)
    seq = data["sequence"]
    ann = data["annotation"]
    label = data["label"]

    ann_cb = coarse_bin_annotation(ann, BIN_SIZE)

    np.savez_compressed(out, sequence=seq, annotation=ann_cb, label=label)

    orig_sum = ann.sum(axis=(0, 2))
    new_sum = ann_cb.sum(axis=(0, 2))

    print(f"\n===== {task} =====")
    print("src:", src)
    print("out:", out)
    print("sequence shape:", seq.shape)
    print("annotation shape:", ann.shape, "->", ann_cb.shape)
    print("label shape:", label.shape)
    print("channel_idx\torig_token_sum\tcoarsebin_token_sum")
    for i, (a, b) in enumerate(zip(orig_sum, new_sum)):
        print(f"{i}\t{int(a)}\t{int(b)}")

print("\nDone.")
