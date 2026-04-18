from pathlib import Path
import numpy as np

ROOT = Path("/home/users/nus/e1538285/scratch/LOGO/02_LOGO_Promoter")
SRC_DIR = ROOT / "data/5_gram_11_knowledge"
OUT_DIR = ROOT / "data/5_gram_11_knowledge_distmask_tsspm100"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TASKS = ["BOTH", "TATA_BOX", "NO_TATA_BOX"]

SEQ_LEN = 600
TSS_POS = 200          # promoter window is [-200, +400] around TSS
MASK_HALF_WIDTH = 100  # mask TSS ±100bp
MASK_L = TSS_POS - MASK_HALF_WIDTH   # 100
MASK_R = TSS_POS + MASK_HALF_WIDTH   # 300

print(f"SEQ_LEN={SEQ_LEN}, TSS_POS={TSS_POS}, mask_range=[{MASK_L}:{MASK_R})")

for task in TASKS:
    src = SRC_DIR / f"epdnew_{task}_Knowledge_5_gram.npz"
    out = OUT_DIR / f"epdnew_{task}_Knowledge_5_gram.npz"

    data = np.load(src, allow_pickle=True)
    seq = data["sequence"]
    ann = data["annotation"].copy()
    label = data["label"]

    orig_total = ann.sum(axis=(0, 2))
    orig_inside = ann[:, :, MASK_L:MASK_R].sum(axis=(0, 2))
    orig_outside = orig_total - orig_inside

    ann[:, :, MASK_L:MASK_R] = 0

    new_total = ann.sum(axis=(0, 2))

    np.savez_compressed(out, sequence=seq, annotation=ann, label=label)

    print(f"\n===== {task} =====")
    print("src:", src)
    print("out:", out)
    print("sequence shape:", seq.shape)
    print("annotation shape:", ann.shape)
    print("label shape:", label.shape)
    print("channel_idx\torig_total\torig_inside_mask\torig_outside_mask\tnew_total")
    for i, (a, b, c, d) in enumerate(zip(orig_total, orig_inside, orig_outside, new_total)):
        print(f"{i}\t{int(a)}\t{int(b)}\t{int(c)}\t{int(d)}")

print("\nDone.")
