from pathlib import Path
import numpy as np

root = Path("/home/users/nus/e1538285/scratch/LOGO")
prep = root / "02_LOGO_Promoter/00_EPDnew_data_prepare.py"

print("===== 00_EPDnew_data_prepare.py : top annotation-related lines =====")
lines = prep.read_text().splitlines()
for a, b in [(1, 90), (430, 470)]:
    print(f"\n--- lines {a}-{b} ---")
    for i in range(a - 1, min(b, len(lines))):
        print(f"{i+1:4d}: {lines[i]}")

def show_npz_info(task):
    fp = root / f"02_LOGO_Promoter/data/5_gram_11_knowledge/epdnew_{task}_Knowledge_5_gram.npz"
    print(f"\n===== NPZ: {fp} =====")
    x = np.load(fp, allow_pickle=True)
    print("keys:", x.files)
    for k in x.files:
        obj = x[k]
        shape = getattr(obj, "shape", None)
        dtype = getattr(obj, "dtype", type(obj))
        print(f"  {k}: shape={shape}, dtype={dtype}")

    ann_key = None
    for k in x.files:
        arr = x[k]
        if hasattr(arr, "ndim") and arr.ndim == 3:
            ann_key = k
            break

    if ann_key is None:
        print("!! did not find 3D annotation array")
        return

    ann = x[ann_key]
    print(f"\nannotation key = {ann_key}, shape = {ann.shape}")

    seq_nonzero = (ann.sum(axis=2) > 0).sum(axis=0)
    token_sum = ann.sum(axis=(0, 2))

    print("\nchannel statistics:")
    print("channel_idx\tseq_nonzero\ttoken_sum")
    for i, (a, b) in enumerate(zip(seq_nonzero, token_sum)):
        print(f"{i}\t{int(a)}\t{int(b)}")

for task in ["BOTH", "TATA_BOX", "NO_TATA_BOX"]:
    show_npz_info(task)
