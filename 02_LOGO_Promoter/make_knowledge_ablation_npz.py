from pathlib import Path
import numpy as np

ROOT = Path("/home/users/nus/e1538285/scratch/LOGO/02_LOGO_Promoter")
SRC_DIR = ROOT / "data/5_gram_11_knowledge"

# 通道映射，来自 00_EPDnew_data_prepare.py 的 include_types 顺序
CHANNEL_MAP = {
    0: "unknown",
    1: "enhancer",
    2: "pseudogene",
    3: "insulator",
    4: "conserved_region",
    5: "protein_binding_site",
    6: "DNAseI_hypersensitive_site",
    7: "nucleotide_cleavage_site",
    8: "silencer",
    9: "gene",
    10: "exon",
    11: "CDS",
    12: "unused_extra"
}

# 第一轮分组
STRUCTURAL_CHANNELS = [2, 4, 9, 10, 11]   # pseudogene, conserved_region, gene, exon, CDS
REGULATORY_CHANNELS = [1, 3, 5, 6, 7, 8]  # enhancer, insulator, protein_binding_site, DNaseI, cleavage, silencer

TASKS = ["BOTH", "TATA_BOX", "NO_TATA_BOX"]

OUT_SPECS = [
    ("5_gram_structural_knowledge", "structural", STRUCTURAL_CHANNELS),
    ("5_gram_regulatory_knowledge", "regulatory", REGULATORY_CHANNELS),
]

def summarize_annotation(ann, title):
    seq_nonzero = (ann.sum(axis=2) > 0).sum(axis=0)
    token_sum = ann.sum(axis=(0, 2))
    print(f"\n=== {title} ===")
    print("channel_idx\tname\tseq_nonzero\ttoken_sum")
    for i in range(ann.shape[1]):
        print(f"{i}\t{CHANNEL_MAP.get(i, 'NA')}\t{int(seq_nonzero[i])}\t{int(token_sum[i])}")

def keep_only_channels(annotation, keep_channels):
    out = np.zeros_like(annotation)
    out[:, keep_channels, :] = annotation[:, keep_channels, :]
    return out

def make_shuffled(annotation, seed=20260413):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(annotation.shape[0])
    # 序列和标签不动，只把 annotation 在样本维度打乱
    return annotation[perm]

for task in TASKS:
    src = SRC_DIR / f"epdnew_{task}_Knowledge_5_gram.npz"
    print(f"\n\n##### processing {src} #####")
    data = np.load(src, allow_pickle=True)
    seq = data["sequence"]
    ann = data["annotation"]
    label = data["label"]

    print("sequence shape:", seq.shape)
    print("annotation shape:", ann.shape)
    print("label shape:", label.shape)
    summarize_annotation(ann, f"original {task}")

    # structural / regulatory
    for out_dir_name, tag, keep_channels in OUT_SPECS:
        out_dir = ROOT / f"data/{out_dir_name}"
        out_dir.mkdir(parents=True, exist_ok=True)

        ann_new = keep_only_channels(ann, keep_channels)

        out_fp = out_dir / f"epdnew_{task}_Knowledge_5_gram.npz"
        np.savez_compressed(out_fp, sequence=seq, annotation=ann_new, label=label)

        print(f"\nwritten: {out_fp}")
        summarize_annotation(ann_new, f"{tag} {task}")

    # shuffled
    out_dir = ROOT / "data/5_gram_shuffled_knowledge"
    out_dir.mkdir(parents=True, exist_ok=True)

    ann_shuf = make_shuffled(ann, seed=20260413 + len(task))
    out_fp = out_dir / f"epdnew_{task}_Knowledge_5_gram.npz"
    np.savez_compressed(out_fp, sequence=seq, annotation=ann_shuf, label=label)

    print(f"\nwritten: {out_fp}")
    summarize_annotation(ann_shuf, f"shuffled {task}")

print("\nDone.")
