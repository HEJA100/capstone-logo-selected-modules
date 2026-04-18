#!/usr/bin/env python3
import os
import sys
import json
import gc
from typing import List, Dict, Tuple

import numpy as np

sys.path.append("../")
from bgi.common.genebank_utils import get_refseq_gff, get_gene_feature_array

TYPE = "P-E"
NGRAM_DEFAULT = 6

# 按论文 promoter knowledge 的 11 类来，故意不加 promoter，避免直接 label leakage
ANNOTATION_TYPES = [
    "CDS",
    "exon",
    "enhancer",
    "insulator",
    "conserved_region",
    "protein_binding_site",
    "pseudogene",
    "DNAseI_hypersensitive_site",
    "nucleotide_cleavage_site",
    "silencer",
    "gene",
]
ANNOTATION_INDEX = {k: i for i, k in enumerate(ANNOTATION_TYPES)}

# GRCh38 NCBI accession mapping
CHR_TO_REFSEQ = {
    "chr1": "NC_000001.11",
    "chr2": "NC_000002.12",
    "chr3": "NC_000003.12",
    "chr4": "NC_000004.12",
    "chr5": "NC_000005.10",
    "chr6": "NC_000006.12",
    "chr7": "NC_000007.14",
    "chr8": "NC_000008.11",
    "chr9": "NC_000009.12",
    "chr10": "NC_000010.11",
    "chr11": "NC_000011.10",
    "chr12": "NC_000012.12",
    "chr13": "NC_000013.11",
    "chr14": "NC_000014.9",
    "chr15": "NC_000015.10",
    "chr16": "NC_000016.10",
    "chr17": "NC_000017.11",
    "chr18": "NC_000018.10",
    "chr19": "NC_000019.10",
    "chr20": "NC_000020.11",
    "chr21": "NC_000021.9",
    "chr22": "NC_000022.11",
    "chrX": "NC_000023.11",
    "chrY": "NC_000024.10",
    "chrM": "NC_012920.1",
    "chrMT": "NC_012920.1",
}

NUMERIC_COLS = [
    "enhancer_start", "enhancer_end",
    "promoter_start", "promoter_end",
    "label",
]


def load_pairs_loose(csv_path: str) -> List[Dict]:
    """
    宽松读取 pairs csv：
    - 修正坏行末尾多余的逗号
    - 只接受 9 列
    """
    rows = []
    with open(csv_path, "r", encoding="utf-8", errors="replace") as fh:
        header = fh.readline().rstrip("\n").split(",")
        if len(header) != 9:
            raise ValueError(f"Unexpected header length in {csv_path}: {len(header)}")

        for line_no, line in enumerate(fh, start=2):
            s = line.rstrip("\n")
            if s.endswith(","):
                s = s[:-1]
            parts = s.split(",")
            if len(parts) != len(header):
                raise ValueError(
                    f"{csv_path}: line {line_no} has {len(parts)} fields, expected {len(header)}"
                )
            row = dict(zip(header, parts))
            for c in NUMERIC_COLS:
                row[c] = int(row[c])
            rows.append(row)
    return rows


def choose_pairs_rows(cell: str, split: str, expected_n: int) -> Tuple[str, List[Dict]]:
    """
    自动选择和 x 行数匹配的 pairs 文件。
    train 优先尝试 augment；test 用 pairs_test。
    """
    if split == "train":
        candidates = [
            f"{cell}/{TYPE}/pairs_train_augment.retained.csv",
            f"{cell}/{TYPE}/pairs_train_augment.csv",
            f"{cell}/{TYPE}/pairs_train.csv",
            f"{cell}/{TYPE}/pairs.csv",
        ]
    elif split == "test":
        candidates = [
            f"{cell}/{TYPE}/test/pairs_test.retained.csv",
            f"{cell}/{TYPE}/pairs_test.csv",
        ]
    else:
        raise ValueError("split must be train or test")

    tried = []
    for path in candidates:
        if not os.path.exists(path):
            continue
        rows = load_pairs_loose(path)
        tried.append((path, len(rows)))
        if len(rows) == expected_n:
            print(f"[OK] matched pairs file: {path} (n={len(rows)})")
            return path, rows

    tried_msg = ", ".join([f"{p}:{n}" for p, n in tried]) if tried else "none found"
    raise ValueError(
        f"No pairs file matched expected_n={expected_n}. Tried: {tried_msg}"
    )


def load_xy(npz_path: str) -> Tuple[np.ndarray, np.ndarray]:
    d = np.load(npz_path, allow_pickle=True)
    x = d["x"]
    y = d["y"]
    print(f"[LOAD] {npz_path}")
    print(f"       x.shape={x.shape}, y.shape={y.shape}")
    return x, y


def build_annotation_tensor(
    rows: List[Dict],
    region_prefix: str,      # "enhancer" or "promoter"
    seq_len_tokens: int,
    ngram: int,
    chr_gff_dict: Dict,
) -> np.ndarray:
    """
    输出 shape = (N, A, L)
    A = len(ANNOTATION_TYPES)
    L = token length, e.g. enhancer 1980 / promoter 972
    """
    n = len(rows)
    a = len(ANNOTATION_TYPES)
    anno = np.zeros((n, a, seq_len_tokens), dtype=np.uint8)

    for i, row in enumerate(rows):
        chrom = row[f"{region_prefix}_chrom"]
        win_start = row[f"{region_prefix}_start"]  # CSV 这里按 0-based start / end-exclusive 来看待
        win_end = row[f"{region_prefix}_end"]
        win_len_bp = win_end - win_start

        refseq_chr = CHR_TO_REFSEQ.get(chrom, chrom)
        hits = get_gene_feature_array(chr_gff_dict, refseq_chr, win_start, win_end)

        if hits is None:
            hits = []

        for hit in hits:
            # get_gene_feature_array 返回格式近似 [annotation, real_start, real_end]
            if len(hit) < 3:
                continue
            anno_type, hit_start, hit_end = hit[0], int(hit[1]), int(hit[2])

            if anno_type not in ANNOTATION_INDEX:
                continue

            # GFF 通常是 1-based inclusive；转成 0-based half-open 近似处理
            hit_start0 = hit_start - 1
            hit_end0 = hit_end

            rel_start = max(0, hit_start0 - win_start)
            rel_end = min(win_len_bp, hit_end0 - win_start)
            if rel_end <= rel_start:
                continue

            # token i 覆盖 [i, i+ngram)
            # 若 token 与注释有任意 overlap，则标 1
            token_left = max(0, rel_start - ngram + 1)
            token_right = min(seq_len_tokens, rel_end)

            if token_right > token_left:
                anno[i, ANNOTATION_INDEX[anno_type], token_left:token_right] = 1

        if (i + 1) % 5000 == 0:
            print(f"[{region_prefix}] processed {i+1}/{n}")

    return anno


def save_knowledge_npz(out_path: str, x: np.ndarray, y: np.ndarray, anno: np.ndarray):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, x=x, y=y, annotation=anno)
    print(f"[SAVE] {out_path}")
    print(f"       x.shape={x.shape}, y.shape={y.shape}, annotation.shape={anno.shape}")


def main():
    if len(sys.argv) < 3:
        print("Usage: python 03_make_knowledge_npz.py CELL split [ngram] [gff_path]")
        print("Example: python 03_make_knowledge_npz.py tB train 6 /scratch/users/nus/e1538285/LOGO/data/hg38/GCF_000001405.39_GRCh38.p13_genomic.gff")
        sys.exit(1)

    cell = sys.argv[1]
    split = sys.argv[2]  # train / test
    ngram = int(sys.argv[3]) if len(sys.argv) >= 4 else NGRAM_DEFAULT
    gff_path = (
        sys.argv[4]
        if len(sys.argv) >= 5
        else "/scratch/users/nus/e1538285/LOGO/data/hg38/GCF_000001405.39_GRCh38.p13_genomic.gff"
    )

    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")

    if split == "train":
        base_dir = f"{cell}/{TYPE}/{ngram}_gram"
    else:
        base_dir = f"{cell}/{TYPE}/test/{ngram}_gram"

    enhancer_npz = os.path.join(base_dir, f"enhancer_Seq_{ngram}_gram.npz")
    promoter_npz = os.path.join(base_dir, f"promoter_Seq_{ngram}_gram.npz")

    if not os.path.exists(enhancer_npz):
        raise FileNotFoundError(f"Missing {enhancer_npz}")
    if not os.path.exists(promoter_npz):
        raise FileNotFoundError(f"Missing {promoter_npz}")

    # 先读普通 x/y
    enhancer_x, enhancer_y = load_xy(enhancer_npz)
    promoter_x, promoter_y = load_xy(promoter_npz)

    if len(enhancer_x) != len(promoter_x):
        raise ValueError("enhancer and promoter sample counts do not match")
    if len(enhancer_y) != len(promoter_y):
        raise ValueError("enhancer and promoter labels do not match")
    if not np.array_equal(enhancer_y, promoter_y):
        raise ValueError("enhancer_y and promoter_y are not identical")

    expected_n = len(enhancer_x)
    pairs_path, rows = choose_pairs_rows(cell, split, expected_n)

    print(f"[INFO] using pairs file: {pairs_path}")
    print(f"[INFO] sample count: {expected_n}")
    print(f"[INFO] enhancer token length: {enhancer_x.shape[1]}")
    print(f"[INFO] promoter token length: {promoter_x.shape[1]}")
    print(f"[INFO] annotation types ({len(ANNOTATION_TYPES)}): {ANNOTATION_TYPES}")

    print(f"[INFO] loading gff: {gff_path}")
    chr_gff_dict = get_refseq_gff(gff_path, ANNOTATION_TYPES)

    # enhancer knowledge
    enhancer_anno = build_annotation_tensor(
        rows=rows,
        region_prefix="enhancer",
        seq_len_tokens=enhancer_x.shape[1],
        ngram=ngram,
        chr_gff_dict=chr_gff_dict,
    )
    enhancer_out = os.path.join(base_dir, f"enhancer_Seq_{ngram}_gram_knowledge.npz")
    save_knowledge_npz(enhancer_out, enhancer_x, enhancer_y, enhancer_anno)

    del enhancer_x, enhancer_y, enhancer_anno
    gc.collect()

    # promoter knowledge
    promoter_anno = build_annotation_tensor(
        rows=rows,
        region_prefix="promoter",
        seq_len_tokens=promoter_x.shape[1],
        ngram=ngram,
        chr_gff_dict=chr_gff_dict,
    )
    promoter_out = os.path.join(base_dir, f"promoter_Seq_{ngram}_gram_knowledge.npz")
    save_knowledge_npz(promoter_out, promoter_x, promoter_y, promoter_anno)

    del promoter_x, promoter_y, promoter_anno
    gc.collect()

    meta_path = os.path.join(base_dir, f"knowledge_meta_{ngram}_gram.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "cell": cell,
                "split": split,
                "ngram": ngram,
                "annotation_types": ANNOTATION_TYPES,
                "pairs_path": pairs_path,
                "gff_path": gff_path,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[SAVE] {meta_path}")
    print("[DONE]")


if __name__ == "__main__":
    main()