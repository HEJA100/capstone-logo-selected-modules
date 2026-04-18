import os
import gzip
import tensorflow as tf
from pyfaidx import Fasta

# =========================
# Paths
# =========================
TRAIN_HUMAN = os.path.expanduser(
    "~/scratch/LOGO/05_LOGO_Variant_Prioritization/my_repro/CADD_data/GRCh37/humanDerived_InDels.vcf.gz"
)
TRAIN_SIM = os.path.expanduser(
    "~/scratch/LOGO/05_LOGO_Variant_Prioritization/my_repro/CADD_data/GRCh37/simulation_InDels.vcf.gz"
)
FASTA = "/home/users/nus/e1538285/scratch/LOGO/04_LOGO_Chromatin_Feature/1. script/04_LOGO_Chrom_predict/Genomics/male.hg19.fasta"

OUTDIR = os.path.expanduser(
    "~/scratch/LOGO/05_LOGO_Variant_Prioritization/my_repro/CADD_redo/data_small"
)
os.makedirs(OUTDIR, exist_ok=True)

# =========================
# Small clean split
# =========================
TRAIN_N = 2000
VAL_N = 500

# Sequence settings
NGRAM = 3
STRIDE = 1
PADDING_SEQ_LEN = 257   # gives 514bp raw window -> ~512 tokens after 3-mer
alphabet = {"N": 0, "A": 1, "G": 2, "C": 3, "T": 4}

fa = Fasta(FASTA, as_raw=True, sequence_always_upper=True)

# =========================
# Helpers
# =========================
def norm_chr(chrom: str) -> str:
    c = chrom if chrom.startswith("chr") else f"chr{chrom}"
    if c == "chrMT":
        c = "chrM"
    return c

def fetch_ref_window(chrom: str, pos1: int, padding_seq_len: int = 257) -> str:
    c = norm_chr(chrom)
    start = max(1, pos1 - padding_seq_len)
    end = pos1 + padding_seq_len - 1
    seq = fa[c][start-1:end]
    seq = str(seq).upper()
    target_len = 2 * padding_seq_len
    if len(seq) < target_len:
        seq = seq + "N" * (target_len - len(seq))
    return seq

def apply_alt(ref_window: str, ref: str, alt: str, padding_seq_len: int = 257) -> str:
    center0 = padding_seq_len - 1
    ref_len = len(ref)
    alt_seq = ref_window[:center0] + alt + ref_window[center0 + ref_len:]
    target_len = len(ref_window)
    if len(alt_seq) > target_len:
        alt_seq = alt_seq[:target_len]
    elif len(alt_seq) < target_len:
        alt_seq = alt_seq + "N" * (target_len - len(alt_seq))
    return alt_seq

def seq_to_ints(seq: str):
    return [alphabet.get(b, 0) for b in seq]

def build_alt_type(seq_len: int, ref_len: int, alt_len: int, padding_seq_len: int = 257):
    arr = [0] * seq_len
    center0 = padding_seq_len - 1
    span = max(ref_len, alt_len)
    for i in range(center0, min(seq_len, center0 + span)):
        arr[i] = 1
    return arr

def kmer_tokenize(int_seq, ngram=3, stride=1):
    out = []
    for i in range(0, len(int_seq) - ngram + 1, stride):
        token = 0
        for x in int_seq[i:i+ngram]:
            token = token * 5 + x
        out.append(token)
    return out

def serialize_example(seq, alt_seq, alt_type, label):
    feat = {
        "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
        "seq": tf.train.Feature(int64_list=tf.train.Int64List(value=seq)),
        "alt_seq": tf.train.Feature(int64_list=tf.train.Int64List(value=alt_seq)),
        "alt_type": tf.train.Feature(int64_list=tf.train.Int64List(value=alt_type)),
    }
    ex = tf.train.Example(features=tf.train.Features(feature=feat))
    return ex.SerializeToString()

def parse_vcf_records(path, label, start_idx=0, limit=None):
    opener = gzip.open if path.endswith(".gz") else open
    seen = 0
    yielded = 0
    with opener(path, "rt") as f:
        for line in f:
            if not line or line.startswith("#"):
                continue
            chrom, pos, _id, ref, alt = line.rstrip("\n").split("\t")[:5]
            if len(ref) == len(alt):
                continue  # only keep indels
            if seen < start_idx:
                seen += 1
                continue
            yield chrom, int(pos), ref, alt, label
            seen += 1
            yielded += 1
            if limit is not None and yielded >= limit:
                break

def write_tfrecord(records, out_path):
    written = 0
    skipped = 0
    with tf.io.TFRecordWriter(out_path) as w:
        for chrom, pos, ref, alt, label in records:
            try:
                ref_window = fetch_ref_window(chrom, pos, PADDING_SEQ_LEN)
                alt_window = apply_alt(ref_window, ref, alt, PADDING_SEQ_LEN)

                seq_int = seq_to_ints(ref_window)
                alt_int = seq_to_ints(alt_window)
                alt_type = build_alt_type(len(seq_int), len(ref), len(alt), PADDING_SEQ_LEN)

                seq_tok = kmer_tokenize(seq_int, NGRAM, STRIDE)
                alt_tok = kmer_tokenize(alt_int, NGRAM, STRIDE)
                alt_type_tok = alt_type[:len(seq_tok)]

                # keep only mutation-related alt tokens, zero elsewhere
                alt_tok = [a * t for a, t in zip(alt_tok, alt_type_tok)]

                ex = serialize_example(seq_tok, alt_tok, alt_type_tok, label)
                w.write(ex)
                written += 1
            except Exception:
                skipped += 1
    print(out_path, "written =", written, "skipped =", skipped)

# =========================
# Main
# =========================
if __name__ == "__main__":
    write_tfrecord(
        parse_vcf_records(TRAIN_HUMAN, label=0, start_idx=0, limit=TRAIN_N),
        os.path.join(OUTDIR, "train_human.tfrecord")
    )
    write_tfrecord(
        parse_vcf_records(TRAIN_SIM, label=1, start_idx=0, limit=TRAIN_N),
        os.path.join(OUTDIR, "train_sim.tfrecord")
    )
    write_tfrecord(
        parse_vcf_records(TRAIN_HUMAN, label=0, start_idx=TRAIN_N, limit=VAL_N),
        os.path.join(OUTDIR, "val_human.tfrecord")
    )
    write_tfrecord(
        parse_vcf_records(TRAIN_SIM, label=1, start_idx=TRAIN_N, limit=VAL_N),
        os.path.join(OUTDIR, "val_sim.tfrecord")
    )
    print("=== clean small CADD TFRecord generation finished ===")
