import gzip
import os
import random
import tensorflow as tf
from pyfaidx import Fasta

# ===== paths =====
FASTA_PATH = os.path.expanduser(
    "~/scratch/LOGO/04_LOGO_Chromatin_Feature/1. script/04_LOGO_Chrom_predict/Genomics/male.hg19.fasta"
)

TRAIN_HUMAN = os.path.expanduser(
    "~/scratch/LOGO/05_LOGO_Variant_Prioritization/my_repro/CADD_data/GRCh37/humanDerived_InDels.vcf.gz"
)
TRAIN_SIM = os.path.expanduser(
    "~/scratch/LOGO/05_LOGO_Variant_Prioritization/my_repro/CADD_data/GRCh37/simulation_InDels.vcf.gz"
)

VALID_FILE = os.path.expanduser(
    "~/scratch/LOGO/05_LOGO_Variant_Prioritization/2. data/5. CADD_indel_Clinvar + 1000G/clinvar_20201003.hg19_multianno.txt.noncoding_fold0_valid.indel"
)
TEST_FILE = os.path.expanduser(
    "~/scratch/LOGO/05_LOGO_Variant_Prioritization/2. data/5. CADD_indel_Clinvar + 1000G/clinvar_20201003.hg19_multianno.txt.noncoding_fold0_test.indel"
)

OUTDIR = os.path.expanduser(
    "~/scratch/LOGO/05_LOGO_Variant_Prioritization/my_repro/CADD_work/local_tfrecord_fold0_512"
)

os.makedirs(OUTDIR, exist_ok=True)

# ===== params =====
PADDING_SEQ_LEN = 257
NGRAM = 3
STRIDE = 1
TRAIN_LIMIT = 20000   # each class: small starter subset
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
fa = Fasta(FASTA_PATH)

alphabet = {'N': 0, 'A': 1, 'G': 2, 'C': 3, 'T': 4}

def norm_chr(chrom: str) -> str:
    chrom = str(chrom).strip()
    if chrom.startswith("chr"):
        return chrom
    if chrom in {"M", "MT"}:
        return "chrM"
    return "chr" + chrom

def fetch_ref_window(chrom: str, pos1: int, padding_seq_len: int = 500) -> str:
    c = norm_chr(chrom)
    start = max(1, pos1 - padding_seq_len)
    end = pos1 + padding_seq_len - 1
    seq = fa[c][start-1:end].seq.upper()
    if len(seq) < 2 * padding_seq_len:
        seq = seq + "N" * (2 * padding_seq_len - len(seq))
    return seq

def apply_alt(ref_window: str, ref: str, alt: str, padding_seq_len: int = 500) -> str:
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

def build_alt_type(seq_len: int, ref_len: int, alt_len: int, padding_seq_len: int = 500):
    arr = [0] * seq_len
    center0 = padding_seq_len - 1
    span = max(ref_len, alt_len)
    for i in range(center0, min(seq_len, center0 + span)):
        arr[i] = 1
    return arr

def kmer_tokenize(int_seq, ngram=3, stride=1):
    max_vocab = 5 ** ngram
    out = []
    for i in range(0, len(int_seq) - ngram + 1, stride):
        token = 0
        ok = True
        for x in int_seq[i:i+ngram]:
            if x < 0 or x > 4:
                ok = False
                break
            token = token * 5 + x
        out.append(token if ok else 0)
    return out

def serialize_example(seq, alt_seq, alt_type, label):
    feature = {
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
        'seq': tf.train.Feature(int64_list=tf.train.Int64List(value=seq)),
        'alt_seq': tf.train.Feature(int64_list=tf.train.Int64List(value=alt_seq)),
        'alt_type': tf.train.Feature(int64_list=tf.train.Int64List(value=alt_type)),
    }
    ex = tf.train.Example(features=tf.train.Features(feature=feature))
    return ex.SerializeToString()

def parse_vcf_records(path, label, limit=None):
    n = 0
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            chrom, pos, _id, ref, alt = line.rstrip("\n").split("\t")[:5]
            yield chrom, int(pos), ref, alt, label
            n += 1
            if limit and n >= limit:
                break

def parse_indel_txt(path):
    with open(path) as fh:
        for line in fh:
            chrom, pos, name, ref, alt = line.rstrip("\n").split("\t")[:5]
            label = 0 if name.startswith("Benign") else 1 if name.startswith("Pathogenic") else -1
            yield chrom, int(pos), ref, alt, label

def write_tfrecord(records, out_path):
    n_ok = 0
    n_skip = 0
    with tf.io.TFRecordWriter(out_path) as w:
        for chrom, pos, ref, alt, label in records:
            if label < 0:
                n_skip += 1
                continue
            try:
                ref_window = fetch_ref_window(chrom, pos, PADDING_SEQ_LEN)
                alt_window = apply_alt(ref_window, ref, alt, PADDING_SEQ_LEN)

                seq_int = seq_to_ints(ref_window)
                alt_int = seq_to_ints(alt_window)
                alt_type = build_alt_type(len(seq_int), len(ref), len(alt), PADDING_SEQ_LEN)

                seq_tok = kmer_tokenize(seq_int, NGRAM, STRIDE)
                alt_tok = kmer_tokenize(alt_int, NGRAM, STRIDE)
                alt_type_tok = alt_type[:len(seq_tok)]

                ex = serialize_example(seq_tok, alt_tok, alt_type_tok, label)
                w.write(ex)
                n_ok += 1
            except Exception:
                n_skip += 1
    print(out_path, "written =", n_ok, "skipped =", n_skip)

if __name__ == "__main__":
    write_tfrecord(parse_vcf_records(TRAIN_HUMAN, label=0, limit=TRAIN_LIMIT),
                   os.path.join(OUTDIR, "humanDerived_InDels_fold0_small.tfrecord"))

    write_tfrecord(parse_vcf_records(TRAIN_SIM, label=1, limit=TRAIN_LIMIT),
                   os.path.join(OUTDIR, "simulation_InDels_fold0_small.tfrecord"))

    write_tfrecord(parse_indel_txt(VALID_FILE),
                   os.path.join(OUTDIR, "clinvar_InDel_fold0_valid_local.tfrecord"))

    write_tfrecord(parse_indel_txt(TEST_FILE),
                   os.path.join(OUTDIR, "clinvar_InDel_fold0_test_local.tfrecord"))

    print("=== local fold0 tfrecord generation finished ===")
