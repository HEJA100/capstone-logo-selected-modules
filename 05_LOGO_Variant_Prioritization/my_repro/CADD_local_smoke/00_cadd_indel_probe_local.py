import gzip
import os
from pyfaidx import Fasta

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

fa = Fasta(FASTA_PATH)

def norm_chr(chrom: str) -> str:
    chrom = str(chrom).strip()
    if chrom.startswith("chr"):
        return chrom
    if chrom in {"M", "MT"}:
        return "chrM"
    return "chr" + chrom

def fetch_context(chrom: str, pos1: int, flank: int = 10) -> str:
    c = norm_chr(chrom)
    start = max(1, pos1 - flank)
    end = pos1 + flank
    seq = fa[c][start-1:end].seq.upper()
    return seq

def probe_vcf(path: str, n: int = 5):
    print(f"\n=== PROBE VCF: {path} ===")
    count = 0
    with gzip.open(path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            chrom, pos, _id, ref, alt = line.rstrip("\n").split("\t")[:5]
            pos = int(pos)
            seq = fetch_context(chrom, pos, flank=10)
            print(chrom, pos, ref, alt, seq)
            count += 1
            if count >= n:
                break

def probe_indel_txt(path: str, n: int = 5):
    print(f"\n=== PROBE INDEL TXT: {path} ===")
    count = 0
    with open(path, "r") as f:
        for line in f:
            chrom, pos, name, ref, alt = line.rstrip("\n").split("\t")[:5]
            pos = int(pos)
            seq = fetch_context(chrom, pos, flank=10)
            print(chrom, pos, name, ref, alt, seq)
            count += 1
            if count >= n:
                break

probe_vcf(TRAIN_HUMAN, 5)
probe_vcf(TRAIN_SIM, 5)
probe_indel_txt(VALID_FILE, 5)
probe_indel_txt(TEST_FILE, 5)

print("\n=== probe finished ===")
