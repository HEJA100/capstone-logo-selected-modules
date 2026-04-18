#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/users/nus/e1538285/scratch/LOGO"
cd "$ROOT"

STAMP=$(date +%Y%m%d_%H%M%S)

echo "=== step 1: backup current trainer files ==="
cp 02_LOGO_Promoter/01_PromID_trainer.py 02_LOGO_Promoter/01_PromID_trainer.py.bak_${STAMP}
cp 02_LOGO_Promoter/02_PromID_trainer_knowledge.py 02_LOGO_Promoter/02_PromID_trainer_knowledge.py.bak_${STAMP}

echo "=== step 2: patch trainer files (idempotent) ==="
python - <<'PY'
from pathlib import Path

ROOT = Path("/home/users/nus/e1538285/scratch/LOGO")
p1 = ROOT / "02_LOGO_Promoter/01_PromID_trainer.py"
p2 = ROOT / "02_LOGO_Promoter/02_PromID_trainer_knowledge.py"

CFG = """# ===== pretrain control =====
USE_PRETRAIN = True
PRETRAIN_PATH = '../99_PreTrain_Model_Weight/LOGO_5_gram_2_layer_8_heads_256_dim_weights_32-0.885107.hdf5'
# To switch to the better full-run checkpoint later, only change PRETRAIN_PATH.

"""

def add_cfg_once(text: str) -> str:
    if "USE_PRETRAIN =" in text and "PRETRAIN_PATH =" in text:
        return text
    marker = "def load_npz_dataset_for_classification("
    if marker not in text:
        raise RuntimeError("Cannot find insertion marker for config block.")
    return text.replace(marker, CFG + marker, 1)

# -------- patch 01 --------
t1 = p1.read_text()

t1 = add_cfg_once(t1)

old1 = """            model = model_def(vocab_size=vocab_size, embedding_size=128, hidden_size=256, num_hidden_layers=2, num_heads=8)
            model.load_weights('../99_PreTrain_Model_Weight/LOGO_5_gram_2_layer_8_heads_256_dim_weights_32-0.885107.hdf5', by_name=True)
            print('compiling...')"""

new1 = """            model = model_def(vocab_size=vocab_size, embedding_size=128, hidden_size=256, num_hidden_layers=2, num_heads=8)

            if USE_PRETRAIN:
                print(f'loading pretrain weights from: {PRETRAIN_PATH}')
                model.load_weights(PRETRAIN_PATH, by_name=True)
            else:
                print('USE_PRETRAIN = False, training from random initialization')

            print('compiling...')"""

if old1 in t1:
    t1 = t1.replace(old1, new1, 1)

p1.write_text(t1)

# -------- patch 02 --------
t2 = p2.read_text()

t2 = add_cfg_once(t2)

old2 = """            model = model_def(vocab_size=vocab_size)
            print('compiling...')"""

new2 = """            model = model_def(vocab_size=vocab_size)

            if USE_PRETRAIN:
                print(f'loading pretrain weights from: {PRETRAIN_PATH}')
                model.load_weights(PRETRAIN_PATH, by_name=True)
            else:
                print('USE_PRETRAIN = False, training from random initialization')

            print('compiling...')"""

if old2 in t2:
    t2 = t2.replace(old2, new2, 1)

# -------- enable BOTH block in 02 only once --------
start_marker = "# ==================== 1. BOTH  ===================="
end_marker   = "# ==================== 2. TATA BOX ===================="

s = t2.find(start_marker)
e = t2.find(end_marker)

if s == -1 or e == -1 or e <= s:
    raise RuntimeError("Cannot find BOTH/TATA block in 02.")

region = t2[s:e]
if "train_data_file = 'epdnew_BOTH_Knowledge_" not in region:
    lines = region.splitlines(True)
    new_lines = []
    for line in lines:
        if "====================" in line:
            new_lines.append(line)
            continue
        stripped = line.lstrip(" \t")
        indent = line[:len(line)-len(stripped)]
        if stripped.startswith("#"):
            rest = stripped[1:]
            if rest.startswith(" "):
                rest = rest[1:]
            line = indent + rest
        new_lines.append(line)
    new_region = "".join(new_lines)
    t2 = t2[:s] + new_region + t2[e:]

p2.write_text(t2)

print("Patched:")
print(" -", p1)
print(" -", p2)
PY

echo "=== step 3: write PBS job A (pretrain) ==="
cat > pbs_promoter_A_pretrain_${STAMP}.sh <<EOF
#!/usr/bin/env bash
#PBS -N promA_${STAMP}
#PBS -q gpu
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1
#PBS -l walltime=24:00:00
#PBS -j oe

set -euo pipefail

ROOT="/home/users/nus/e1538285/scratch/LOGO"
cd "\$ROOT"

source ~/.bashrc 2>/dev/null || true
if [ -f "\$HOME/.anaconda/etc/profile.d/conda.sh" ]; then
  source "\$HOME/.anaconda/etc/profile.d/conda.sh"
elif [ -f "\$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  source "\$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "\$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "\$HOME/miniconda3/etc/profile.d/conda.sh"
fi
conda activate logo

echo "===== A job starts ====="
date
hostname
nvidia-smi || true

perl -0pi -e 's/USE_PRETRAIN = False/USE_PRETRAIN = True/g' \
  02_LOGO_Promoter/01_PromID_trainer.py \
  02_LOGO_Promoter/02_PromID_trainer_knowledge.py

grep -n "USE_PRETRAIN =" \
  02_LOGO_Promoter/01_PromID_trainer.py \
  02_LOGO_Promoter/02_PromID_trainer_knowledge.py

cd 02_LOGO_Promoter

python -u 01_PromID_trainer.py |& tee A_pretrain_sequence.log
python -u 02_PromID_trainer_knowledge.py |& tee A_pretrain_knowledge.log

cd "\$ROOT"

mkdir -p 02_LOGO_Promoter/result_snapshots/A_pretrain_${STAMP}/models
mkdir -p 02_LOGO_Promoter/result_snapshots/A_pretrain_${STAMP}/files

cp -a 02_LOGO_Promoter/data/promoter_best_model_gene_bert_* \
      02_LOGO_Promoter/result_snapshots/A_pretrain_${STAMP}/models/ 2>/dev/null || true

cp -a \
  02_LOGO_Promoter/A_pretrain_sequence.log \
  02_LOGO_Promoter/A_pretrain_knowledge.log \
  02_LOGO_Promoter/*.csv \
  02_LOGO_Promoter/*.png \
  02_LOGO_Promoter/result \
  02_LOGO_Promoter/result_snapshots/A_pretrain_${STAMP}/files/ 2>/dev/null || true

echo "===== A job finished ====="
date
EOF

echo "=== step 4: write PBS job B (random-init) ==="
cat > pbs_promoter_B_random_${STAMP}.sh <<EOF
#!/usr/bin/env bash
#PBS -N promB_${STAMP}
#PBS -q gpu
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1
#PBS -l walltime=24:00:00
#PBS -j oe

set -euo pipefail

ROOT="/home/users/nus/e1538285/scratch/LOGO"
cd "\$ROOT"

source ~/.bashrc 2>/dev/null || true
if [ -f "\$HOME/.anaconda/etc/profile.d/conda.sh" ]; then
  source "\$HOME/.anaconda/etc/profile.d/conda.sh"
elif [ -f "\$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  source "\$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "\$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "\$HOME/miniconda3/etc/profile.d/conda.sh"
fi
conda activate logo

echo "===== B job starts ====="
date
hostname
nvidia-smi || true

perl -0pi -e 's/USE_PRETRAIN = True/USE_PRETRAIN = False/g' \
  02_LOGO_Promoter/01_PromID_trainer.py \
  02_LOGO_Promoter/02_PromID_trainer_knowledge.py

grep -n "USE_PRETRAIN =" \
  02_LOGO_Promoter/01_PromID_trainer.py \
  02_LOGO_Promoter/02_PromID_trainer_knowledge.py

cd 02_LOGO_Promoter

python -u 01_PromID_trainer.py |& tee B_random_sequence.log
python -u 02_PromID_trainer_knowledge.py |& tee B_random_knowledge.log

cd "\$ROOT"

mkdir -p 02_LOGO_Promoter/result_snapshots/B_random_${STAMP}/models
mkdir -p 02_LOGO_Promoter/result_snapshots/B_random_${STAMP}/files

cp -a 02_LOGO_Promoter/data/promoter_best_model_gene_bert_* \
      02_LOGO_Promoter/result_snapshots/B_random_${STAMP}/models/ 2>/dev/null || true

cp -a \
  02_LOGO_Promoter/B_random_sequence.log \
  02_LOGO_Promoter/B_random_knowledge.log \
  02_LOGO_Promoter/*.csv \
  02_LOGO_Promoter/*.png \
  02_LOGO_Promoter/result \
  02_LOGO_Promoter/result_snapshots/B_random_${STAMP}/files/ 2>/dev/null || true

# restore default to True for later use
perl -0pi -e 's/USE_PRETRAIN = False/USE_PRETRAIN = True/g' \
  02_LOGO_Promoter/01_PromID_trainer.py \
  02_LOGO_Promoter/02_PromID_trainer_knowledge.py

echo "===== B job finished ====="
date
EOF

echo "=== step 5: write PBS job Summary ==="
cat > pbs_promoter_summary_${STAMP}.sh <<EOF
#!/usr/bin/env bash
#PBS -N promS_${STAMP}
#PBS -q normal
#PBS -l select=1:ncpus=1:mem=4gb
#PBS -l walltime=01:00:00
#PBS -j oe

set -euo pipefail

ROOT="/home/users/nus/e1538285/scratch/LOGO"
cd "\$ROOT"

OUT="02_LOGO_Promoter/result_snapshots/ablation_summary_${STAMP}.txt"
mkdir -p 02_LOGO_Promoter/result_snapshots

{
  echo "===== promoter ablation summary ====="
  date
  echo
  echo "=== USE_PRETRAIN flags after all jobs ==="
  grep -n "USE_PRETRAIN =" \
    02_LOGO_Promoter/01_PromID_trainer.py \
    02_LOGO_Promoter/02_PromID_trainer_knowledge.py || true
  echo
  echo "=== snapshots ==="
  find 02_LOGO_Promoter/result_snapshots -maxdepth 2 -type d | sort
  echo
  echo "=== tail: A_pretrain_sequence.log ==="
  tail -n 40 02_LOGO_Promoter/A_pretrain_sequence.log 2>/dev/null || true
  echo
  echo "=== tail: A_pretrain_knowledge.log ==="
  tail -n 40 02_LOGO_Promoter/A_pretrain_knowledge.log 2>/dev/null || true
  echo
  echo "=== tail: B_random_sequence.log ==="
  tail -n 40 02_LOGO_Promoter/B_random_sequence.log 2>/dev/null || true
  echo
  echo "=== tail: B_random_knowledge.log ==="
  tail -n 40 02_LOGO_Promoter/B_random_knowledge.log 2>/dev/null || true
  echo
  echo "=== current promoter CSVs ==="
  ls -lah 02_LOGO_Promoter/*.csv 2>/dev/null || true
  echo
  echo "=== current promoter PNGs ==="
  ls -lah 02_LOGO_Promoter/*.png 2>/dev/null || true
} > "\$OUT"

echo "Summary written to: \$OUT"
EOF

chmod +x pbs_promoter_A_pretrain_${STAMP}.sh
chmod +x pbs_promoter_B_random_${STAMP}.sh
chmod +x pbs_promoter_summary_${STAMP}.sh

echo "=== step 6: quick patch verification ==="
grep -n "USE_PRETRAIN\|PRETRAIN_PATH\|loading pretrain weights\|random initialization" \
  02_LOGO_Promoter/01_PromID_trainer.py \
  02_LOGO_Promoter/02_PromID_trainer_knowledge.py

echo
echo "=== submitting jobs with dependency chain ==="
AID=$(qsub pbs_promoter_A_pretrain_${STAMP}.sh)
BID=$(qsub -W depend=afterok:${AID} pbs_promoter_B_random_${STAMP}.sh)
SID=$(qsub -W depend=afterok:${BID} pbs_promoter_summary_${STAMP}.sh)

echo "A job id: ${AID}"
echo "B job id: ${BID}  (afterok A)"
echo "S job id: ${SID}  (afterok B)"

echo
echo "=== qstat ==="
qstat -u "$USER" || true

echo
echo "Created files:"
echo "  pbs_promoter_A_pretrain_${STAMP}.sh"
echo "  pbs_promoter_B_random_${STAMP}.sh"
echo "  pbs_promoter_summary_${STAMP}.sh"
