#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/users/nus/e1538285/scratch/LOGO"
cd "$ROOT"

python - <<'PY'
from pathlib import Path

root = Path("/home/users/nus/e1538285/scratch/LOGO")
src = root / "02_LOGO_Promoter/02_PromID_trainer_knowledge.py"

specs = {
    "structural": "./data/5_gram_structural_knowledge",
    "regulatory": "./data/5_gram_regulatory_knowledge",
    "shuffled": "./data/5_gram_shuffled_knowledge",
}

base_text = src.read_text()

for tag, data_path in specs.items():
    t = base_text

    # 1) 补 PRETRAIN_VOCAB_SIZE
    old_pretrain = "PRETRAIN_PATH = '../99_PreTrain_Model_Weight/LOGO_5_gram_2_layer_8_heads_256_dim_weights_32-0.885107.hdf5'\n"
    if "PRETRAIN_VOCAB_SIZE = 3138" not in t:
        if old_pretrain not in t:
            raise RuntimeError(f"[{tag}] could not find PRETRAIN_PATH line")
        t = t.replace(old_pretrain, old_pretrain + "PRETRAIN_VOCAB_SIZE = 3138\n", 1)

    # 2) 修误导性 print
    t = t.replace(
        "print('USE_PRETRAIN = True, training from random initialization')",
        "print('USE_PRETRAIN = False, training from random initialization')"
    )

    # 3) 修 vocab mismatch：只替换建模这一行，别再整段匹配
    old_model_line = "            model = model_def(vocab_size=vocab_size)\n"
    new_model_line = (
        "            model_vocab_size = PRETRAIN_VOCAB_SIZE if USE_PRETRAIN else vocab_size\n"
        "            print(f'build model with vocab_size={model_vocab_size} (data vocab_size={vocab_size})')\n"
        "            model = model_def(vocab_size=model_vocab_size)\n"
    )
    if old_model_line in t:
        t = t.replace(old_model_line, new_model_line)
    elif "model = model_def(vocab_size=model_vocab_size)" not in t:
        raise RuntimeError(f"[{tag}] could not find model = model_def(vocab_size=vocab_size)")

    # 4) 改 data_path 到对应派生数据目录
    old_data_path = "    data_path = './data/' + '{}_gram_11_knowledge'.format(ngram)\n"
    if old_data_path in t:
        t = t.replace(old_data_path, f"    data_path = '{data_path}'\n")
    elif data_path not in t:
        raise RuntimeError(f"[{tag}] could not patch data_path")

    # 5) 输出模型文件名加 tag，避免互相覆盖
    old_filename = "        filename = './data/promoter_best_model_gene_bert_{}_fold_{}_gram_{}_11_Knowledge.h5'.format(str(k_fold), str(ngram), task_name)\n"
    new_filename = f"        filename = './data/promoter_best_model_gene_bert_{{}}_fold_{{}}_gram_{{}}_11_Knowledge_{tag}.h5'.format(str(k_fold), str(ngram), task_name)\n"
    if old_filename in t:
        t = t.replace(old_filename, new_filename)
    elif f"_11_Knowledge_{tag}.h5" not in t:
        raise RuntimeError(f"[{tag}] could not patch filename")

    out = root / f"02_LOGO_Promoter/02_PromID_trainer_knowledge_{tag}.py"
    out.write_text(t)
    print("written", out)

PY

for tag in structural regulatory shuffled; do
cat > "pbs_promoter_K_${tag}.sh" <<EOF
#!/usr/bin/env bash
#PBS -N K_${tag}
#PBS -q normal
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1
#PBS -l walltime=04:00:00
#PBS -j oe

set -euo pipefail

ROOT="\${PBS_O_WORKDIR:-/scratch/users/nus/e1538285/LOGO}"
PYTHON="/home/users/nus/e1538285/miniconda3/envs/logo/bin/python"

echo "ROOT=\$ROOT"
pwd
ls -ld "\$ROOT" || true
cd "\$ROOT"
pwd

echo "PYTHON=\$PYTHON"
"\$PYTHON" -V
which tee || true
nvidia-smi || true

cd 02_LOGO_Promoter

: > K_${tag}.log
: > K_${tag}_exit_code.txt

rc=999
set +e
env PYTHONUNBUFFERED=1 "\$PYTHON" -u "02_PromID_trainer_knowledge_${tag}.py" 2>&1 | tee -a "K_${tag}.log"
rc=\${PIPESTATUS[0]}
set -e

echo "\$rc" > "K_${tag}_exit_code.txt"
echo "knowledge_${tag}_rc=\$rc"

cd "\$ROOT"
mkdir -p "02_LOGO_Promoter/result_snapshots/K_${tag}_latest/models"
mkdir -p "02_LOGO_Promoter/result_snapshots/K_${tag}_latest/files"

cp -a "02_LOGO_Promoter/K_${tag}.log" \
      "02_LOGO_Promoter/K_${tag}_exit_code.txt" \
      "02_LOGO_Promoter/result_snapshots/K_${tag}_latest/files/" 2>/dev/null || true

cp -a 02_LOGO_Promoter/data/*"_${tag}.h5" \
      "02_LOGO_Promoter/result_snapshots/K_${tag}_latest/models/" 2>/dev/null || true

exit "\$rc"
EOF

chmod +x "pbs_promoter_K_${tag}.sh"

cat > "pbs_promoter_K_${tag}_summary.sh" <<EOF
#!/usr/bin/env bash
#PBS -N KS_${tag}
#PBS -q normal
#PBS -l select=1:ncpus=1:mem=4gb
#PBS -l walltime=00:30:00
#PBS -j oe

set -euo pipefail

ROOT="\${PBS_O_WORKDIR:-/scratch/users/nus/e1538285/LOGO}"
cd "\$ROOT"

mkdir -p 02_LOGO_Promoter/result_snapshots
OUT="02_LOGO_Promoter/result_snapshots/knowledge_${tag}_summary.txt"

{
  echo "===== knowledge ${tag} summary ====="
  date
  echo

  echo "=== exit code ==="
  cat "02_LOGO_Promoter/K_${tag}_exit_code.txt" 2>/dev/null || true
  echo

  echo "=== log status ==="
  ls -lh "02_LOGO_Promoter/K_${tag}.log" 2>/dev/null || true
  stat "02_LOGO_Promoter/K_${tag}.log" 2>/dev/null || true
  echo

  echo "=== tail log ==="
  tail -n 120 "02_LOGO_Promoter/K_${tag}.log" 2>/dev/null || true
  echo

  echo "=== saved models ==="
  find 02_LOGO_Promoter/data -maxdepth 1 -type f -name "*_${tag}.h5" | sort | tail -40
  echo

  echo "=== model count ==="
  find 02_LOGO_Promoter/data -maxdepth 1 -type f -name "*_${tag}.h5" | wc -l
  echo

  echo "=== last save events ==="
  grep -n "saving model to" "02_LOGO_Promoter/K_${tag}.log" 2>/dev/null | tail -40 || true
  echo

  echo "=== last Eval lines ==="
  grep -n "Eval:" "02_LOGO_Promoter/K_${tag}.log" 2>/dev/null | tail -20 || true
} > "\$OUT"

echo "Summary written to: \$OUT"
EOF

chmod +x "pbs_promoter_K_${tag}_summary.sh"
done

echo "done"
