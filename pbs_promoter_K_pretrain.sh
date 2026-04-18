#!/usr/bin/env bash
#PBS -N promK
#PBS -q normal
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1
#PBS -l walltime=04:00:00
#PBS -j oe

set -euo pipefail

ROOT="${PBS_O_WORKDIR:-/scratch/users/nus/e1538285/LOGO}"
PYTHON="/home/users/nus/e1538285/miniconda3/envs/logo/bin/python"

echo "ROOT=$ROOT"
pwd
ls -ld "$ROOT" || true
cd "$ROOT"
pwd

echo "PYTHON=$PYTHON"
"$PYTHON" -V
which tee || true
nvidia-smi || true

# 强制 knowledge 这次使用 pretrain
perl -0pi -e 's/USE_PRETRAIN = False/USE_PRETRAIN = True/g' \
  02_LOGO_Promoter/02_PromID_trainer_knowledge.py

# 只打印真正的配置行，避免把 print 语句也匹配出来
grep -n '^USE_PRETRAIN =' 02_LOGO_Promoter/02_PromID_trainer_knowledge.py || true
grep -n '^PRETRAIN_PATH =' 02_LOGO_Promoter/02_PromID_trainer_knowledge.py || true

cd 02_LOGO_Promoter

# 先创建日志文件，避免 tail 时提示不存在
: > K_pretrain_knowledge.log
: > knowledge_exit_code.txt

rc=999
set +e
env PYTHONUNBUFFERED=1 "$PYTHON" -u 02_PromID_trainer_knowledge.py 2>&1 | tee -a K_pretrain_knowledge.log
rc=${PIPESTATUS[0]}
set -e

echo "$rc" > knowledge_exit_code.txt
echo "knowledge_rc=$rc"

cd "$ROOT"
mkdir -p 02_LOGO_Promoter/result_snapshots/K_pretrain_latest/models
mkdir -p 02_LOGO_Promoter/result_snapshots/K_pretrain_latest/files

cp -a 02_LOGO_Promoter/K_pretrain_knowledge.log \
      02_LOGO_Promoter/knowledge_exit_code.txt \
      02_LOGO_Promoter/result_snapshots/K_pretrain_latest/files/ 2>/dev/null || true

cp -a 02_LOGO_Promoter/data/promoter_best_model_gene_bert_*Knowledge_11_Knowledge.h5 \
      02_LOGO_Promoter/result_snapshots/K_pretrain_latest/models/ 2>/dev/null || true

exit "$rc"
