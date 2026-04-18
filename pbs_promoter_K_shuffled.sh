#!/usr/bin/env bash
#PBS -N K_shuffled
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

cd 02_LOGO_Promoter

: > K_shuffled.log
: > K_shuffled_exit_code.txt

rc=999
set +e
env PYTHONUNBUFFERED=1 "$PYTHON" -u "02_PromID_trainer_knowledge_shuffled.py" 2>&1 | tee -a "K_shuffled.log"
rc=${PIPESTATUS[0]}
set -e

echo "$rc" > "K_shuffled_exit_code.txt"
echo "knowledge_shuffled_rc=$rc"

cd "$ROOT"
mkdir -p "02_LOGO_Promoter/result_snapshots/K_shuffled_latest/models"
mkdir -p "02_LOGO_Promoter/result_snapshots/K_shuffled_latest/files"

cp -a "02_LOGO_Promoter/K_shuffled.log"       "02_LOGO_Promoter/K_shuffled_exit_code.txt"       "02_LOGO_Promoter/result_snapshots/K_shuffled_latest/files/" 2>/dev/null || true

cp -a 02_LOGO_Promoter/data/*"_shuffled.h5"       "02_LOGO_Promoter/result_snapshots/K_shuffled_latest/models/" 2>/dev/null || true

exit "$rc"
