#!/usr/bin/env bash
#PBS -N K_regulatory
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

: > K_regulatory.log
: > K_regulatory_exit_code.txt

rc=999
set +e
env PYTHONUNBUFFERED=1 "$PYTHON" -u "02_PromID_trainer_knowledge_regulatory.py" 2>&1 | tee -a "K_regulatory.log"
rc=${PIPESTATUS[0]}
set -e

echo "$rc" > "K_regulatory_exit_code.txt"
echo "knowledge_regulatory_rc=$rc"

cd "$ROOT"
mkdir -p "02_LOGO_Promoter/result_snapshots/K_regulatory_latest/models"
mkdir -p "02_LOGO_Promoter/result_snapshots/K_regulatory_latest/files"

cp -a "02_LOGO_Promoter/K_regulatory.log"       "02_LOGO_Promoter/K_regulatory_exit_code.txt"       "02_LOGO_Promoter/result_snapshots/K_regulatory_latest/files/" 2>/dev/null || true

cp -a 02_LOGO_Promoter/data/*"_regulatory.h5"       "02_LOGO_Promoter/result_snapshots/K_regulatory_latest/models/" 2>/dev/null || true

exit "$rc"
