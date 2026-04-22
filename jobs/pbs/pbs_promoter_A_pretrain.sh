#!/usr/bin/env bash
#PBS -N promA
#PBS -q normal
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1
#PBS -l walltime=12:00:00
#PBS -j oe

set -euo pipefail

ROOT="${PBS_O_WORKDIR:-/scratch/users/nus/e1538285/LOGO}"
echo "ROOT=$ROOT"
pwd
ls -ld "$ROOT" || true
cd "$ROOT"
pwd

PYTHON="/home/users/nus/e1538285/miniconda3/envs/logo/bin/python"
echo "PYTHON=$PYTHON"
$PYTHON -V
which perl || true
which tee || true

date
hostname
nvidia-smi || true

perl -0pi -e 's/USE_PRETRAIN = False/USE_PRETRAIN = True/g' \
  02_LOGO_Promoter/01_PromID_trainer.py \
  02_LOGO_Promoter/02_PromID_trainer_knowledge.py

cd 02_LOGO_Promoter

$PYTHON -u 01_PromID_trainer.py |& tee A_pretrain_sequence.log
$PYTHON -u 02_PromID_trainer_knowledge.py |& tee A_pretrain_knowledge.log

cd "$ROOT"
mkdir -p 02_LOGO_Promoter/result_snapshots/A_pretrain/models
mkdir -p 02_LOGO_Promoter/result_snapshots/A_pretrain/files

cp -a 02_LOGO_Promoter/data/promoter_best_model_gene_bert_* \
      02_LOGO_Promoter/result_snapshots/A_pretrain/models/ 2>/dev/null || true

cp -a \
  02_LOGO_Promoter/A_pretrain_sequence.log \
  02_LOGO_Promoter/A_pretrain_knowledge.log \
  02_LOGO_Promoter/*.csv \
  02_LOGO_Promoter/*.png \
  02_LOGO_Promoter/result \
  02_LOGO_Promoter/result_snapshots/A_pretrain/files/ 2>/dev/null || true

date
