#!/usr/bin/env bash
#PBS -N promA_20260412_230611
#PBS -q gpu
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1
#PBS -l walltime=24:00:00
#PBS -j oe

set -euo pipefail

ROOT="/home/users/nus/e1538285/scratch/LOGO"
cd "$ROOT"

source ~/.bashrc 2>/dev/null || true
if [ -f "$HOME/.anaconda/etc/profile.d/conda.sh" ]; then
  source "$HOME/.anaconda/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
fi
conda activate logo

echo "===== A job starts ====="
date
hostname
nvidia-smi || true

perl -0pi -e 's/USE_PRETRAIN = False/USE_PRETRAIN = True/g'   02_LOGO_Promoter/01_PromID_trainer.py   02_LOGO_Promoter/02_PromID_trainer_knowledge.py

grep -n "USE_PRETRAIN ="   02_LOGO_Promoter/01_PromID_trainer.py   02_LOGO_Promoter/02_PromID_trainer_knowledge.py

cd 02_LOGO_Promoter

python -u 01_PromID_trainer.py |& tee A_pretrain_sequence.log
python -u 02_PromID_trainer_knowledge.py |& tee A_pretrain_knowledge.log

cd "$ROOT"

mkdir -p 02_LOGO_Promoter/result_snapshots/A_pretrain_20260412_230611/models
mkdir -p 02_LOGO_Promoter/result_snapshots/A_pretrain_20260412_230611/files

cp -a 02_LOGO_Promoter/data/promoter_best_model_gene_bert_*       02_LOGO_Promoter/result_snapshots/A_pretrain_20260412_230611/models/ 2>/dev/null || true

cp -a   02_LOGO_Promoter/A_pretrain_sequence.log   02_LOGO_Promoter/A_pretrain_knowledge.log   02_LOGO_Promoter/*.csv   02_LOGO_Promoter/*.png   02_LOGO_Promoter/result   02_LOGO_Promoter/result_snapshots/A_pretrain_20260412_230611/files/ 2>/dev/null || true

echo "===== A job finished ====="
date
