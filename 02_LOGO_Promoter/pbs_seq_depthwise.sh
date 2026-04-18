#!/usr/bin/env bash
#PBS -N seq_depthwise
#PBS -q normal
#PBS -l select=1:ncpus=8:mem=64gb:ngpus=1
#PBS -l walltime=08:00:00
#PBS -j oe

set -euo pipefail
cd /home/users/nus/e1538285/scratch/LOGO/02_LOGO_Promoter

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "/apps/anaconda3/etc/profile.d/conda.sh" ]; then
    source "/apps/anaconda3/etc/profile.d/conda.sh"
fi

conda activate logo
export PYTHONPATH=/home/users/nus/e1538285/scratch/LOGO:${PYTHONPATH:-}
mkdir -p /home/users/nus/e1538285/scratch/LOGO/02_LOGO_Promoter/locality_logs

python /home/users/nus/e1538285/scratch/LOGO/02_LOGO_Promoter/01_PromID_trainer_locality_seq_depthwise.py 2>&1 | tee /home/users/nus/e1538285/scratch/LOGO/02_LOGO_Promoter/locality_logs/seq_depthwise.log
