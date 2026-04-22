#!/usr/bin/env bash
#PBS -N promS_20260412_230611
#PBS -q normal
#PBS -l select=1:ncpus=1:mem=4gb
#PBS -l walltime=01:00:00
#PBS -j oe

set -euo pipefail

ROOT="/home/users/nus/e1538285/scratch/LOGO"
cd "$ROOT"

OUT="02_LOGO_Promoter/result_snapshots/ablation_summary_20260412_230611.txt"
mkdir -p 02_LOGO_Promoter/result_snapshots

{
  echo "===== promoter ablation summary ====="
  date
  echo
  echo "=== USE_PRETRAIN flags after all jobs ==="
  grep -n "USE_PRETRAIN ="     02_LOGO_Promoter/01_PromID_trainer.py     02_LOGO_Promoter/02_PromID_trainer_knowledge.py || true
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
} > "$OUT"

echo "Summary written to: $OUT"
