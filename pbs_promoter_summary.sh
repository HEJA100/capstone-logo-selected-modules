#!/usr/bin/env bash
#PBS -N promSum
#PBS -q normal
#PBS -l select=1:ncpus=1:mem=4gb
#PBS -l walltime=00:30:00
#PBS -j oe

set -euo pipefail

ROOT="${PBS_O_WORKDIR:-/scratch/users/nus/e1538285/LOGO}"
echo "ROOT=$ROOT"
pwd
ls -ld "$ROOT" || true
cd "$ROOT"
pwd

OUT="02_LOGO_Promoter/result_snapshots/ablation_summary.txt"
mkdir -p 02_LOGO_Promoter/result_snapshots

{
  echo "===== promoter ablation summary ====="
  date
  echo
  echo "=== tail A_pretrain_sequence.log ==="
  tail -n 40 02_LOGO_Promoter/A_pretrain_sequence.log 2>/dev/null || true
  echo
  echo "=== tail A_pretrain_knowledge.log ==="
  tail -n 40 02_LOGO_Promoter/A_pretrain_knowledge.log 2>/dev/null || true
  echo
  echo "=== tail B_random_sequence.log ==="
  tail -n 40 02_LOGO_Promoter/B_random_sequence.log 2>/dev/null || true
  echo
  echo "=== tail B_random_knowledge.log ==="
  tail -n 40 02_LOGO_Promoter/B_random_knowledge.log 2>/dev/null || true
  echo
  echo "=== snapshots ==="
  find 02_LOGO_Promoter/result_snapshots -maxdepth 2 -type d | sort
} > "$OUT"

echo "Summary written to: $OUT"
