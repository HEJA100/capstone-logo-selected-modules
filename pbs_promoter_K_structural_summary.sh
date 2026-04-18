#!/usr/bin/env bash
#PBS -N KS_structural
#PBS -q normal
#PBS -l select=1:ncpus=1:mem=4gb
#PBS -l walltime=00:30:00
#PBS -j oe

set -euo pipefail

ROOT="${PBS_O_WORKDIR:-/scratch/users/nus/e1538285/LOGO}"
cd "$ROOT"

mkdir -p 02_LOGO_Promoter/result_snapshots
OUT="02_LOGO_Promoter/result_snapshots/knowledge_structural_summary.txt"

{
  echo "===== knowledge structural summary ====="
  date
  echo

  echo "=== exit code ==="
  cat "02_LOGO_Promoter/K_structural_exit_code.txt" 2>/dev/null || true
  echo

  echo "=== log status ==="
  ls -lh "02_LOGO_Promoter/K_structural.log" 2>/dev/null || true
  stat "02_LOGO_Promoter/K_structural.log" 2>/dev/null || true
  echo

  echo "=== tail log ==="
  tail -n 120 "02_LOGO_Promoter/K_structural.log" 2>/dev/null || true
  echo

  echo "=== saved models ==="
  find 02_LOGO_Promoter/data -maxdepth 1 -type f -name "*_structural.h5" | sort | tail -40
  echo

  echo "=== model count ==="
  find 02_LOGO_Promoter/data -maxdepth 1 -type f -name "*_structural.h5" | wc -l
  echo

  echo "=== last save events ==="
  grep -n "saving model to" "02_LOGO_Promoter/K_structural.log" 2>/dev/null | tail -40 || true
  echo

  echo "=== last Eval lines ==="
  grep -n "Eval:" "02_LOGO_Promoter/K_structural.log" 2>/dev/null | tail -20 || true
} > "$OUT"

echo "Summary written to: $OUT"
