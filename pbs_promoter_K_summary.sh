#!/usr/bin/env bash
#PBS -N promKsum
#PBS -q normal
#PBS -l select=1:ncpus=1:mem=4gb
#PBS -l walltime=00:30:00
#PBS -j oe

set -euo pipefail

ROOT="${PBS_O_WORKDIR:-/scratch/users/nus/e1538285/LOGO}"
cd "$ROOT"

mkdir -p 02_LOGO_Promoter/result_snapshots
OUT="02_LOGO_Promoter/result_snapshots/knowledge_summary.txt"

{
  echo "===== knowledge summary ====="
  date
  echo

  echo "=== exit code ==="
  cat 02_LOGO_Promoter/knowledge_exit_code.txt 2>/dev/null || true
  echo

  echo "=== knowledge log status ==="
  ls -lh 02_LOGO_Promoter/K_pretrain_knowledge.log 2>/dev/null || true
  stat 02_LOGO_Promoter/K_pretrain_knowledge.log 2>/dev/null || true
  echo

  echo "=== tail: knowledge log ==="
  tail -n 120 02_LOGO_Promoter/K_pretrain_knowledge.log 2>/dev/null || true
  echo

  echo "=== saved knowledge models ==="
  find 02_LOGO_Promoter/data -maxdepth 1 -type f -name 'promoter_best_model_gene_bert_*Knowledge_11_Knowledge.h5' | sort | tail -40
  echo

  echo "=== count of saved knowledge models ==="
  find 02_LOGO_Promoter/data -maxdepth 1 -type f -name 'promoter_best_model_gene_bert_*Knowledge_11_Knowledge.h5' | wc -l
  echo

  echo "=== last save events ==="
  grep -n "saving model to" 02_LOGO_Promoter/K_pretrain_knowledge.log 2>/dev/null | tail -40 || true
  echo

  echo "=== last Eval lines ==="
  grep -n "Eval:" 02_LOGO_Promoter/K_pretrain_knowledge.log 2>/dev/null | tail -20 || true
} > "$OUT"

echo "Summary written to: $OUT"
