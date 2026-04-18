set -euo pipefail

OUT="02_promoter_audit_$(date +%Y%m%d_%H%M%S).txt"

{
  echo "=== BASIC ==="
  date
  pwd
  echo

  echo "=== DIR SIZE ==="
  du -sh 02_LOGO_Promoter 2>/dev/null || true
  echo

  echo "=== SUBDIR TREE (maxdepth=2) ==="
  find 02_LOGO_Promoter -maxdepth 2 -type d | sort
  echo

  echo "=== SCRIPT FILES ==="
  find 02_LOGO_Promoter -type f \( -name "*.py" -o -name "*.sh" -o -name "*.ipynb" -o -name "*.md" \) | sort
  echo

  echo "=== RESULT-LIKE FILES ==="
  find 02_LOGO_Promoter -type f \( -name "*.csv" -o -name "*.tsv" -o -name "*.log" -o -name "*.out" -o -name "*.txt" -o -name "*.png" -o -name "*.pdf" -o -name "*.npz" -o -name "*.h5" \) | sort
  echo

  echo "=== KEYWORD SEARCH: PRETRAIN / LOAD WEIGHTS / CHECKPOINT ==="
  find 02_LOGO_Promoter -type f \( -name "*.py" -o -name "*.sh" \) -print0 | \
    xargs -0 grep -nH -E "load_weights|checkpoint|ckpt|restore|pretrain|pre-trained|bert_model|init_checkpoint|restore_best_weights" 2>/dev/null || true
  echo

  echo "=== KEYWORD SEARCH: BASELINES / TASK NAMES ==="
  find 02_LOGO_Promoter -type f \( -name "*.py" -o -name "*.sh" -o -name "*.md" \) -print0 | \
    xargs -0 grep -nH -E "BiLSTM|LSTM|CNN|PromID|DeeReCT|baseline|TATA|NO_TATA|BOTH|Knowledge|knowledge|5_gram|fold|promoter" 2>/dev/null || true
  echo

  echo "=== KEYWORD SEARCH: ARGS / FLAGS ==="
  find 02_LOGO_Promoter -type f -name "*.py" -print0 | \
    xargs -0 grep -nH -E "argparse|flags\.DEFINE|tf\.app\.flags|parser\.add_argument" 2>/dev/null || true
  echo

  echo "=== POSSIBLE SUMMARY FILES IN WHOLE REPO ==="
  find . -type f | grep -Ei "summary|promoter|bilstm|cnn|both|tata|knowledge|5_gram|f1|precision|recall" | sort || true
  echo

} > "$OUT"

echo "Audit written to: $OUT"
tail -n 120 "$OUT"
