#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/users/nus/e1538285/scratch/LOGO
PROM=$ROOT/02_LOGO_Promoter
LIB=$ROOT/bgi/bert4keras
LOGDIR=$PROM/locality_logs

mkdir -p "$LOGDIR"

python - <<'PY'
from pathlib import Path
import re

ROOT = Path("/home/users/nus/e1538285/scratch/LOGO")
PROM = ROOT / "02_LOGO_Promoter"
LIB = ROOT / "bgi" / "bert4keras"

def replace_between(block: str, start_marker: str, end_marker: str, replacement: str, label: str) -> str:
    if start_marker not in block or end_marker not in block:
        raise RuntimeError(f"cannot patch {label}")
    i = block.index(start_marker)
    j = block.index(end_marker, i)
    return block[:i] + replacement + block[j:]

# ------------------------------------------------------------------
# 1) Create a separate locality-enabled backend:
#    bgi/bert4keras/models_locality.py
# ------------------------------------------------------------------
src = LIB / "models.py"
dst = LIB / "models_locality.py"
s = src.read_text()

# Add locality_mode arg to classes that already have custom_conv_layer
s = s.replace(
    "            custom_conv_layer=False,\n            use_position_ids=True,",
    "            custom_conv_layer=False,\n            locality_mode='none',\n            use_position_ids=True,"
)

# Backward compatibility:
# - if old code sets custom_conv_layer=True and locality_mode not given -> treat as multi
# - otherwise locality_mode controls whether conv is on
s = s.replace(
    "        self.custom_conv_layer = custom_conv_layer",
    "        self.locality_mode = 'multi' if (locality_mode == 'none' and custom_conv_layer) else locality_mode\n"
    "        self.custom_conv_layer = (self.locality_mode != 'none')"
)

# ---------- Patch BERT embedding block ----------
bert_start = s.index("class BERT(Transformer):")
multi_start = s.index("class Multi_Inputs_BERT(Transformer):")
bert_block = s[bert_start:multi_start]

bert_block = replace_between(
    bert_block,
    """        x = self.apply(
            inputs=x,
            layer=Embedding,
            input_dim=self.vocab_size,
            output_dim=self.embedding_size,
            embeddings_initializer=self.initializer,
            mask_zero=True,
            name='Embedding-Token'
        )
""",
    "        if self.type_vocab_size > 0:\n",
    """        x = self.apply(
            inputs=x,
            layer=Embedding,
            input_dim=self.vocab_size,
            output_dim=self.embedding_size,
            embeddings_initializer=self.initializer,
            mask_zero=True,
            name='Embedding-Token'
        )

        locality_mode = getattr(self, 'locality_mode', 'none')
        if locality_mode == 'multi':
            x1 = self.apply(
                inputs=x,
                layer=keras.layers.Conv1D,
                filters=self.embedding_size,
                kernel_size=2,
                strides=1,
                padding='same',
                name='Embedding-Token-Conv1D-2'
            )
            x2 = self.apply(
                inputs=x,
                layer=keras.layers.Conv1D,
                filters=self.embedding_size,
                kernel_size=3,
                strides=1,
                padding='same',
                name='Embedding-Token-Conv1D-3'
            )
            x3 = self.apply(
                inputs=x,
                layer=keras.layers.Conv1D,
                filters=self.embedding_size,
                kernel_size=5,
                strides=1,
                padding='same',
                name='Embedding-Token-Conv1D-5'
            )
            token_inputs = [x1, x2, x3]
        elif locality_mode == 'single':
            x_local = self.apply(
                inputs=x,
                layer=keras.layers.Conv1D,
                filters=self.embedding_size,
                kernel_size=3,
                strides=1,
                padding='same',
                name='Embedding-Token-Conv1D-3'
            )
            token_inputs = [x_local]
        elif locality_mode == 'depthwise':
            x_local = self.apply(
                inputs=x,
                layer=keras.layers.SeparableConv1D,
                filters=self.embedding_size,
                kernel_size=3,
                strides=1,
                padding='same',
                name='Embedding-Token-SeparableConv1D-3'
            )
            token_inputs = [x_local]
        else:
            token_inputs = [x]

        s = self.apply(
            inputs=s,
            layer=Embedding,
            input_dim=self.segment_vocab_size,
            output_dim=self.embedding_size,
            embeddings_initializer=self.initializer,
            name='Embedding-Segment'
        )
        print("x: ", x)
        print("s: ", s)

        if self.use_segment_ids:
            embedding_inputs = token_inputs + [s]
        else:
            embedding_inputs = list(token_inputs)

""",
    "BERT apply_embeddings",
)

s = s[:bert_start] + bert_block + s[multi_start:]

# ---------- Patch Multi_Inputs_BERT embedding block ----------
multi_start = s.index("class Multi_Inputs_BERT(Transformer):")
next_class = s.index("\nclass ", multi_start + 1)
multi_block = s[multi_start:next_class]

multi_block = replace_between(
    multi_block,
    """        x = self.apply(
            inputs=x,
            layer=Embedding,
            input_dim=self.vocab_size,
            output_dim=self.embedding_size,
            embeddings_initializer=self.initializer,
            mask_zero=True,
            name='Embedding-Token'
        )
""",
    "        if len(self.multi_inputs) > 0:\n",
    """        x = self.apply(
            inputs=x,
            layer=Embedding,
            input_dim=self.vocab_size,
            output_dim=self.embedding_size,
            embeddings_initializer=self.initializer,
            mask_zero=True,
            name='Embedding-Token'
        )

        locality_mode = getattr(self, 'locality_mode', 'none')
        if locality_mode == 'multi':
            x1 = self.apply(
                inputs=x,
                layer=keras.layers.Conv1D,
                filters=self.embedding_size,
                kernel_size=2,
                strides=1,
                padding='same',
                name='Embedding-Token-Conv1D-2'
            )
            x2 = self.apply(
                inputs=x,
                layer=keras.layers.Conv1D,
                filters=self.embedding_size,
                kernel_size=3,
                strides=1,
                padding='same',
                name='Embedding-Token-Conv1D-3'
            )
            x3 = self.apply(
                inputs=x,
                layer=keras.layers.Conv1D,
                filters=self.embedding_size,
                kernel_size=5,
                strides=1,
                padding='same',
                name='Embedding-Token-Conv1D-5'
            )
            token_inputs = [x1, x2, x3]
        elif locality_mode == 'single':
            x_local = self.apply(
                inputs=x,
                layer=keras.layers.Conv1D,
                filters=self.embedding_size,
                kernel_size=3,
                strides=1,
                padding='same',
                name='Embedding-Token-Conv1D-3'
            )
            token_inputs = [x_local]
        elif locality_mode == 'depthwise':
            x_local = self.apply(
                inputs=x,
                layer=keras.layers.SeparableConv1D,
                filters=self.embedding_size,
                kernel_size=3,
                strides=1,
                padding='same',
                name='Embedding-Token-SeparableConv1D-3'
            )
            token_inputs = [x_local]
        else:
            token_inputs = [x]

        s = self.apply(
            inputs=s,
            layer=Embedding,
            input_dim=2,
            output_dim=self.embedding_size,
            embeddings_initializer=self.initializer,
            name='Embedding-Segment'
        )

        embedding_inputs = list(token_inputs)
        if self.use_segment_ids:
            embedding_inputs = token_inputs + [s]

""",
    "Multi_Inputs_BERT apply_embeddings",
)

s = s[:multi_start] + multi_block + s[next_class:]

# ---------- Patch FFN block ----------
old_ffn = """        if self.custom_conv_layer:
            x = self.apply(
                inputs=x,
                layer=keras.layers.Conv1D,
                filters=self.hidden_size,
                kernel_size=3,
                strides=1,
                padding='same',
                name=feed_forward_name + '_Conv1D'
            )
        else:
            x = self.apply(
                inputs=x,
                layer=FeedForward,
                units=self.intermediate_size,
                activation=self.hidden_act,
                kernel_initializer=self.initializer,
                name=feed_forward_name
            )
"""

new_ffn = """        locality_mode = getattr(self, 'locality_mode', 'none')
        if locality_mode == 'none':
            x = self.apply(
                inputs=x,
                layer=FeedForward,
                units=self.intermediate_size,
                activation=self.hidden_act,
                kernel_initializer=self.initializer,
                name=feed_forward_name
            )
        elif locality_mode == 'depthwise':
            x = self.apply(
                inputs=x,
                layer=keras.layers.SeparableConv1D,
                filters=self.hidden_size,
                kernel_size=3,
                strides=1,
                padding='same',
                name=feed_forward_name + '_SeparableConv1D'
            )
        else:
            x = self.apply(
                inputs=x,
                layer=keras.layers.Conv1D,
                filters=self.hidden_size,
                kernel_size=3,
                strides=1,
                padding='same',
                name=feed_forward_name + '_Conv1D'
            )
"""

s = s.replace(old_ffn, new_ffn)
dst.write_text(s)
print(f"[OK] wrote {dst}")

# ------------------------------------------------------------------
# 2) Create trainer variants
# ------------------------------------------------------------------
def strip_pretrain(text: str) -> str:
    text = re.sub(r'^(\s*)USE_PRETRAIN\s*=\s*True', r'\1USE_PRETRAIN = False', text, flags=re.M)
    text = re.sub(r'^\s*model\.load_weights\([^\n]*by_name=True[^\n]*\)\n', '', text, flags=re.M)
    return text

def patch_seq(mode: str):
    src = PROM / "01_PromID_trainer.py"
    text = src.read_text()

    text = text.replace(
        "from bgi.bert4keras.models import build_transformer_model",
        f"from bgi.bert4keras.models_locality import build_transformer_model\n\nLOCALITY_MODE = '{mode}'\nRUN_FAMILY = 'sequence_only'"
    )
    text = strip_pretrain(text)

    text = text.replace(
        '"custom_conv_layer": False,',
        '"custom_conv_layer": LOCALITY_MODE != "none",\n        "locality_mode": LOCALITY_MODE,'
    )

    text = text.replace(
        "./data/kfold_{}_train_and_valid_index_{}_gram_{}.npz'.format(str(k_fold), str(ngram), task_name)",
        "./data/kfold_{}_train_and_valid_index_{}_gram_{}_{}_{}.npz'.format(str(k_fold), str(ngram), task_name, RUN_FAMILY, LOCALITY_MODE)"
    )

    text = text.replace(
        "./data/promoter_best_model_gene_bert_{}_fold_{}_gram_{}.h5'.format(str(k_fold), str(ngram), task_name)",
        "./data/promoter_best_model_gene_bert_{}_fold_{}_gram_{}_{}_{}.h5'.format(str(k_fold), str(ngram), task_name, RUN_FAMILY, LOCALITY_MODE)"
    )

    out = PROM / f"01_PromID_trainer_locality_seq_{mode}.py"
    out.write_text(text)
    print(f"[OK] wrote {out}")

def patch_struct(mode: str):
    candidates = [
        PROM / "02_PromID_trainer_knowledge_structural.py",
        PROM / "02_PromID_trainer_knowledge.py",
    ]
    src = None
    for p in candidates:
        if p.exists():
            src = p
            break
    if src is None:
        raise RuntimeError("cannot find structural knowledge trainer source")

    text = src.read_text()

    text = text.replace(
        "from bgi.bert4keras.models import build_transformer_model",
        f"from bgi.bert4keras.models_locality import build_transformer_model\n\nLOCALITY_MODE = '{mode}'\nRUN_FAMILY = 'structural_knowledge'"
    )
    text = strip_pretrain(text)

    text = text.replace(
        '"custom_conv_layer": True,',
        '"custom_conv_layer": LOCALITY_MODE != "none",\n        "locality_mode": LOCALITY_MODE,'
    )

    # fallback: if only generic knowledge trainer exists, redirect folder names
    if src.name == "02_PromID_trainer_knowledge.py":
        text = text.replace("11_knowledge", "structural_knowledge")

    text = text.replace(
        "./data/kfold_{}_train_and_valid_index_{}_gram_{}_11_Knowledge.npz'.format(str(k_fold), str(ngram), task_name)",
        "./data/kfold_{}_train_and_valid_index_{}_gram_{}_{}_{}.npz'.format(str(k_fold), str(ngram), task_name, RUN_FAMILY, LOCALITY_MODE)"
    )

    text = text.replace(
        "./data/promoter_best_model_gene_bert_{}_fold_{}_gram_{}_11_Knowledge.h5'.format(str(k_fold), str(ngram), task_name)",
        "./data/promoter_best_model_gene_bert_{}_fold_{}_gram_{}_{}_{}.h5'.format(str(k_fold), str(ngram), task_name, RUN_FAMILY, LOCALITY_MODE)"
    )

    out = PROM / f"02_PromID_trainer_locality_structural_{mode}.py"
    out.write_text(text)
    print(f"[OK] wrote {out}")

for mode in ["multi", "single", "depthwise"]:
    patch_seq(mode)

for mode in ["none", "single", "depthwise"]:
    patch_struct(mode)

# ------------------------------------------------------------------
# 3) Create parser
# ------------------------------------------------------------------
parser = PROM / "parse_locality_ablation_logs.py"
parser.write_text(r'''
import csv
import math
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path("/home/users/nus/e1538285/scratch/LOGO")
PROM = ROOT / "02_LOGO_Promoter"
LOGDIR = PROM / "locality_logs"

def infer_task(line: str):
    u = line.upper()
    if "NONTATA" in u or "NO_TATA" in u or "NO-TATA" in u:
        return "NO_TATA_BOX"
    if "TATA_BOX" in u or ("TATA" in u and "NONTATA" not in u and "NO_TATA" not in u):
        return "TATA_BOX"
    if "EPDNEW_BOTH" in u or "BOTH" in u:
        return "BOTH"
    return None

def parse_eval(line: str):
    m = re.search(r"Eval:\s*\[([^\]]+)\]", line)
    if not m:
        return None
    parts = [x.strip() for x in m.group(1).split(",")]
    if len(parts) < 5:
        return None
    vals = [float(x) for x in parts[:5]]
    return {
        "loss": vals[0],
        "acc": vals[1],
        "precision": vals[2],
        "recall": vals[3],
        "f1": vals[4],
    }

def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")

def sd(xs):
    if len(xs) <= 1:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))

rows = []
current_task = {}

for path in sorted(LOGDIR.glob("*.log")):
    stem = path.stem
    if "_" not in stem:
        continue
    branch_short, mode = stem.split("_", 1)
    branch = "sequence_only" if branch_short == "seq" else "structural_knowledge"
    current_task[path.name] = None

    with path.open() as f:
        for line in f:
            task = infer_task(line)
            if task is not None:
                current_task[path.name] = task

            ev = parse_eval(line)
            if ev is None:
                continue

            task = current_task[path.name]
            if task is None:
                continue

            rows.append({
                "task": task,
                "branch": branch,
                "locality_mode": mode,
                **ev,
                "source": path.name,
            })

# add existing corners if present
final_table = PROM / "promoter_final_comparison_table.csv"
if final_table.exists():
    with final_table.open() as f:
        for r in csv.DictReader(f):
            if r.get("model") == "LOGO_sequence_only":
                rows.append({
                    "task": r["task"],
                    "branch": "sequence_only",
                    "locality_mode": "none",
                    "loss": float("nan"),
                    "acc": float(r["acc_mean"]),
                    "precision": float(r["precision_mean"]),
                    "recall": float(r["recall_mean"]),
                    "f1": float(r["f1_mean"]),
                    "source": "existing_promoter_final_comparison_table.csv",
                })

knowledge_table = PROM / "knowledge_ablation_results.csv"
if knowledge_table.exists():
    with knowledge_table.open() as f:
        for r in csv.DictReader(f):
            if r.get("knowledge_type") == "structural":
                rows.append({
                    "task": r["task"],
                    "branch": "structural_knowledge",
                    "locality_mode": "multi",
                    "loss": float("nan"),
                    "acc": float(r["acc_mean"]),
                    "precision": float(r["precision_mean"]),
                    "recall": float(r["recall_mean"]),
                    "f1": float(r["f1_mean"]),
                    "source": "existing_knowledge_ablation_results.csv",
                })

grouped = defaultdict(list)
for r in rows:
    grouped[(r["task"], r["branch"], r["locality_mode"])].append(r)

out_csv = PROM / "locality_ablation_results.csv"
with out_csv.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow([
        "task", "branch", "locality_mode", "n_folds",
        "acc_mean", "acc_sd",
        "precision_mean", "precision_sd",
        "recall_mean", "recall_sd",
        "f1_mean", "f1_sd",
        "sources"
    ])
    for key in sorted(grouped):
        xs = grouped[key]
        w.writerow([
            key[0], key[1], key[2], len(xs),
            mean([x["acc"] for x in xs]), sd([x["acc"] for x in xs]),
            mean([x["precision"] for x in xs]), sd([x["precision"] for x in xs]),
            mean([x["recall"] for x in xs]), sd([x["recall"] for x in xs]),
            mean([x["f1"] for x in xs]), sd([x["f1"] for x in xs]),
            ";".join(sorted({x["source"] for x in xs})),
        ])

print(f"[OK] wrote {out_csv}")
''')
print(f"[OK] wrote {parser}")

PY

# ------------------------------------------------------------------
# 4) Create PBS scripts
# ------------------------------------------------------------------
make_pbs () {
    local job_name="$1"
    local py_file="$2"
    local log_file="$3"
    cat > "$PROM/pbs_${job_name}.sh" <<EOF
#!/usr/bin/env bash
#PBS -N ${job_name}
#PBS -q normal
#PBS -l select=1:ncpus=8:mem=64gb:ngpus=1
#PBS -l walltime=08:00:00
#PBS -j oe

set -euo pipefail
cd $ROOT

if [ -f "\$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "\$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "\$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "\$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "/apps/anaconda3/etc/profile.d/conda.sh" ]; then
    source "/apps/anaconda3/etc/profile.d/conda.sh"
fi

conda activate logo
export PYTHONPATH=$ROOT:\$PYTHONPATH
mkdir -p $LOGDIR

python $PROM/${py_file} 2>&1 | tee $LOGDIR/${log_file}
EOF
    chmod +x "$PROM/pbs_${job_name}.sh"
    echo "[OK] wrote $PROM/pbs_${job_name}.sh"
}

make_pbs "seq_multi"                "01_PromID_trainer_locality_seq_multi.py"                "seq_multi.log"
make_pbs "seq_single"               "01_PromID_trainer_locality_seq_single.py"               "seq_single.log"
make_pbs "seq_depthwise"            "01_PromID_trainer_locality_seq_depthwise.py"            "seq_depthwise.log"
make_pbs "structural_none"          "02_PromID_trainer_locality_structural_none.py"          "structural_none.log"
make_pbs "structural_single"        "02_PromID_trainer_locality_structural_single.py"        "structural_single.log"
make_pbs "structural_depthwise"     "02_PromID_trainer_locality_structural_depthwise.py"     "structural_depthwise.log"

# ------------------------------------------------------------------
# 5) Create one-click submit script
# ------------------------------------------------------------------
cat > "$PROM/submit_locality_ablation_jobs.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

PROM=/home/users/nus/e1538285/scratch/LOGO/02_LOGO_Promoter

jid1=$(qsub "$PROM/pbs_seq_multi.sh")
jid2=$(qsub "$PROM/pbs_seq_single.sh")
jid3=$(qsub "$PROM/pbs_seq_depthwise.sh")
jid4=$(qsub "$PROM/pbs_structural_none.sh")
jid5=$(qsub "$PROM/pbs_structural_single.sh")
jid6=$(qsub "$PROM/pbs_structural_depthwise.sh")

echo "submitted:"
echo "  $jid1  seq_multi"
echo "  $jid2  seq_single"
echo "  $jid3  seq_depthwise"
echo "  $jid4  structural_none"
echo "  $jid5  structural_single"
echo "  $jid6  structural_depthwise"
EOF

chmod +x "$PROM/submit_locality_ablation_jobs.sh"
echo "[OK] wrote $PROM/submit_locality_ablation_jobs.sh"

echo
echo "All setup finished."
echo "Next:"
echo "  bash $PROM/submit_locality_ablation_jobs.sh"
echo
echo "After all jobs finish:"
echo "  conda activate logo"
echo "  python $PROM/parse_locality_ablation_logs.py"
