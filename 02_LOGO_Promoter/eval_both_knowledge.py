import os
import math
import inspect
import importlib.util
import numpy as np
import tensorflow as tf

BASE = "02_LOGO_Promoter"
DATA_FILE = os.path.join(BASE, "data/5_gram_11_knowledge/epdnew_BOTH_Knowledge_5_gram.npz")
IDX_TMPL = os.path.join(BASE, "data/kfold_{fold}_train_and_valid_index_5_gram_epdnew_BOTH_Knowledge_11_Knowledge.npz")
MODEL_TMPL = os.path.join(BASE, "data/promoter_best_model_gene_bert_{fold}_fold_5_gram_epdnew_BOTH_Knowledge_11_Knowledge.h5")

# load module
spec = importlib.util.spec_from_file_location(
    "kmod",
    os.path.join(BASE, "02_PromID_trainer_knowledge.py")
)
kmod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kmod)

print("=== DIAGNOSTIC ===")
print("model_def signature:", inspect.signature(kmod.model_def))
print("annotation_size:", getattr(kmod, "annotation_size", None))

# load full dataset using the repo helper, matching original pipeline style
x_all, anno_all, y_all = kmod.load_all_data(
    [DATA_FILE],
    ngram=5,
    only_one_slice=True,
    ngram_index=None,
    masked=False
)

print("x_all shape:", x_all.shape)
print("anno_all shape:", anno_all.shape)
print("y_all shape:", y_all.shape)

def pick_valid_key(keys):
    for k in keys:
        lk = k.lower()
        if "valid" in lk:
            return k
    for k in keys:
        lk = k.lower()
        if "test" in lk:
            return k
    # fallback: second array if present
    if len(keys) >= 2:
        return keys[1]
    return keys[0]

results = []
batch_size = 256

for fold in range(10):
    idx_file = IDX_TMPL.format(fold=fold)
    model_file = MODEL_TMPL.format(fold=fold)

    if not os.path.exists(idx_file):
        print(f"missing index file: {idx_file}")
        continue
    if not os.path.exists(model_file):
        print(f"missing model file: {model_file}")
        continue

    idx_npz = np.load(idx_file)
    idx_keys = list(idx_npz.files)
    valid_key = pick_valid_key(idx_keys)
    valid_idx = idx_npz[valid_key]

    x_valid = x_all[valid_idx]
    anno_valid = anno_all[valid_idx]
    y_valid = y_all[valid_idx]

    ds = kmod.load_npz_dataset_for_classification(
        x_promoter_data_all=x_valid,
        annotation_data_all=anno_valid,
        y_data_all=y_valid,
        promoter_seq_len=x_valid.shape[1],
        annotation_size=getattr(kmod, "annotation_size", anno_valid.shape[1]),
        ngram=5,
        only_one_slice=True,
        ngram_index=None,
        shuffle=False,
        seq_len=x_valid.shape[1],
        num_classes=1,
        masked=False,
    )
    ds = ds.batch(batch_size).map(kmod.parse_function).prefetch(tf.data.AUTOTUNE)

    # build model and load weights
    model = kmod.model_def()
    try:
        model.load_weights(model_file)
    except Exception as e:
        print(f"fold {fold}: load_weights failed, trying load_model -> {e}")
        model = tf.keras.models.load_model(
            model_file,
            custom_objects={"f1_score": kmod.f1_score},
            compile=False
        )

    model.compile(
        loss="binary_crossentropy",
        optimizer=tf.keras.optimizers.Adam(1e-4),
        metrics=[
            "acc",
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
            kmod.f1_score
        ]
    )

    steps = math.ceil(len(x_valid) / batch_size)
    ev = model.evaluate(ds, steps=steps, verbose=0)

    # loss, acc, precision, recall, f1
    row = [fold] + [float(v) for v in ev[:5]]
    results.append(row)
    print(f"fold={fold},loss={row[1]:.6f},acc={row[2]:.6f},precision={row[3]:.6f},recall={row[4]:.6f},f1={row[5]:.6f}")

print("\n=== CSV_FOLD_RESULTS ===")
print("fold,loss,acc,precision,recall,f1")
for row in results:
    print(",".join([str(row[0])] + [f"{v:.6f}" for v in row[1:]]))

if results:
    arr = np.array([r[1:] for r in results], dtype=float)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0, ddof=0)

    print("\n=== CSV_SUMMARY ===")
    print("task,model,n_folds,loss_mean,acc_mean,precision_mean,recall_mean,f1_mean,loss_sd,acc_sd,precision_sd,recall_sd,f1_sd")
    print(
        "BOTH,LOGO_knowledge_enabled,{},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f}".format(
            len(results),
            mean[0], mean[1], mean[2], mean[3], mean[4],
            std[0], std[1], std[2], std[3], std[4]
        )
    )
