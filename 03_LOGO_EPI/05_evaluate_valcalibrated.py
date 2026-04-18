import os
import sys
import gc
import csv
from pathlib import Path
import importlib.util

import numpy as np
import sklearn.metrics as sk_metrics
import tensorflow as tf

print("==========================================================")
print("Starting validation-calibrated evaluation on original test")
print("==========================================================")

# 保证能 import 到 bgi
sys.path.append("../")

# 动态导入训练脚本（不会触发 __main__）
script_name = "04_LOGO_EPI_train_conv1d_concat_atcg_valauc.py"
spec = importlib.util.spec_from_file_location("logo_module", script_name)
logo_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(logo_module)

from bgi.common.refseq_utils import get_word_dict_for_n_gram_number

# ==== 全局配置 ====
NGRAM = 6
TYPE = "P-E"
CELLS = ["tB", "FoeT", "Mon", "nCD4", "tCD4", "tCD8"]
NUM_ENSEMBL = 10
BATCH_SIZE = 128
NUM_PARALLEL_CALLS = 16
ENHANCER_RESIZED_LEN = 2000
PROMOTER_RESIZED_LEN = 1000

# Dynamic allocation of video memory
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

word_dict = get_word_dict_for_n_gram_number(n_gram=NGRAM)
logo_module.vocab_size = len(word_dict) + 10

enhancer_seq_len = ENHANCER_RESIZED_LEN // NGRAM // NGRAM * NGRAM
promoter_seq_len = PROMOTER_RESIZED_LEN // NGRAM // NGRAM * NGRAM

out_dir = Path("optimization_all_celltypes")
(out_dir / "val_threshold_tables").mkdir(parents=True, exist_ok=True)
(out_dir / "val_threshold_notes").mkdir(parents=True, exist_ok=True)

summary_rows = []

def load_region_pair(base_dir, ngram=6):
    enhancer_file = f"{base_dir}/enhancer_Seq_{ngram}_gram.npz"
    promoter_file = f"{base_dir}/promoter_Seq_{ngram}_gram.npz"

    region1_seq, label = logo_module.load_all_data(
        [enhancer_file], ngram=ngram, only_one_slice=True, ngram_index=1
    )
    region2_seq, _ = logo_module.load_all_data(
        [promoter_file], ngram=ngram, only_one_slice=True, ngram_index=1
    )
    return region1_seq, region2_seq, label

def build_dataset(region1_seq, region2_seq, label):
    dataset = logo_module.load_npz_dataset_for_classification(
        region1_seq,
        region2_seq,
        label,
        enhancer_seq_len,
        promoter_seq_len,
        ngram=NGRAM,
        only_one_slice=True,
        ngram_index=1,
        shuffle=False,
        seq_len=0,
        num_classes=1,
        masked=False,
    )
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.map(map_func=logo_module.parse_function,
                          num_parallel_calls=NUM_PARALLEL_CALLS)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

def predict_scores(weight_path, region1_seq, region2_seq, label):
    model = logo_module.model_def(vocab_size=logo_module.vocab_size)
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(lr=0.00001),
        metrics=['acc', logo_module.f1_score, tf.keras.metrics.AUC(), logo_module.average_precision]
    )
    model.load_weights(weight_path)

    dataset = build_dataset(region1_seq, region2_seq, label)
    steps_per_epoch = len(label) // BATCH_SIZE + 1
    score = model.predict(dataset, steps=steps_per_epoch, verbose=0)
    score = score[:len(label)].reshape(-1)

    tf.keras.backend.clear_session()
    del model
    gc.collect()
    return score

def select_best_threshold(y_true, y_score):
    best_t = 0.50
    best_f1 = -1.0
    for t in np.arange(0.01, 1.00, 0.01):
        y_pred = (y_score > t).astype(int)
        f1 = sk_metrics.f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(np.round(t, 2))
    return best_t, best_f1

def get_original_test_dir(cell, type_):
    test_original = Path(cell) / type_ / "test_original" / f"{NGRAM}_gram"
    if test_original.exists():
        return str(test_original)
    return str(Path(cell) / type_ / "test" / f"{NGRAM}_gram")

for cell in CELLS:
    print("\n" + "=" * 60)
    print(f"Processing {cell}")
    print("=" * 60)

    train_base = f"{cell}/{TYPE}/{NGRAM}_gram"
    test_base = get_original_test_dir(cell, TYPE)

    # ---- 1) 读取训练集，用保存好的 kfold index 重建 OOF validation scores ----
    train_region1, train_region2, train_label = load_region_pair(train_base, ngram=NGRAM)
    oof_scores = np.zeros(len(train_label), dtype=np.float32)

    for fold in range(NUM_ENSEMBL):
        idx_path = f"{cell}/{TYPE}/kfold_{fold}_train_and_valid_index.npz"
        weight_path = f"{cell}/{TYPE}/opt_valauc/best_model_gene_bert_{fold}.h5"

        if not os.path.exists(idx_path):
            raise FileNotFoundError(f"Missing kfold index: {idx_path}")
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"Missing weight: {weight_path}")

        idx_data = np.load(idx_path)
        valid_idx = idx_data["test"]

        fold_scores = predict_scores(
            weight_path,
            train_region1[valid_idx],
            train_region2[valid_idx],
            train_label[valid_idx],
        )
        oof_scores[valid_idx] = fold_scores
        print(f"{cell} fold {fold}: validation scores done")

    val_auprc = sk_metrics.average_precision_score(train_label, oof_scores)
    best_t, val_best_f1 = select_best_threshold(train_label, oof_scores)

    print(f"{cell} validation-calibrated threshold = {best_t:.2f}")
    print(f"{cell} OOF validation AUPRC = {val_auprc:.6f}")
    print(f"{cell} OOF validation F1@best_t = {val_best_f1:.6f}")

    # ---- 2) 在原始不平衡 test 上做 bagging 评估 ----
    test_region1, test_region2, test_label = load_region_pair(test_base, ngram=NGRAM)
    bag_scores = np.zeros((NUM_ENSEMBL, len(test_label)), dtype=np.float32)

    for fold in range(NUM_ENSEMBL):
        weight_path = f"{cell}/{TYPE}/opt_valauc/best_model_gene_bert_{fold}.h5"
        bag_scores[fold, :] = predict_scores(
            weight_path,
            test_region1,
            test_region2,
            test_label,
        )
        print(f"{cell} fold {fold}: test scores done")

    test_vote_score = np.mean(bag_scores, axis=0)
    test_auprc = sk_metrics.average_precision_score(test_label, test_vote_score)
    test_f1_default = sk_metrics.f1_score(test_label, (test_vote_score > 0.5).astype(int))
    test_f1_valcal = sk_metrics.f1_score(test_label, (test_vote_score > best_t).astype(int))

    print(f"{cell} test AUPRC = {test_auprc:.6f}")
    print(f"{cell} test F1@0.50 = {test_f1_default:.6f}")
    print(f"{cell} test F1@val_best_t({best_t:.2f}) = {test_f1_valcal:.6f}")

    # ---- 3) 保存每个 cell 的详细分数 ----
    np.savez(
        out_dir / "val_threshold_tables" / f"{cell}_valcalibrated_scores.npz",
        train_label=train_label,
        oof_scores=oof_scores,
        test_label=test_label,
        test_vote_score=test_vote_score,
        best_threshold=best_t,
        val_auprc=val_auprc,
        val_best_f1=val_best_f1,
        test_auprc=test_auprc,
        test_f1_default=test_f1_default,
        test_f1_valcal=test_f1_valcal,
    )

    summary_rows.append({
        "cell_type": cell,
        "best_threshold_from_validation": best_t,
        "val_oof_auprc": val_auprc,
        "val_oof_f1_at_best_t": val_best_f1,
        "test_auprc": test_auprc,
        "test_f1_default_0.5": test_f1_default,
        "test_f1_valcalibrated": test_f1_valcal,
        "delta_f1": test_f1_valcal - test_f1_default,
        "test_source": test_base,
    })

# ---- 4) 写总表 ----
csv_path = out_dir / "val_threshold_tables" / "table_epi_valcalibrated_summary.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "cell_type",
            "best_threshold_from_validation",
            "val_oof_auprc",
            "val_oof_f1_at_best_t",
            "test_auprc",
            "test_f1_default_0.5",
            "test_f1_valcalibrated",
            "delta_f1",
            "test_source",
        ],
    )
    writer.writeheader()
    writer.writerows(summary_rows)

print("\n==========================================================")
print(f"Saved summary to: {csv_path}")
print("Validation-calibrated evaluation finished.")
print("==========================================================")

# 避免 TensorFlow 收尾死锁
import os as _os
import sys as _sys
print("Forcing exit to prevent TensorFlow deadlock.")
_sys.stdout.flush()
_os._exit(0)
