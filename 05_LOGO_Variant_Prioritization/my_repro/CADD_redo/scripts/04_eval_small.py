import os
import importlib.util
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score, average_precision_score

BASE = "/home/users/nus/e1538285/scratch/LOGO/05_LOGO_Variant_Prioritization/my_repro/CADD_redo"
TRAIN_SCRIPT = os.path.join(BASE, "scripts", "02_train_smoke.py")

VALID_FILE = os.path.join(BASE, "data_small_compat", "clinvar_InDel_fold0_valid_local.tfrecord")
TEST_FILE  = os.path.join(BASE, "data_small_compat", "clinvar_InDel_fold0_test_local.tfrecord")

feature_description = {
    "label": tf.io.FixedLenFeature([1], tf.int64),
    "seq": tf.io.VarLenFeature(tf.int64),
    "alt_seq": tf.io.VarLenFeature(tf.int64),
    "alt_type": tf.io.VarLenFeature(tf.int64),
}

def read_tfrecord(path):
    y_list, seq_list, alt_list, typ_list = [], [], [], []
    ds = tf.data.TFRecordDataset(path)
    for raw in ds:
        ex = tf.io.parse_single_example(raw, feature_description)
        y = int(ex["label"].numpy()[0])
        seq = tf.sparse.to_dense(ex["seq"]).numpy()
        alt = tf.sparse.to_dense(ex["alt_seq"]).numpy()
        typ = tf.sparse.to_dense(ex["alt_type"]).numpy()
        y_list.append(y)
        seq_list.append(seq)
        alt_list.append(alt)
        typ_list.append(typ)

    y = np.array(y_list, dtype=np.int32)
    seq = np.array(seq_list, dtype=np.int32)
    alt = np.array(alt_list, dtype=np.int32)
    typ = np.array(typ_list, dtype=np.int32)
    return y, seq, alt, typ

def report(name, y, pred):
    print(f"\n===== {name} =====")
    print("n =", len(y))
    print("label_mean =", float(y.mean()))
    print("pred_mean  =", float(pred.mean()))
    print("pred_min   =", float(pred.min()))
    print("pred_max   =", float(pred.max()))
    print("pred>0.5   =", float((pred > 0.5).mean()))
    print("pred>0.1   =", float((pred > 0.1).mean()))
    print("pred>0.9   =", float((pred > 0.9).mean()))
    try:
        auroc = roc_auc_score(y, pred)
    except Exception as e:
        auroc = f"ERROR: {e}"
    try:
        auprc = average_precision_score(y, pred)
    except Exception as e:
        auprc = f"ERROR: {e}"
    print("AUROC      =", auroc)
    print("AUPRC      =", auprc)

print("=== importing and running smoke training script ===")
spec = importlib.util.spec_from_file_location("redo_train_smoke", TRAIN_SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

if not hasattr(mod, "model"):
    raise RuntimeError("Could not find `model` in 02_train_smoke.py after execution.")

model = mod.model

valid_y, valid_seq, valid_alt, valid_typ = read_tfrecord(VALID_FILE)
test_y,  test_seq,  test_alt,  test_typ  = read_tfrecord(TEST_FILE)

valid_pred = model.predict([valid_seq, valid_alt, valid_typ], batch_size=128).reshape(-1)
test_pred  = model.predict([test_seq,  test_alt,  test_typ],  batch_size=128).reshape(-1)

report("VALID_SMALL", valid_y, valid_pred)
report("TEST_SMALL", test_y, test_pred)
