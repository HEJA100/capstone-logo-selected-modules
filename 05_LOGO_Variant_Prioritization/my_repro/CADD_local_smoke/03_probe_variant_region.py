import os
import tensorflow as tf
import numpy as np

BASE = os.path.expanduser(
    "~/scratch/LOGO/05_LOGO_Variant_Prioritization/my_repro/CADD_work/local_tfrecord_fold0"
)

FILES = [
    os.path.join(BASE, "humanDerived_InDels_fold0_small.tfrecord"),
    os.path.join(BASE, "simulation_InDels_fold0_small.tfrecord"),
    os.path.join(BASE, "clinvar_InDel_fold0_valid_local.tfrecord"),
    os.path.join(BASE, "clinvar_InDel_fold0_test_local.tfrecord"),
]

feature_description = {
    "label": tf.io.VarLenFeature(tf.int64),
    "seq": tf.io.VarLenFeature(tf.int64),
    "alt_seq": tf.io.VarLenFeature(tf.int64),
    "alt_type": tf.io.VarLenFeature(tf.int64),
}

def parse_example(example_proto):
    ex = tf.io.parse_single_example(example_proto, feature_description)
    out = {}
    for k, v in ex.items():
        out[k] = tf.sparse.to_dense(v).numpy()
    return out

for path in FILES:
    print("\n===== FILE =====")
    print(path)
    ds = tf.data.TFRecordDataset(path)

    for i, raw in enumerate(ds.take(3)):
        ex = parse_example(raw)
        seq = ex["seq"]
        alt = ex["alt_seq"]
        typ = ex["alt_type"]
        label = ex["label"].tolist()

        diff_idx = np.where(seq != alt)[0]
        type_idx = np.where(typ != 0)[0]

        print(f"example {i}")
        print(" label            :", label)
        print(" n_diff_tokens    :", len(diff_idx))
        print(" n_type_nonzero   :", len(type_idx))

        if len(diff_idx) > 0:
            print(" first_diff_idx   :", diff_idx[:10].tolist())
        else:
            print(" first_diff_idx   : []")

        if len(type_idx) > 0:
            print(" first_type_idx   :", type_idx[:10].tolist())
        else:
            print(" first_type_idx   : []")

        mid = len(seq) // 2
        s = max(0, mid - 10)
        e = min(len(seq), mid + 10)

        print(" seq[mid-10:mid+10] :", seq[s:e].tolist())
        print(" alt[mid-10:mid+10] :", alt[s:e].tolist())
        print(" typ[mid-10:mid+10] :", typ[s:e].tolist())
