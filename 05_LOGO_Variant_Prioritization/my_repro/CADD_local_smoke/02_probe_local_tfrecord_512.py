import os
import tensorflow as tf

BASE = os.path.expanduser(
    "~/scratch/LOGO/05_LOGO_Variant_Prioritization/my_repro/CADD_work/local_tfrecord_fold0_512"
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
        out[k] = tf.sparse.to_dense(v)
    return out

for path in FILES:
    print("\n===== FILE =====")
    print(path)
    ds = tf.data.TFRecordDataset(path).map(parse_example)

    for i, ex in enumerate(ds.take(3)):
        label = ex["label"].numpy().tolist()
        seq = ex["seq"].numpy()
        alt_seq = ex["alt_seq"].numpy()
        alt_type = ex["alt_type"].numpy()

        print(f"example {i}")
        print(" label    :", label)
        print(" seq_len  :", len(seq))
        print(" alt_len  :", len(alt_seq))
        print(" type_len :", len(alt_type))
        print(" seq[:10] :", seq[:10].tolist())
        print(" alt[:10] :", alt_seq[:10].tolist())
        print(" typ[:10] :", alt_type[:10].tolist())
