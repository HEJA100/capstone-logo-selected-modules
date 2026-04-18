import os
import sys
import importlib.util
import numpy as np
import tensorflow as tf

REPO_ROOT = os.path.expanduser("~/scratch/LOGO")
sys.path.insert(0, REPO_ROOT)

AUTHOR_02 = os.path.join(
    os.path.expanduser("~/scratch/LOGO/05_LOGO_Variant_Prioritization/1. script/05_LOGO_CADD"),
    "02_cadd_classification_transformer_tfrecord.py"
)

LOCAL_BASE = os.path.expanduser(
    "~/scratch/LOGO/05_LOGO_Variant_Prioritization/my_repro/CADD_work/local_tfrecord_fold0_512_full"
)
WEIGHT = os.path.expanduser(
    "~/scratch/LOGO/05_LOGO_Variant_Prioritization/my_repro/CADD_work/full_train_512_run/weights_epoch02.h5"
)

VALID_TXT = os.path.expanduser(
    "~/scratch/LOGO/05_LOGO_Variant_Prioritization/2. data/5. CADD_indel_Clinvar + 1000G/clinvar_20201003.hg19_multianno.txt.noncoding_fold0_valid.indel"
)
TEST_TXT = os.path.expanduser(
    "~/scratch/LOGO/05_LOGO_Variant_Prioritization/2. data/5. CADD_indel_Clinvar + 1000G/clinvar_20201003.hg19_multianno.txt.noncoding_fold0_test.indel"
)

SEQ_LEN = 512
BATCH_SIZE = 128
VALID_STEPS = 23
TEST_STEPS = 23

spec = importlib.util.spec_from_file_location("cadd02", AUTHOR_02)
cadd02 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cadd02)

def load_labels(txt_path):
    y = []
    with open(txt_path) as f:
        for line in f:
            name = line.rstrip("\n").split("\t")[2]
            if name.startswith("Benign"):
                y.append(0)
            elif name.startswith("Pathogenic"):
                y.append(1)
            else:
                raise ValueError(f"Unknown label name: {name}")
    return np.array(y, dtype=np.int64)

def single_file_dataset_fixed(input_file, seq_len=512):
    d = tf.data.TFRecordDataset(input_file)

    def single_example_parser(serialized_example):
        name_to_features = {
            'label': tf.io.FixedLenFeature([], tf.int64),
            'seq': tf.io.FixedLenFeature([seq_len], tf.int64),
            'alt_seq': tf.io.FixedLenFeature([seq_len], tf.int64),
            'alt_type': tf.io.FixedLenFeature([seq_len], tf.int64),
        }
        example = tf.io.parse_single_example(serialized_example, name_to_features)
        ref_seq = example['seq']
        alt_seq = example['alt_seq']
        alt_type = example['alt_type']
        return ref_seq, alt_seq, alt_type

    return d.map(single_example_parser, num_parallel_calls=tf.data.AUTOTUNE)

def create_pred_dataset_fixed(file_names, batch_size, seq_len=512):
    if isinstance(file_names, str):
        file_names = [file_names]

    ds_list = [single_file_dataset_fixed(f, seq_len=seq_len) for f in file_names]
    dataset = ds_list[0]
    for ds in ds_list[1:]:
        dataset = dataset.concatenate(ds)

    def _select_x(ref_seq, alt_seq, alt_type):
        return {
            'Input-Token-ALT': ref_seq,
            'Input-Token-Alt-ALT': alt_seq,
            'Input-Segment-ALT': alt_type,
        }

    dataset = dataset.map(_select_x, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

word_index_from = 10
word_dict = cadd02.get_word_dict_for_n_gram_alphabet(n_gram=3, word_index_from=word_index_from)
vocab_size = len(word_dict) + word_index_from

model = cadd02.get_model(
    num_classes=1,
    embedding_dims=128,
    hidden_layers=1,
    vocab_size=vocab_size,
    activation='sigmoid'
)
model.load_weights(WEIGHT)

valid_file = os.path.join(LOCAL_BASE, "clinvar_InDel_fold0_valid_local.tfrecord")
test_file = os.path.join(LOCAL_BASE, "clinvar_InDel_fold0_test_local.tfrecord")

valid_y = load_labels(VALID_TXT)
test_y = load_labels(TEST_TXT)

valid_ds = create_pred_dataset_fixed([valid_file], batch_size=BATCH_SIZE, seq_len=SEQ_LEN)
test_ds = create_pred_dataset_fixed([test_file], batch_size=BATCH_SIZE, seq_len=SEQ_LEN)

valid_pred = model.predict(valid_ds, steps=VALID_STEPS).reshape(-1)[:len(valid_y)]
test_pred = model.predict(test_ds, steps=TEST_STEPS).reshape(-1)[:len(test_y)]

for tag, y_true, y_pred in [
    ("valid", valid_y, valid_pred),
    ("test", test_y, test_pred),
]:
    print(f"\n===== {tag.upper()} =====")
    print("n =", len(y_true))
    print("label_mean =", y_true.mean())
    print("pred_mean  =", y_pred.mean())
    print("pred_min   =", y_pred.min())
    print("pred_max   =", y_pred.max())
    print("pred>0.5   =", (y_pred > 0.5).mean())
    print("pred>0.1   =", (y_pred > 0.1).mean())
    print("pred>0.9   =", (y_pred > 0.9).mean())
