import os
import sys
import csv
import importlib.util
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

OUTDIR = os.path.expanduser(
    "~/scratch/LOGO/05_LOGO_Variant_Prioritization/my_repro/CADD_work/full_train_512_run"
)
os.makedirs(OUTDIR, exist_ok=True)

SEQ_LEN = 512

spec = importlib.util.spec_from_file_location("cadd02", AUTHOR_02)
cadd02 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cadd02)

train_files = [
    os.path.join(LOCAL_BASE, "humanDerived_InDels_fold0_small.tfrecord"),
    os.path.join(LOCAL_BASE, "simulation_InDels_fold0_small.tfrecord"),
]
valid_files = [
    os.path.join(LOCAL_BASE, "clinvar_InDel_fold0_valid_local.tfrecord"),
]
test_files = [
    os.path.join(LOCAL_BASE, "clinvar_InDel_fold0_test_local.tfrecord"),
]

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
        label = example['label']
        ref_seq = example['seq']
        alt_seq = example['alt_seq']
        alt_type = example['alt_type']
        return ref_seq, alt_seq, alt_type, label

    return d.map(single_example_parser, num_parallel_calls=tf.data.AUTOTUNE)

def create_classifier_dataset_fixed(file_names, batch_size, seq_len=512, is_training=True, shuffle_size=10000):
    if isinstance(file_names, str):
        file_names = [file_names]

    ds_list = [single_file_dataset_fixed(f, seq_len=seq_len) for f in file_names]
    dataset = ds_list[0]
    for ds in ds_list[1:]:
        dataset = dataset.concatenate(ds)

    def _select_data_from_record(ref_seq, alt_seq, alt_type, label):
        x = {
            'Input-Token-ALT': ref_seq,
            'Input-Token-Alt-ALT': alt_seq,
            'Input-Segment-ALT': alt_type,
        }
        y = label
        return x, y

    dataset = dataset.map(_select_data_from_record, num_parallel_calls=tf.data.AUTOTUNE)
    if is_training:
        dataset = dataset.shuffle(shuffle_size)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

word_index_from = 10
word_dict = cadd02.get_word_dict_for_n_gram_alphabet(n_gram=3, word_index_from=word_index_from)
vocab_size = len(word_dict) + word_index_from

train_dataset = create_classifier_dataset_fixed(
    train_files, batch_size=128, seq_len=SEQ_LEN, is_training=True, shuffle_size=10000
)
valid_dataset = create_classifier_dataset_fixed(
    valid_files, batch_size=128, seq_len=SEQ_LEN, is_training=False, shuffle_size=1
)
test_dataset = create_classifier_dataset_fixed(
    test_files, batch_size=128, seq_len=SEQ_LEN, is_training=False, shuffle_size=1
)

model = cadd02.get_model(
    num_classes=1,
    embedding_dims=128,
    hidden_layers=1,
    vocab_size=vocab_size,
    activation='sigmoid'
)

model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=cadd02.Adam(2e-6),
    metrics=['accuracy', tf.keras.metrics.AUC()]
)

ckpt = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(OUTDIR, "weights_epoch{epoch:02d}.h5"),
    monitor="val_auc",
    mode="max",
    save_best_only=False,
    save_weights_only=True,
    verbose=1
)

csv_logger = tf.keras.callbacks.CSVLogger(
    os.path.join(OUTDIR, "train_history.csv"), append=False
)

history = model.fit(
    train_dataset,
    validation_data=valid_dataset,
    epochs=2,
    steps_per_epoch=28711,
    validation_steps=23,
    callbacks=[ckpt, csv_logger],
    verbose=1
)

test_metrics = model.evaluate(test_dataset, steps=23, verbose=1)

with open(os.path.join(OUTDIR, "test_metrics.txt"), "w") as f:
    f.write(str(test_metrics) + "\n")

print("=== full training finished ===")
print(history.history)
print("test_metrics =", test_metrics)
