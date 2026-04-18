import os
import sys
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

print("AUTHOR_02:", AUTHOR_02)
print("train_files:", train_files)
print("valid_files:", valid_files)

def single_file_dataset_fixed(input_file, seq_len=998):
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

    d = d.map(single_example_parser)
    return d

def create_classifier_dataset_fixed(file_names, batch_size, seq_len=998, is_training=True, shuffle_size=1000):
    dataset = single_file_dataset_fixed(file_names, seq_len=seq_len)

    def _select_data_from_record(ref_seq, alt_seq, alt_type, label):
        x = {
            'Input-Token-ALT': ref_seq,
            'Input-Token-Alt-ALT': alt_seq,
            'Input-Segment-ALT': alt_type,
        }
        y = label
        return x, y

    dataset = dataset.map(_select_data_from_record, num_parallel_calls=16)
    if is_training:
        dataset = dataset.shuffle(shuffle_size)
    dataset = dataset.batch(batch_size)
    return dataset

word_index_from = 10
word_dict = cadd02.get_word_dict_for_n_gram_alphabet(n_gram=3, word_index_from=word_index_from)
vocab_size = len(word_dict) + word_index_from
print("vocab_size =", vocab_size)

train_dataset = create_classifier_dataset_fixed(
    train_files, batch_size=8, seq_len=SEQ_LEN, is_training=True, shuffle_size=1000
).take(1)

valid_dataset = create_classifier_dataset_fixed(
    valid_files, batch_size=8, seq_len=SEQ_LEN, is_training=False, shuffle_size=1
).take(1)

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

print("=== start fixed smoke fit ===")
history = model.fit(
    train_dataset,
    validation_data=valid_dataset,
    epochs=1,
    steps_per_epoch=1,
    validation_steps=1,
    verbose=1
)

print("=== fixed smoke fit finished ===")
print(history.history)
