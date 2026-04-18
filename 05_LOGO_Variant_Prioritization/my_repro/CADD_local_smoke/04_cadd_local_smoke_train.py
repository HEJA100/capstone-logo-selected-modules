import os
import sys
import importlib.util
import tensorflow as tf

# 让作者原始 02 脚本能找到 repo 里的 bgi 包
REPO_ROOT = os.path.expanduser("~/scratch/LOGO")
sys.path.insert(0, REPO_ROOT)

AUTHOR_02 = os.path.join(
    os.path.expanduser("~/scratch/LOGO/05_LOGO_Variant_Prioritization/1. script/05_LOGO_CADD"),
    "02_cadd_classification_transformer_tfrecord.py"
)

LOCAL_BASE = os.path.expanduser(
    "~/scratch/LOGO/05_LOGO_Variant_Prioritization/my_repro/CADD_work/local_tfrecord_fold0"
)

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

print("AUTHOR_02:", AUTHOR_02)
print("train_files:", train_files)
print("valid_files:", valid_files)
print("test_files :", test_files)

# 作者原始 02 的 3-gram vocab 逻辑
word_index_from = 10
word_dict = cadd02.get_word_dict_for_n_gram_alphabet(n_gram=3, word_index_from=word_index_from)
vocab_size = len(word_dict) + word_index_from
print("vocab_size =", vocab_size)

# 数据集：尽量沿用作者原始函数
train_dataset = cadd02.create_classifier_dataset(
    train_files,
    batch_size=8,
    is_training=True,
    epochs=1,
    shuffle_size=1000
).take(1)

valid_dataset = cadd02.create_classifier_dataset(
    valid_files,
    batch_size=8,
    is_training=False,
    epochs=1,
    shuffle_size=1
).take(1)

# 模型：沿用作者原始 02 的 get_model
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

print("=== start smoke fit ===")
history = model.fit(
    train_dataset,
    validation_data=valid_dataset,
    epochs=1,
    steps_per_epoch=1,
    validation_steps=1,
    verbose=1
)

print("=== smoke fit finished ===")
print(history.history)
