# -*- coding:utf-8 -*-
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Activation, Dropout, BatchNormalization, Reshape, Permute, concatenate, Lambda, Layer
import tensorflow.keras.backend as K
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics

sys.path.append("../")
from bgi.bert4keras.models import build_transformer_model
from bgi.common.callbacks import LRSchedulerPerStep
from bgi.common.refseq_utils import get_word_dict_for_n_gram_number

include_types = ['enhancer', 'promoter', 'pseudogene', 'insulator', 'conserved_region', 
                 'protein_binding_site', 'DNAseI_hypersensitive_site', 'nucleotide_cleavage_site', 
                 'silencer', 'gene', 'exon', 'CDS']

def load_npz_data_for_classification(file_name, ngram=3, only_one_slice=True, ngram_index=None, masked=True):
    x_data_all = []
    anno_data_all = []
    y_data_all = []
    if str(file_name).endswith('.npz') is False or os.path.exists(file_name) is False:
        return x_data_all, None, y_data_all

    loaded = np.load(file_name)
    x_data = loaded['x']
    x_annotation = loaded['annotation']
    y_data = loaded['y']

    print("Load: ", file_name)
    print("X: ", x_data.shape)
    print("Annotation: ", x_annotation.shape)
    print("Y: ", y_data.shape)
    if only_one_slice is True:
        for ii in range(ngram):
            if ngram_index is not None and ii != ngram_index:
                continue
            kk = ii
            slice_indexes = []
            max_slice_seq_len = x_data.shape[1] // ngram * ngram
            for gg in range(kk, max_slice_seq_len, ngram):
                slice_indexes.append(gg)
            x_data_slice = x_data[:, slice_indexes]
            x_data_all.append(x_data_slice)
            x_annotation_slice = x_annotation[:, :, slice_indexes]
            anno_data_all.append(x_annotation_slice)
            y_data_all.append(y_data)
    else:
        x_data_all.append(x_data)
        anno_data_all.append(x_annotation)
        y_data_all.append(y_data)

    return x_data_all, anno_data_all, y_data_all

def load_all_data(record_names: list, ngram=3, only_one_slice=True, ngram_index=None, masked=False):
    x_data_all = []
    x_annotation_all = []
    y_data_all = []
    for file_name in record_names:
        x_data, anno_data, y_data = load_npz_data_for_classification(file_name, ngram, only_one_slice, ngram_index, masked=masked)
        x_data_all.extend(x_data)
        x_annotation_all.extend(anno_data)
        y_data_all.extend(y_data)

    x_data_all = np.concatenate(x_data_all)
    x_annotation_all = np.concatenate(x_annotation_all)
    y_data_all = np.concatenate(y_data_all)
    return x_data_all, x_annotation_all, y_data_all

def load_npz_dataset_for_classification(x_enhancer_data_all, x_enhancer_annotation_all, x_promoter_data_all, x_promoter_annotation_all,
                                        y_data_all, enhancer_seq_len, promoter_seq_len, annotation_size,
                                        ngram=5, only_one_slice=True, ngram_index=None, shuffle=False, seq_len=200, num_classes=1, masked=True):
    if num_classes == 1:
        y_data_all = np.reshape(y_data_all, (y_data_all.shape[0], 1))

    def data_generator():
        total_size = len(x_enhancer_data_all)
        indexes = np.arange(total_size)
        if shuffle is True:
            np.random.shuffle(indexes)

        ii = 0
        while True:
            if ii < total_size:
                index = indexes[ii]
            else:
                total_size = len(x_enhancer_data_all)
                indexes = np.arange(total_size)
                if shuffle is True:
                    np.random.shuffle(indexes)
                ii = 0
                index = indexes[ii]

            x_enhancer = x_enhancer_data_all[index]
            x_enhancer_annotation = x_enhancer_annotation_all[index]
            x_promoter = x_promoter_data_all[index]
            x_promoter_annotation = x_promoter_annotation_all[index]
            segment_promoter = np.zeros_like(x_promoter)
            segment_enhancer = np.zeros_like(x_enhancer)
            y = y_data_all[index]
            ii += 1
            yield x_enhancer, x_enhancer_annotation, segment_enhancer, x_promoter, x_promoter_annotation, segment_promoter, y

    classes_shape = tf.TensorShape([num_classes]) if num_classes != 1 else tf.TensorShape([1])
    dataset = tf.data.Dataset.from_generator(data_generator,
                                             output_types=(tf.int16, tf.int16,  tf.int16,  tf.int16, tf.int16, tf.int16, tf.int32),
                                             output_shapes=(
                                                 tf.TensorShape([enhancer_seq_len]),
                                                 tf.TensorShape([annotation_size, enhancer_seq_len]),
                                                 tf.TensorShape([enhancer_seq_len]),
                                                 tf.TensorShape([promoter_seq_len]),
                                                 tf.TensorShape([annotation_size, promoter_seq_len]),
                                                 tf.TensorShape([promoter_seq_len]),
                                                 classes_shape
                                             ))
    return dataset

def parse_function(x_enhancer, annotation_enhancer, segment_enhancer, x_promoter, annotation_promoter, segment_x, y):
    x = {
        'Input-Token_Enhancer': x_enhancer,
        'Input-Segment_Enhancer': segment_enhancer,
        'Input-Token_Promoter': x_promoter,
        'Input-Segment_Promoter': segment_x,
    }
    exclude_annotation = []
    index = 0
    for ii in range(annotation_enhancer.shape[1]):
        if ii < annotation_enhancer.shape[1]: 
            gene_type = K.zeros_like(annotation_enhancer[:, ii, :], dtype='int16') if ii in exclude_annotation else annotation_enhancer[:, ii, :]
            x['Input-Token-Type_Enhancer_{}'.format(index)] = gene_type
            index += 1

    index = 0
    for ii in range(annotation_promoter.shape[1]):
        if ii < annotation_promoter.shape[1]:
            gene_type = K.zeros_like(annotation_promoter[:, ii, :], dtype='int16') if ii in exclude_annotation else annotation_promoter[:, ii, :]
            x['Input-Token-Type_Promoter_{}'.format(index)] = gene_type
            index += 1

    y = {'CLS-Activation': y}
    return x, y

def f1_score(y_true, y_pred):
    y_true = tf.cast(y_true, 'float')
    y_pred = tf.cast(y_pred, 'float')
    TP = K.sum(tf.cast(K.equal(y_true, 1) & K.equal(K.round(y_pred), 1), 'float'))
    FP = K.sum(tf.cast(K.equal(y_true, 0) & K.equal(K.round(y_pred), 1), 'float'))
    FN = K.sum(tf.cast(K.equal(y_true, 1) & K.equal(K.round(y_pred), 0), 'float'))
    P = TP / (TP + FP + K.epsilon())
    R = TP / (TP + FN + K.epsilon())
    F1 = 2 * P * R / (P + R + K.epsilon())
    return F1

def average_precision(y_true, y_pred):
    y_true = tf.cast(y_true, 'float')
    y_pred = tf.cast(y_pred, 'float')
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def model_def(embedding_size=128, hidden_size=128, num_heads=8, num_hidden_layers=1, vocab_size=10000, drop_rate=0.25, annotation_size=11):
    multi_inputs = [2] * annotation_size
    config = {
        "attention_probs_dropout_prob": 0, "hidden_act": "gelu", "hidden_dropout_prob": 0,
        "embedding_size": embedding_size, "hidden_size": hidden_size, "initializer_range": 0.02,
        "intermediate_size": 512, "max_position_embeddings": 1024 * 2, "num_attention_heads": num_heads,
        "num_hidden_layers": num_hidden_layers, "num_hidden_groups": 1, "net_structure_type": 0,
        "gap_size": 0, "num_memory_blocks": 0, "inner_group_num": 1, "down_scale_factor": 1,
        "type_vocab_size": 0, "vocab_size": vocab_size, "custom_masked_sequence": False,
        "custom_conv_layer": True, "use_segment_ids": True, "use_position_ids": True, "multi_inputs": multi_inputs
    }
    bert_enhancer = build_transformer_model(configs=config, model='multi_inputs_bert', return_keras_model=False)
    bert_promoter = build_transformer_model(configs=config, model='multi_inputs_bert', return_keras_model=False)

    x_promoter = tf.keras.layers.Input(shape=(None,), name='Input-Token_Promoter')
    s_promoter = tf.keras.layers.Input(shape=(None,), name='Input-Segment_Promoter')
    x_enhancer = tf.keras.layers.Input(shape=(None,), name='Input-Token_Enhancer')
    s_segment = tf.keras.layers.Input(shape=(None,), name='Input-Segment_Enhancer')

    inputs = [x_promoter, s_promoter, x_enhancer, s_segment]
    promoter_inputs = [x_promoter, s_promoter]
    enhancer_inputs = [x_enhancer, s_segment]

    for ii in range(annotation_size):
        name = 'Input-Token-Type_Enhancer_{}'.format(ii)
        input = tf.keras.layers.Input(shape=(None,), name=name)
        enhancer_inputs.append(input)
        inputs.append(input)

    for ii in range(annotation_size):
        name = 'Input-Token-Type_Promoter_{}'.format(ii)
        input = tf.keras.layers.Input(shape=(None,), name=name)
        promoter_inputs.append(input)
        inputs.append(input)

    bert_enhancer.set_inputs(enhancer_inputs)
    bert_promoter.set_inputs(promoter_inputs)
    enhancer_output = bert_enhancer.model(enhancer_inputs)
    promoter_output = bert_promoter.model(promoter_inputs)

    promoter_output = Lambda(lambda x: x[:, 0])(promoter_output)
    enhancer_output = Lambda(lambda x: x[:, 0])(enhancer_output)

    x = concatenate([promoter_output, enhancer_output])
    x = BatchNormalization()(x)
    x = Dropout(drop_rate)(x)
    output = Dense(1, activation='sigmoid', name='CLS-Activation')(x)
    model = tf.keras.models.Model(inputs, output)
    return model

def train_kfold(CELL, TYPE, batch_size=256, annotation_size=11, epochs=10, ngram=6, vocab_size=10000, ENHANCER_RESIZED_LEN=2000, PROMOTER_RESIZED_LEN=1000):
    num_gpu = 1
    strategy = tf.distribute.MirroredStrategy()
    if strategy.num_replicas_in_sync >= 1:
        num_gpu = strategy.num_replicas_in_sync
    GLOBAL_BATCH_SIZE = batch_size * num_gpu
    num_parallel_calls = 16

    data_path = CELL + '/' + TYPE + '/' + '{}_gram'.format(ngram)
    train_enhancer_files = [data_path + '/enhancer_Seq_{}_gram_knowledge.npz'.format(ngram)]
    train_promoter_files = [data_path + '/promoter_Seq_{}_gram_knowledge.npz'.format(ngram)]

    region1_seq, annotation_1, label = load_all_data(train_enhancer_files, ngram=ngram, only_one_slice=True, ngram_index=1)
    region2_seq, annotation_2, _ = load_all_data(train_promoter_files, ngram=ngram, only_one_slice=True, ngram_index=1)

    seed = 7
    np.random.seed(seed)
    X = np.hstack([region1_seq, region2_seq])
    Annotation = np.concatenate([annotation_1, annotation_2], axis=-1)
    Y = label

    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    enhancer_seq_len = ENHANCER_RESIZED_LEN // ngram // ngram * ngram
    promoter_seq_len = PROMOTER_RESIZED_LEN // ngram // ngram * ngram

    k_fold = 0
    for train, test in kfold.split(X, Y):
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=3)

        x_train_data, x_train_annotation, y_train_data = X[train], Annotation[train], Y[train]
        x_valid_data, x_valid_annotation, y_valid_data = X[test], Annotation[test], Y[test]

        with strategy.scope():
            model = model_def(vocab_size=vocab_size, annotation_size=annotation_size)
            model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.00001), metrics=['acc', f1_score, tf.keras.metrics.AUC()])

        filename = CELL + '/' + TYPE + '/best_knowledge_model_{}.h5'.format(str(k_fold)) # ⚠️改了名字防止覆盖04跑出来的纯序列权重
        modelCheckpoint = ModelCheckpoint(filename, monitor='val_acc', save_best_only=True, verbose=1)

        region1_seq = x_train_data[:, 0:enhancer_seq_len]
        region1_seq_annotation = x_train_annotation[:, :, 0:enhancer_seq_len]
        region2_seq = x_train_data[:, enhancer_seq_len:]
        region2_seq_annotation = x_train_annotation[:, :, enhancer_seq_len:]

        region1_seq_valid = x_valid_data[:, 0:enhancer_seq_len]
        region1_seq_annotation_valid = x_valid_annotation[:, :, 0:enhancer_seq_len]
        region2_seq_valid = x_valid_data[:, enhancer_seq_len:]
        region2_seq_annotation_valid = x_valid_annotation[:, :, enhancer_seq_len:]

        train_dataset = load_npz_dataset_for_classification(
            region1_seq, region1_seq_annotation, region2_seq, region2_seq_annotation, y_train_data,
            enhancer_seq_len, promoter_seq_len, ngram=ngram, only_one_slice=True, ngram_index=1,
            shuffle=True, seq_len=0, masked=False, annotation_size=annotation_size)
        train_dataset = train_dataset.batch(GLOBAL_BATCH_SIZE).map(map_func=parse_function, num_parallel_calls=num_parallel_calls).prefetch(tf.data.experimental.AUTOTUNE)

        valid_dataset = load_npz_dataset_for_classification(
            region1_seq_valid, region1_seq_annotation_valid, region2_seq_valid, region2_seq_annotation_valid, y_valid_data,
            enhancer_seq_len, promoter_seq_len, ngram=ngram, only_one_slice=True, ngram_index=1,
            shuffle=False, seq_len=0, num_classes=1, masked=False, annotation_size=annotation_size)
        valid_dataset = valid_dataset.batch(GLOBAL_BATCH_SIZE).map(map_func=parse_function, num_parallel_calls=num_parallel_calls).prefetch(tf.data.experimental.AUTOTUNE)

        train_steps_per_epoch = len(y_train_data) // GLOBAL_BATCH_SIZE
        valid_steps_per_epoch = len(y_valid_data) // GLOBAL_BATCH_SIZE

        model_train_history = model.fit(train_dataset, steps_per_epoch=train_steps_per_epoch, epochs=epochs,
                                        validation_data=valid_dataset, validation_steps=valid_steps_per_epoch,
                                        callbacks=[modelCheckpoint, early_stopping], verbose=2)
        
        k_fold += 1
        
        # 🌟 核心防泄漏魔法：清理计算图
        tf.keras.backend.clear_session()
        print(f"Fold {k_fold-1} finished and session cleared.")

# 🌟 引入动态阈值寻优技术
def optimized_bagging_predict(label, bag_score):
    vote_score = np.mean(bag_score, axis=0)
    auprc = metrics.average_precision_score(label, vote_score)
    best_f1, best_t = 0, 0.5
    for t in np.arange(0.01, 1.0, 0.01):
        temp_pred = (vote_score > t).astype(int)
        temp_f1 = metrics.f1_score(label, temp_pred)
        if temp_f1 > best_f1:
            best_f1 = temp_f1
            best_t = t
    print(f"🎯 Threshold Optimized! Best Cutoff is: {best_t:.2f}")
    return best_f1, auprc

# 🌟 被作者遗忘的 evaluate 算分函数（已被我们重构补齐）
def evaluate(CELL, TYPE, NUM_ENSEMBL=10, ngram=6, batch_size=128, annotation_size=11, num_parallel_calls=16, ENHANCER_RESIZED_LEN=2000, PROMOTER_RESIZED_LEN=1000):
    data_path = CELL + '/' + TYPE + '/test/' + '{}_gram'.format(ngram)
    test_enhancer_files = [data_path + '/enhancer_Seq_{}_gram_knowledge.npz'.format(ngram)]
    test_promoter_files = [data_path + '/promoter_Seq_{}_gram_knowledge.npz'.format(ngram)]

    region1_seq, annotation_1, label = load_all_data(test_enhancer_files, ngram=ngram, only_one_slice=True, ngram_index=1)
    region2_seq, annotation_2, _ = load_all_data(test_promoter_files, ngram=ngram, only_one_slice=True, ngram_index=1)

    bag_score = np.zeros((NUM_ENSEMBL, label.shape[0]))
    enhancer_seq_len = ENHANCER_RESIZED_LEN // ngram // ngram * ngram
    promoter_seq_len = PROMOTER_RESIZED_LEN // ngram // ngram * ngram

    for t in range(NUM_ENSEMBL):
        model = model_def(vocab_size=vocab_size, annotation_size=annotation_size)
        model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.00001), metrics=['acc', f1_score, tf.keras.metrics.AUC()])

        weight_path = CELL + '/' + TYPE + '/best_knowledge_model_' + str(t) + '.h5'
        if os.path.exists(weight_path):
            model.load_weights(weight_path)
        else:
            print(f"Warning: Model weight {weight_path} not found!")
            continue

        valid_dataset = load_npz_dataset_for_classification(
            region1_seq, annotation_1, region2_seq, annotation_2, label,
            enhancer_seq_len, promoter_seq_len, ngram=ngram, only_one_slice=True, ngram_index=1,
            shuffle=False, seq_len=0, num_classes=1, masked=False, annotation_size=annotation_size)
        valid_dataset = valid_dataset.batch(batch_size).map(map_func=parse_function, num_parallel_calls=num_parallel_calls).prefetch(tf.data.experimental.AUTOTUNE)

        steps_per_epoch = len(label) // batch_size + 1
        score = model.predict(valid_dataset, steps=steps_per_epoch)
        bag_score[t, :] = score[:len(label)].reshape(-1)
        
        tf.keras.backend.clear_session()

    f1, auprc = optimized_bagging_predict(label, bag_score)
    print("--------------------------------------------------")
    print(f"👑 FINAL RESULTS for {CELL} (Knowledge Model):")
    print(f"👑 AUPRC: {auprc:.4f} | MAX F1: {f1:.4f}")
    print("--------------------------------------------------")

if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)

    ngram = 6
    word_dict = get_word_dict_for_n_gram_number(n_gram=ngram)
    vocab_size = len(word_dict) + 10
    annotation_size = 11

    # 🌟 支持命令行传入特定的细胞系，避免一次性跑爆 16 个小时
    if len(sys.argv) > 1:
        CELLs = [sys.argv[1]]
    else:
        CELLs = ['tB', 'FoeT', 'Mon', 'nCD4', 'tCD4', 'tCD8']
        
    TYPE = 'P-E'
    for CELL in CELLs:
        print(f"\n=================================================")
        print(f"🚀 [Knowledge Model] Starting Training for {CELL}")
        print(f"=================================================")
        train_kfold(CELL, TYPE, batch_size=128, epochs=10, vocab_size=vocab_size, annotation_size=annotation_size)
        
        print(f"\n=================================================")
        print(f"💯 [Knowledge Model] Skip evaluation for now (no test knowledge yet) for {CELL}")
        print(f"=================================================")
        # evaluate(CELL, TYPE, NUM_ENSEMBL=10, ngram=6, batch_size=128, annotation_size=annotation_size)

    # 🌟 终极防死锁魔法：强制断电
    print("Forcing exit to prevent TensorFlow deadlock.")
    sys.stdout.flush()
    os._exit(0)