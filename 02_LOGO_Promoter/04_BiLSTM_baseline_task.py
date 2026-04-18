import os
import sys
import csv
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

if len(sys.argv) < 2:
    raise SystemExit("Usage: python 04_BiLSTM_baseline_task.py <TASK>, e.g. BOTH / TATA_BOX / NO_TATA_BOX")

TASK = sys.argv[1]
BASE = "02_LOGO_Promoter"

DATA_FILE = os.path.join(BASE, "data/5_gram_11_knowledge", f"epdnew_{TASK}_Knowledge_5_gram.npz")
KFOLD_TMPL = os.path.join(BASE, "data", f"kfold_{{fold}}_train_and_valid_index_5_gram_epdnew_{TASK}_Knowledge_11_Knowledge.npz")

OUT_FOLD = os.path.join(BASE, f"bilstm_{TASK.lower()}_fold_results.csv")
OUT_SUMMARY = os.path.join(BASE, f"bilstm_{TASK.lower()}_summary.csv")

def expand_5gram_slices(seq, y, ngram=5):
    xs = []
    ys = []
    max_len = (seq.shape[1] // ngram) * ngram
    for ii in range(ngram):
        idxs = list(range(ii, max_len, ngram))
        xs.append(seq[:, idxs])
        ys.append(y)
    x_all = np.concatenate(xs, axis=0).astype("int32")
    y_all = np.concatenate(ys, axis=0).astype("int32")
    return x_all, y_all

def build_bilstm(vocab_size, input_len):
    inp = tf.keras.Input(shape=(input_len,), dtype="int32", name="tokens")
    x = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=128, name="embed")(inp)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64, return_sequences=True, dropout=0.2)
    )(x)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(32, dropout=0.2)
    )(x)
    x = tf.keras.layers.Dropout(0.30)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

loaded = np.load(DATA_FILE)
seq = loaded["sequence"]
y = loaded["label"]

x_all, y_all = expand_5gram_slices(seq, y, ngram=5)
vocab_size = int(x_all.max()) + 1
input_len = x_all.shape[1]

print("=== DIAGNOSTIC ===")
print("TASK:", TASK)
print("DATA_FILE:", DATA_FILE)
print("raw sequence shape:", seq.shape)
print("expanded x_all shape:", x_all.shape)
print("expanded y_all shape:", y_all.shape)
print("input_len:", input_len)
print("vocab_size:", vocab_size)

results = []

for fold in range(10):
    idx_file = KFOLD_TMPL.format(fold=fold)
    idx = np.load(idx_file)
    train_idx = idx["train"]
    test_idx = idx["test"]

    x_train = x_all[train_idx]
    y_train = y_all[train_idx]
    x_test = x_all[test_idx]
    y_test = y_all[test_idx]

    x_tr, x_val, y_tr, y_val = train_test_split(
        x_train, y_train,
        test_size=0.1,
        random_state=SEED,
        stratify=y_train
    )

    model = build_bilstm(vocab_size=vocab_size, input_len=input_len)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=2,
            restore_best_weights=True
        )
    ]

    model.fit(
        x_tr, y_tr,
        validation_data=(x_val, y_val),
        epochs=8,
        batch_size=256,
        verbose=2,
        callbacks=callbacks
    )

    prob = model.predict(x_test, batch_size=1024, verbose=0).ravel()
    pred = (prob >= 0.5).astype("int32")

    acc = accuracy_score(y_test, pred)
    prec = precision_score(y_test, pred, zero_division=0)
    rec = recall_score(y_test, pred, zero_division=0)
    f1 = f1_score(y_test, pred, zero_division=0)

    results.append([fold, acc, prec, rec, f1])
    print(f"fold={fold},acc={acc:.6f},precision={prec:.6f},recall={rec:.6f},f1={f1:.6f}")

with open(OUT_FOLD, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["fold", "acc", "precision", "recall", "f1"])
    w.writerows(results)

arr = np.array([r[1:] for r in results], dtype=float)
mean = arr.mean(axis=0)
std = arr.std(axis=0, ddof=0)

with open(OUT_SUMMARY, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow([
        "task", "model", "n_folds",
        "acc_mean", "precision_mean", "recall_mean", "f1_mean",
        "acc_sd", "precision_sd", "recall_sd", "f1_sd"
    ])
    w.writerow([
        TASK, "BiLSTM", len(results),
        f"{mean[0]:.6f}", f"{mean[1]:.6f}", f"{mean[2]:.6f}", f"{mean[3]:.6f}",
        f"{std[0]:.6f}", f"{std[1]:.6f}", f"{std[2]:.6f}", f"{std[3]:.6f}"
    ])

print("\n=== CSV_FOLD_RESULTS ===")
print("fold,acc,precision,recall,f1")
for row in results:
    print(",".join([str(row[0])] + [f"{v:.6f}" for v in row[1:]]))

print("\n=== CSV_SUMMARY ===")
print("task,model,n_folds,acc_mean,precision_mean,recall_mean,f1_mean,acc_sd,precision_sd,recall_sd,f1_sd")
print(
    f"{TASK},BiLSTM,{len(results)},"
    f"{mean[0]:.6f},{mean[1]:.6f},{mean[2]:.6f},{mean[3]:.6f},"
    f"{std[0]:.6f},{std[1]:.6f},{std[2]:.6f},{std[3]:.6f}"
)
