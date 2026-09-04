"""
federated/lstm_experiment.py
Temporal model experiment: does looking at a SEQUENCE of consecutive flows
help detect multi-stage / slow attacks that a single flow does not reveal?

    python -m federated.lstm_experiment            (~5-10 min on CPU)

Design
  * Flows are taken in the order they appear in each CIC-IDS2017 CSV, which is
    chronological within a capture day, and cut into non-overlapping windows of
    SEQ_LEN flows.  A window is labelled ATTACK if any flow in it is an attack.
  * Model: masked LSTM(64) over the 30 scaled features per step -> Dense(1).
  * Baseline for the same windows: "any flow in the window flagged by the
    federated hybrid model" (so the comparison is window-level vs window-level).
  * Chronological split per file: first 70% of windows train, last 30% test,
    so the model never sees the future.
Writes federated/logs/lstm_experiment.json.  Centralized experiment only; a
federated version is a straight FedAvg extension of client.py (same weights API).
"""
import glob
import json
import os
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import joblib
import numpy as np
import pandas as pd

from federated import config as C
from federated.prepare_data import _normalise_label

SEQ_LEN = 10
EPOCHS = 4
BATCH = 512
MAX_WINDOWS_PER_FILE = 25000        # keeps the experiment to a few minutes on CPU


def load_windows(scaler):
    Xs, ys, fams, split = [], [], [], []
    for f in sorted(glob.glob(os.path.join(C.RAW_DATASET_DIR, "*.csv"))):
        header = pd.read_csv(f, nrows=0, encoding_errors="replace").columns
        wanted = {c.strip(): c for c in header}
        df = pd.read_csv(f, usecols=[wanted[c] for c in C.SELECTED_COLUMNS] + [wanted["Label"]],
                         encoding_errors="replace")
        df.columns = df.columns.str.strip()
        feats = df[C.SELECTED_COLUMNS].astype("float64").replace([np.inf, -np.inf], np.nan)
        ok = ~feats.isna().any(axis=1)
        feats, lab = feats[ok].values, df.loc[ok, "Label"].map(_normalise_label).map(C.ATTACK_FAMILY)
        y = (lab.values != "BENIGN").astype(np.int8)
        fam = lab.map(C.FAMILY_IDS).values.astype(np.int8)
        n = (len(y) // SEQ_LEN) * SEQ_LEN
        X = np.clip(scaler.transform(feats[:n]), 0, 1).astype(np.float32).reshape(-1, SEQ_LEN, len(C.SELECTED_COLUMNS))
        yw = y[:n].reshape(-1, SEQ_LEN)
        fw = fam[:n].reshape(-1, SEQ_LEN)
        if len(X) > MAX_WINDOWS_PER_FILE:          # keep chronology: evenly spaced windows
            idx = np.linspace(0, len(X) - 1, MAX_WINDOWS_PER_FILE).astype(int)
            X, yw, fw = X[idx], yw[idx], fw[idx]
        cut = int(len(X) * 0.7)
        Xs.append(X); ys.append(yw); fams.append(fw)
        split.append(np.r_[np.zeros(cut, dtype=bool), np.ones(len(X) - cut, dtype=bool)])
        print(f"  {os.path.basename(f)[:50]:50s} windows={len(X):>6,} attack-windows={int(yw.max(1).sum()):>6,}")
    return (np.concatenate(Xs), np.concatenate(ys), np.concatenate(fams), np.concatenate(split))


def metrics(y, pred):
    y = y.astype(bool); pred = pred.astype(bool)
    tp = (pred & y).sum(); fp = (pred & ~y).sum(); fn = (~pred & y).sum(); tn = (~pred & ~y).sum()
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    return {"accuracy": round(float((tp + tn) / len(y)), 4), "precision": round(float(prec), 4),
            "recall": round(float(rec), 4), "f1": round(float(2 * prec * rec / max(prec + rec, 1e-9)), 4),
            "fpr": round(float(fp / max(fp + tn, 1)), 4)}


def main():
    from tensorflow.keras import layers, models
    scaler = joblib.load(os.path.join(C.DATA_DIR, "scaler.pkl"))
    print("Building flow windows (chronological)...")
    X, yw, fw, is_test = load_windows(scaler)
    y = yw.max(axis=1)                                    # window label
    Xtr, ytr, Xte, yte = X[~is_test], y[~is_test], X[is_test], y[is_test]
    print(f"train windows={len(ytr):,} ({ytr.mean():.1%} attack)  test windows={len(yte):,} ({yte.mean():.1%} attack)")

    model = models.Sequential([
        layers.Input(shape=(SEQ_LEN, X.shape[-1])),
        layers.LSTM(64, return_sequences=False),
        layers.Dropout(0.2),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy")
    pos_w = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
    t0 = time.time()
    model.fit(Xtr, ytr, epochs=EPOCHS, batch_size=BATCH, verbose=0, class_weight={0: 1.0, 1: pos_w})
    train_s = time.time() - t0
    p = model.predict(Xte, batch_size=4096, verbose=0).ravel()
    lstm_m = metrics(yte, p > 0.5)

    # Window-level baseline with the deployed federated hybrid model (flow-wise, any-hit)
    from core.predictor import predict_batch
    flat = Xte.reshape(-1, X.shape[-1])
    raw = scaler.inverse_transform(flat)
    hyb = predict_batch(pd.DataFrame(raw, columns=C.SELECTED_COLUMNS))
    flow_attack = (hyb["label"].values == "ATTACK").reshape(-1, SEQ_LEN)
    base_m = metrics(yte, flow_attack.any(axis=1))

    # Slow / multi-stage families: DoS slowloris & Slowhttptest are inside "DoS"; report per family
    fte = fw[is_test]
    per_family = {}
    for fid, name in C.ID_TO_FAMILY.items():
        if name == "BENIGN":
            continue
        m = (fte == fid).any(axis=1)
        if m.sum() >= 30:
            per_family[name] = {"windows": int(m.sum()),
                                "lstm_recall": round(float((p[m] > 0.5).mean()), 4),
                                "hybrid_anyflow_recall": round(float(flow_attack.any(axis=1)[m].mean()), 4)}

    out = {"seq_len": SEQ_LEN, "epochs": EPOCHS, "train_windows": int(len(ytr)), "test_windows": int(len(yte)),
           "train_seconds": round(train_s, 1), "lstm_window_level": lstm_m,
           "hybrid_anyflow_window_level": base_m, "per_family": per_family,
           "note": ("Chronological 70/30 split per capture day; window = 10 consecutive flows; "
                    "ATTACK if any flow in the window is an attack.")}
    os.makedirs(C.LOG_DIR, exist_ok=True)
    with open(os.path.join(C.LOG_DIR, "lstm_experiment.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    model.save(os.path.join(C.FL_MODEL_DIR, "lstm_sequence.keras"))
    print("\n| Detector (window level) | Accuracy | Precision | Recall | F1 | FPR |")
    print("|---|---|---|---|---|---|")
    for name, m in (("LSTM over 10-flow windows", lstm_m), ("Federated hybrid, any flow in window", base_m)):
        print(f"| {name} | {m['accuracy']} | {m['precision']} | {m['recall']} | {m['f1']} | {m['fpr']} |")
    print("per family:", json.dumps(per_family, indent=1))
    print("Written federated/logs/lstm_experiment.json")


if __name__ == "__main__":
    main()
