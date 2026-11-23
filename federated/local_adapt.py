"""
federated/local_adapt.py
Local adaptation node: teach the global model what NORMAL traffic looks like
on this deployment network, without the traffic leaving the machine.

    python -m federated.local_adapt --train federated/data/local_train.npz \\
                                    --test  federated/data/local_test.npz  [--rounds 10]

Why: live testing showed false alarms on modern benign traffic the 2017 lab
never contained (an AI-assistant streaming API, DNS, Windows update sharing).
The federated design already has the remedy: the deployment network joins as
one more node.

What the node does (one more bagging node, run in-process):
  * its data = its OWN captured normal traffic (network/capture_local.py, never
    shared) + PUBLIC labelled CIC-IDS2017 training flows (attacks and benign
    from the node partitions; never the server hold-out);
  * it grows new XGBoost trees on top of the current global booster (3 parallel
    trees per round, same parameters and per-node class weight as the other
    nodes); the server appends them, exactly as in FedXgbBagging;
  * the server then recalibrates the cut-off and Platt scaling on its own
    calibration half (federated/calibrate.py).
The autoencoder is unchanged (its live alarms are handled by learning mode).

Acceptance is measured, not assumed: the report half of the hold-out must keep
its accuracy and per-family detection, and a SEPARATE later capture
(--test, never used for training) shows the live false-alarm change.
Writes the candidate to hybrid_model_local/ and federated/logs/local_adaptation.json.
"""
import argparse
import json
import os
import shutil

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import joblib
import numpy as np
import xgboost as xgb

from federated import config as C
from federated.calibrate import calibrate
from federated.data_utils import load_node, load_test
from federated.evaluate import metrics, per_family_recall
from federated.models import reconstruction_error

OUT_DIR = os.path.join(C.ROOT_DIR, "hybrid_model_local")
SRC_DIR = C.HYBRID_DIR
PUBLIC_PER_FAMILY = 20_000
PUBLIC_BENIGN = 60_000
LOCAL_WEIGHT_SHARE = 0.20      # own traffic carries 20% of the node's training weight


def _scale(scaler, X_raw):
    return np.clip(scaler.transform(np.asarray(X_raw, dtype=np.float64)), 0.0, 1.0).astype(np.float32)


def public_sample(seed=0):
    """Stratified sample of the public CIC-IDS2017 training partitions (no hold-out rows)."""
    rng = np.random.default_rng(seed)
    Xs, ys, fs = [], [], []
    for n in range(C.NUM_NODES):
        X, y, f = load_node(n)
        Xs.append(X); ys.append(y); fs.append(f)
    X, y, f = np.concatenate(Xs), np.concatenate(ys), np.concatenate(fs)
    keep = []
    for fid in np.unique(f):
        idx = np.flatnonzero(f == fid)
        cap = PUBLIC_BENIGN if fid == 0 else PUBLIC_PER_FAMILY
        keep.append(idx if len(idx) <= cap else rng.choice(idx, cap, replace=False))
    k = np.concatenate(keep)
    return X[k], y[k].astype(np.float32), f[k]


def hybrid_eval(model_dir, X, y, fam):
    from tensorflow.keras.models import load_model
    from core.decision import decide, load_thresholds, calibrated_prob
    cfg = load_thresholds(os.path.join(model_dir, "threshold.json"))
    b = xgb.Booster(); b.load_model(os.path.join(model_dir, "xgb_model.json"))
    p = b.predict(xgb.DMatrix(X))
    mse = reconstruction_error(load_model(os.path.join(model_dir, "autoencoder.keras")), X)
    _, fin, att = decide(mse, p, cfg)
    clf = att & (calibrated_prob(p, cfg) > 0.5)
    return att, clf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--rounds", type=int, default=10)
    a = ap.parse_args()

    scaler = joblib.load(os.path.join(C.DATA_DIR, "scaler.pkl"))
    loc = np.load(a.train)
    X_loc = _scale(scaler, loc["X"])
    Xp, yp, fp = public_sample()
    X_node = np.concatenate([X_loc, Xp]); y_node = np.concatenate([np.zeros(len(X_loc), np.float32), yp])
    print(f"local node data: {len(X_loc):,} own normal snapshots + {len(Xp):,} public flows "
          f"({int(yp.sum()):,} attacks)", flush=True)

    # --- node side: grow trees on top of the global booster (one bagging node) ---
    params = dict(C.XGB_PARAMS)
    params["scale_pos_weight"] = float(len(Xp) - yp.sum()) / max(float(yp.sum()), 1.0)   # public benign : attacks
    base = xgb.Booster(); base.load_model(os.path.join(SRC_DIR, "xgb_model.json"))
    n_before = len(base.get_dump())
    # a few hundred own snapshots would vanish next to ~140k public flows: give them a fixed share of the weight
    w_local = LOCAL_WEIGHT_SHARE * len(Xp) / ((1 - LOCAL_WEIGHT_SHARE) * max(len(X_loc), 1))
    w = np.concatenate([np.full(len(X_loc), w_local, np.float32), np.ones(len(Xp), np.float32)])
    bst = xgb.train(params, xgb.DMatrix(X_node, label=y_node, weight=w), num_boost_round=a.rounds, xgb_model=base)
    print(f"trees: {n_before} -> {len(bst.get_dump())}", flush=True)

    # --- server side: install candidate, recalibrate on the calibration half ---
    os.makedirs(OUT_DIR, exist_ok=True)
    for f in ("autoencoder.keras", "scaler.pkl", "selected_columns.pkl", "family_prototypes.json"):
        shutil.copy2(os.path.join(SRC_DIR, f), OUT_DIR)
    bst.save_model(os.path.join(OUT_DIR, "xgb_model.json"))
    calibrate(OUT_DIR, verbose=True)

    # --- acceptance: report half (dataset) and separate live capture ---
    Xr, yr, fr = load_test("report")
    res = {"rounds": a.rounds, "trees_before": n_before, "trees_after": len(bst.get_dump()),
           "local_weight_share": LOCAL_WEIGHT_SHARE,
           "local_train_snapshots": int(len(X_loc)), "public_flows": int(len(Xp))}
    for name, d in (("before", SRC_DIR), ("after", OUT_DIR)):
        att, _ = hybrid_eval(d, Xr, yr, fr)
        res[f"report_half_{name}"] = {**metrics(yr, att), "per_family": per_family_recall(fr, att)}
    tst = np.load(a.test)
    Xt = _scale(scaler, tst["X"])
    for name, d in (("before", SRC_DIR), ("after", OUT_DIR)):
        att, clf = hybrid_eval(d, Xt, np.zeros(len(Xt)), np.zeros(len(Xt)))
        res[f"live_test_{name}"] = {"snapshots": int(len(Xt)), "flagged": int(att.sum()),
                                    "flagged_pct": round(float(att.mean()) * 100, 2),
                                    "classifier_flags": int(clf.sum()),
                                    "classifier_flags_pct": round(float(clf.mean()) * 100, 2)}
    os.makedirs(C.LOG_DIR, exist_ok=True)
    with open(os.path.join(C.LOG_DIR, "local_adaptation.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(json.dumps({k: v for k, v in res.items() if not k.startswith("report_half")}, indent=1))
    for k in ("report_half_before", "report_half_after"):
        m = res[k]
        print(k, {x: m[x] for x in ("accuracy", "precision", "recall", "f1", "fpr")}, m["per_family"])


if __name__ == "__main__":
    main()
