"""
federated/evaluate.py
Centralized vs Federated comparison on the server hold-out set.

    python -m federated.evaluate                # trains the centralized baseline too
    python -m federated.evaluate --no-central   # only evaluate saved models

Compares, on the same test.npz:
  * Phase 1 models (hybrid_model/, trained on the DDoS day only)
  * Centralized Phase 2 models (all node data pooled, same budget as federated)
  * Federated Phase 2 global models (hybrid_model_fl/)
for Autoencoder-only, XGBoost-only and the hybrid rule, plus recall per attack
family and the communication cost read from federated/logs/.
Writes federated/logs/comparison.json.
"""
import argparse
import json
import os
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import joblib
import numpy as np

from core.decision import decide, load_thresholds
from federated import config as C
from federated.data_utils import load_test, load_all_nodes
from federated.metrics_logger import read_log
from federated.models import build_autoencoder, reconstruction_error, anomaly_threshold


def metrics(y, pred):
    y = y.astype(bool); pred = pred.astype(bool)
    tp = (pred & y).sum(); fp = (pred & ~y).sum(); fn = (~pred & y).sum(); tn = (~pred & ~y).sum()
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    return {"accuracy": round(float((tp + tn) / len(y)), 4), "precision": round(float(prec), 4),
            "recall": round(float(rec), 4), "f1": round(float(2 * prec * rec / max(prec + rec, 1e-9)), 4),
            "fpr": round(float(fp / max(fp + tn, 1)), 4)}


def per_family_recall(fam, pred):
    out = {}
    for fid, name in C.ID_TO_FAMILY.items():
        m = fam == fid
        if m.any() and name != "BENIGN":
            out[name] = round(float(pred[m].mean()), 4)
    return out


def xgb_proba(model, X):
    """Works for a sklearn XGBClassifier or a raw Booster."""
    import xgboost as xgb
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.predict(xgb.DMatrix(X))


def evaluate_pair(name, ae, xgbm, thr_cfg, X, y, fam, results):
    mse = reconstruction_error(ae, X)
    ml = xgb_proba(xgbm, X)
    ae_pred = mse > thr_cfg["anomaly_multiplier"] * thr_cfg["ae_threshold"]
    ml_pred = ml > thr_cfg.get("ml_threshold", 0.5)
    _, _, hy_pred = decide(mse, ml, thr_cfg)
    results[name] = {
        "ae_threshold": thr_cfg["ae_threshold"],
        "autoencoder_only": metrics(y, ae_pred),
        "xgboost_only": metrics(y, ml_pred),
        "hybrid": metrics(y, hy_pred),
        "hybrid_recall_per_family": per_family_recall(fam, hy_pred),
    }
    h = results[name]["hybrid"]
    print(f"  {name:28s} AE acc={results[name]['autoencoder_only']['accuracy']:.4f}  "
          f"XGB acc={results[name]['xgboost_only']['accuracy']:.4f}  "
          f"HYBRID acc={h['accuracy']:.4f} recall={h['recall']:.4f} fpr={h['fpr']:.4f}")


def phase1_models(X_test_raw_scaler_note):
    """Load Phase 1 artifacts. They used their own scaler fitted on the DDoS day,
    so the test features must be re-scaled with THAT scaler."""
    from tensorflow.keras.models import load_model
    d = C.HYBRID_DIR
    backup = d + "_phase1_backup"
    if os.path.exists(os.path.join(backup, "autoencoder.keras")):
        d = backup
    ae = load_model(os.path.join(d, "autoencoder.keras"))
    xgbm = joblib.load(os.path.join(d, "xgb_model.pkl"))
    scaler = joblib.load(os.path.join(d, "scaler.pkl"))
    cfg = load_thresholds(os.path.join(d, "threshold.json"))
    return ae, xgbm, scaler, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-central", action="store_true")
    args = ap.parse_args()

    X, y, fam = load_test("report")           # never the calibration half
    fl_scaler = joblib.load(os.path.join(C.DATA_DIR, "scaler.pkl"))
    X_raw = fl_scaler.inverse_transform(X)          # to re-scale for Phase 1 models
    results = {"test_rows": int(len(y)), "test_attacks": int(y.sum()),
               "family_counts": {C.ID_TO_FAMILY[int(k)]: int(v) for k, v in zip(*np.unique(fam, return_counts=True))}}
    print(f"Hold-out set: {len(y):,} rows, {int(y.sum()):,} attacks\n")

    # 1) Phase 1 (DDoS-day only) -------------------------------------------------
    try:
        ae1, xgb1, sc1, cfg1 = phase1_models(None)
        X1 = np.clip(sc1.transform(X_raw), 0, 1).astype(np.float32)
        evaluate_pair("phase1_ddos_only", ae1, xgb1, cfg1, X1, y, fam, results)
    except Exception as e:
        print(f"  phase1 models skipped: {e}")

    # 2) Centralized Phase 2 ---------------------------------------------------------
    if not args.no_central:
        import xgboost as xgb
        Xtr, ytr, _ = load_all_nodes()
        t0 = time.time()
        ae_c = build_autoencoder(C.AE_INPUT_DIM)
        ae_c.fit(Xtr[ytr == 0], Xtr[ytr == 0], epochs=C.AE_ROUNDS * C.AE_LOCAL_EPOCHS,
                 batch_size=C.AE_BATCH_SIZE, verbose=0)
        thr_c = anomaly_threshold(reconstruction_error(ae_c, Xtr[ytr == 0]), C.AE_THRESHOLD_PERCENTILE)
        params = {k: v for k, v in C.XGB_PARAMS.items() if k != "num_parallel_tree"}
        params["scale_pos_weight"] = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
        n_trees = C.XGB_ROUNDS * C.NUM_NODES * C.XGB_TREES_PER_ROUND
        xgb_c = xgb.train(params, xgb.DMatrix(Xtr, label=ytr), num_boost_round=n_trees)
        central_seconds = time.time() - t0
        from core.decision import best_ml_threshold
        Xc, yc, _ = load_test("calib")
        best_c = best_ml_threshold(yc, xgb_c.predict(xgb.DMatrix(Xc)), max_fpr=0.01)
        cfg_c = {"ae_threshold": thr_c, "anomaly_multiplier": C.ANOMALY_MULTIPLIER,
                 "ml_threshold": best_c["ml_threshold"], "decision_threshold": 0.5}
        evaluate_pair("centralized_phase2", ae_c, xgb_c, cfg_c, X, y, fam, results)
        results["centralized_phase2"]["train_seconds"] = round(central_seconds, 1)
        results["centralized_phase2"]["trees"] = n_trees
        results["centralized_phase2"]["training_rows_moved_to_server"] = int(len(ytr))

    # 3) Federated Phase 2 -------------------------------------------------------------
    fl_ae_path = os.path.join(C.FL_MODEL_DIR, "autoencoder.keras")
    fl_xgb_path = os.path.join(C.FL_MODEL_DIR, "xgb_model.json")
    if os.path.exists(fl_ae_path) and os.path.exists(fl_xgb_path):
        import xgboost as xgb
        from tensorflow.keras.models import load_model
        ae_f = load_model(fl_ae_path)
        bst = xgb.Booster(); bst.load_model(fl_xgb_path)
        cfg_f = load_thresholds(os.path.join(C.FL_MODEL_DIR, "threshold.json"))
        evaluate_pair("federated_phase2", ae_f, bst, cfg_f, X, y, fam, results)
        comm = {}
        for run in ("ae", "ae_dp", "xgb"):
            log = read_log(run)
            if log:
                comm[run] = {
                    "rounds": len(log["rounds"]),
                    "bytes_total": log.get("bytes_total") or sum(
                        r.get("bytes_down_total", 0) + r.get("bytes_up_total", 0) for r in log["rounds"]),
                    "final_model_bytes": log.get("final_model_bytes"),
                    "dp": log.get("dp"), "tls": log.get("tls"),
                }
        results["federated_phase2"]["communication"] = comm
        results["federated_phase2"]["training_rows_moved_to_server"] = 0
    else:
        print("  federated models not found; run python -m federated.run_federated first")

    os.makedirs(C.LOG_DIR, exist_ok=True)
    out = os.path.join(C.LOG_DIR, "comparison.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Markdown summary ---------------------------------------------------------------
    print("\n| Model | Detector | Accuracy | Precision | Recall | F1 | FPR |")
    print("|---|---|---|---|---|---|---|")
    for name in ("phase1_ddos_only", "centralized_phase2", "federated_phase2"):
        if name in results:
            for det in ("autoencoder_only", "xgboost_only", "hybrid"):
                m = results[name][det]
                print(f"| {name} | {det} | {m['accuracy']:.4f} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | {m['fpr']:.4f} |")
    print("\nHybrid recall per attack family:")
    for name in ("phase1_ddos_only", "centralized_phase2", "federated_phase2"):
        if name in results:
            print(f"  {name:22s} {results[name]['hybrid_recall_per_family']}")
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()
