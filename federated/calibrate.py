"""
federated/calibrate.py
Server-side calibration of the global model's operating point.

    python -m federated.calibrate            # uses hybrid_model_fl/

Reads the federated global Autoencoder and XGBoost booster, computes on the
CALIBRATION half of the hold-out set
  * ae_threshold  : 90th percentile of benign reconstruction error
  * ml_threshold  : P(attack) cutoff with the best F1 under a 1% FPR budget
and writes them to <model_dir>/threshold.json.  Called automatically at the
end of federated/server.py; run it by hand after changing the multiplier or
the FPR budget.  Raw traffic never leaves the nodes: the calibration split is
the server's own labelled hold-out.
"""
import argparse
import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np

from core.decision import best_ml_threshold, decide
from federated import config as C
from federated.data_utils import load_test
from federated.models import reconstruction_error, anomaly_threshold


def calibrate(model_dir: str = C.FL_MODEL_DIR, max_fpr: float = 0.01, verbose: bool = True) -> dict:
    import xgboost as xgb
    from tensorflow.keras.models import load_model

    X, y, _ = load_test("calib")
    cfg = {"anomaly_multiplier": C.ANOMALY_MULTIPLIER, "decision_threshold": 0.5,
           "threshold_percentile": C.AE_THRESHOLD_PERCENTILE, "calibration_rows": int(len(y)),
           "max_fpr_budget": max_fpr}

    ae_path = os.path.join(model_dir, "autoencoder.keras")
    if os.path.exists(ae_path):
        ae = load_model(ae_path)
        mse = reconstruction_error(ae, X)
        cfg["ae_threshold"] = anomaly_threshold(mse[y == 0], C.AE_THRESHOLD_PERCENTILE)
    else:
        mse = None
        cfg["ae_threshold"] = 0.000114912769

    xgb_path = os.path.join(model_dir, "xgb_model.json")
    if os.path.exists(xgb_path):
        bst = xgb.Booster(); bst.load_model(xgb_path)
        p = bst.predict(xgb.DMatrix(X))
        best = best_ml_threshold(y, p, max_fpr=max_fpr)
        cfg["ml_threshold"] = best["ml_threshold"]
        cfg["ml_calibration"] = best
    else:
        p = None
        cfg["ml_threshold"] = 0.5

    if mse is not None and p is not None:
        _, _, att = decide(mse, p, cfg)
        yb = y.astype(bool)
        cfg["calibration_hybrid"] = {
            "recall": float(att[yb].mean()), "fpr": float(att[~yb].mean()),
            "accuracy": float((att == yb).mean()),
        }

    with open(os.path.join(model_dir, "threshold.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    if verbose:
        print(f"[calibrate] ae_threshold={cfg['ae_threshold']:.4e}  ml_threshold={cfg['ml_threshold']:.4f}")
        if "ml_calibration" in cfg:
            b = cfg["ml_calibration"]
            print(f"[calibrate] xgb @cutoff: f1={b['f1']:.4f} recall={b['recall']:.4f} fpr={b['fpr']:.4f}")
        if "calibration_hybrid" in cfg:
            h = cfg["calibration_hybrid"]
            print(f"[calibrate] hybrid on calib split: acc={h['accuracy']:.4f} recall={h['recall']:.4f} fpr={h['fpr']:.4f}")
        print(f"[calibrate] written {os.path.join(model_dir, 'threshold.json')}")
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=C.FL_MODEL_DIR)
    ap.add_argument("--max-fpr", type=float, default=0.01)
    args = ap.parse_args()
    calibrate(args.model_dir, args.max_fpr)


if __name__ == "__main__":
    main()
