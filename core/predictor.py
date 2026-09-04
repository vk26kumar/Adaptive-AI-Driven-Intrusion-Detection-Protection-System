"""
core/predictor.py
Loads the hybrid model (Autoencoder + XGBoost) once and exposes:

    predict_features(feature_dict) -> dict      one flow
    predict_batch(DataFrame)       -> DataFrame many flows (evaluation / tests)
    explain_features(feature_dict) -> dict      per-prediction explanation

Model directory: hybrid_model/ (override with env IDS_MODEL_DIR).  Both the
Phase 1 artifacts (xgb_model.pkl) and the Phase 2 federated export
(xgb_model.json booster) are supported.

Decision rule lives in core/decision.py.  Thresholds come from
threshold.json (see core/decision.load_thresholds for the accepted formats).
"""
import os
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import joblib
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from tensorflow.keras.models import load_model

from core.decision import decide, load_thresholds, confidence

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.environ.get("IDS_MODEL_DIR", os.path.join(_BASE, "hybrid_model"))

# -- Load artifacts once ------------------------------------------------------
_autoencoder   = load_model(os.path.join(MODEL_DIR, "autoencoder.keras"))
_scaler        = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
_selected_cols = list(joblib.load(os.path.join(MODEL_DIR, "selected_columns.pkl")))
THRESHOLDS     = load_thresholds(os.path.join(MODEL_DIR, "threshold.json"))

_json_path = os.path.join(MODEL_DIR, "xgb_model.json")
if os.path.exists(_json_path):
    _booster = xgb.Booster()
    _booster.load_model(_json_path)
    XGB_SOURCE = "federated booster (xgb_model.json)"
else:
    _clf = joblib.load(os.path.join(MODEL_DIR, "xgb_model.pkl"))
    _booster = _clf.get_booster()
    XGB_SOURCE = "phase 1 classifier (xgb_model.pkl)"

MODEL_INFO = {}
_info_path = os.path.join(MODEL_DIR, "model_info.json")
if os.path.exists(_info_path):
    with open(_info_path, encoding="utf-8") as _f:
        MODEL_INFO = json.load(_f)

# Global feature importance of the classifier (gain, normalised). Replaces the
# hard-coded constants of Phase 1; the dashboard shows the top entries.
_gain = _booster.get_score(importance_type="gain")
_total_gain = sum(_gain.values()) or 1.0
_fnames = _booster.feature_names or [f"f{i}" for i in range(len(_selected_cols))]
_fmap = {fn: col for fn, col in zip(_fnames, _selected_cols)}
GLOBAL_IMPORTANCE = dict(sorted(
    ((_fmap.get(k, k), v / _total_gain) for k, v in _gain.items()),
    key=lambda kv: -kv[1]))
SHAP_WEIGHTS = dict(list(GLOBAL_IMPORTANCE.items())[:6])   # backwards-compatible name


# -- Helpers ------------------------------------------------------------------
def _to_frame(rows) -> pd.DataFrame:
    """List of dicts / DataFrame -> ordered, cleaned DataFrame of 30 columns."""
    df = pd.DataFrame(rows) if not isinstance(rows, pd.DataFrame) else rows.copy()
    for c in _selected_cols:
        if c not in df.columns:
            df[c] = 0.0
    df = df[_selected_cols].astype("float64")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df.fillna(0.0)


def _scale(df: pd.DataFrame) -> np.ndarray:
    # Live traffic can exceed the training range; clip so the AE input stays in [0,1]
    return np.clip(_scaler.transform(df.values), 0.0, 1.0).astype(np.float32)


def _ml_prob(X: np.ndarray) -> np.ndarray:
    return _booster.predict(xgb.DMatrix(X, feature_names=_fnames))


def _mse(X: np.ndarray) -> np.ndarray:
    if len(X) <= 512:
        # Direct call avoids Keras predict() per-call overhead (~80 ms -> ~2 ms per flow)
        rec = _autoencoder(X, training=False).numpy()
    else:
        rec = _autoencoder.predict(X, batch_size=8192, verbose=0)
    return np.mean(np.square(X - rec), axis=1)


# -- Public API -----------------------------------------------------------------
def predict_batch(rows) -> pd.DataFrame:
    """Vectorised prediction. Returns columns mse, dl_score, ml_prob, final_score, label."""
    X = _scale(_to_frame(rows))
    mse = _mse(X)
    ml = _ml_prob(X)
    dl, fin, is_attack = decide(mse, ml, THRESHOLDS)
    return pd.DataFrame({
        "mse": mse, "dl_score": dl, "ml_prob": ml, "final_score": fin,
        "label": np.where(is_attack, "ATTACK", "NORMAL"),
    })


def predict_features(feature_dict: dict) -> dict:
    """Single-flow prediction used by the sniffer and the simulator."""
    t0 = time.perf_counter()
    out = predict_batch([feature_dict]).iloc[0]
    is_attack = out["label"] == "ATTACK"
    return {
        "label":       str(out["label"]),
        "dl_score":    round(float(out["dl_score"]), 4),
        "ml_prob":     round(float(out["ml_prob"]), 4),
        "final_score": round(float(out["final_score"]), 4),
        "mse":         float(out["mse"]),
        "confidence":  confidence(float(out["final_score"]), is_attack),
        "latency_ms":  round((time.perf_counter() - t0) * 1000, 2),
    }


def explain_features(feature_dict: dict, top_k: int = 6) -> dict:
    """
    Per-prediction explanation.
      ml_contrib : XGBoost SHAP values (TreeSHAP via pred_contribs) for this flow,
                   positive = pushes towards ATTACK.
      dl_contrib : share of the Autoencoder reconstruction error carried by each
                   feature, i.e. which features looked least like normal traffic.
    """
    X = _scale(_to_frame([feature_dict]))
    contribs = _booster.predict(xgb.DMatrix(X, feature_names=_fnames), pred_contribs=True)[0]
    ml = {col: float(v) for col, v in zip(_selected_cols, contribs[:-1])}   # last = bias
    rec = _autoencoder.predict(X, verbose=0)[0]
    sq = np.square(X[0] - rec)
    share = sq / (sq.sum() or 1.0)
    dl = {col: float(v) for col, v in zip(_selected_cols, share)}
    return {
        "ml_contrib": dict(sorted(ml.items(), key=lambda kv: -abs(kv[1]))[:top_k]),
        "ml_bias": float(contribs[-1]),
        "dl_contrib": dict(sorted(dl.items(), key=lambda kv: -kv[1])[:top_k]),
    }


def explain_lime(feature_dict: dict, top_k: int = 6, n_samples: int = 500, kernel_width: float = 0.75,
                 seed: int = 0) -> dict:
    """
    LIME-style local explanation of the FINAL hybrid score (not just XGBoost):
    perturb the scaled feature vector around the flow, weight samples by
    proximity (exponential kernel on Euclidean distance), fit a weighted ridge
    regression, and return its coefficients.  Positive = pushes the final score
    (and therefore the ATTACK verdict) up.  Self-contained: no lime dependency.
    """
    rng = np.random.default_rng(seed)
    x0 = _scale(_to_frame([feature_dict]))[0]
    Z = np.clip(x0 + rng.normal(0.0, 0.15, size=(n_samples, len(x0))), 0.0, 1.0).astype(np.float32)
    Z[0] = x0
    mse = _mse(Z)
    ml = _ml_prob(Z)
    _, fin, _ = decide(mse, ml, THRESHOLDS)
    d = np.linalg.norm(Z - x0, axis=1)
    w = np.exp(-(d ** 2) / (kernel_width ** 2))
    Xd = Z - x0                                    # local coordinates
    A = Xd.T @ (Xd * w[:, None]) + 1e-3 * np.eye(len(x0))
    coef = np.linalg.solve(A, Xd.T @ (w * (fin - fin[0])))
    lime = {col: float(c) for col, c in zip(_selected_cols, coef)}
    return {
        "intercept": float(fin[0]),
        "coefficients": dict(sorted(lime.items(), key=lambda kv: -abs(kv[1]))[:top_k]),
        "local_fit_r2": float(1 - np.sum(w * (fin - fin[0] - Xd @ coef) ** 2) / max(np.sum(w * (fin - fin[0]) ** 2), 1e-12)),
    }
