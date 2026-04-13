"""
core/predictor.py
Loads the trained hybrid model (Autoencoder + XGBoost) once and exposes
a single predict_features() function used by both the sniffer and simulator.
"""
import os
import json
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# ── Absolute path to project root ───────────────────────────────────────────
_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_HM   = os.path.join(_BASE, "hybrid_model")

# ── Load artifacts once at import time ──────────────────────────────────────
_autoencoder   = load_model(os.path.join(_HM, "autoencoder.keras"))
_xgb           = joblib.load(os.path.join(_HM, "xgb_model.pkl"))
_scaler        = joblib.load(os.path.join(_HM, "scaler.pkl"))
_selected_cols = joblib.load(os.path.join(_HM, "selected_columns.pkl"))

with open(os.path.join(_HM, "threshold.json")) as _f:
    _threshold = json.load(_f)["threshold"]

# ── SHAP feature weights (static, from training) ────────────────────────────
SHAP_WEIGHTS = {
    'Fwd Packets/s':               0.312,
    'Destination Port':            0.241,
    'Fwd IAT Total':               0.188,
    'Total Length of Fwd Packets': 0.134,
    'Fwd Packet Length Mean':      0.082,
    'Init_Win_bytes_forward':      0.043,
}


def predict_features(feature_dict: dict) -> dict:
    """
    Hybrid prediction: Autoencoder (primary) + XGBoost (secondary booster).

    The autoencoder was trained on 90th-percentile threshold of normal traffic
    reconstruction error — it is the primary decision maker.
    XGBoost acts as a secondary signal: if it fires (ml_prob > 0.5) it boosts
    the final score, but it does not suppress a strong DL signal.

    Scoring:
        dl_score    = clip(dl_error / (threshold * 2), 0, 1)
        final_score = dl_score  +  0.3 * ml_prob  (capped at 1.0)
        ATTACK      = final_score > 0.5
    """
    # ── Build feature row ────────────────────────────────────────────────────
    row = {col: 0.0 for col in _selected_cols}
    for k, v in feature_dict.items():
        if k in row:
            row[k] = float(v) if v is not None else 0.0

    df = pd.DataFrame([row])[_selected_cols]
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    data = _scaler.transform(df)

    # ── Deep Learning — primary decision maker ───────────────────────────────
    recon    = _autoencoder.predict(data, verbose=0)
    dl_error = float(np.mean(np.square(data - recon)))

    # Scale: error at threshold → dl_score 0.5 | error at 2×threshold → 1.0
    dl_score = float(np.clip(dl_error / _threshold, 0.0, 1.0))

    # ── Machine Learning — secondary booster ────────────────────────────────
    ml_prob = float(_xgb.predict_proba(data)[0, 1])

    # ── Ensemble ─────────────────────────────────────────────────────────────
    # DL is the primary signal. ML can only ADD to it, not cancel it.
    # If DL says 0.55 and ML says 0.0  → 0.55 + 0.00 = 0.55  → ATTACK ✅
    # If DL says 0.55 and ML says 0.8  → 0.55 + 0.24 = 0.79  → ATTACK ✅
    # If DL says 0.1  and ML says 0.0  → 0.10 + 0.00 = 0.10  → NORMAL ✅
    # If DL says 0.1  and ML says 0.9  → 0.10 + 0.27 = 0.37  → NORMAL ✅
    final_score = float(np.clip(dl_score + 0.3 * ml_prob, 0.0, 1.0))

    label      = "ATTACK" if final_score > 0.5 else "NORMAL"
    confidence = round(final_score * 100 if label == "ATTACK"
                       else (1 - final_score) * 100, 1)

    return {
        'label':       label,
        'dl_score':    round(dl_score,    4),
        'ml_prob':     round(ml_prob,     4),
        'final_score': round(final_score, 4),
        'confidence':  confidence,
    }