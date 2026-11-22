"""
core/decision.py
The hybrid decision rule, as pure functions with no model loading so it can be
unit-tested and reused by the predictor, the evaluator and the federated server.

Rule (chosen empirically on CIC-IDS2017, see federated/evaluate.py):

    dl_score    = clip( mse / (2 * ANOMALY_MULTIPLIER * ae_threshold), 0, 1 )
                  -> 0.5 exactly when mse == ANOMALY_MULTIPLIER x threshold
    ml_score    = piecewise-linear rescaling of XGBoost P(attack) so that
                  ml_score == 0.5 exactly when P(attack) == ml_threshold
    final_score = max(dl_score, ml_score)
    label       = ATTACK if final_score > 0.5

In words: a flow is an attack when EITHER the supervised model recognises a
known signature OR the reconstruction error is far (10x) above what benign
traffic produces.  The two detectors are complementary rather than averaged,
so the anomaly detector can still flag a zero-day the classifier has never
seen, while a confident classifier is never diluted by a quiet autoencoder.

Why ml_threshold is not simply 0.5: in federated bagging every node appends
trees each round, and nodes that see almost no attacks push probabilities
down.  The ranking (AUC) is excellent but the raw probabilities sit low, so
the server calibrates the cutoff on its calibration split (federated/calibrate.py)
and publishes it in threshold.json together with the autoencoder threshold.
"""
from __future__ import annotations

import json
import os

import numpy as np

DEFAULTS = {
    "ae_threshold": 0.000114912769,   # 90th percentile benign MSE (Phase 1 value)
    "anomaly_multiplier": 10.0,
    "ml_threshold": 0.5,
    "decision_threshold": 0.5,
}


def load_thresholds(path: str) -> dict:
    """Read threshold.json. Accepts the Phase 1 single-key format too."""
    cfg = dict(DEFAULTS)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        if "ae_threshold" in raw:
            cfg.update({k: raw[k] for k in DEFAULTS if k in raw})
            if "ml_platt" in raw:
                cfg["ml_platt"] = raw["ml_platt"]
        elif "threshold" in raw and raw["threshold"] < 0.01:
            # Legacy file that actually holds the reconstruction threshold
            cfg["ae_threshold"] = float(raw["threshold"])
        # else: legacy file holding the 0.5 ensemble cutoff -> keep defaults
    return cfg


def dl_score(mse, ae_threshold: float, anomaly_multiplier: float = 10.0):
    """Reconstruction error -> [0, 1]; 0.5 at multiplier x threshold."""
    return np.clip(np.asarray(mse, dtype=float) / (2.0 * anomaly_multiplier * ae_threshold), 0.0, 1.0)


def ml_score(ml_prob, ml_threshold: float = 0.5):
    """P(attack) -> [0, 1] with the calibrated cutoff mapped to exactly 0.5."""
    p = np.clip(np.asarray(ml_prob, dtype=float), 0.0, 1.0)
    t = float(np.clip(ml_threshold, 1e-6, 1 - 1e-6))
    below = 0.5 * p / t
    above = 0.5 + 0.5 * (p - t) / (1.0 - t)
    return np.where(p <= t, below, above)


def final_score(dl, ml):
    return np.maximum(np.asarray(dl, dtype=float), np.asarray(ml, dtype=float))


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def platt(ml_prob, a: float, b: float):
    """Platt scaling: calibrated P(attack) = sigmoid(a * logit(p) + b), a > 0."""
    return 1.0 / (1.0 + np.exp(-(a * _logit(ml_prob) + b)))


def fit_platt(y_true, ml_prob) -> dict:
    """Fit Platt scaling on a labelled calibration set; returns {'a', 'b'}.
    Federated bagging ranks flows almost perfectly but leaves raw probabilities
    far too low (node 3's benign-only trees pull every probability down), so a
    clear port scan gets P = 0.26.  The decision cutoff copes with that, but the
    score is then not a probability: margins look thin and the prevention
    engine's 0.75 rule almost never fires.  Platt scaling is monotone, so every
    ATTACK/NORMAL decision is unchanged; only the scores become calibrated."""
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(C=1e6).fit(_logit(ml_prob)[:, None], np.asarray(y_true).astype(int))
    a, b = float(lr.coef_[0, 0]), float(lr.intercept_[0])
    if a <= 0:                                   # would invert the ranking: refuse
        return {"a": 1.0, "b": 0.0}
    return {"a": a, "b": b}


def calibrated_prob(ml_prob, cfg: dict):
    """XGBoost P(attack) after the Platt scaling in cfg (identity if none)."""
    pl = cfg.get("ml_platt")
    return platt(ml_prob, pl["a"], pl["b"]) if pl else np.clip(np.asarray(ml_prob, dtype=float), 0.0, 1.0)


def decide(mse, ml_prob, cfg: dict):
    """Vectorised decision. Returns (dl, final, is_attack).
    ml_prob is the RAW booster probability; if cfg carries Platt parameters both
    the probability and the cutoff are mapped through it (monotone, so the
    decision is identical and only the score scale changes)."""
    dl = dl_score(mse, cfg["ae_threshold"], cfg["anomaly_multiplier"])
    ml = ml_score(calibrated_prob(ml_prob, cfg), float(calibrated_prob(cfg.get("ml_threshold", 0.5), cfg)))
    fin = final_score(dl, ml)
    return dl, fin, fin > cfg["decision_threshold"]


def confidence(final: float, is_attack: bool) -> float:
    return round(float(final * 100 if is_attack else (1 - final) * 100), 1)


MIN_FAMILY_ROWS = 20     # families smaller than this on the calibration set are too noisy to steer the cutoff
MACRO_TOLERANCE = 0.01   # accept cutoffs within 1 point of the best achievable macro recall


def best_ml_threshold(y_true, ml_prob, max_fpr: float = 0.01, family=None) -> dict:
    """
    Pick the P(attack) cutoff on a calibration set among cutoffs whose benign
    false-positive rate stays under max_fpr.

    family=None  -> best F1 (original Phase 2 rule).
    family=array -> family-aware rule: keep only cutoffs whose MACRO recall
                    (mean detection rate over attack families with >=
                    MIN_FAMILY_ROWS calibration rows) is within
                    MACRO_TOLERANCE of the best achievable, then take the best
                    F1 among them.  In words: maximise F1, but never by giving
                    up a rare family.  Plain F1 is blind to Bot and WebAttack
                    (~0.4% of attacks): in the Nov 2026 retrain it chose a
                    cutoff on a cliff edge that cost Bot 26 and WebAttack 38
                    recall points for +0.0002 F1.  Pure macro recall overshoots
                    the other way (cutoff pushed to the FPR budget edge for
                    1-2 points of recall), hence the tolerance.

    Returns {'ml_threshold', 'f1', 'recall', 'fpr'} (+ 'macro_recall' and
    'family_recall' when family is given).
    """
    y = np.asarray(y_true).astype(bool)
    p = np.asarray(ml_prob, dtype=float)
    fams = []
    if family is not None:
        family = np.asarray(family)
        fams = [f for f in np.unique(family[y]) if (family == f).sum() >= MIN_FAMILY_ROWS]
    cands = []
    for t in np.unique(np.round(np.quantile(p, np.linspace(0.001, 0.999, 999)), 6)):
        pred = p > t
        tp = (pred & y).sum(); fp = (pred & ~y).sum(); fn = (~pred & y).sum(); tn = (~pred & ~y).sum()
        fpr = fp / max(fp + tn, 1)
        if fpr > max_fpr:
            continue
        prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        cand = {"ml_threshold": float(t), "f1": float(f1), "recall": float(rec), "fpr": float(fpr)}
        if fams:
            fr = {int(f): float(pred[family == f].mean()) for f in fams}
            cand["family_recall"] = fr
            cand["macro_recall"] = float(np.mean(list(fr.values())))
        cands.append(cand)
    if not cands:
        return {"ml_threshold": 0.5, "f1": 0.0, "recall": 0.0, "fpr": 0.0}
    if fams:
        floor = max(c["macro_recall"] for c in cands) - MACRO_TOLERANCE
        cands = [c for c in cands if c["macro_recall"] >= floor]
    return max(cands, key=lambda c: c["f1"])
