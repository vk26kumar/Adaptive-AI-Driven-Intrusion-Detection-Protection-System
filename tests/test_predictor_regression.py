"""
tests/test_predictor_regression.py
Replays real CIC-IDS2017 flows through the deployed predictor and fails if the
model stops detecting attacks or floods benign traffic with alerts.  This is
the guard that would have caught the Phase 1 threshold.json bug.

Run:  python -m pytest tests -q
Needs the federated partitions (python -m federated.prepare_data); skipped otherwise.
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

TEST_NPZ = os.path.join(ROOT, "federated", "data", "test.npz")
N_SAMPLE = 20000
MIN_RECALL = 0.90          # on the multi-attack hold-out set
MAX_FPR = 0.05
MIN_RECALL_PER_FAMILY = {"DDoS": 0.95, "DoS": 0.90, "PortScan": 0.90, "BruteForce": 0.80}


@pytest.fixture(scope="module")
def sample():
    if not os.path.exists(TEST_NPZ):
        pytest.skip("federated/data/test.npz missing; run python -m federated.prepare_data")
    import joblib
    d = np.load(TEST_NPZ)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(d["y"]), size=min(N_SAMPLE, len(d["y"])), replace=False)
    scaler = joblib.load(os.path.join(ROOT, "federated", "data", "scaler.pkl"))
    cols = list(joblib.load(os.path.join(ROOT, "federated", "data", "selected_columns.pkl")))
    import pandas as pd
    X_raw = scaler.inverse_transform(d["X"][idx])            # predictor scales itself
    return pd.DataFrame(X_raw, columns=cols), d["y"][idx], d["family"][idx]


@pytest.fixture(scope="module")
def predictions(sample):
    from core.predictor import predict_batch
    df, y, fam = sample
    out = predict_batch(df)
    return (out["label"].values == "ATTACK"), y.astype(bool), fam


def test_decision_rule_is_pure():
    from core.decision import decide
    cfg = {"ae_threshold": 1e-4, "anomaly_multiplier": 10.0, "ml_threshold": 0.5, "decision_threshold": 0.5}
    dl, fin, att = decide(np.array([1e-4, 1e-3, 5e-3]), np.array([0.0, 0.0, 0.0]), cfg)
    assert list(att) == [False, False, True]               # 1x thr no, 10x thr borderline, 50x thr yes
    _, _, att2 = decide(np.array([0.0]), np.array([0.9]), cfg)
    assert att2[0]                                          # confident classifier alone is enough


def test_threshold_file_is_reconstruction_threshold():
    from core.predictor import THRESHOLDS
    assert THRESHOLDS["ae_threshold"] < 0.05, (
        "threshold.json holds a decision cutoff, not the autoencoder reconstruction threshold")


def test_overall_recall_and_fpr(predictions):
    pred, y, _ = predictions
    recall = pred[y].mean()
    fpr = pred[~y].mean()
    print(f"\nrecall={recall:.4f} fpr={fpr:.4f} on {len(y)} sampled flows")
    assert recall >= MIN_RECALL, f"attack recall {recall:.3f} < {MIN_RECALL}"
    assert fpr <= MAX_FPR, f"benign false-positive rate {fpr:.3f} > {MAX_FPR}"


def test_per_family_recall(predictions):
    pred, y, fam = predictions
    from federated.config import FAMILY_IDS
    failures = []
    for name, floor in MIN_RECALL_PER_FAMILY.items():
        m = fam == FAMILY_IDS[name]
        if m.sum() < 50:
            continue
        r = pred[m].mean()
        print(f"{name:12s} n={int(m.sum()):6d} recall={r:.4f}")
        if r < floor:
            failures.append(f"{name}: {r:.3f} < {floor}")
    assert not failures, "; ".join(failures)


def test_single_flow_api_matches_batch(sample):
    from core.predictor import predict_features, predict_batch
    df, _, _ = sample
    row = df.iloc[0].to_dict()
    single = predict_features(row)
    batch = predict_batch([row]).iloc[0]
    assert single["label"] == batch["label"]
    assert abs(single["final_score"] - float(batch["final_score"])) < 1e-3
    assert "latency_ms" in single
