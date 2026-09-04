"""
tests/test_phase2_modules.py
Prevention engine policy (dry-run), privacy accounting, LIME explanation,
event store.  None of these touch the real firewall or need the dataset.
"""
import math
import os
import sys
import tempfile

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


def test_prevention_policy_dry_run(monkeypatch, tmp_path):
    import core.prevention as P
    monkeypatch.setattr(P, "LOG_DIR", str(tmp_path))
    monkeypatch.setattr(P, "STATE_PATH", str(tmp_path / "blocked.json"))
    monkeypatch.setattr(P, "AUDIT_PATH", str(tmp_path / "audit.jsonl"))
    eng = P.PreventionEngine(enabled=True, dry_run=True)
    attacker = "45.33.32.156"        # public address (RFC 5737 test ranges count as private in ipaddress)
    # below MIN_HITS: nothing happens
    for _ in range(P.MIN_HITS - 1):
        assert eng.observe(attacker, 0.99, "DDoS") is None
    rec = eng.observe(attacker, 0.99, "DDoS")
    assert rec is not None and rec.mode == "dry-run" and rec.ip == attacker
    assert attacker in eng.blocked
    # low score never blocks
    assert eng.observe("185.199.108.1", 0.6, "Other") is None
    # private / loopback / simulator addresses are never blocked
    for ip in ("192.168.1.5", "10.0.0.2", "127.0.0.1", "SIM:10.0.99.1"):
        for _ in range(P.MIN_HITS + 1):
            assert eng.observe(ip, 0.99, "DDoS") is None
    assert eng.unblock(attacker) is True
    assert attacker not in eng.blocked
    assert (tmp_path / "audit.jsonl").read_text().count('"block"') == 1


def test_prevention_disabled_by_default():
    from core.prevention import PreventionEngine
    eng = PreventionEngine()
    assert eng.enabled is False and eng.dry_run is True


def test_epsilon_accounting_monotonic():
    from federated.dp_analysis import epsilon
    e_small_noise = epsilon(0.01, 15)
    e_big_noise = epsilon(1.0, 15)
    assert e_big_noise < e_small_noise               # more noise -> stronger privacy
    assert epsilon(1.0, 30) > epsilon(1.0, 15)       # more rounds -> weaker privacy
    assert math.isinf(epsilon(0.0, 10))
    assert 5 < epsilon(1.0, 15) < 60                 # sanity range for z=1, 15 rounds, delta=1e-5


def test_event_store_roundtrip(tmp_path):
    from core.persistence import EventStore
    s = EventStore(str(tmp_path / "ev.sqlite"))
    s.log_flow({"src_ip": "1.2.3.4", "dst_port": 80, "proto": "TCP", "label": "ATTACK",
                "final_score": 0.9, "ml_prob": 0.9, "dl_score": 0.1, "latency_ms": 2.0, "features": {"a": 1}},
               source="live", attack_type="DDoS")
    s.log_flow({"label": "NORMAL", "final_score": 0.1}, source="sim", truth="NORMAL")
    s.log_action("block", "1.2.3.4", {"mode": "dry-run"})
    assert s.counts() == {"flows": 2, "attacks": 1, "actions": 1}
    assert s.recent_alerts(5)[0][2] == "1.2.3.4"


@pytest.mark.skipif(not os.path.exists(os.path.join(ROOT, "hybrid_model", "autoencoder.keras")),
                    reason="model artifacts missing")
def test_lime_explanation_has_expected_shape():
    from core.predictor import explain_lime
    from core.simulator import simulate_packet
    r = simulate_packet(profile="PortScan")
    e = explain_lime(r["features"], top_k=5, n_samples=200)
    assert len(e["coefficients"]) == 5
    assert 0.0 <= e["intercept"] <= 1.0
    assert all(isinstance(v, float) for v in e["coefficients"].values())
