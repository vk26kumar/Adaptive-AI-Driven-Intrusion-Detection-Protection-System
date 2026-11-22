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


def test_every_simulator_profile_gets_its_true_verdict():
    """Each dashboard profile (one real flow per family) must be classified as
    its ground truth; a demo button that shows a miss is a visible defect.
    Phase 2 used per-feature medians and the Bot profile came out NORMAL."""
    from core.simulator import simulate_packet, PROFILES
    wrong = {}
    for profile in PROFILES:
        r = simulate_packet(profile=profile)
        if r["label"] != r["truth"]:
            wrong[profile] = (r["label"], r["final_score"])
    assert not wrong, f"profiles misclassified: {wrong}"


def test_family_aware_calibration_keeps_rare_families():
    """Plain F1 trades a rare family away for a tiny F1 gain; the family-aware
    rule must not (core.decision.best_ml_threshold, family= argument)."""
    import numpy as np
    from core.decision import best_ml_threshold
    rng = np.random.default_rng(0)
    benign = rng.uniform(0.00, 0.30, 20000)
    common = rng.uniform(0.60, 1.00, 5000)          # big family, clearly separated
    rare = rng.uniform(0.27, 0.33, 100)              # small family, overlaps benign tail
    p = np.concatenate([benign, common, rare])
    y = np.r_[np.zeros(20000), np.ones(5100)]
    fam = np.r_[np.zeros(20000), np.ones(5000), np.full(100, 2)]
    plain = best_ml_threshold(y, p, max_fpr=0.10)
    aware = best_ml_threshold(y, p, max_fpr=0.10, family=fam)
    rare_recall = lambda t: float((rare > t).mean())
    assert rare_recall(plain["ml_threshold"]) < 0.6            # F1 gives the rare family up
    assert rare_recall(aware["ml_threshold"]) > 0.9            # family-aware keeps it
    assert aware["fpr"] <= 0.10


def test_platt_scaling_keeps_every_decision():
    """Platt scaling must only rescale scores: identical ATTACK/NORMAL decisions,
    cutoff mapped consistently, and a monotone (non-inverting) fit."""
    import numpy as np
    from core.decision import decide, fit_platt, platt
    rng = np.random.default_rng(1)
    p = np.r_[rng.beta(1, 30, 5000), rng.beta(3, 8, 1000)]          # low raw probs, attacks slightly higher
    y = np.r_[np.zeros(5000), np.ones(1000)]
    pl = fit_platt(y, p)
    assert pl["a"] > 0
    mse = rng.uniform(0, 2e-4, len(p))
    base = {"ae_threshold": 1e-4, "anomaly_multiplier": 10.0, "ml_threshold": 0.15, "decision_threshold": 0.5}
    _, f0, d0 = decide(mse, p, base)
    _, f1, d1 = decide(mse, p, {**base, "ml_platt": pl})
    assert np.array_equal(d0, d1)
    assert np.all(np.diff(platt(np.sort(p), pl["a"], pl["b"])) >= 0)


def test_scan_detector_flags_vertical_scan_not_browsing():
    from core.scan_detector import ScanDetector
    d = ScanDetector(window_s=10, min_ports=20, alert_every=20)
    alerts = [d.observe("10.0.0.9", "10.0.0.5", p, 100 + p * 0.01) for p in range(1, 61)]
    fired = [a for a in alerts if a]
    assert len(fired) == 3 and fired[0]["distinct_ports"] == 20          # at 20, 40, 60 ports
    # a browser: many hosts on 443, and repeated connections to one host on few ports
    b = ScanDetector(window_s=10, min_ports=20)
    assert not any(b.observe("10.0.0.5", f"93.184.0.{i}", 443, 200 + i * 0.05) for i in range(100))
    assert not any(b.observe("10.0.0.5", "93.184.1.1", 443 if i % 2 else 80, 300 + i * 0.05) for i in range(100))
    # slow probing spread beyond the window is not a burst
    s = ScanDetector(window_s=10, min_ports=20)
    assert not any(s.observe("10.0.0.9", "10.0.0.6", p, 400 + p * 1.0) for p in range(1, 40))


def test_live_result_names_the_flow_client_not_the_packet_sender():
    """A probe becomes classifiable on the victim's RST; the result must still
    name the scanner as source (prevention would otherwise target the victim)."""
    from scapy.all import IP, TCP
    from network import features as F
    from network.sniffer import make_packet_handler
    from queue import Queue
    F.reset_flow_cache()
    q = Queue(); h = make_packet_handler(q)
    syn = IP(src="45.33.32.156", dst="10.0.0.5") / TCP(sport=51516, dport=8082, flags="S"); syn.time = 7000.0
    rst = IP(src="10.0.0.5", dst="45.33.32.156") / TCP(sport=8082, dport=51516, flags="RA"); rst.time = 7000.0001
    h(syn); h(rst)
    r = q.get_nowait()
    assert r["src_ip"] == "45.33.32.156" and r["dst_ip"] == "10.0.0.5" and r["dst_port"] == 8082


def test_live_baseline_can_only_raise_the_threshold(tmp_path, monkeypatch):
    import json
    import network.baseline as B
    base = {"ae_threshold": 5e-5, "anomaly_multiplier": 10.0, "ml_threshold": 0.16, "decision_threshold": 0.5}
    p = tmp_path / "b.json"
    monkeypatch.setattr(B, "BASELINE_PATH", str(p))
    assert B.load_live_thresholds(base) is base                         # no file: unchanged
    p.write_text(json.dumps({"ae_threshold_live": 2e-4}))
    assert B.load_live_thresholds(base)["ae_threshold"] == 2e-4         # raised
    p.write_text(json.dumps({"ae_threshold_live": 1e-6}))
    assert B.load_live_thresholds(base)["ae_threshold"] == 5e-5         # never lowered
