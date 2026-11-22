"""
network/baseline.py
Learning mode: fit the LIVE autoencoder threshold to this network's normal traffic.

    python -m network.baseline --minutes 5          (use the network normally meanwhile)
    python -m network.baseline --clear              (back to the dataset threshold)

Why: the autoencoder learned "normal" from the 2017 CIC testbed.  Live testing
on a 2026 office Wi-Fi showed that every remaining false positive came from the
autoencoder (XGBoost gave P(attack) ~ 0 on all of them), concentrated on
long-lived TLS keep-alive connections, a traffic pattern the 2017 lab barely had.
That is domain shift, and the standard remedy for an anomaly detector is a
per-deployment baseline -- the same idea as federated nodes knowing their own
normal traffic.

What it does: captures unicast traffic through the real sniffer pipeline for
--minutes, collects the autoencoder reconstruction error of every classified
flow snapshot, and sets
    ae_threshold_live = max(dataset ae_threshold, 90th percentile of local MSE)
i.e. the SAME definition the server uses on its calibration split, applied to
local traffic, and never more sensitive than the validated threshold.  Written
to logs/live_baseline.json; network/sniffer.py applies it to live traffic only.
The dataset model, its threshold.json and every reported dataset metric are
untouched; XGBoost and the behavioural scan detector are unaffected.

Caveat: whatever happens during learning is taken as normal.  Run it when the
network is known to be clean; flows flagged by XGBoost or the scan detector
during learning are excluded from the baseline.
"""
import argparse
import json
import os
import threading
import time
from queue import Queue, Empty

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINE_PATH = os.environ.get("IDS_LIVE_BASELINE", os.path.join(ROOT, "logs", "live_baseline.json"))
MIN_SNAPSHOTS = 200


def load_live_thresholds(base: dict) -> dict:
    """THRESHOLDS with the local live baseline applied (or `base` unchanged)."""
    try:
        with open(BASELINE_PATH, encoding="utf-8") as f:
            b = json.load(f)
        thr = float(b["ae_threshold_live"])
        if thr >= base["ae_threshold"]:
            return {**base, "ae_threshold": thr, "live_baseline": b.get("created")}
    except (OSError, KeyError, ValueError, TypeError):
        pass
    return base


def learn(minutes: float, iface=None) -> dict:
    from core.predictor import THRESHOLDS
    from network.sniffer import start_sniffing
    q, stop = Queue(maxsize=200000), threading.Event()
    th = threading.Thread(target=start_sniffing, args=(q, stop, iface),
                          kwargs={"use_live_baseline": False}, daemon=True)   # learn against the DATASET threshold
    th.start()
    print(f"Learning the local baseline for {minutes:g} min; use the network normally...")
    time.sleep(minutes * 60)
    stop.set(); th.join(timeout=10)
    mse, excluded = [], 0
    while True:
        try:
            r = q.get_nowait()
        except Empty:
            break
        if r.get("detector") or r.get("ml_prob", 0) > 0.5:  # possible attack: keep it out of "normal"
            excluded += 1
            continue
        mse.append(r["mse"])
    if len(mse) < MIN_SNAPSHOTS:
        raise SystemExit(f"Only {len(mse)} flow snapshots captured (need {MIN_SNAPSHOTS}); learn for longer.")
    p90 = float(np.percentile(mse, 90))
    thr = max(THRESHOLDS["ae_threshold"], p90)
    out = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "minutes": minutes, "snapshots": len(mse),
           "excluded_suspicious": excluded, "local_mse_p90": p90,
           "dataset_ae_threshold": THRESHOLDS["ae_threshold"], "ae_threshold_live": thr,
           "local_snapshots_over_dataset_rule": round(float(np.mean(np.array(mse) > 10 * THRESHOLDS["ae_threshold"])), 4),
           "local_snapshots_over_live_rule": round(float(np.mean(np.array(mse) > 10 * thr)), 4)}
    os.makedirs(os.path.dirname(BASELINE_PATH), exist_ok=True)
    with open(BASELINE_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=5.0)
    ap.add_argument("--clear", action="store_true", help="remove the live baseline")
    a = ap.parse_args()
    if a.clear:
        if os.path.exists(BASELINE_PATH):
            os.remove(BASELINE_PATH)
        print("Live baseline removed; live traffic uses the dataset threshold.")
        return
    print(json.dumps(learn(a.minutes), indent=2))
    print(f"Written {BASELINE_PATH}. Restart the dashboard / main.py to apply it.")


if __name__ == "__main__":
    main()
