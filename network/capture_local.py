"""
network/capture_local.py
Record this computer's own traffic as flow features (no packets, no payloads),
for the local adaptation node (federated/local_adapt.py).

    python -m network.capture_local --minutes 10 --out federated/data/local_train.npz
    python -m network.capture_local --minutes 5  --out federated/data/local_test.npz

Every live check the sniffer makes is stored: the 30 model features plus where
the connection went.  Loopback traffic and port-scan-rule alerts are skipped.
The capture is assumed to be normal traffic, so record it on a network you
trust.  The file stays on this computer (federated/data is git-ignored).
"""
import argparse
import os
import threading
import time
from queue import Queue, Empty

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np

from network.features import SELECTED_COLUMNS
from network.sniffer import start_sniffing


def capture(minutes: float, out: str, generate_web: bool = True) -> dict:
    q, stop = Queue(maxsize=500000), threading.Event()
    th = threading.Thread(target=start_sniffing, args=(q, stop, None), kwargs={"use_live_baseline": False},
                          daemon=True)
    th.start()
    gen_stop = threading.Event()
    if generate_web:                                    # some ordinary browsing on top of what the PC already does
        from network.live_check import _benign_traffic
        threading.Thread(target=_benign_traffic, args=(gen_stop,), daemon=True).start()
    end = time.time() + minutes * 60
    rows, meta = [], []
    while time.time() < end or not q.empty():
        try:
            r = q.get(timeout=1.0)
        except Empty:
            if time.time() >= end:
                break
            continue
        if r.get("detector") or str(r.get("dst_ip", "")).startswith("127.") or str(r.get("src_ip", "")).startswith("127."):
            continue
        rows.append([float(r["features"].get(c, 0.0)) for c in SELECTED_COLUMNS])
        meta.append((r.get("dst_ip", ""), int(r.get("dst_port", 0) or 0), r.get("proto", ""),
                     float(r.get("ml_prob", 0.0)), float(r.get("dl_score", 0.0)), r.get("label", "")))
        if time.time() >= end:
            gen_stop.set(); stop.set()
    gen_stop.set(); stop.set(); th.join(timeout=10)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    np.savez_compressed(out, X=np.asarray(rows, dtype=np.float64),
                        dst_ip=np.asarray([m[0] for m in meta], dtype="U45"),
                        dst_port=np.asarray([m[1] for m in meta], dtype=np.int32),
                        proto=np.asarray([m[2] for m in meta], dtype="U4"),
                        ml_prob=np.asarray([m[3] for m in meta]), dl_score=np.asarray([m[4] for m in meta]),
                        label=np.asarray([m[5] for m in meta], dtype="U6"))
    labels = [m[5] for m in meta]
    return {"snapshots": len(rows), "flagged_by_current_model": labels.count("ATTACK"),
            "classifier_flags": int(sum(1 for m in meta if m[5] == "ATTACK" and m[3] > 0.5)),
            "destinations": len({(m[0], m[1]) for m in meta}), "out": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=10)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-web", action="store_true", help="do not generate extra web requests")
    a = ap.parse_args()
    print(capture(a.minutes, a.out, not a.no_web))


if __name__ == "__main__":
    main()
