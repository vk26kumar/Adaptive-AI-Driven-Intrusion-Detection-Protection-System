"""
federated/make_prototypes.py
Build one "typical flow" per attack family for the dashboard's Attack Simulator.

    python -m federated.make_prototypes
Writes federated/data/family_prototypes.json (copied to hybrid_model/ on export).

The prototype is the family MEDOID: the real flow (from the server's
calibration half) closest, in scaled feature space (L1), to the family's
per-feature median.  Phase 2 originally used the per-feature median itself,
but that vector is not a flow anybody captured: each feature comes from a
different flow, so it mixes incompatible values (e.g. Min Packet Length 0 with
6-byte packets in both directions) and the Bot "median flow" fell into benign
territory.  The medoid is guaranteed to be real traffic and still central to
its family.  The per-feature median is kept in the JSON for reference.
"""
import json
import os

import joblib
import numpy as np

from federated import config as C
from federated.data_utils import load_test


def _raw(scaler, X_scaled: np.ndarray) -> np.ndarray:
    # float64 before inverse scaling: features such as Fwd Header Length span
    # ~1e9, so float32 would lose whole bytes.
    return scaler.inverse_transform(np.asarray(X_scaled, dtype=np.float64))


def main():
    X, y, fam = load_test("calib")
    scaler = joblib.load(os.path.join(C.DATA_DIR, "scaler.pkl"))
    protos = {}
    for fid, name in C.ID_TO_FAMILY.items():
        m = fam == fid
        if not m.any():
            continue
        Xs = X[m].astype(np.float64)
        med = np.median(Xs, axis=0)
        i = int(np.argmin(np.abs(Xs - med).sum(axis=1)))
        medoid = _raw(scaler, Xs[i:i + 1])[0]
        median_raw = _raw(scaler, med[None, :])[0]
        protos[name] = {
            "count": int(m.sum()),
            "method": "medoid (real flow nearest the family median, scaled L1)",
            "features": {c: float(round(v, 6)) for c, v in zip(C.SELECTED_COLUMNS, medoid)},
            "median": {c: float(round(v, 6)) for c, v in zip(C.SELECTED_COLUMNS, median_raw)},
        }
    out = os.path.join(C.DATA_DIR, "family_prototypes.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(protos, f, indent=2)
    for name, p in protos.items():
        f_ = p["features"]
        print(f"{name:10s} n={p['count']:>7,}  port={f_['Destination Port']:.0f}  "
              f"fwd_pkts={f_['Total Fwd Packets']:.0f}  fwd_len_mean={f_['Fwd Packet Length Mean']:.1f}  "
              f"pkts/s={f_['Fwd Packets/s']:.1f}  win_fwd={f_['Init_Win_bytes_forward']:.0f}")
    print(f"written {out}")


if __name__ == "__main__":
    main()
