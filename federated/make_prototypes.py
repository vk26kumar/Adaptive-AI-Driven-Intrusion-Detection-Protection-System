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
territory.  The medoid is real traffic and still central to its family.

Exact values: test.npz stores features scaled to float32, and inverting that
shifts large-range features (Fwd Header Length spans ~1e9) by hundreds of
bytes.  So the medoid's feature values are taken from test_raw.npz, the exact
CSV values of the hold-out written by prepare_data (checked against test.npz
before use).  Without it the script falls back to the inverted (approximate)
values and says so.
The per-feature median is kept in the JSON for reference.
"""
import json
import os

import joblib
import numpy as np

from federated import config as C
from federated.data_utils import load_test, load_test_raw


def _exact_calib_rows(scaler, X_calib: np.ndarray):
    """Raw float64 features of the calibration half (test_raw.npz), or None."""
    raw = load_test_raw("calib")
    if raw is None:
        print("  test_raw.npz missing (python -m federated.prepare_data --test-raw-only); "
              "using inverted float32 values (approximate)")
        return None
    if raw.shape != X_calib.shape or not np.allclose(
            scaler.transform(raw).astype(np.float32), X_calib, atol=1e-6):
        print("  test_raw.npz does not match test.npz; using inverted values (approximate)")
        return None
    return raw


def main():
    X, y, fam = load_test("calib")
    scaler = joblib.load(os.path.join(C.DATA_DIR, "scaler.pkl"))
    exact = _exact_calib_rows(scaler, X)
    protos = {}
    for fid, name in C.ID_TO_FAMILY.items():
        idx = np.flatnonzero(fam == fid)
        if not len(idx):
            continue
        Xs = X[idx].astype(np.float64)
        med = np.median(Xs, axis=0)
        i = idx[int(np.argmin(np.abs(Xs - med).sum(axis=1)))]
        if exact is not None:
            medoid = exact[i]
        else:
            medoid = scaler.inverse_transform(X[i:i + 1].astype(np.float64))[0]
        median_raw = scaler.inverse_transform(med[None, :])[0]
        protos[name] = {
            "count": int(len(idx)),
            "method": ("medoid (real flow nearest the family median, scaled L1); "
                       + ("exact CSV values" if exact is not None else "approximate (inverted float32)")),
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
              f"pkts/s={f_['Fwd Packets/s']:.1f}  hdr={f_['Fwd Header Length']:.0f}")
    print(f"written {out}")


if __name__ == "__main__":
    main()
