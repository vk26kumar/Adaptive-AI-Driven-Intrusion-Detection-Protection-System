"""
federated/make_prototypes.py
Build one "typical flow" per attack family from the hold-out set (median of
every feature, in raw dataset units).  The dashboard's Attack Simulator starts
from these real profiles instead of hand-invented numbers.

    python -m federated.make_prototypes
Writes federated/data/family_prototypes.json (copied to hybrid_model/ on export).
"""
import json
import os

import joblib
import numpy as np

from federated import config as C
from federated.data_utils import load_test


def main():
    X, y, fam = load_test()
    scaler = joblib.load(os.path.join(C.DATA_DIR, "scaler.pkl"))
    X_raw = scaler.inverse_transform(X)
    protos = {}
    for fid, name in C.ID_TO_FAMILY.items():
        m = fam == fid
        if not m.any():
            continue
        med = np.median(X_raw[m], axis=0)
        protos[name] = {
            "count": int(m.sum()),
            "features": {c: float(round(v, 6)) for c, v in zip(C.SELECTED_COLUMNS, med)},
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
