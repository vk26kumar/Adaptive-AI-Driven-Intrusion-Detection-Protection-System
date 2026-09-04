"""
federated/data_utils.py
Loading helpers for the node partitions written by prepare_data.py.

The server-side hold-out (test.npz) is used for two different jobs, so it is
split deterministically in half:
  * "calib"  (even rows) -> choosing thresholds / operating points
  * "report" (odd rows)  -> the numbers we publish
Nothing that is tuned on the calibration half is ever measured on it.
"""
import os
import sys

import numpy as np

from federated.config import DATA_DIR, NUM_NODES


def _load(path: str):
    if not os.path.exists(path):
        sys.exit(f"{path} not found. Run:  python -m federated.prepare_data")
    d = np.load(path)
    return d["X"], d["y"], d["family"]


def load_node(node_id: int):
    """Return (X_scaled, y_binary, family_id) for one federated node."""
    if not 0 <= node_id < NUM_NODES:
        raise ValueError(f"node_id must be in 0..{NUM_NODES - 1}")
    return _load(os.path.join(DATA_DIR, f"node_{node_id}.npz"))


def load_test(split: str | None = None):
    """Global hold-out. split=None -> all rows, 'calib' -> even rows, 'report' -> odd rows."""
    X, y, fam = _load(os.path.join(DATA_DIR, "test.npz"))
    if split is None:
        return X, y, fam
    if split == "calib":
        sl = slice(0, None, 2)
    elif split == "report":
        sl = slice(1, None, 2)
    else:
        raise ValueError("split must be None, 'calib' or 'report'")
    return X[sl], y[sl], fam[sl]


def load_all_nodes():
    """Pooled training data (used only for the centralized baseline)."""
    parts = [load_node(i) for i in range(NUM_NODES)]
    return (np.concatenate([p[0] for p in parts]),
            np.concatenate([p[1] for p in parts]),
            np.concatenate([p[2] for p in parts]))
