"""
core/simulator.py
Attack Simulator: builds a flow feature vector and runs the hybrid model.

Phase 2: instead of inventing values, each traffic profile starts from the
MEDIAN real flow of that attack family in CIC-IDS2017 (hybrid_model/
family_prototypes.json, produced by federated.make_prototypes).  The user can
then override the visible knobs (port, rate, packet length, duration, packet
count).  Because the intended family is known, the dashboard's confusion
matrix can be updated with real ground truth.
"""
import json
import os

from core.predictor import predict_features, MODEL_DIR
from network.features import SELECTED_COLUMNS

PROFILES = ["Normal", "DDoS", "DoS", "PortScan", "BruteForce", "Bot", "WebAttack"]
_PROFILE_TO_FAMILY = {"Normal": "BENIGN"}

_proto_path = os.path.join(MODEL_DIR, "family_prototypes.json")
if os.path.exists(_proto_path):
    with open(_proto_path, encoding="utf-8") as _f:
        _PROTOTYPES = {k: v["features"] for k, v in json.load(_f).items()}
else:
    _PROTOTYPES = {}

# Fallback prototypes (used only if the JSON is missing): rough CIC-IDS2017 medians.
_FALLBACK = {
    "BENIGN":     {"Destination Port": 53, "Total Fwd Packets": 2, "Total Length of Fwd Packets": 80,
                   "Fwd Packet Length Max": 40, "Fwd Packet Length Mean": 40, "Bwd Packet Length Max": 100,
                   "Bwd Packet Length Min": 100, "Bwd Packet Length Mean": 100, "Avg Bwd Segment Size": 100,
                   "Min Packet Length": 40, "Packet Length Mean": 70, "Average Packet Size": 93,
                   "Fwd Packets/s": 20000, "Fwd IAT Total": 100, "Fwd IAT Mean": 100, "Fwd IAT Max": 100,
                   "Fwd Header Length": 16, "Fwd Header Length.1": 16, "Subflow Fwd Packets": 2,
                   "Subflow Fwd Bytes": 80, "Subflow Bwd Packets": 2, "Down/Up Ratio": 1,
                   "Init_Win_bytes_forward": -1, "Init_Win_bytes_backward": -1, "act_data_pkt_fwd": 2,
                   "Avg Fwd Segment Size": 40},
    "DDoS":       {"Destination Port": 80, "Total Fwd Packets": 3, "Total Length of Fwd Packets": 26,
                   "Fwd Packet Length Max": 20, "Fwd Packet Length Mean": 8.67, "Fwd Packet Length Std": 10.3,
                   "Bwd Packet Length Max": 11601, "Bwd Packet Length Mean": 2320, "Avg Bwd Segment Size": 2320,
                   "Packet Length Mean": 1300, "Average Packet Size": 1500, "Fwd Packets/s": 30,
                   "Fwd IAT Total": 90000, "Fwd IAT Mean": 45000, "Fwd IAT Std": 60000, "Fwd IAT Max": 89000,
                   "Bwd IAT Mean": 20000, "Flow IAT Std": 30000, "Fwd Header Length": 72,
                   "Fwd Header Length.1": 72, "Subflow Fwd Packets": 3, "Subflow Fwd Bytes": 26,
                   "Subflow Bwd Packets": 5, "Down/Up Ratio": 1, "Init_Win_bytes_forward": 29200,
                   "Init_Win_bytes_backward": 229, "act_data_pkt_fwd": 1, "Avg Fwd Segment Size": 8.67},
}


def _base(family: str) -> dict:
    src = _PROTOTYPES.get(family) or _FALLBACK.get(family) or _FALLBACK["BENIGN"]
    row = {c: 0.0 for c in SELECTED_COLUMNS}
    row.update({k: float(v) for k, v in src.items() if k in row})
    return row


def simulate_packet(profile: str = "Normal", dst_port=None, pkt_rate=None, pkt_len=None,
                    flow_duration_ms=None, fwd_pkts=None, proto: int = 6) -> dict:
    """
    Build a feature vector for `profile` and predict it.  Any knob left as
    None keeps the dataset value for that profile.
    Returns the prediction dict plus 'features', 'profile', 'truth', 'family'.
    """
    family = _PROFILE_TO_FAMILY.get(profile, profile)
    f = _base(family)

    if dst_port is not None:
        f["Destination Port"] = float(dst_port)
    if fwd_pkts is not None:
        n = float(max(int(fwd_pkts), 1))
        f["Total Fwd Packets"] = f["Subflow Fwd Packets"] = n
        f["act_data_pkt_fwd"] = min(f.get("act_data_pkt_fwd", n), n)
        f["Fwd Header Length"] = f["Fwd Header Length.1"] = n * (20 if proto == 6 else 8)
    if pkt_len is not None:
        L = float(pkt_len)
        f["Fwd Packet Length Mean"] = f["Avg Fwd Segment Size"] = L
        f["Fwd Packet Length Max"] = max(L, f.get("Fwd Packet Length Max", 0.0))
        f["Total Length of Fwd Packets"] = f["Subflow Fwd Bytes"] = L * f["Total Fwd Packets"]
        f["Packet Length Mean"] = f["Average Packet Size"] = (L + f.get("Bwd Packet Length Mean", 0.0)) / 2
    if flow_duration_ms is not None:
        dur_us = float(flow_duration_ms) * 1000.0
        n = max(f["Total Fwd Packets"], 1.0)
        f["Fwd IAT Total"] = dur_us
        f["Fwd IAT Mean"] = dur_us / max(n - 1, 1)
        f["Fwd IAT Max"] = f["Fwd IAT Mean"] * 1.5
        if pkt_rate is None:
            f["Fwd Packets/s"] = n / max(dur_us / 1e6, 1e-6)
    if pkt_rate is not None:
        f["Fwd Packets/s"] = float(pkt_rate)
        if flow_duration_ms is None:
            n = max(f["Total Fwd Packets"], 1.0)
            dur_us = n / max(float(pkt_rate), 1e-6) * 1e6
            f["Fwd IAT Total"] = dur_us
            f["Fwd IAT Mean"] = dur_us / max(n - 1, 1)
            f["Fwd IAT Max"] = f["Fwd IAT Mean"] * 1.5
    if proto == 17:
        f["Init_Win_bytes_forward"] = f["Init_Win_bytes_backward"] = -1.0

    result = predict_features(f)
    result.update({"features": f, "profile": profile, "family": family,
                   "truth": "NORMAL" if family == "BENIGN" else "ATTACK"})
    return result
