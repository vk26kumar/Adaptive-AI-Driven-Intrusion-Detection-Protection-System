"""
core/simulator.py
Builds synthetic feature vectors from user-tuned slider values.
Values match the CIC-IDS2017 dataset feature distribution.
"""
from network.features import SELECTED_COLUMNS
from core.predictor import predict_features

_BRUTEFORCE_PORTS = {22, 23, 21, 3389, 5900}
_SQLINJECT_PORTS  = {3306, 5432, 1433, 1521}


def simulate_packet(
    pkt_len: int          = 1200,
    dst_port: int         = 80,
    flow_duration_ms: int = 500,
    pkt_rate: float       = 50,
    proto: int            = 6,
    fwd_pkts: int         = 5,
) -> dict:

    # ── Classify attack type ──────────────────────────────────────────────────
    is_ddos       = pkt_rate > 1000
    is_bruteforce = dst_port in _BRUTEFORCE_PORTS and pkt_rate > 200
    is_sqlinject  = dst_port in _SQLINJECT_PORTS  and pkt_rate > 200
    is_portscan   = pkt_rate > 100 and not is_ddos and not is_bruteforce and not is_sqlinject
    is_attack     = is_ddos or is_bruteforce or is_sqlinject or is_portscan

    # ── Dataset-realistic values per attack type ──────────────────────────────
    # internal_pkts drives Fwd IAT Total — must be large for autoencoder to fire
    # effective_rate drives Fwd Packets/s and IAT Mean
    if is_ddos:
        internal_pkts  = 5000
        effective_rate = pkt_rate * 40.0    # e.g. 3000×40 = 120,000 pkts/s
    elif is_bruteforce or is_sqlinject:
        internal_pkts  = 2000
        effective_rate = 80_000.0           # mid-range attack scale in dataset
    elif is_portscan:
        internal_pkts  = 3000
        effective_rate = 30_000.0           # lower-range attack scale in dataset
    else:
        internal_pkts  = fwd_pkts
        effective_rate = pkt_rate * 40.0

    # ── Inter-arrival times ───────────────────────────────────────────────────
    iat_mean  = 1.0 / max(effective_rate, 1.0)
    iat_total = iat_mean * max(internal_pkts - 1, 1)
    iat_std   = iat_mean * 0.05 if is_attack else iat_mean * 1.5

    # ── Feature signatures ────────────────────────────────────────────────────
    win_fwd  = 0       if is_attack else 65535
    win_bwd  = 0       if is_attack else 65535
    down_up  = 0.0     if is_attack else 1.0
    bwd_mean = 0.0     if is_attack else pkt_len * 0.8
    bwd_max  = 0.0     if is_attack else float(pkt_len)
    bwd_min  = 0.0     if is_attack else float(pkt_len // 2)
    bwd_pkts = 0       if is_attack else 1
    pkt_std         = pkt_len * 0.03 if is_attack else pkt_len * 0.35
    total_fwd_bytes = pkt_len * internal_pkts

    features = {col: 0.0 for col in SELECTED_COLUMNS}
    features.update({
        'Destination Port':              float(dst_port),
        'Total Fwd Packets':             float(internal_pkts),
        'Total Length of Fwd Packets':   float(total_fwd_bytes),
        'Subflow Fwd Packets':           float(internal_pkts),
        'Subflow Fwd Bytes':             float(total_fwd_bytes),
        'act_data_pkt_fwd':              float(internal_pkts),
        'Fwd Packet Length Max':         float(pkt_len),
        'Fwd Packet Length Mean':        float(pkt_len),
        'Fwd Packet Length Std':         float(pkt_std),
        'Avg Fwd Segment Size':          float(pkt_len),
        'Min Packet Length':             float(max(pkt_len // 2, 1)),
        'Packet Length Mean':            float(pkt_len),
        'Average Packet Size':           float(pkt_len),
        'Bwd Packet Length Max':         float(bwd_max),
        'Bwd Packet Length Min':         float(bwd_min),
        'Bwd Packet Length Mean':        float(bwd_mean),
        'Avg Bwd Segment Size':          float(bwd_mean),
        'Subflow Bwd Packets':           float(bwd_pkts),
        'Fwd Packets/s':                 float(effective_rate),
        'Fwd IAT Total':                 float(iat_total),
        'Fwd IAT Mean':                  float(iat_mean),
        'Fwd IAT Std':                   float(iat_std),
        'Fwd IAT Max':                   float(iat_mean * 2),
        'Bwd IAT Mean':                  0.0 if is_attack else float(iat_mean),
        'Flow IAT Std':                  float(iat_std),
        'Init_Win_bytes_forward':        float(win_fwd),
        'Init_Win_bytes_backward':       float(win_bwd),
        'Down/Up Ratio':                 float(down_up),
        'Fwd Header Length':             float(40 if proto == 6 else 28),
        'Fwd Header Length.1':           float(40 if proto == 6 else 28),
    })

    result = predict_features(features)
    result['features'] = features
    return result