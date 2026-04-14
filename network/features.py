"""
network/features.py
Real-time feature extractor from live Scapy packets.
Maps to the 30 selected_columns used by the hybrid model.

FIX: Original code used duration = 1e-6 when ts == start_ts (first packet).
     This made Fwd Packets/s = 1,000,000+ which is the #1 SHAP feature —
     driving the autoencoder to score 1.000 on every new connection.
     Fixed by:
       1. Skipping flows with < 2 forward packets (no meaningful rate yet)
       2. Clamping minimum duration to 0.5s for realistic rate computation
"""
import time
import numpy as np

# All 30 feature columns the model was trained on
SELECTED_COLUMNS = [
    'Down/Up Ratio', 'Bwd IAT Mean', 'Flow IAT Std', 'Fwd Packets/s',
    'Min Packet Length', 'Fwd Packet Length Std', 'Packet Length Mean',
    'Bwd Packet Length Mean', 'Avg Bwd Segment Size', 'Bwd Packet Length Max',
    'Fwd Header Length', 'Destination Port', 'Fwd IAT Mean',
    'Average Packet Size', 'Subflow Bwd Packets', 'Bwd Packet Length Min',
    'Init_Win_bytes_backward', 'Subflow Fwd Packets', 'Fwd IAT Max',
    'Fwd Header Length.1', 'Init_Win_bytes_forward', 'Total Fwd Packets',
    'Fwd IAT Std', 'act_data_pkt_fwd', 'Fwd Packet Length Max',
    'Fwd IAT Total', 'Subflow Fwd Bytes', 'Total Length of Fwd Packets',
    'Fwd Packet Length Mean', 'Avg Fwd Segment Size'
]

# Flow state tracker for per-flow statistics
_flow_cache = {}

def extract_features(packet) -> dict | None:
    """
    Extract the 30 model features from a raw Scapy packet.
    Returns a dict keyed by SELECTED_COLUMNS, or None if packet is irrelevant.
    """
    try:
        if not packet.haslayer("IP"):
            return None

        ip = packet["IP"]
        pkt_len = len(packet)
        ts = time.time()

        proto_num = 0
        sport, dport = 0, 0
        tcp_win_fwd, tcp_win_bwd = 0, 0
        header_len = ip.ihl * 4 if hasattr(ip, 'ihl') else 20

        if packet.haslayer("TCP"):
            proto_num = 6
            tcp = packet["TCP"]
            sport, dport = tcp.sport, tcp.dport
            tcp_win_fwd = tcp.window
            header_len = header_len + (tcp.dataofs * 4 if hasattr(tcp, 'dataofs') else 20)
        elif packet.haslayer("UDP"):
            proto_num = 17
            udp = packet["UDP"]
            sport, dport = udp.sport, udp.dport
            header_len = header_len + 8
        else:
            return None

        flow_key = (ip.src, ip.dst, sport, dport, proto_num)
        rev_key  = (ip.dst, ip.src, dport, sport, proto_num)

        if flow_key not in _flow_cache:
            _flow_cache[flow_key] = {
                'fwd_pkts': [], 'bwd_pkts': [],
                'fwd_ts': [],   'bwd_ts': [],
                'start_ts': ts
            }

        flow = _flow_cache[flow_key]
        rev  = _flow_cache.get(rev_key, None)

        if rev is None:
            flow['fwd_pkts'].append(pkt_len)
            flow['fwd_ts'].append(ts)
        else:
            rev['bwd_pkts'].append(pkt_len)
            rev['bwd_ts'].append(ts)

        fwd_pkts = flow['fwd_pkts']
        bwd_pkts = flow['bwd_pkts']
        fwd_ts   = flow['fwd_ts']
        bwd_ts   = flow['bwd_ts']

        # Need at least 2 forward packets for meaningful rate/IAT computation
        if len(fwd_pkts) < 2:
            return None

        all_lens   = fwd_pkts + bwd_pkts
        fwd_lens   = fwd_pkts if fwd_pkts else [0]
        bwd_lens   = bwd_pkts if bwd_pkts else [0]
        fwd_iat    = np.diff(fwd_ts).tolist() if len(fwd_ts) > 1 else [0]
        bwd_iat    = np.diff(bwd_ts).tolist() if len(bwd_ts) > 1 else [0]
        all_ts     = sorted(fwd_ts + bwd_ts)
        flow_iat   = np.diff(all_ts).tolist() if len(all_ts) > 1 else [0]

        # Clamp to 0.5s minimum so bursts don't produce millions pkts/s
        raw_duration = ts - flow['start_ts']
        duration = max(raw_duration, 0.5)

        features = {
            'Destination Port':              dport,
            'Total Fwd Packets':             len(fwd_pkts),
            'Total Length of Fwd Packets':   sum(fwd_lens),
            'Fwd Packet Length Max':         max(fwd_lens),
            'Fwd Packet Length Mean':        float(np.mean(fwd_lens)),
            'Fwd Packet Length Std':         float(np.std(fwd_lens)) if len(fwd_lens) > 1 else 0,
            'Bwd Packet Length Max':         max(bwd_lens),
            'Bwd Packet Length Min':         min(bwd_lens),
            'Bwd Packet Length Mean':        float(np.mean(bwd_lens)),
            'Min Packet Length':             min(all_lens) if all_lens else 0,
            'Packet Length Mean':            float(np.mean(all_lens)) if all_lens else 0,
            'Fwd Packets/s':                 len(fwd_pkts) / duration,
            'Fwd Header Length':             header_len,
            'Fwd Header Length.1':           header_len,
            'Fwd IAT Total':                 sum(fwd_iat),
            'Fwd IAT Mean':                  float(np.mean(fwd_iat)),
            'Fwd IAT Std':                   float(np.std(fwd_iat)) if len(fwd_iat) > 1 else 0,
            'Fwd IAT Max':                   max(fwd_iat),
            'Bwd IAT Mean':                  float(np.mean(bwd_iat)),
            'Flow IAT Std':                  float(np.std(flow_iat)) if len(flow_iat) > 1 else 0,
            'Average Packet Size':           float(np.mean(all_lens)) if all_lens else 0,
            'Avg Fwd Segment Size':          float(np.mean(fwd_lens)),
            'Avg Bwd Segment Size':          float(np.mean(bwd_lens)),
            'Subflow Fwd Packets':           len(fwd_pkts),
            'Subflow Fwd Bytes':             sum(fwd_lens),
            'Subflow Bwd Packets':           len(bwd_pkts),
            'Down/Up Ratio':                 len(bwd_pkts) / max(len(fwd_pkts), 1),
            'Init_Win_bytes_forward':        tcp_win_fwd,
            'Init_Win_bytes_backward':       tcp_win_bwd,
            'act_data_pkt_fwd':              len([p for p in fwd_lens if p > header_len]),
        }

        if len(_flow_cache) > 5000:
            oldest = list(_flow_cache.keys())[0]
            del _flow_cache[oldest]

        return features

    except Exception:
        return None


def build_feature_row(feature_dict: dict) -> dict:
    """Fill missing columns with 0 and return ordered dict."""
    row = {col: 0.0 for col in SELECTED_COLUMNS}
    for k, v in feature_dict.items():
        if k in row:
            row[k] = float(v) if v is not None else 0.0
    return row