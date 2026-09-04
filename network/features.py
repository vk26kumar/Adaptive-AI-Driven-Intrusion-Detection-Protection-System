"""
network/features.py
Real-time flow feature extractor for live Scapy packets, producing the 30
CIC-IDS2017 features the model was trained on.

Phase 2 rewrite.  The Phase 1 extractor produced values in different units
from the training data, which is why live traffic never resembled the dataset:

  * Times: CICFlowMeter reports durations and inter-arrival times in
    MICROSECONDS.  Phase 1 used seconds (1e6 times too small).
  * Packet lengths: the dataset's "packet length" is the transport PAYLOAD size
    in bytes.  Phase 1 used the whole Ethernet frame length.
  * Header length: total TCP/UDP header bytes across forward packets, not the
    header of the latest packet.
  * Init_Win_bytes_backward was never filled (always 0); UDP flows use -1 as
    in the dataset.
  * Flows never expired; the cache evicted an arbitrary key.  Flows now time
    out after FLOW_TIMEOUT_S of inactivity (CICFlowMeter default 120 s).
"""
import time
from collections import OrderedDict

import numpy as np

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
    'Fwd Packet Length Mean', 'Avg Fwd Segment Size',
]

MAX_FLOWS       = 5000     # bound on tracked flows
MAX_PKTS_PER_FLOW = 4096   # bound on per-flow history
FLOW_TIMEOUT_S  = 120.0    # idle time after which a flow is dropped
MIN_FWD_PACKETS = 2        # need 2 packets before rates / IATs mean anything

_flow_cache: "OrderedDict[tuple, dict]" = OrderedDict()


def _new_flow(ts: float) -> dict:
    return {
        'fwd_len': [], 'bwd_len': [], 'fwd_ts': [], 'bwd_ts': [],
        'fwd_hdr': 0, 'fwd_data_pkts': 0,
        'win_fwd': None, 'win_bwd': None,
        'start_ts': ts, 'last_ts': ts,
    }


def _expire(now: float) -> None:
    """Drop idle flows and enforce the size bound (oldest first)."""
    for key in list(_flow_cache.keys()):
        if now - _flow_cache[key]['last_ts'] > FLOW_TIMEOUT_S:
            del _flow_cache[key]
        else:
            break  # OrderedDict is kept in last-seen order
    while len(_flow_cache) > MAX_FLOWS:
        _flow_cache.popitem(last=False)


def _stats(values):
    if not values:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    a = np.asarray(values, dtype=float)
    return float(a.sum()), float(a.mean()), float(a.std()) if len(a) > 1 else 0.0, float(a.max()), float(a.min())


def extract_features(packet) -> dict | None:
    """
    Update the flow this packet belongs to and return the 30 features for
    that flow (dataset units), or None if the packet is not TCP/UDP over IP or
    the flow is still too short to describe.
    """
    try:
        if not packet.haslayer("IP"):
            return None
        ip = packet["IP"]
        now = time.time()

        if packet.haslayer("TCP"):
            l4 = packet["TCP"]; proto = 6
            hdr_len = int(l4.dataofs) * 4 if l4.dataofs else 20
            window = int(l4.window)
        elif packet.haslayer("UDP"):
            l4 = packet["UDP"]; proto = 17
            hdr_len = 8
            window = -1                       # CICFlowMeter convention for UDP
        else:
            return None
        payload_len = len(bytes(l4.payload))
        sport, dport = int(l4.sport), int(l4.dport)

        fwd_key = (ip.src, ip.dst, sport, dport, proto)
        rev_key = (ip.dst, ip.src, dport, sport, proto)

        # Direction is defined by the first packet seen for the 5-tuple.
        if fwd_key in _flow_cache:
            key, flow, is_fwd = fwd_key, _flow_cache[fwd_key], True
        elif rev_key in _flow_cache:
            key, flow, is_fwd = rev_key, _flow_cache[rev_key], False
        else:
            key, flow, is_fwd = fwd_key, _new_flow(now), True
            _flow_cache[key] = flow
        _flow_cache.move_to_end(key)
        flow['last_ts'] = now

        if is_fwd:
            if len(flow['fwd_len']) < MAX_PKTS_PER_FLOW:
                flow['fwd_len'].append(payload_len); flow['fwd_ts'].append(now)
            flow['fwd_hdr'] += hdr_len
            if payload_len > 0:
                flow['fwd_data_pkts'] += 1
            if flow['win_fwd'] is None:
                flow['win_fwd'] = window
        else:
            if len(flow['bwd_len']) < MAX_PKTS_PER_FLOW:
                flow['bwd_len'].append(payload_len); flow['bwd_ts'].append(now)
            if flow['win_bwd'] is None:
                flow['win_bwd'] = window

        _expire(now)

        fwd_len, bwd_len = flow['fwd_len'], flow['bwd_len']
        if len(fwd_len) < MIN_FWD_PACKETS:
            return None

        US = 1e6
        fwd_iat = (np.diff(flow['fwd_ts']) * US).tolist() if len(flow['fwd_ts']) > 1 else []
        bwd_iat = (np.diff(flow['bwd_ts']) * US).tolist() if len(flow['bwd_ts']) > 1 else []
        all_ts = sorted(flow['fwd_ts'] + flow['bwd_ts'])
        flow_iat = (np.diff(all_ts) * US).tolist() if len(all_ts) > 1 else []
        duration_us = max((all_ts[-1] - flow['start_ts']) * US, 1.0)

        f_sum, f_mean, f_std, f_max, _ = _stats(fwd_len)
        _, b_mean, _, b_max, b_min = _stats(bwd_len)
        fi_sum, fi_mean, fi_std, fi_max, _ = _stats(fwd_iat)
        _, bi_mean, _, _, _ = _stats(bwd_iat)
        _, _, flow_iat_std, _, _ = _stats(flow_iat)
        all_len = fwd_len + bwd_len
        n_fwd, n_bwd = len(fwd_len), len(bwd_len)

        # Destination port of the flow = destination of the FIRST packet
        flow_dport = key[3]

        return {
            'Destination Port':            float(flow_dport),
            'Total Fwd Packets':           float(n_fwd),
            'Total Length of Fwd Packets': f_sum,
            'Fwd Packet Length Max':       f_max,
            'Fwd Packet Length Mean':      f_mean,
            'Fwd Packet Length Std':       f_std,
            'Bwd Packet Length Max':       b_max,
            'Bwd Packet Length Min':       b_min,
            'Bwd Packet Length Mean':      b_mean,
            'Min Packet Length':           float(min(all_len)),
            'Packet Length Mean':          float(np.mean(all_len)),
            'Average Packet Size':         float(np.mean(all_len)),
            'Fwd Packets/s':               n_fwd / (duration_us / US),
            'Fwd Header Length':           float(flow['fwd_hdr']),
            'Fwd Header Length.1':         float(flow['fwd_hdr']),
            'Fwd IAT Total':               fi_sum,
            'Fwd IAT Mean':                fi_mean,
            'Fwd IAT Std':                 fi_std,
            'Fwd IAT Max':                 fi_max,
            'Bwd IAT Mean':                bi_mean,
            'Flow IAT Std':                flow_iat_std,
            'Avg Fwd Segment Size':        f_mean,
            'Avg Bwd Segment Size':        b_mean,
            'Subflow Fwd Packets':         float(n_fwd),
            'Subflow Fwd Bytes':           f_sum,
            'Subflow Bwd Packets':         float(n_bwd),
            'Down/Up Ratio':               float(n_bwd // max(n_fwd, 1)),   # integer ratio as in the dataset
            'Init_Win_bytes_forward':      float(flow['win_fwd'] if flow['win_fwd'] is not None else -1),
            'Init_Win_bytes_backward':     float(flow['win_bwd'] if flow['win_bwd'] is not None else -1),
            'act_data_pkt_fwd':            float(flow['fwd_data_pkts']),
        }
    except Exception:
        return None


def build_feature_row(feature_dict: dict) -> dict:
    """Fill missing columns with 0 and return an ordered dict."""
    row = {col: 0.0 for col in SELECTED_COLUMNS}
    for k, v in feature_dict.items():
        if k in row:
            row[k] = float(v) if v is not None else 0.0
    return row


def reset_flow_cache() -> None:
    _flow_cache.clear()
