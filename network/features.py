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
  * Flows never expired; the cache evicted an arbitrary key.

Final-phase fixes, found by live capture with Npcap (network/live_check.py):

  * Timestamps are the CAPTURE time of the packet (packet.time from Npcap), not
    the time Python processes it.  Packets processed in a burst all got
    near-identical times, so a normal TCP handshake looked like "2 packets in
    1 us = 2,000,000 packets/s" and was flagged.
  * Active timeout: like CICFlowMeter, a flow is closed after FLOW_ACTIVE_TIMEOUT_S
    (120 s) of total duration and a new one starts.  Without it a long download
    became one giant flow that no training record resembles.
  * Exact running statistics (count, sum, sum of squares, min, max) replace
    per-flow packet lists capped at 4096, which made counts saturate and rates
    wrong on long flows; per-packet cost is now O(1).  Standard deviations are
    sample standard deviations, as in CICFlowMeter (Apache commons-math).
  * A TCP flow whose first captured packet is not a pure SYN (a SYN-ACK or a
    mid-connection packet) was already open when capture started; it is
    oriented by port (lower port = service) instead of by that first packet,
    which may be server -> client.
"""
import math
import time
from collections import OrderedDict

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

MAX_FLOWS            = 5000     # bound on tracked flows
FLOW_TIMEOUT_S       = 120.0    # idle time after which a flow is dropped
FLOW_ACTIVE_TIMEOUT_S = 120.0   # CICFlowMeter flow timeout: max duration of one flow record
MIN_FLOW_PACKETS     = 2        # packets (either direction) before a flow is classified.
                                # Phase 2 required 2 FORWARD packets, which made port scans invisible:
                                # a probe is SYN -> RST, i.e. 1 forward + 1 backward packet
                                # (the CIC-IDS2017 PortScan medoid has Total Fwd Packets = 1).

_flow_cache: "OrderedDict[tuple, dict]" = OrderedDict()


class _Stat:
    """Running count / sum / sum of squares / min / max."""
    __slots__ = ("n", "s", "ss", "mn", "mx")

    def __init__(self):
        self.n, self.s, self.ss, self.mn, self.mx = 0, 0.0, 0.0, math.inf, -math.inf

    def add(self, v: float) -> None:
        self.n += 1; self.s += v; self.ss += v * v
        if v < self.mn: self.mn = v
        if v > self.mx: self.mx = v

    @property
    def mean(self) -> float:
        return self.s / self.n if self.n else 0.0

    @property
    def std(self) -> float:          # sample std (n - 1), 0 for fewer than 2 values
        if self.n < 2:
            return 0.0
        return math.sqrt(max(self.ss - self.s * self.s / self.n, 0.0) / (self.n - 1))

    @property
    def max(self) -> float:
        return self.mx if self.n else 0.0

    @property
    def min(self) -> float:
        return self.mn if self.n else 0.0


def _new_flow(ts: float) -> dict:
    return {
        'fwd_len': _Stat(), 'bwd_len': _Stat(), 'all_len': _Stat(),
        'fwd_iat': _Stat(), 'bwd_iat': _Stat(), 'flow_iat': _Stat(),
        'fwd_last': None, 'bwd_last': None, 'any_last': None,
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


def _packet_time(packet) -> float:
    t = getattr(packet, "time", None)
    try:
        return float(t) if t else time.time()
    except (TypeError, ValueError):
        return time.time()


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
        now = _packet_time(packet)

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
        payload_len = float(len(bytes(l4.payload)))
        sport, dport = int(l4.sport), int(l4.dport)

        fwd_key = (ip.src, ip.dst, sport, dport, proto)
        rev_key = (ip.dst, ip.src, dport, sport, proto)

        # Direction = first packet seen for the 5-tuple (CICFlowMeter), except
        # TCP flows already open at capture start: orient those by port.
        if fwd_key in _flow_cache:
            key, is_fwd = fwd_key, True
        elif rev_key in _flow_cache:
            key, is_fwd = rev_key, False
        else:
            fl_bits = int(l4.flags) if proto == 6 else 0
            mid_flow = proto == 6 and not (fl_bits & 0x02 and not fl_bits & 0x10)   # not a pure SYN (SYN-ACK = server side)
            key, is_fwd = (rev_key, False) if (mid_flow and sport < dport) else (fwd_key, True)
            _flow_cache[key] = _new_flow(now)
        flow = _flow_cache[key]
        if now - flow['start_ts'] > FLOW_ACTIVE_TIMEOUT_S:               # CICFlowMeter active timeout
            flow = _flow_cache[key] = _new_flow(now)
        _flow_cache.move_to_end(key)
        flow['last_ts'] = now

        if flow['any_last'] is not None:
            flow['flow_iat'].add((now - flow['any_last']) * 1e6)
        flow['any_last'] = now
        flow['all_len'].add(payload_len)
        if is_fwd:
            flow['fwd_len'].add(payload_len)
            if flow['fwd_last'] is not None:
                flow['fwd_iat'].add((now - flow['fwd_last']) * 1e6)
            flow['fwd_last'] = now
            flow['fwd_hdr'] += hdr_len
            if payload_len > 0:
                flow['fwd_data_pkts'] += 1
            if flow['win_fwd'] is None:
                flow['win_fwd'] = window
        else:
            flow['bwd_len'].add(payload_len)
            if flow['bwd_last'] is not None:
                flow['bwd_iat'].add((now - flow['bwd_last']) * 1e6)
            flow['bwd_last'] = now
            if flow['win_bwd'] is None:
                flow['win_bwd'] = window

        _expire(now)

        fl, bl, al = flow['fwd_len'], flow['bwd_len'], flow['all_len']
        if fl.n < 1 or fl.n + bl.n < MIN_FLOW_PACKETS:
            return None
        fi, bi = flow['fwd_iat'], flow['bwd_iat']
        duration_s = max(flow['any_last'] - flow['start_ts'], 1e-6)

        return {
            'Destination Port':            float(key[3]),       # destination of the flow's first packet
            'Total Fwd Packets':           float(fl.n),
            'Total Length of Fwd Packets': fl.s,
            'Fwd Packet Length Max':       fl.max,
            'Fwd Packet Length Mean':      fl.mean,
            'Fwd Packet Length Std':       fl.std,
            'Bwd Packet Length Max':       bl.max,
            'Bwd Packet Length Min':       bl.min,
            'Bwd Packet Length Mean':      bl.mean,
            'Min Packet Length':           al.min,
            'Packet Length Mean':          al.mean,
            'Average Packet Size':         al.mean,
            'Fwd Packets/s':               fl.n / duration_s,
            'Fwd Header Length':           float(flow['fwd_hdr']),
            'Fwd Header Length.1':         float(flow['fwd_hdr']),
            'Fwd IAT Total':               fi.s,
            'Fwd IAT Mean':                fi.mean,
            'Fwd IAT Std':                 fi.std,
            'Fwd IAT Max':                 fi.max,
            'Bwd IAT Mean':                bi.mean,
            'Flow IAT Std':                flow['flow_iat'].std,
            'Avg Fwd Segment Size':        fl.mean,
            'Avg Bwd Segment Size':        bl.mean,
            'Subflow Fwd Packets':         float(fl.n),
            'Subflow Fwd Bytes':           fl.s,
            'Subflow Bwd Packets':         float(bl.n),
            'Down/Up Ratio':               float(bl.n // max(fl.n, 1)),   # integer ratio as in the dataset
            'Init_Win_bytes_forward':      float(flow['win_fwd'] if flow['win_fwd'] is not None else -1),
            'Init_Win_bytes_backward':     float(flow['win_bwd'] if flow['win_bwd'] is not None else -1),
            'act_data_pkt_fwd':            float(flow['fwd_data_pkts']),
            # flow orientation (not model features; build_feature_row drops them):
            # the client -> server direction of the FLOW, not of this packet
            '_src': key[0], '_dst': key[1], '_sport': key[2], '_proto': key[4],
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


def flow_key(packet):
    """Direction-independent key of the flow a packet belongs to (or None)."""
    try:
        ip = packet["IP"]
        l4 = packet["TCP"] if packet.haslayer("TCP") else packet["UDP"]
        a, b = (ip.src, int(l4.sport)), (ip.dst, int(l4.dport))
        return (min(a, b), max(a, b), 6 if packet.haslayer("TCP") else 17)
    except Exception:
        return None


def reset_flow_cache() -> None:
    _flow_cache.clear()
