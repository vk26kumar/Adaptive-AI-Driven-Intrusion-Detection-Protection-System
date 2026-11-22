"""
network/sniffer.py
Live packet capture.  For every TCP/UDP packet: update the flow, extract the
30 features, run the hybrid model and push the result on the shared queue.

Phase 2: the Phase 1 "ML gate" (ATTACK only if XGBoost >= 0.30 AND score >=
0.75) is gone.  It existed to hide autoencoder false positives caused by the
unit mismatch in the old extractor, and it also made live detection
impossible because the DDoS-only classifier never fired on other attacks.
The retrained multi-attack model plus the corrected extractor make the normal
decision rule usable on live traffic.  A stricter cutoff can still be applied
with the IDS_LIVE_THRESHOLD environment variable (default 0.5).
"""
import os
import time

from scapy.all import sniff

from network.features import extract_features, build_feature_row, flow_key, _packet_time
from core.predictor import predict_features, THRESHOLDS
from core.scan_detector import ScanDetector, is_connection_attempt

LIVE_THRESHOLD = float(os.environ.get("IDS_LIVE_THRESHOLD", "0.5"))
# Scope: unicast TCP/UDP.  Every CIC-IDS2017 attack class is unicast; LAN
# discovery multicast/broadcast from other devices (mDNS 224.0.0.251, SSDP /
# WS-Discovery 239.255.255.250, 255.255.255.255) is outside what the model was
# trained to judge and was the bulk of live false positives on a shared Wi-Fi.
# The prevention engine never blocks such addresses either.  IDS_INCLUDE_MULTICAST=1
# restores the old behaviour.
def _local_broadcasts() -> list:
    """Subnet-directed broadcast addresses of the local networks (e.g. 172.20.7.255
    for 172.20.0.0/21), from the routing table Scapy reads."""
    try:
        from scapy.all import conf
        from scapy.utils import ltoa
        out = set()
        for net, mask, *_ in conf.route.routes:
            if 0 < mask < 0xFFFFFFFF and (net >> 24) not in (0, 127, 224):
                out.add(ltoa((net | (~mask & 0xFFFFFFFF)) & 0xFFFFFFFF))
        return sorted(out)
    except Exception:
        return []


if os.environ.get("IDS_INCLUDE_MULTICAST", "0") == "1":
    BPF_FILTER = "ip and (tcp or udp)"
else:
    BPF_FILTER = "ip and (tcp or udp) and not ip multicast and not dst host 255.255.255.255" + "".join(
        f" and not dst host {b}" for b in _local_broadcasts())

# Per-flow inference throttle.  One model call costs ~15-20 ms, i.e. ~60 calls/s,
# while a single bulk download delivers thousands of packets/s; classifying every
# packet made the sniffer fall behind and starve all other flows.  Flow
# statistics are still updated on EVERY packet; the model runs on a flow for its
# first EAGER_PACKETS packets (scans and short attacks are decided there) and
# then at most once per FLOW_INFER_INTERVAL_S.
EAGER_PACKETS = int(os.environ.get("IDS_EAGER_PACKETS", "10"))
FLOW_INFER_INTERVAL_S = float(os.environ.get("IDS_FLOW_INFER_INTERVAL", "1.0"))
_MAX_TRACKED = 20000


def should_classify(state: dict, key, now: float) -> bool:
    """Throttle decision for one packet of flow `key` (state is mutated)."""
    n, last = state.get(key, (0, 0.0))
    n += 1
    go = n <= EAGER_PACKETS or now - last >= FLOW_INFER_INTERVAL_S
    state[key] = (n, now if go else last)
    if len(state) > _MAX_TRACKED:                       # bounded memory
        for k in list(state)[: len(state) // 2]:
            del state[k]
    return go


def _scan_alert(packet, alert: dict, t0: float) -> dict:
    """Queue item for a behavioural port-scan alert (same shape as a model result)."""
    row = build_feature_row({"Destination Port": alert["highest_port"]})
    return {
        "label": "ATTACK", "final_score": 1.0, "confidence": 100.0,
        "ml_prob": 0.0, "dl_score": 0.0, "mse": 0.0,
        "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
        "src_ip": alert["src"], "dst_ip": alert["dst"], "dst_port": alert["highest_port"],
        "proto": "TCP" if packet.haslayer("TCP") else "UDP", "pkt_len": len(packet),
        "features": row, "source": "live",
        "attack_type": "PortScan",
        "detector": (f"behaviour: {alert['distinct_ports']} distinct ports of {alert['dst']} "
                     f"in {alert['window_s']:.0f} s (ports {alert['lowest_port']}-{alert['highest_port']})"),
    }


def make_packet_handler(result_queue, use_live_baseline: bool = True):
    throttle = {}
    scans = ScanDetector()
    # live-only autoencoder threshold learned on this network (network/baseline.py), if any
    if use_live_baseline:
        from network.baseline import load_live_thresholds
        cfg = load_live_thresholds(THRESHOLDS)
    else:
        cfg = THRESHOLDS

    def _put(item):
        try:
            result_queue.put_nowait(item)
        except Exception:
            pass  # queue full: drop rather than block the capture thread

    def process_packet(packet):
        try:
            t0 = time.perf_counter()
            if packet.haslayer("IP") and is_connection_attempt(packet):     # every packet, before throttling
                l4 = packet["TCP"] if packet.haslayer("TCP") else packet["UDP"]
                alert = scans.observe(packet["IP"].src, packet["IP"].dst, int(l4.dport), _packet_time(packet))
                if alert:
                    _put(_scan_alert(packet, alert, t0))
            raw = extract_features(packet)          # always: keeps flow statistics exact
            if raw is None:
                return
            if not should_classify(throttle, flow_key(packet), time.monotonic()):
                return
            row = build_feature_row(raw)
            result = predict_features(row, cfg)
            if result["final_score"] <= LIVE_THRESHOLD:
                result["label"] = "NORMAL"
            result["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
            # the FLOW's client -> server direction, not this packet's: a scan probe
            # becomes classifiable on the victim's RST, and blaming the packet's
            # source would name (and could block) the victim
            result["src_ip"]   = raw.get("_src", packet["IP"].src)
            result["dst_ip"]   = raw.get("_dst", packet["IP"].dst)
            result["dst_port"] = int(raw["Destination Port"])
            result["proto"]    = "TCP" if packet.haslayer("TCP") else "UDP"
            result["pkt_len"]  = len(packet)
            result["features"] = row
            result["source"]   = "live"
        except Exception:
            return
        _put(result)

    return process_packet


def start_sniffing(result_queue, stop_event, iface=None, use_live_baseline: bool = True):
    """Blocking; returns when stop_event is set."""
    handler = make_packet_handler(result_queue, use_live_baseline)
    while not stop_event.is_set():
        sniff(prn=handler, store=False, timeout=2, filter=BPF_FILTER, iface=iface)
