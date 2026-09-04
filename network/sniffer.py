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

from network.features import extract_features, build_feature_row
from core.predictor import predict_features

LIVE_THRESHOLD = float(os.environ.get("IDS_LIVE_THRESHOLD", "0.5"))
BPF_FILTER = "ip and (tcp or udp)"


def make_packet_handler(result_queue):
    def process_packet(packet):
        try:
            t0 = time.perf_counter()
            raw = extract_features(packet)
            if raw is None:
                return
            row = build_feature_row(raw)
            result = predict_features(row)
            if result["final_score"] <= LIVE_THRESHOLD:
                result["label"] = "NORMAL"
            result["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
            result["src_ip"]   = packet["IP"].src
            result["dst_ip"]   = packet["IP"].dst
            result["dst_port"] = int(raw["Destination Port"])
            result["proto"]    = "TCP" if packet.haslayer("TCP") else "UDP"
            result["pkt_len"]  = len(packet)
            result["features"] = row
            result["source"]   = "live"
        except Exception:
            return
        try:
            result_queue.put_nowait(result)
        except Exception:
            pass  # queue full: drop rather than block the capture thread

    return process_packet


def start_sniffing(result_queue, stop_event, iface=None):
    """Blocking; returns when stop_event is set."""
    handler = make_packet_handler(result_queue)
    while not stop_event.is_set():
        sniff(prn=handler, store=False, timeout=2, filter=BPF_FILTER, iface=iface)
