"""
network/sniffer.py
Packet sniffing module — captures live packets and pushes results to a queue.
"""
from scapy.all import sniff
from network.features import extract_features, build_feature_row
from core.predictor import predict_features

# ── Higher threshold for LIVE traffic to avoid false positives ───────────────
# Simulator packets are injected directly so this only affects real packets
_LIVE_ATTACK_THRESHOLD = 0.75


def make_packet_handler(result_queue):
    """Returns a packet handler that pushes prediction results to queue."""

    def process_packet(packet):
        try:
            raw_features = extract_features(packet)
            if raw_features is None:
                return

            feature_row = build_feature_row(raw_features)
            result = predict_features(feature_row)

            # ── For live traffic use stricter threshold to avoid FP ───────
            # Simulator injects directly — never goes through here
            if result['final_score'] < _LIVE_ATTACK_THRESHOLD:
                result['label'] = 'NORMAL'

            result['src_ip']   = packet["IP"].src if packet.haslayer("IP") else "?"
            result['dst_ip']   = packet["IP"].dst if packet.haslayer("IP") else "?"
            result['dst_port'] = int(raw_features.get('Destination Port', 0))
            result['proto']    = ('TCP' if packet.haslayer('TCP')
                                  else 'UDP' if packet.haslayer('UDP') else 'OTHER')
            result['pkt_len']  = len(packet)
            result['features'] = feature_row

        except Exception:
            pass
        else:
            result_queue.put(result)

    return process_packet


def start_sniffing(result_queue, stop_event):
    """Blocking call — runs until stop_event is set."""
    handler = make_packet_handler(result_queue)
    while not stop_event.is_set():
        sniff(prn=handler, store=False, timeout=2,
              filter="ip and (tcp or udp)")