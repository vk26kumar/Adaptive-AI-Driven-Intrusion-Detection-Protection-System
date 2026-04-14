"""
network/sniffer.py
Packet sniffing module — captures live packets and pushes results to a queue.

ROOT CAUSE OF FALSE POSITIVES (fixed here):
--------------------------------------------
The autoencoder was trained on CIC-IDS2017 lab data from 2017. Real-world
traffic in 2026 looks structurally different — different OS TCP stacks,
different application patterns, different timing. The autoencoder has never
seen this traffic, so reconstruction error is always high → DL Score = 1.000
for nearly every real packet, regardless of whether it's an attack.

Meanwhile, XGBoost (ML Prob) correctly says 0.0% for all normal packets
because it was trained with supervised labels and generalises better.

FIX — For live sniffer traffic, we use XGBoost as the primary gate:
  - If ML Prob < 0.3  → NORMAL (XGBoost is confident it's benign)
  - If ML Prob >= 0.3 → use the full ensemble score with threshold 0.75

This means the autoencoder's anomaly signal only matters when XGBoost
is already suspicious. This completely eliminates the false positives
caused by the autoencoder misfiring on modern real-world traffic.

The Attack Simulator bypasses this entirely (injects directly into session
state) so simulator predictions are unaffected.
"""
from scapy.all import sniff
from network.features import extract_features, build_feature_row
from core.predictor import predict_features

# Thresholds for live traffic only (simulator bypasses sniffer entirely)
_ML_GATE_THRESHOLD   = 0.30   # XGBoost must be >= 30% confident before we consider DL
_LIVE_SCORE_THRESHOLD = 0.75  # Even then, final ensemble score must exceed 0.75


def make_packet_handler(result_queue):
    """Returns a packet handler that pushes prediction results to queue."""

    def process_packet(packet):
        try:
            raw_features = extract_features(packet)
            if raw_features is None:
                return

            feature_row = build_feature_row(raw_features)
            result = predict_features(feature_row)

            ml_prob    = result.get('ml_prob', 0.0)
            final_score = result.get('final_score', 0.0)

            # ── Live traffic gate ─────────────────────────────────────────
            # XGBoost is the primary gate for real traffic.
            # The autoencoder alone cannot reliably distinguish modern
            # real-world traffic from attacks — it was trained on 2017
            # lab data and produces high reconstruction error for any
            # traffic it hasn't seen before.
            #
            # Rule: ATTACK only if BOTH conditions hold:
            #   1. XGBoost is suspicious (ml_prob >= 0.30)
            #   2. Final ensemble score >= 0.75
            if not (ml_prob >= _ML_GATE_THRESHOLD and final_score >= _LIVE_SCORE_THRESHOLD):
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