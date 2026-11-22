"""
tests/test_features.py
Unit tests for the live flow extractor: units and direction handling must
match the CIC-IDS2017 conventions the model was trained on.
"""
import os
import sys
import time

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

scapy = pytest.importorskip("scapy.all")
from scapy.all import IP, TCP, UDP, Raw  # noqa: E402

from network import features as F  # noqa: E402


@pytest.fixture(autouse=True)
def clean_cache():
    F.reset_flow_cache()
    yield
    F.reset_flow_cache()


def test_first_packet_returns_none_and_second_returns_features():
    p = IP(src="10.0.0.1", dst="10.0.0.2") / TCP(sport=40000, dport=80, window=64240) / Raw(b"x" * 100)
    assert F.extract_features(p) is None
    f = F.extract_features(p)
    assert f is not None
    assert f["Total Fwd Packets"] == 2
    assert f["Destination Port"] == 80
    assert f["Init_Win_bytes_forward"] == 64240
    assert f["Init_Win_bytes_backward"] == -1          # no reply seen yet


def test_payload_not_frame_length():
    p = IP(src="10.0.0.1", dst="10.0.0.2") / TCP(sport=40001, dport=443) / Raw(b"x" * 100)
    F.extract_features(p)
    f = F.extract_features(p)
    assert f["Fwd Packet Length Mean"] == 100          # payload bytes, not the 154-byte frame
    assert f["Total Length of Fwd Packets"] == 200
    assert f["Fwd Header Length"] == 40                 # 2 x 20-byte TCP headers


def _at(pkt, t):
    pkt.time = t          # capture timestamp, as Npcap sets it
    return pkt


def test_iat_in_microseconds():
    p = lambda: IP(src="10.0.0.1", dst="10.0.0.2") / TCP(sport=40002, dport=80) / Raw(b"a")
    F.extract_features(_at(p(), 1000.0))
    f = F.extract_features(_at(p(), 1000.05))
    assert abs(f["Fwd IAT Total"] - 50_000) < 1        # 50 ms expressed in us
    assert abs(f["Fwd Packets/s"] - 40) < 0.01         # 2 packets over 50 ms


def test_uses_capture_time_not_processing_time():
    """Packets captured 200 ms apart but processed back to back (a backlog)
    must keep their real spacing; Phase 2 used processing time and saw a
    handshake as millions of packets per second."""
    p = lambda: IP(src="10.0.0.1", dst="10.0.0.2") / TCP(sport=40010, dport=443)
    F.extract_features(_at(p(), 2000.0))
    f = F.extract_features(_at(p(), 2000.2))
    assert abs(f["Fwd Packets/s"] - 10) < 0.01


def test_active_timeout_starts_a_new_flow():
    p = lambda: IP(src="10.0.0.1", dst="10.0.0.2") / TCP(sport=40011, dport=443) / Raw(b"z" * 10)
    for i in range(5):
        F.extract_features(_at(p(), 3000.0 + i))
    F.extract_features(_at(p(), 3000.0 + F.FLOW_ACTIVE_TIMEOUT_S + 5))     # past 120 s: new record
    f = F.extract_features(_at(p(), 3000.0 + F.FLOW_ACTIVE_TIMEOUT_S + 6))
    assert f["Total Fwd Packets"] == 2


def test_counts_do_not_saturate_and_std_is_sample_std():
    p = lambda n: IP(src="10.0.0.1", dst="10.0.0.2") / TCP(sport=40012, dport=80) / Raw(b"x" * n)
    for i in range(5000):
        f = F.extract_features(_at(p(10 if i % 2 else 30), 4000.0 + i * 0.001))
    assert f["Total Fwd Packets"] == 5000
    assert f["Total Length of Fwd Packets"] == 5000 * 20
    import statistics
    assert abs(f["Fwd Packet Length Std"] - statistics.stdev([10, 30] * 2500)) < 1e-6


def test_reverse_direction_counts_as_backward():
    fwd = IP(src="10.0.0.1", dst="10.0.0.2") / TCP(sport=40003, dport=22, window=1000) / Raw(b"a" * 10)
    rev = IP(src="10.0.0.2", dst="10.0.0.1") / TCP(sport=22, dport=40003, window=2000) / Raw(b"b" * 30)
    F.extract_features(fwd)
    F.extract_features(rev)
    f = F.extract_features(fwd)
    assert f["Total Fwd Packets"] == 2
    assert f["Subflow Bwd Packets"] == 1
    assert f["Bwd Packet Length Mean"] == 30
    assert f["Init_Win_bytes_backward"] == 2000
    assert f["Destination Port"] == 22                  # port of the first packet's destination
    assert f["Down/Up Ratio"] == 0                      # 1 // 2


def test_udp_window_is_minus_one():
    p = IP(src="10.0.0.1", dst="8.8.8.8") / UDP(sport=50000, dport=53) / Raw(b"q" * 40)
    F.extract_features(p)
    f = F.extract_features(p)
    assert f["Init_Win_bytes_forward"] == -1
    assert f["Fwd Header Length"] == 16


def test_cache_is_bounded():
    for i in range(F.MAX_FLOWS + 50):
        p = IP(src="10.0.0.1", dst="10.0.1.%d" % (i % 250)) / TCP(sport=1000 + i, dport=80)
        F.extract_features(p)
    assert len(F._flow_cache) <= F.MAX_FLOWS


def test_flow_already_open_at_capture_start_is_oriented_by_port():
    """First captured packet is server -> client mid-download (ACK, no SYN):
    the flow must still be client -> server:443, not a flood towards the
    client's ephemeral port."""
    F.reset_flow_cache()
    down = IP(src="3.5.254.213", dst="10.0.0.5") / TCP(sport=443, dport=58807, flags="A") / Raw(b"d" * 1460)
    up = IP(src="10.0.0.5", dst="3.5.254.213") / TCP(sport=58807, dport=443, flags="A")
    F.extract_features(down); F.extract_features(up); F.extract_features(down)
    feats = F.extract_features(up)
    assert feats is not None
    assert feats["Destination Port"] == 443
    assert feats["Bwd Packet Length Mean"] == 1460          # the download is backward traffic
    assert feats["Total Length of Fwd Packets"] == 0        # client only sends ACKs


def test_inference_throttle_eager_then_rate_limited():
    from network.sniffer import should_classify, EAGER_PACKETS, FLOW_INFER_INTERVAL_S
    state, t = {}, 100.0
    assert all(should_classify(state, "k", t) for _ in range(EAGER_PACKETS))   # first packets: always
    assert not should_classify(state, "k", t + 0.01)                          # bulk: throttled
    assert should_classify(state, "k", t + FLOW_INFER_INTERVAL_S + 0.01)      # but re-checked regularly
    assert should_classify(state, "other", t)                                 # per flow, not global


def test_syn_ack_as_first_packet_is_oriented_to_the_server_port():
    synack = IP(src="93.184.216.34", dst="10.0.0.5") / TCP(sport=443, dport=51000, flags="SA")
    ack = IP(src="10.0.0.5", dst="93.184.216.34") / TCP(sport=51000, dport=443, flags="A")
    F.extract_features(_at(synack, 5000.0)); F.extract_features(_at(ack, 5000.01))
    f = F.extract_features(_at(ack.copy(), 5000.02))
    assert f["Destination Port"] == 443


def test_port_scan_probe_syn_rst_is_classified():
    """One SYN answered by one RST is a complete port-scan probe and must
    produce features (Phase 2 waited for a 2nd forward packet that never comes)."""
    syn = IP(src="10.0.0.9", dst="10.0.0.5") / TCP(sport=51515, dport=8081, flags="S")
    rst = IP(src="10.0.0.5", dst="10.0.0.9") / TCP(sport=8081, dport=51515, flags="RA")
    assert F.extract_features(_at(syn, 6000.0)) is None
    f = F.extract_features(_at(rst, 6000.00005))
    assert f is not None and f["Total Fwd Packets"] == 1 and f["Destination Port"] == 8081
