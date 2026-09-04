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


def test_iat_in_microseconds():
    p = IP(src="10.0.0.1", dst="10.0.0.2") / TCP(sport=40002, dport=80) / Raw(b"a")
    F.extract_features(p)
    time.sleep(0.05)
    f = F.extract_features(p)
    assert 30_000 < f["Fwd IAT Total"] < 500_000       # ~50 ms expressed in us
    assert f["Fwd Packets/s"] < 100                    # 2 packets over ~50 ms


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
