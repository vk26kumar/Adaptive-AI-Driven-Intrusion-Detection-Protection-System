"""
core/scan_detector.py
Behavioural port-scan detector for live traffic (vertical scans: one source
probing many destination ports of one host).

Why a rule next to the ML model: live testing (network/live_check.py) showed
that the per-flow model does not recognise scanning BEHAVIOUR but the
CIC-IDS2017 testbed: 99.9% of its PortScan flows carry the attacker's TCP window
(29200 = Linux/Kali default, or 1024 = nmap SYN scan) and a 6-byte Ethernet
padding on the RST.  A connect() scan from Windows (window 65535) over loopback
(no padding) is a per-flow twin of a dataset PortScan flow, yet scores 0.009;
changing only those two fields flips it to 0.94.  A single probe flow carries no
evidence of scanning at all: the evidence is the NUMBER of distinct ports one
source touches in a short time, which no per-flow feature contains.  That is
what this detector counts (the idea of Snort's sfPortscan).

Policy: an alert when one source opens connection attempts (TCP SYN without
ACK, or UDP datagrams) to >= MIN_PORTS distinct ports of one destination host
within WINDOW_S seconds; then again every further ALERT_EVERY new ports, so a
sustained scan keeps feeding the prevention engine (which blocks after 3
alerts in 60 s).  Horizontal sweeps (one port, many hosts) are NOT flagged:
a browser legitimately contacts dozens of hosts on 443 within seconds.
"""
from __future__ import annotations

import os
from collections import defaultdict, deque

WINDOW_S = float(os.environ.get("IDS_SCAN_WINDOW_S", "10"))
MIN_PORTS = int(os.environ.get("IDS_SCAN_MIN_PORTS", "20"))
ALERT_EVERY = int(os.environ.get("IDS_SCAN_ALERT_EVERY", "20"))
_MAX_PAIRS = 50000


class ScanDetector:
    def __init__(self, window_s: float = WINDOW_S, min_ports: int = MIN_PORTS, alert_every: int = ALERT_EVERY):
        self.window_s, self.min_ports, self.alert_every = window_s, min_ports, alert_every
        self._events = defaultdict(deque)        # (src, dst) -> deque[(ts, port)]
        self._next_alert = {}                    # (src, dst) -> distinct-port count that triggers next alert

    def observe(self, src: str, dst: str, dport: int, ts: float) -> dict | None:
        """Record one connection attempt; return an alert dict when the source
        is scanning dst, else None."""
        key = (src, dst)
        ev = self._events[key]
        ev.append((ts, int(dport)))
        while ev and ts - ev[0][0] > self.window_s:
            ev.popleft()
        ports = {p for _, p in ev}
        need = self._next_alert.get(key, self.min_ports)
        if len(ev) and len(ports) < self.min_ports:
            self._next_alert[key] = self.min_ports          # scan stopped / window slid: reset
        if len(self._events) > _MAX_PAIRS:                   # bounded memory: drop the stalest half
            for k in list(self._events)[: _MAX_PAIRS // 2]:
                self._events.pop(k, None); self._next_alert.pop(k, None)
        if len(ports) >= need:
            self._next_alert[key] = len(ports) + self.alert_every
            return {"src": src, "dst": dst, "distinct_ports": len(ports), "window_s": self.window_s,
                    "lowest_port": min(ports), "highest_port": max(ports)}
        return None


def is_connection_attempt(packet) -> bool:
    """TCP SYN without ACK, or any UDP datagram."""
    try:
        if packet.haslayer("TCP"):
            f = int(packet["TCP"].flags)
            return bool(f & 0x02) and not f & 0x10
        return packet.haslayer("UDP")
    except Exception:
        return False
