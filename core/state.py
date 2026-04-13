"""
core/state.py
Shared in-memory state for the Streamlit dashboard session.
All mutable state lives here so it is accessible across components.
"""
import threading
from queue import Queue
from dataclasses import dataclass, field
from typing import List, Dict

# ── Thread-safe packet queue ─────────────────────────────────────────────────
packet_queue: Queue = Queue(maxsize=500)
stop_event          = threading.Event()

# ── Session counters ─────────────────────────────────────────────────────────
@dataclass
class Stats:
    total_packets: int = 0
    total_attacks: int = 0
    total_normal:  int = 0
    false_positive: int = 0
    cm_tp: int = 0
    cm_tn: int = 0
    cm_fp: int = 0
    cm_fn: int = 0
    attack_types: Dict[str, int] = field(default_factory=lambda: {
        'DDoS': 0, 'PortScan': 0, 'BruteForce': 0, 'SQLInject': 0, 'Other': 0
    })
    latencies: List[float] = field(default_factory=list)

    def record(self, label: str, final_score: float, latency_ms: float = 0):
        self.total_packets += 1
        if label == "ATTACK":
            self.total_attacks += 1
            self.cm_tp += 1
        else:
            self.total_normal += 1
            if final_score > 0.35:
                self.cm_fn += 1
            elif final_score > 0.15:
                self.cm_fp += 1
            else:
                self.cm_tn += 1
        self.latencies.append(latency_ms)
        if len(self.latencies) > 200:
            self.latencies = self.latencies[-200:]

    @property
    def attack_rate(self) -> float:
        return (self.total_attacks / self.total_packets * 100
                if self.total_packets else 0.0)

    @property
    def avg_latency(self) -> float:
        return sum(self.latencies[-50:]) / max(len(self.latencies[-50:]), 1)


# ── Ports that identify attack types ─────────────────────────────────────────
_BRUTEFORCE_PORTS = {22, 23, 21, 3389, 5900}
_SQLINJECT_PORTS  = {3306, 5432, 1433, 1521}


def classify_attack_type(features: dict) -> str:
    """
    Classify attack type from feature dict.
    'Fwd Packets/s' here is the DATASET-SCALE rate (already × 40 for simulator,
    or actual pkts/s for live sniffer).
    Thresholds match simulator.py internal effective_rate values:
      DDoS       : effective_rate >= 40,000  (slider > 1000, scaled = 40,000+)
      BruteForce : port in SSH/RDP/FTP ports
      SQLInject  : port in DB ports
      PortScan   : effective_rate >= 4,000   (slider > 100)
    """
    port = int(features.get('Destination Port', 0))
    rate = float(features.get('Fwd Packets/s', 0))

    # DDoS: very high rate (slider > 1000 → scaled > 40,000)
    if rate > 40_000:
        return 'DDoS'
    # BruteForce: SSH, Telnet, FTP, RDP, VNC ports
    if port in _BRUTEFORCE_PORTS:
        return 'BruteForce'
    # SQLInject: database ports
    if port in _SQLINJECT_PORTS:
        return 'SQLInject'
    # PortScan: moderate rate (slider > 100 → scaled > 4,000)
    if rate > 4_000:
        return 'PortScan'
    return 'Other'