"""
core/state.py
Shared in-memory state for the dashboard session.

Phase 2 change: the confusion matrix is only updated when the true label is
known (simulator injections and labelled replays).  Live packets have no
ground truth, so they are counted separately instead of being guessed from
the score as Phase 1 did.
"""
import threading
from dataclasses import dataclass, field
from queue import Queue
from typing import Dict, List, Optional

packet_queue: Queue = Queue(maxsize=500)
stop_event = threading.Event()

ATTACK_FAMILIES = ["DDoS", "DoS", "PortScan", "BruteForce", "Bot", "WebAttack", "Other"]


@dataclass
class Stats:
    total_packets: int = 0
    total_attacks: int = 0
    total_normal: int = 0
    unlabeled: int = 0            # live packets: no ground truth available
    cm_tp: int = 0
    cm_tn: int = 0
    cm_fp: int = 0
    cm_fn: int = 0
    attack_types: Dict[str, int] = field(default_factory=lambda: {k: 0 for k in ATTACK_FAMILIES})
    latencies: List[float] = field(default_factory=list)

    def record(self, label: str, final_score: float, latency_ms: float = 0.0,
               truth: Optional[str] = None) -> None:
        self.total_packets += 1
        if label == "ATTACK":
            self.total_attacks += 1
        else:
            self.total_normal += 1

        if truth is None:
            self.unlabeled += 1
        else:
            pred_attack = label == "ATTACK"
            true_attack = truth == "ATTACK"
            if pred_attack and true_attack:
                self.cm_tp += 1
            elif pred_attack and not true_attack:
                self.cm_fp += 1
            elif not pred_attack and true_attack:
                self.cm_fn += 1
            else:
                self.cm_tn += 1

        self.latencies.append(float(latency_ms))
        if len(self.latencies) > 500:
            self.latencies = self.latencies[-500:]

    @property
    def attack_rate(self) -> float:
        return self.total_attacks / self.total_packets * 100 if self.total_packets else 0.0

    @property
    def avg_latency(self) -> float:
        recent = self.latencies[-100:]
        return sum(recent) / len(recent) if recent else 0.0

    @property
    def labeled(self) -> int:
        return self.cm_tp + self.cm_tn + self.cm_fp + self.cm_fn

    def live_metrics(self) -> Dict[str, float]:
        """Accuracy / precision / recall / F1 over labelled samples only."""
        n = self.labeled
        if n == 0:
            return {}
        tp, fp, fn, tn = self.cm_tp, self.cm_fp, self.cm_fn, self.cm_tn
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        return {
            "Accuracy": (tp + tn) / n * 100,
            "Precision": prec * 100,
            "Recall": rec * 100,
            "F1-Score": (2 * prec * rec / (prec + rec) * 100) if prec + rec else 0.0,
        }


# -- Heuristic attack-type labelling for LIVE traffic ---------------------------
# The federated model is binary (ATTACK / NORMAL).  For live flows the attack
# family is inferred from port and rate; the simulator knows the family exactly.
_BRUTEFORCE_PORTS = {21, 22, 23, 3389, 5900}
_WEB_PORTS = {80, 443, 8080, 8443}
_IRC_PORTS = {6667, 6668, 6669, 8080}


def classify_attack_type(features: dict) -> str:
    port = int(features.get("Destination Port", 0))
    rate = float(features.get("Fwd Packets/s", 0.0))
    fwd_pkts = float(features.get("Total Fwd Packets", 0.0))
    fwd_len_mean = float(features.get("Fwd Packet Length Mean", 0.0))

    if port in _BRUTEFORCE_PORTS:
        return "BruteForce"
    if fwd_pkts <= 3 and fwd_len_mean <= 6 and port not in _WEB_PORTS:
        return "PortScan"                       # tiny SYN-style probes
    if port in _WEB_PORTS and rate > 1000:
        return "DDoS"
    if port in _WEB_PORTS and fwd_pkts >= 20:
        return "DoS"                            # slow HTTP style
    if port in _WEB_PORTS and fwd_len_mean > 200:
        return "WebAttack"
    if port in _IRC_PORTS:
        return "Bot"
    return "Other"
