"""
network/live_check.py
Live validation of the whole capture pipeline (Npcap/Scapy -> network.features
-> core.predictor), the same path the dashboard and main.py use.

    python -m network.live_check --scenario benign            # web traffic on the default NIC
    python -m network.live_check --scenario portscan          # TCP connect scan of THIS machine on loopback

benign   : captures on the default interface while making ordinary HTTPS
           requests; reports the share of packets/flows classified ATTACK
           (a live false-positive estimate).
portscan : captures on the Npcap loopback adapter while scanning 127.0.0.1
           ports 1..N with plain TCP connects (no other host is touched);
           reports whether, and how fast, the scanner's flows are flagged.
Writes logs/live_check_<scenario>.json.  Needs Npcap (https://npcap.com); with
"restrict to administrators" unchecked at install time no admin rights are needed.
"""
import argparse
import json
import os
import socket
import threading
import time
import urllib.request
from queue import Queue, Empty

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from network.sniffer import start_sniffing           # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOOPBACK_IFACE = r"\Device\NPF_Loopback"
BENIGN_URLS = ["https://www.wikipedia.org", "https://www.python.org", "https://pypi.org",
               "https://www.github.com", "https://www.unb.ca"]


def _benign_traffic(stop):
    while not stop.is_set():
        for u in BENIGN_URLS:
            try:
                urllib.request.urlopen(u, timeout=5).read(20000)
            except Exception:
                pass
            time.sleep(1.0)


def _portscan(ports, delay):
    for p in range(1, ports + 1):
        s = socket.socket(); s.settimeout(0.2)
        try:
            s.connect(("127.0.0.1", p))
        except OSError:
            pass
        finally:
            s.close()
        time.sleep(delay)


def run(scenario: str, seconds: float, ports: int, delay: float) -> dict:
    q, stop = Queue(maxsize=100000), threading.Event()
    iface = LOOPBACK_IFACE if scenario == "portscan" else None
    th = threading.Thread(target=start_sniffing, args=(q, stop, iface), daemon=True)
    th.start()
    time.sleep(3)                                        # let the capture start
    t0 = time.time()
    if scenario == "benign":
        gen_stop = threading.Event()
        g = threading.Thread(target=_benign_traffic, args=(gen_stop,), daemon=True); g.start()
        time.sleep(seconds); gen_stop.set()
    else:
        _portscan(ports, delay)
    time.sleep(3)
    stop.set(); th.join(timeout=10)

    rows = []
    while True:
        try:
            rows.append(q.get_nowait())
        except Empty:
            break
    flows = {}
    for r in rows:
        k = (r["src_ip"], r["dst_ip"], r["dst_port"], r["proto"])
        f = flows.setdefault(k, {"packets": 0, "attack_packets": 0, "max_score": 0.0})
        f["packets"] += 1
        f["attack_packets"] += r["label"] == "ATTACK"
        f["max_score"] = max(f["max_score"], r["final_score"])
    n_flows = len(flows)
    flagged = sum(1 for f in flows.values() if f["attack_packets"])
    lat = sorted(r["latency_ms"] for r in rows)
    out = {"scenario": scenario, "interface": iface or "default", "duration_s": round(time.time() - t0, 1),
           "packets": len(rows), "packets_attack": sum(r["label"] == "ATTACK" for r in rows),
           "flows": n_flows, "flows_flagged": flagged,
           "latency_ms_median": lat[len(lat) // 2] if lat else None,
           "latency_ms_p95": lat[int(len(lat) * 0.95)] if lat else None,
           "flagged_flows": [{"src": k[0], "dst": k[1], "dst_port": k[2], "proto": k[3], **v}
                             for k, v in flows.items() if v["attack_packets"]][:50]}
    if scenario == "portscan":
        model_rows = [r for r in rows if not r.get("detector")]
        alerts = [r for r in rows if r.get("detector")]
        probe_keys = {(r["src_ip"], r["dst_ip"], r["dst_port"]) for r in model_rows
                      if r["dst_ip"] == "127.0.0.1" and 1 <= r["dst_port"] <= ports}
        out["ports_scanned"] = ports
        out["probe_flows_classified_by_model"] = len(probe_keys)
        out["probe_flows_flagged_by_model"] = len({(r["src_ip"], r["dst_ip"], r["dst_port"]) for r in model_rows
                                                   if r["label"] == "ATTACK" and (r["src_ip"], r["dst_ip"], r["dst_port"]) in probe_keys})
        out["behaviour_alerts"] = [r["detector"] for r in alerts]
        out["scan_detected"] = bool(alerts)
    os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
    with open(os.path.join(ROOT, "logs", f"live_check_{scenario}.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=["benign", "portscan"], required=True)
    ap.add_argument("--seconds", type=float, default=30, help="benign: how long to generate traffic")
    ap.add_argument("--ports", type=int, default=300, help="portscan: scan ports 1..N on 127.0.0.1")
    ap.add_argument("--delay", type=float, default=0.01, help="portscan: seconds between connects")
    a = ap.parse_args()
    print(json.dumps(run(a.scenario, a.seconds, a.ports, a.delay), indent=2))


if __name__ == "__main__":
    main()
