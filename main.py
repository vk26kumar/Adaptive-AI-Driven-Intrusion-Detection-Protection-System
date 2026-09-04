"""
main.py -- headless live IDS (no dashboard).  Prints one line per classified flow.

    python main.py            (Administrator / root required for packet capture)
    IDS_LIVE_THRESHOLD=0.7 python main.py
"""
import os
import sys
import threading
import time
from queue import Empty

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.state import packet_queue, stop_event, classify_attack_type  # noqa: E402
from core.prevention import engine as prevention  # noqa: E402
from core.persistence import store  # noqa: E402
from network.sniffer import start_sniffing  # noqa: E402

# IDS_AUTOBLOCK=1 enables prevention (dry-run unless IDS_APPLY_FIREWALL=1)
prevention.enabled = os.environ.get("IDS_AUTOBLOCK", "0") == "1"
prevention.dry_run = os.environ.get("IDS_APPLY_FIREWALL", "0") != "1"


def main():
    t = threading.Thread(target=start_sniffing, args=(packet_queue, stop_event), daemon=True)
    t.start()
    print(f"A-IDAPS-FL headless IDS running (auto-block={prevention.enabled}, dry-run={prevention.dry_run}). Ctrl+C to stop.")
    try:
        while True:
            try:
                r = packet_queue.get(timeout=1.0)
            except Empty:
                continue
            tag = "ATTACK" if r["label"] == "ATTACK" else "normal"
            atype = classify_attack_type(r["features"]) if r["label"] == "ATTACK" else None
            extra = f" type={atype}" if atype else ""
            store.log_flow(r, source="live", attack_type=atype)
            if atype:
                blk = prevention.observe(r["src_ip"], r["final_score"], atype)
                if blk:
                    extra += f"  -> {'BLOCKED' if blk.mode != 'dry-run' else 'would block'} {blk.ip}"
            print(f"{time.strftime('%H:%M:%S')} {tag:6s} {r['src_ip']}:{r['dst_port']} {r['proto']} "
                  f"score={r['final_score']:.3f} ml={r['ml_prob']:.3f} dl={r['dl_score']:.3f} "
                  f"{r['latency_ms']:.1f}ms{extra}")
    except KeyboardInterrupt:
        stop_event.set()
        print("stopped.")


if __name__ == "__main__":
    main()
