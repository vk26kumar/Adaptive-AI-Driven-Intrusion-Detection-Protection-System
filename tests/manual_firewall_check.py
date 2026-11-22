"""Real Windows Firewall test of core/prevention.py: block -> verify -> undo -> verify.
Not collected by pytest (needs Administrator for the real block; without it the engine must
record the failure and not claim the block is active).

    python tests/manual_firewall_check.py <repo_dir> <result.json>     (from an elevated prompt)
"""
import ctypes
import json
import subprocess
import sys
import tempfile
import time

repo, out_path = sys.argv[1], sys.argv[2]
sys.path.insert(0, repo)
import core.prevention as P  # noqa: E402

tmp = tempfile.mkdtemp(prefix="aidaps_fw_")
P.LOG_DIR, P.STATE_PATH, P.AUDIT_PATH = tmp, tmp + "\\blocked.json", tmp + "\\audit.jsonl"

IP = "45.33.32.156"          # scanme.nmap.org: public host meant for security testing; inbound-only rule
res = {"admin": bool(ctypes.windll.shell32.IsUserAnAdmin()), "ip": IP}


def rule_exists(rule):
    p = subprocess.run(["netsh", "advfirewall", "firewall", "show", "rule", f"name={rule}"],
                       capture_output=True, text=True)
    return p.returncode == 0 and IP in p.stdout and "Block" in p.stdout, p.stdout.strip()[:600]


eng = P.PreventionEngine(enabled=True, dry_run=False)
recs = [eng.observe(IP, 0.99, "PortScan") for _ in range(P.MIN_HITS)]
rec = recs[-1]
res["fired_after_hits"] = next((i + 1 for i, r in enumerate(recs) if r), None)
res["record"] = {"mode": rec.mode, "active": rec.active, "error": rec.error, "rule": rec.rule} if rec else None
if rec:
    exists, text = rule_exists(rec.rule)
    res["rule_in_windows_firewall_after_block"] = exists
    res["netsh_show_rule"] = text
    # private LAN address must be refused even with 3 strong alerts
    lan = [eng.observe("192.168.1.50", 0.99, "DoS") for _ in range(P.MIN_HITS)]
    res["private_ip_blocked"] = any(lan)
    time.sleep(1)
    res["unblock_ok"] = eng.unblock(IP)
    res["rule_in_windows_firewall_after_undo"] = rule_exists(rec.rule)[0]
    res["state_after_undo"] = list(eng.blocked)
with open(P.AUDIT_PATH, encoding="utf-8") as f:
    res["audit_log"] = [json.loads(l) for l in f]
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(res, f, indent=2)
print(json.dumps(res, indent=2))
