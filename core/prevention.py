"""
core/prevention.py
Autonomous intrusion PREVENTION: block the source of a confirmed attack at the
host firewall, with an allowlist, a dry-run mode and full undo.

Backends
  Windows : netsh advfirewall firewall add rule ... remoteip=<ip> action=block
  Linux   : iptables -I INPUT -s <ip> -j DROP      (root required)
  dry-run : records the decision without touching the firewall (default)

Policy
  * only block when final_score >= BLOCK_THRESHOLD and the source appeared in
    at least MIN_HITS alerts within WINDOW_S seconds (one noisy flow is not enough)
  * never block private/loopback addresses, the host's own addresses, anything
    in the allowlist, or simulator addresses
  * every action is appended to logs/prevention.jsonl and to the SQLite event
    store so the security analyst can audit and reverse it
"""
from __future__ import annotations

import ipaddress
import json
import os
import platform
import socket
import subprocess
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(_ROOT, "logs")
STATE_PATH = os.path.join(LOG_DIR, "blocked_ips.json")
AUDIT_PATH = os.path.join(LOG_DIR, "prevention.jsonl")
RULE_PREFIX = "AIDAPS_BLOCK_"

BLOCK_THRESHOLD = float(os.environ.get("IDS_BLOCK_THRESHOLD", "0.75"))
MIN_HITS = int(os.environ.get("IDS_BLOCK_MIN_HITS", "3"))
WINDOW_S = float(os.environ.get("IDS_BLOCK_WINDOW_S", "60"))
DEFAULT_ALLOWLIST = {"SIM:10.0.99.1"}


def _local_addresses() -> set[str]:
    addrs = {"127.0.0.1", "::1"}
    try:
        addrs.update(i[4][0] for i in socket.getaddrinfo(socket.gethostname(), None))
    except Exception:
        pass
    return addrs


def is_blockable(ip: str, allowlist: set[str]) -> tuple[bool, str]:
    if ip in allowlist or ip.startswith("SIM:"):
        return False, "allowlisted"
    try:
        a = ipaddress.ip_address(ip)
    except ValueError:
        return False, "not an IP address"
    if a.is_loopback or a.is_link_local or a.is_multicast or a.is_unspecified:
        return False, "loopback/link-local/multicast"
    if a.is_private:
        return False, "private address (LAN peers are never auto-blocked)"
    return True, "ok"


@dataclass
class BlockRecord:
    ip: str
    time: float
    score: float
    hits: int
    attack_type: str
    mode: str                 # "dry-run" | "windows" | "linux"
    rule: str = ""
    active: bool = True
    error: str = ""


@dataclass
class PreventionEngine:
    enabled: bool = False              # master switch (dashboard toggle)
    dry_run: bool = True               # True: decide + log only
    allowlist: set = field(default_factory=lambda: set(DEFAULT_ALLOWLIST) | _local_addresses())
    blocked: dict = field(default_factory=dict)   # ip -> BlockRecord
    _hits: dict = field(default_factory=lambda: defaultdict(deque))
    _lock: threading.Lock = field(default_factory=threading.Lock)

    # ---------------------------------------------------------------- policy --
    def observe(self, src_ip: str, final_score: float, attack_type: str = "Other") -> BlockRecord | None:
        """Feed every ATTACK verdict here. Returns a BlockRecord when a block happens."""
        if not self.enabled or final_score < BLOCK_THRESHOLD:
            return None
        now = time.time()
        with self._lock:
            q = self._hits[src_ip]
            q.append(now)
            while q and now - q[0] > WINDOW_S:
                q.popleft()
            hits = len(q)
            if hits < MIN_HITS or src_ip in self.blocked:
                return None
            ok, why = is_blockable(src_ip, self.allowlist)
            if not ok:
                self._audit({"event": "skip", "ip": src_ip, "reason": why, "score": final_score})
                return None
            return self._block(src_ip, final_score, hits, attack_type)

    # ---------------------------------------------------------------- actions --
    def _block(self, ip, score, hits, attack_type) -> BlockRecord:
        mode = "dry-run" if self.dry_run else platform.system().lower()
        rec = BlockRecord(ip=ip, time=time.time(), score=score, hits=hits, attack_type=attack_type, mode=mode,
                          rule=f"{RULE_PREFIX}{ip.replace('.', '_').replace(':', '_')}")
        if not self.dry_run:
            rec.error = _firewall_block(ip, rec.rule)
            if rec.error:
                rec.active = False
        self.blocked[ip] = rec
        self._audit({"event": "block", **asdict(rec)})
        self._save()
        return rec

    def unblock(self, ip: str) -> bool:
        with self._lock:
            rec = self.blocked.get(ip)
            if not rec:
                return False
            err = "" if (rec.mode == "dry-run" or not rec.active) else _firewall_unblock(ip, rec.rule)
            rec.active = False
            rec.error = err
            self._audit({"event": "unblock", "ip": ip, "error": err})
            del self.blocked[ip]
            self._save()
            return not err

    def unblock_all(self) -> None:
        for ip in list(self.blocked):
            self.unblock(ip)

    # ---------------------------------------------------------------- persistence
    def _save(self):
        os.makedirs(LOG_DIR, exist_ok=True)
        with open(STATE_PATH, "w", encoding="utf-8") as f:
            json.dump({ip: asdict(r) for ip, r in self.blocked.items()}, f, indent=2)

    def load(self):
        if os.path.exists(STATE_PATH):
            with open(STATE_PATH, encoding="utf-8") as f:
                self.blocked = {ip: BlockRecord(**r) for ip, r in json.load(f).items()}
        return self

    @staticmethod
    def _audit(entry: dict):
        os.makedirs(LOG_DIR, exist_ok=True)
        entry = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), **entry}
        with open(AUDIT_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")


# -------------------------------------------------------------------- firewall --
def _run(cmd: list[str]) -> str:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        return "" if p.returncode == 0 else (p.stderr or p.stdout).strip()[:300]
    except Exception as e:  # noqa: BLE001
        return str(e)


def _firewall_block(ip: str, rule: str) -> str:
    sysname = platform.system().lower()
    if sysname == "windows":
        return _run(["netsh", "advfirewall", "firewall", "add", "rule", f"name={rule}", "dir=in",
                     "action=block", f"remoteip={ip}", "enable=yes"])
    if sysname == "linux":
        return _run(["iptables", "-I", "INPUT", "-s", ip, "-j", "DROP", "-m", "comment", "--comment", rule])
    return f"unsupported platform {sysname}"


def _firewall_unblock(ip: str, rule: str) -> str:
    sysname = platform.system().lower()
    if sysname == "windows":
        return _run(["netsh", "advfirewall", "firewall", "delete", "rule", f"name={rule}"])
    if sysname == "linux":
        return _run(["iptables", "-D", "INPUT", "-s", ip, "-j", "DROP", "-m", "comment", "--comment", rule])
    return f"unsupported platform {sysname}"


# module-level singleton used by the dashboard and main.py
engine = PreventionEngine().load()
