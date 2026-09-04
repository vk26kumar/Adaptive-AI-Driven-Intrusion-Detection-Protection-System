"""
core/persistence.py
Persistent event store (SQLite) so verdicts, alerts and prevention actions
survive a dashboard restart -- the "no persistent logging" limitation of Phase 1.

    from core.persistence import store
    store.log_flow(result_dict, source="live")
    store.recent_alerts(20)
"""
import json
import os
import sqlite3
import threading
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.environ.get("IDS_DB_PATH", os.path.join(_ROOT, "logs", "ids_events.sqlite"))

_SCHEMA = """
CREATE TABLE IF NOT EXISTS flows (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL, source TEXT, src_ip TEXT, dst_ip TEXT, dst_port INTEGER, proto TEXT,
    label TEXT, final_score REAL, ml_prob REAL, dl_score REAL, latency_ms REAL,
    attack_type TEXT, truth TEXT, features TEXT
);
CREATE INDEX IF NOT EXISTS flows_ts ON flows(ts);
CREATE INDEX IF NOT EXISTS flows_label ON flows(label);
CREATE TABLE IF NOT EXISTS actions (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL, action TEXT, ip TEXT, detail TEXT
);
"""


class EventStore:
    def __init__(self, path: str = DB_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        self._lock = threading.Lock()
        with self._conn() as c:
            c.executescript(_SCHEMA)

    def _conn(self):
        return sqlite3.connect(self.path, timeout=5, check_same_thread=False)

    def log_flow(self, r: dict, source: str = "live", attack_type: str = None, truth: str = None) -> None:
        row = (time.time(), source, r.get("src_ip"), r.get("dst_ip"), r.get("dst_port"), r.get("proto"),
               r.get("label"), r.get("final_score"), r.get("ml_prob"), r.get("dl_score"), r.get("latency_ms"),
               attack_type, truth, json.dumps(r.get("features", {})) if r.get("label") == "ATTACK" else None)
        with self._lock, self._conn() as c:
            c.execute("INSERT INTO flows (ts,source,src_ip,dst_ip,dst_port,proto,label,final_score,ml_prob,"
                      "dl_score,latency_ms,attack_type,truth,features) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", row)

    def log_action(self, action: str, ip: str, detail: dict = None) -> None:
        with self._lock, self._conn() as c:
            c.execute("INSERT INTO actions (ts, action, ip, detail) VALUES (?,?,?,?)",
                      (time.time(), action, ip, json.dumps(detail or {})))

    def recent_alerts(self, n: int = 20):
        with self._conn() as c:
            return c.execute("SELECT ts, source, src_ip, dst_port, proto, final_score, attack_type FROM flows "
                             "WHERE label='ATTACK' ORDER BY ts DESC LIMIT ?", (n,)).fetchall()

    def counts(self) -> dict:
        with self._conn() as c:
            total, attacks = c.execute("SELECT COUNT(*), SUM(label='ATTACK') FROM flows").fetchone()
            actions = c.execute("SELECT COUNT(*) FROM actions").fetchone()[0]
        return {"flows": total or 0, "attacks": attacks or 0, "actions": actions or 0}


store = EventStore()
