"""
federated/metrics_logger.py
Append-only JSON log of every federated round.  The dashboard's
"Federated Nodes" panel and evaluate.py both read these files.
"""
import json
import os
import time

from federated.config import LOG_DIR


class RoundLogger:
    def __init__(self, run_name: str):
        os.makedirs(LOG_DIR, exist_ok=True)
        self.path = os.path.join(LOG_DIR, f"{run_name}_rounds.json")
        self.data = {"run": run_name, "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                     "rounds": [], "finished": None}
        self._flush()

    def log_round(self, server_round: int, **fields):
        entry = {"round": server_round, "time": time.strftime("%H:%M:%S")}
        entry.update({k: _plain(v) for k, v in fields.items()})
        self.data["rounds"].append(entry)
        self._flush()

    def finish(self, **fields):
        self.data["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self.data.update({k: _plain(v) for k, v in fields.items()})
        self._flush()

    def _flush(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2)


def _plain(v):
    """Make numpy scalars / dicts JSON serialisable."""
    if hasattr(v, "item"):
        return v.item()
    if isinstance(v, dict):
        return {k: _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    return v


def read_log(run_name: str):
    path = os.path.join(LOG_DIR, f"{run_name}_rounds.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)
