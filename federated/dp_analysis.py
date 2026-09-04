"""
federated/dp_analysis.py
Privacy budget (epsilon, delta) for the differential-privacy runs and the
accuracy each one reached.

    python -m federated.dp_analysis

Mechanism: Flower's DifferentialPrivacyServerSideFixedClipping clips every
client update to L2 norm <= C and adds Gaussian noise N(0, (z*C/n)^2) to the
average of n client updates, which is the standard Gaussian mechanism with
noise multiplier z on a query of sensitivity C/n (client-level DP, all n
clients participate every round, no subsampling).

Accounting (Mironov 2017, Renyi DP):
    one round:   eps_RDP(alpha) = alpha / (2 z^2)
    R rounds:    eps_RDP(alpha) = R * alpha / (2 z^2)
    to (eps, delta):  eps = min_alpha [ R*alpha/(2 z^2) + log(1/delta)/(alpha-1) ]

Writes federated/logs/dp_analysis.json and prints a table.
"""
import glob
import json
import math
import os
import re

from federated import config as C
from federated.metrics_logger import read_log

DELTA = 1e-5


def epsilon(noise_multiplier: float, rounds: int, delta: float = DELTA) -> float:
    if noise_multiplier <= 0:
        return math.inf
    best = math.inf
    for alpha in [1.0 + x / 100.0 for x in range(1, 100)] + list(range(2, 512)):
        eps = rounds * alpha / (2.0 * noise_multiplier ** 2) + math.log(1.0 / delta) / (alpha - 1.0)
        best = min(best, eps)
    return best


def _parse_setting(fname: str):
    m = re.search(r"noise([0-9.]+)_clip([0-9.]+)(?:_r(\d+))?", fname)
    if not m:
        return None
    return float(m.group(1)), float(m.group(2))


def main():
    rows = []
    base = read_log("ae")
    if base and base["rounds"]:
        last = base["rounds"][-1]
        rows.append({"setting": "no DP", "noise_multiplier": 0.0, "clipping_norm": None,
                     "rounds": len(base["rounds"]) - 1, "epsilon": math.inf, "delta": DELTA,
                     "benign_mse": last["benign_mse_mean"], "ae_recall": last["recall"], "ae_fpr": last["fpr"]})
    for path in sorted(glob.glob(os.path.join(C.LOG_DIR, "ae_dp_noise*_rounds.json"))):
        st = _parse_setting(os.path.basename(path))
        if not st:
            continue
        with open(path, encoding="utf-8") as f:
            log = json.load(f)
        r = [x for x in log["rounds"] if x["round"] > 0]
        if not r:
            continue
        last = r[-1]
        z, clip = st
        rows.append({"setting": f"noise {z}, clip {clip}, {len(r)} rounds", "noise_multiplier": z,
                     "clipping_norm": clip, "rounds": len(r), "epsilon": epsilon(z, len(r)), "delta": DELTA,
                     "benign_mse": last["benign_mse_mean"], "ae_recall": last["recall"], "ae_fpr": last["fpr"]})

    out = {"delta": DELTA, "num_clients": C.NUM_NODES,
           "note": ("Client-level (eps, delta)-DP via RDP accounting of the Gaussian mechanism with all "
                    "clients participating each round; smaller epsilon = stronger privacy. "
                    f"noise std per weight = noise_multiplier * clipping_norm / {C.NUM_NODES}."),
           "runs": rows}
    os.makedirs(C.LOG_DIR, exist_ok=True)
    with open(os.path.join(C.LOG_DIR, "dp_analysis.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"Client-level DP, delta={DELTA}, {C.NUM_NODES} clients\n")
    print("| Setting | epsilon | benign MSE | AE recall | AE FPR |")
    print("|---|---|---|---|---|")
    for r in rows:
        eps = "inf" if math.isinf(r["epsilon"]) else f"{r['epsilon']:.1f}"
        print(f"| {r['setting']} | {eps} | {r['benign_mse']:.2e} | {r['ae_recall']:.3f} | {r['ae_fpr']:.4f} |")
    print("\nReading: eps < 1 strong privacy, 1-10 moderate, > 10 weak, > 100 essentially none.")
    print("Written federated/logs/dp_analysis.json")


if __name__ == "__main__":
    main()
