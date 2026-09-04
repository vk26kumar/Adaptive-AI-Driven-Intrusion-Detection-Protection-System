"""
federated/comm_analysis.py
Communication overhead and convergence speed of federated training versus
shipping raw traffic to a central server.

    python -m federated.comm_analysis

Reads federated/logs/{ae,xgb}_rounds.json and federated/data/summary.json.
Writes federated/logs/comm_analysis.json and prints a table.
"""
import json
import os

from federated import config as C
from federated.metrics_logger import read_log

BYTES_PER_FEATURE = 4      # float32
FEATURES = len(C.SELECTED_COLUMNS)


def rounds_to_reach(rounds, key, fraction=0.99):
    """First round at which `key` reaches `fraction` of its final value."""
    vals = [(r["round"], r.get(key, 0.0)) for r in rounds if r["round"] > 0]
    if not vals:
        return None
    final = vals[-1][1]
    for rnd, v in vals:
        if v >= fraction * final:
            return rnd
    return vals[-1][0]


def main():
    with open(os.path.join(C.DATA_DIR, "summary.json"), encoding="utf-8") as f:
        summary = json.load(f)
    train_rows = sum(n["rows"] for n in summary["nodes"].values())
    raw_bytes = train_rows * (FEATURES * BYTES_PER_FEATURE + 1)      # features + label

    out = {"centralized": {"training_rows": train_rows, "raw_bytes_uploaded": raw_bytes,
                           "raw_megabytes": round(raw_bytes / 1e6, 1)},
           "federated": {}}

    ae = read_log("ae")
    if ae:
        r = [x for x in ae["rounds"] if x["round"] > 0]
        per_round = ae.get("bytes_per_round") or (r[0]["bytes_down_total"] + r[0]["bytes_up_total"])
        out["federated"]["autoencoder"] = {
            "rounds": len(r), "bytes_per_round_all_nodes": per_round,
            "bytes_per_node_per_round": ae["rounds"][0].get("bytes_per_client"),
            "bytes_total": per_round * len(r),
            "rounds_to_99pct_of_final_recall": rounds_to_reach(ae["rounds"], "recall"),
            "final_recall": r[-1]["recall"], "final_fpr": r[-1]["fpr"],
        }
    xgb = read_log("xgb")
    if xgb:
        r = [x for x in xgb["rounds"] if x["round"] > 0]
        down = sum(x.get("bytes_down_total", 0) for x in r)
        # each node uploads only its new trees; the final model / (rounds*nodes) approximates one batch
        final_model = xgb.get("final_model_bytes") or r[-1]["global_model_bytes"]
        up = final_model  # every tree in the final model was uploaded exactly once
        out["federated"]["xgboost"] = {
            "rounds": len(r), "bytes_down_total": down, "bytes_up_total_estimate": up,
            "bytes_total": down + up, "final_model_bytes": final_model,
            "final_trees": r[-1].get("num_trees"),
            "rounds_to_99pct_of_final_auc": rounds_to_reach(xgb["rounds"], "auc"),
            "final_auc": r[-1].get("auc"),
        }
    fed_total = sum(v["bytes_total"] for v in out["federated"].values())
    out["federated"]["bytes_total_all_models"] = fed_total
    out["federated"]["megabytes_total"] = round(fed_total / 1e6, 1)
    out["ratio_federated_to_raw_upload"] = round(fed_total / raw_bytes, 3) if raw_bytes else None
    out["privacy_note"] = ("Federated traffic consists of model parameters and trees only; no flow record, "
                           "IP address or port ever leaves a node.")

    with open(os.path.join(C.LOG_DIR, "comm_analysis.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("| Quantity | Value |")
    print("|---|---|")
    print(f"| Centralized: raw rows uploaded | {train_rows:,} ({raw_bytes/1e6:.1f} MB) |")
    if "autoencoder" in out["federated"]:
        a = out["federated"]["autoencoder"]
        print(f"| Federated AE: rounds / bytes per round / total | {a['rounds']} / {a['bytes_per_round_all_nodes']/1e3:.0f} KB / {a['bytes_total']/1e6:.1f} MB |")
        print(f"| Federated AE: rounds to 99% of final recall | {a['rounds_to_99pct_of_final_recall']} |")
    if "xgboost" in out["federated"]:
        x = out["federated"]["xgboost"]
        print(f"| Federated XGB: rounds / total traffic / final model | {x['rounds']} / {x['bytes_total']/1e6:.1f} MB / {x['final_model_bytes']/1e6:.2f} MB |")
        print(f"| Federated XGB: rounds to 99% of final AUC | {x['rounds_to_99pct_of_final_auc']} |")
    print(f"| Federated total vs raw upload | {fed_total/1e6:.1f} MB vs {raw_bytes/1e6:.1f} MB (ratio {out['ratio_federated_to_raw_upload']}) |")
    print("Written federated/logs/comm_analysis.json")


if __name__ == "__main__":
    main()
