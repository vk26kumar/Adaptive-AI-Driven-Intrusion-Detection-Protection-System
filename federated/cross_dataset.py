"""
federated/cross_dataset.py
Cross-dataset generalisation: run the DEPLOYED model (trained and calibrated on
CIC-IDS2017 only, nothing refitted) on CSE-CIC-IDS2018.

    python -m federated.cross_dataset [--dir PATH] [--per-label 100000] [--benign 500000]

CSE-CIC-IDS2018 (CIC + Communications Security Establishment, 2018) was
recorded on a different, AWS-hosted network a year later, with CICFlowMeter-V3.
Its "Processed Traffic Data for ML Algorithms" CSVs use abbreviated column
names; federated.prepare_data.CICIDS2018_COLUMNS maps them to the 2017 names.
16.2 M flows do not fit the 8 GB machine this was written on, so files are read
in chunks and a fixed-seed sample is evaluated: up to --per-label rows of every
attack label (all rows of the small ones) plus --benign benign rows.
Writes federated/logs/cross_dataset_2018.json.
"""
import argparse
import glob
import json
import os
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd

from federated import config as C
from federated.prepare_data import CICIDS2018_COLUMNS, CICIDS2018_LABELS

DEFAULT_DIR = os.environ.get("CICIDS2018_DIR", os.path.join(os.path.dirname(C.ROOT_DIR), "CSE-CIC-IDS2018"))
CHUNK = 500_000


def _metrics(y, pred):
    y = y.astype(bool); pred = pred.astype(bool)
    tp = int((pred & y).sum()); fp = int((pred & ~y).sum()); fn = int((~pred & y).sum()); tn = int((~pred & ~y).sum())
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    return {"accuracy": round((tp + tn) / max(len(y), 1), 4), "precision": round(prec, 4), "recall": round(rec, 4),
            "f1": round(2 * prec * rec / max(prec + rec, 1e-9), 4), "fpr": round(fp / max(fp + tn, 1), 4),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def load_sample(d: str, per_label: int, benign: int, seed: int = 0) -> pd.DataFrame:
    """Chunked read of every 2018 CSV; reservoir-style sampling per label."""
    rng = np.random.default_rng(seed)
    want = list(CICIDS2018_COLUMNS) + ["Label"]
    pools, seen = {}, {}
    files = sorted(glob.glob(os.path.join(d, "*.csv")))
    if not files:
        raise SystemExit(f"No CSVs in {d}. Download CSE-CIC-IDS2018 'Processed Traffic Data for ML Algorithms'.")
    for f in files:
        hdr = pd.read_csv(f, nrows=0).columns
        cols = [c for c in want if c in hdr]
        n_rows = 0
        for ch in pd.read_csv(f, usecols=cols, chunksize=CHUNK,
                              low_memory=False):
            ch = ch[ch["Label"].astype(str).str.strip() != "Label"]          # repeated header rows
            ch = ch.rename(columns=CICIDS2018_COLUMNS)
            ch["Label"] = ch["Label"].astype(str).str.strip().map(lambda x: CICIDS2018_LABELS.get(x, x))
            for c in C.SELECTED_COLUMNS:
                if c == "Fwd Header Length.1":
                    continue
                ch[c] = pd.to_numeric(ch[c], errors="coerce")
            ch["Fwd Header Length.1"] = ch["Fwd Header Length"]
            feats = ch[C.SELECTED_COLUMNS].replace([np.inf, -np.inf], np.nan)
            ch = ch.loc[~feats.isna().any(axis=1)]
            n_rows += len(ch)
            for lab, g in ch.groupby("Label"):
                cap = benign if lab == "BENIGN" else per_label
                seen[lab] = seen.get(lab, 0) + len(g)
                cur = pools.get(lab)
                g = g[C.SELECTED_COLUMNS + ["Label"]].astype({c: "float32" for c in C.SELECTED_COLUMNS})
                cur = g if cur is None else pd.concat([cur, g], ignore_index=True)
                if len(cur) > cap:                    # keep a uniform random subset of everything seen so far
                    cur = cur.sample(n=cap, random_state=int(rng.integers(1 << 31)))
                pools[lab] = cur
        print(f"  {os.path.basename(f)[:48]:48s} rows={n_rows:>10,}", flush=True)
    df = pd.concat(pools.values(), ignore_index=True)
    df.attrs["population"] = seen
    return df


SAMPLE_PATH = os.path.join(C.DATA_DIR, "cross2018_sample.npz")
BATCH = 20_000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=DEFAULT_DIR)
    ap.add_argument("--per-label", type=int, default=15_000)
    ap.add_argument("--benign", type=int, default=150_000)
    ap.add_argument("--resample", action="store_true", help="re-read the CSVs even if a saved sample exists")
    a = ap.parse_args()

    t0 = time.time()
    if a.resample or not os.path.exists(SAMPLE_PATH):
        # Stage 1 (no TensorFlow loaded): chunked read + sample, saved so a rerun skips the 7 GB read
        print(f"Reading CSE-CIC-IDS2018 from {a.dir}", flush=True)
        df = load_sample(a.dir, a.per_label, a.benign)
        pop = df.attrs["population"]
        np.savez_compressed(SAMPLE_PATH, X=df[C.SELECTED_COLUMNS].values.astype(np.float64),
                            label=np.asarray(df["Label"].astype(str).tolist(), dtype="U64"),
                            pop_labels=np.array(list(pop.keys()), dtype=str),
                            pop_counts=np.array(list(pop.values()), dtype=np.int64),
                            caps=np.array([a.per_label, a.benign]))
        del df
        print(f"sample saved to {SAMPLE_PATH}", flush=True)
    d = np.load(SAMPLE_PATH, allow_pickle=True)          # our own file; older saves stored labels as objects
    X, labels = d["X"], d["label"]
    population = dict(zip(d["pop_labels"].tolist(), d["pop_counts"].tolist()))
    a.per_label, a.benign = int(d["caps"][0]), int(d["caps"][1])
    labels_s = pd.Series(labels)
    fam = labels_s.map(C.ATTACK_FAMILY).fillna("Other").values
    y = (fam != "BENIGN")
    print(f"sample: {len(X):,} flows ({y.sum():,} attacks) from a population of {sum(population.values()):,}", flush=True)

    # Stage 2: score in batches (one 700k-row call made several full copies next to
    # TensorFlow and thrashed an 8 GB machine for hours)
    from core.predictor import predict_batch, THRESHOLDS
    from core.decision import ml_score, calibrated_prob
    parts = []
    for i in range(0, len(X), BATCH):
        parts.append(predict_batch(pd.DataFrame(X[i:i + BATCH], columns=C.SELECTED_COLUMNS))[["ml_prob", "dl_score", "label"]])
        print(f"  scored {min(i + BATCH, len(X)):,}/{len(X):,}", flush=True)
    out = pd.concat(parts, ignore_index=True)
    att = out["label"].values == "ATTACK"
    ml_only = ml_score(out["ml_prob"].values, float(calibrated_prob(THRESHOLDS["ml_threshold"], THRESHOLDS))) > 0.5
    ae_only = out["dl_score"].values > 0.5
    df = pd.DataFrame({"Label": labels})

    # Sample-weighted overall numbers: each label weighted back to its share of the full 2018 population
    w = df["Label"].map(lambda l: population[l] / max((df["Label"] == l).sum(), 1)).values

    def weighted(pred):
        tp = (w * (pred & y)).sum(); fp = (w * (pred & ~y)).sum(); fn = (w * (~pred & y)).sum(); tn = (w * (~pred & ~y)).sum()
        prec = tp / max(tp + fp, 1e-9); rec = tp / max(tp + fn, 1e-9)
        return {"accuracy": round(float((tp + tn) / w.sum()), 4), "precision": round(float(prec), 4),
                "recall": round(float(rec), 4), "f1": round(float(2 * prec * rec / max(prec + rec, 1e-9)), 4),
                "fpr": round(float(fp / max(fp + tn, 1e-9)), 4)}

    per_label = {}
    for lab in sorted(df["Label"].unique()):
        m = (df["Label"] == lab).values
        rate = float(att[m].mean())
        per_label[lab] = {"family": C.ATTACK_FAMILY.get(lab, "Other"), "population": int(population[lab]),
                          "sampled": int(m.sum()),
                          ("false_positive_rate" if lab == "BENIGN" else "detection_rate"): round(rate, 4),
                          "xgboost_only": round(float(ml_only[m].mean()), 4),
                          "autoencoder_only": round(float(ae_only[m].mean()), 4)}
    res = {
        "dataset": "CSE-CIC-IDS2018 (Processed Traffic Data for ML Algorithms)",
        "model": "deployed federated hybrid, trained and calibrated on CIC-IDS2017 only (no refit)",
        "sample": {"flows": int(len(X)), "attacks": int(y.sum()), "per_label_cap": a.per_label, "benign_cap": a.benign},
        "population_flows": int(sum(population.values())),
        "hybrid_sample": _metrics(y, att),
        "hybrid_population_weighted": weighted(att),
        "xgboost_only_population_weighted": weighted(ml_only),
        "autoencoder_only_population_weighted": weighted(ae_only),
        "per_label": per_label,
        "seconds": round(time.time() - t0, 1),
    }
    os.makedirs(C.LOG_DIR, exist_ok=True)
    with open(os.path.join(C.LOG_DIR, "cross_dataset_2018.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(json.dumps({k: v for k, v in res.items() if k != "per_label"}, indent=1))
    print("| 2018 label | family | sampled | detected (benign: flagged) | XGB only | AE only |")
    print("|---|---|---|---|---|---|")
    for lab, r in per_label.items():
        rate = r.get("detection_rate", r.get("false_positive_rate"))
        print(f"| {lab} | {r['family']} | {r['sampled']:,} | {rate:.1%} | {r['xgboost_only']:.1%} | {r['autoencoder_only']:.1%} |")
    print("Written federated/logs/cross_dataset_2018.json")


if __name__ == "__main__":
    main()
