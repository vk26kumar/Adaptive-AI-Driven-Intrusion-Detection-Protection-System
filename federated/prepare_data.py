"""
federated/prepare_data.py
Turn the eight raw CIC-IDS2017 CSVs into federated node partitions.

    python -m federated.prepare_data

Steps
  1. Load only the 30 selected features + Label from each CSV.
  2. Clean: strip column names, drop inf/NaN rows, normalise label text.
  3. Map every raw label to a coarse attack family (config.ATTACK_FAMILY).
  4. Fit ONE MinMaxScaler on all rows (same choice as Phase 1) and save it.
  5. Split a stratified 20% global test set (held by the server only).
  6. Distribute the remaining rows over 4 non-IID nodes: each node receives
     the attack families assigned to it plus a random share of benign rows.

Outputs (federated/data/)
  scaler.pkl, selected_columns.pkl, class_map.json,
  node_0.npz .. node_3.npz, test.npz, summary.json
"""
import glob
import json
import os
import re
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from federated.config import (
    RAW_DATASET_DIR, DATA_DIR, SELECTED_COLUMNS, ATTACK_FAMILY, FAMILY_IDS,
    NUM_NODES, NODE_ATTACK_FAMILIES, TEST_FRACTION, RANDOM_SEED,
)


def _normalise_label(raw: str) -> str:
    """'Web Attack � Brute Force' -> 'Web Attack Brute Force'."""
    s = str(raw).strip()
    s = re.sub(r"[^A-Za-z0-9 \-]", " ", s)      # drop the broken dash bytes
    s = re.sub(r"\s+", " ", s).strip()
    return s


# CSE-CIC-IDS2018 uses the same CICFlowMeter features under abbreviated names.
# Map 2018 header -> 2017 header so both releases feed the same pipeline.
# (Schema support only: the 2018 files are not in this repository; run with
#  CICIDS2017_DIR pointing at a 2018 folder and check the family counts.)
CICIDS2018_COLUMNS = {
    "Down/Up Ratio": "Down/Up Ratio", "Bwd IAT Mean": "Bwd IAT Mean", "Flow IAT Std": "Flow IAT Std",
    "Fwd Pkts/s": "Fwd Packets/s", "Pkt Len Min": "Min Packet Length", "Fwd Pkt Len Std": "Fwd Packet Length Std",
    "Pkt Len Mean": "Packet Length Mean", "Bwd Pkt Len Mean": "Bwd Packet Length Mean",
    "Bwd Seg Size Avg": "Avg Bwd Segment Size", "Bwd Pkt Len Max": "Bwd Packet Length Max",
    "Fwd Header Len": "Fwd Header Length", "Dst Port": "Destination Port", "Fwd IAT Mean": "Fwd IAT Mean",
    "Pkt Size Avg": "Average Packet Size", "Subflow Bwd Pkts": "Subflow Bwd Packets",
    "Bwd Pkt Len Min": "Bwd Packet Length Min", "Init Bwd Win Byts": "Init_Win_bytes_backward",
    "Subflow Fwd Pkts": "Subflow Fwd Packets", "Fwd IAT Max": "Fwd IAT Max",
    "Init Fwd Win Byts": "Init_Win_bytes_forward", "Tot Fwd Pkts": "Total Fwd Packets",
    "Fwd IAT Std": "Fwd IAT Std", "Fwd Act Data Pkts": "act_data_pkt_fwd", "Fwd Pkt Len Max": "Fwd Packet Length Max",
    "Fwd IAT Tot": "Fwd IAT Total", "Subflow Fwd Byts": "Subflow Fwd Bytes", "TotLen Fwd Pkts": "Total Length of Fwd Packets",
    "Fwd Pkt Len Mean": "Fwd Packet Length Mean", "Fwd Seg Size Avg": "Avg Fwd Segment Size",
}
CICIDS2018_LABELS = {   # 2018 label text -> 2017 label text used in config.ATTACK_FAMILY
    "Benign": "BENIGN", "DDOS attack-HOIC": "DDoS", "DDOS attack-LOIC-UDP": "DDoS", "DDoS attacks-LOIC-HTTP": "DDoS",
    "DoS attacks-Hulk": "DoS Hulk", "DoS attacks-GoldenEye": "DoS GoldenEye", "DoS attacks-Slowloris": "DoS slowloris",
    "DoS attacks-SlowHTTPTest": "DoS Slowhttptest", "FTP-BruteForce": "FTP-Patator", "SSH-Bruteforce": "SSH-Patator",
    "Bot": "Bot", "Brute Force -Web": "Web Attack Brute Force", "Brute Force -XSS": "Web Attack XSS",
    "SQL Injection": "Web Attack Sql Injection", "Infilteration": "Infiltration",
}


def _read_csv_any_release(f: str) -> pd.DataFrame:
    """Read a CIC-IDS2017 or CSE-CIC-IDS2018 CSV and return 2017-style columns + Label."""
    header = pd.read_csv(f, nrows=0, encoding_errors="replace").columns
    stripped = {c.strip(): c for c in header}
    if "Fwd Header Length.1" in stripped or "Destination Port" in stripped:          # 2017 release
        usecols = [stripped[c] for c in SELECTED_COLUMNS] + [stripped["Label"]]
        df = pd.read_csv(f, usecols=usecols, encoding_errors="replace")
        df.columns = df.columns.str.strip()
        return df
    # 2018 release
    usecols = [stripped[c] for c in CICIDS2018_COLUMNS if c in stripped] + [stripped["Label"]]
    df = pd.read_csv(f, usecols=usecols, encoding_errors="replace", low_memory=False)
    df.columns = df.columns.str.strip()
    df = df.rename(columns=CICIDS2018_COLUMNS)
    df["Fwd Header Length.1"] = df["Fwd Header Length"]           # 2017 duplicate column
    df["Label"] = df["Label"].map(lambda x: CICIDS2018_LABELS.get(str(x).strip(), str(x).strip()))
    df = df[df["Label"] != "Label"]                               # 2018 files repeat the header mid-file
    for c in SELECTED_COLUMNS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[SELECTED_COLUMNS + ["Label"]]


def load_all(raw_dir: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(raw_dir, "*.csv")))
    if not files:
        sys.exit(f"No CSV files found in {raw_dir}. Set CICIDS2017_DIR.")
    frames = []
    for f in files:
        df = _read_csv_any_release(f)
        df["Label"] = df["Label"].map(_normalise_label)
        df["source_file"] = os.path.basename(f)
        frames.append(df)
        print(f"  loaded {os.path.basename(f):55s} rows={len(df):>8,}")
    return pd.concat(frames, ignore_index=True)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    feats = df[SELECTED_COLUMNS].astype("float64").replace([np.inf, -np.inf], np.nan)
    mask = ~feats.isna().any(axis=1)
    df = df.loc[mask].copy()
    df[SELECTED_COLUMNS] = feats.loc[mask]
    print(f"  cleaned: dropped {before - len(df):,} rows with inf/NaN -> {len(df):,} rows")
    unknown = sorted(set(df["Label"]) - set(ATTACK_FAMILY))
    if unknown:
        sys.exit(f"Unmapped labels found: {unknown}. Add them to config.ATTACK_FAMILY.")
    df["family"] = df["Label"].map(ATTACK_FAMILY)
    df["family_id"] = df["family"].map(FAMILY_IDS).astype(np.int8)
    df["is_attack"] = (df["family"] != "BENIGN").astype(np.int8)
    return df


def partition(df: pd.DataFrame, rng: np.random.Generator):
    """Return dict node_id -> DataFrame (non-IID by attack family)."""
    benign = df[df["is_attack"] == 0]
    attacks = df[df["is_attack"] == 1]

    # Benign rows are shared out at random: node 3 (the quiet site) gets the
    # largest benign share so it looks like a low-risk organisation.
    benign_shares = np.array([0.20, 0.20, 0.20, 0.40])
    benign_idx = rng.permutation(benign.index.values)
    cuts = (np.cumsum(benign_shares)[:-1] * len(benign_idx)).astype(int)
    benign_parts = np.split(benign_idx, cuts)

    nodes = {}
    for node_id in range(NUM_NODES):
        fams = NODE_ATTACK_FAMILIES[node_id]
        part = pd.concat([
            benign.loc[benign_parts[node_id]],
            attacks[attacks["family"].isin(fams)],
        ])
        nodes[node_id] = part.sample(frac=1.0, random_state=RANDOM_SEED + node_id)
    return nodes


def save_npz(path: str, df: pd.DataFrame, scaler: MinMaxScaler) -> None:
    X = scaler.transform(df[SELECTED_COLUMNS].values).astype(np.float32)
    np.savez_compressed(
        path,
        X=X,
        y=df["is_attack"].values.astype(np.int8),
        family=df["family_id"].values.astype(np.int8),
    )


def main() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)

    print(f"Reading CIC-IDS2017 from {RAW_DATASET_DIR}")
    df = clean(load_all(RAW_DATASET_DIR))
    print("\nAttack family counts:")
    print(df["family"].value_counts().to_string())

    # One scaler for everybody (the server would normally publish the feature
    # ranges as part of the global model; fitting on all data mirrors Phase 1).
    scaler = MinMaxScaler().fit(df[SELECTED_COLUMNS].values)
    joblib.dump(scaler, os.path.join(DATA_DIR, "scaler.pkl"))
    joblib.dump(list(SELECTED_COLUMNS), os.path.join(DATA_DIR, "selected_columns.pkl"))

    train_df, test_df = train_test_split(
        df, test_size=TEST_FRACTION, stratify=df["family_id"], random_state=RANDOM_SEED
    )
    save_npz(os.path.join(DATA_DIR, "test.npz"), test_df, scaler)
    print(f"\nGlobal test set: {len(test_df):,} rows "
          f"({int(test_df['is_attack'].sum()):,} attacks)")

    nodes = partition(train_df, rng)
    summary = {
        "total_rows": int(len(df)),
        "test_rows": int(len(test_df)),
        "families": {k: int(v) for k, v in df["family"].value_counts().items()},
        "nodes": {},
    }
    for node_id, part in nodes.items():
        save_npz(os.path.join(DATA_DIR, f"node_{node_id}.npz"), part, scaler)
        fam_counts = {k: int(v) for k, v in part["family"].value_counts().items()}
        summary["nodes"][str(node_id)] = {
            "rows": int(len(part)),
            "attacks": int(part["is_attack"].sum()),
            "families": fam_counts,
            "assigned_families": NODE_ATTACK_FAMILIES[node_id],
        }
        print(f"  node_{node_id}: rows={len(part):>9,}  attacks={int(part['is_attack'].sum()):>8,}  {fam_counts}")

    with open(os.path.join(DATA_DIR, "class_map.json"), "w", encoding="utf-8") as f:
        json.dump({"attack_family": ATTACK_FAMILY, "family_ids": FAMILY_IDS}, f, indent=2)
    with open(os.path.join(DATA_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nDone. Partitions written to {DATA_DIR}")


if __name__ == "__main__":
    main()
