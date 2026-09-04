"""
federated/config.py
Single source of truth for Phase 2 (Federated Learning) settings.

Everything that the data preparation, clients, server and evaluation share
lives here so the four members work against the same constants.
"""
import os

# -- Paths -------------------------------------------------------------------
ROOT_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FED_DIR       = os.path.join(ROOT_DIR, "federated")
DATA_DIR      = os.path.join(FED_DIR, "data")
LOG_DIR       = os.path.join(FED_DIR, "logs")
CERT_DIR      = os.path.join(FED_DIR, "certs")
HYBRID_DIR    = os.path.join(ROOT_DIR, "hybrid_model")          # Phase 1 artifacts (kept)
FL_MODEL_DIR  = os.path.join(ROOT_DIR, "hybrid_model_fl")       # Phase 2 global model output

# Raw CIC-IDS2017 CSVs (MachineLearningCVE release). Override with env var.
RAW_DATASET_DIR = os.environ.get(
    "CICIDS2017_DIR",
    os.path.join(os.path.dirname(ROOT_DIR), "MachineLearningCVE"),
)

# -- Features ----------------------------------------------------------------
# The 30 features selected in Phase 1 (Random Forest importance). Order matters:
# it must match hybrid_model/selected_columns.pkl and network/features.py.
SELECTED_COLUMNS = [
    'Down/Up Ratio', 'Bwd IAT Mean', 'Flow IAT Std', 'Fwd Packets/s',
    'Min Packet Length', 'Fwd Packet Length Std', 'Packet Length Mean',
    'Bwd Packet Length Mean', 'Avg Bwd Segment Size', 'Bwd Packet Length Max',
    'Fwd Header Length', 'Destination Port', 'Fwd IAT Mean',
    'Average Packet Size', 'Subflow Bwd Packets', 'Bwd Packet Length Min',
    'Init_Win_bytes_backward', 'Subflow Fwd Packets', 'Fwd IAT Max',
    'Fwd Header Length.1', 'Init_Win_bytes_forward', 'Total Fwd Packets',
    'Fwd IAT Std', 'act_data_pkt_fwd', 'Fwd Packet Length Max',
    'Fwd IAT Total', 'Subflow Fwd Bytes', 'Total Length of Fwd Packets',
    'Fwd Packet Length Mean', 'Avg Fwd Segment Size',
]

# -- Labels ------------------------------------------------------------------
# Raw CIC-IDS2017 label -> coarse attack family used by the dashboard.
# Heartbleed (11 rows) and Infiltration (36 rows) are far too small to learn;
# they stay ATTACK for the binary task but are mapped to "Other".
# NOTE: the raw CSV uses a non-ASCII dash in "Web Attack ? XSS"; labels are
# normalised in prepare_data.py before this lookup.
ATTACK_FAMILY = {
    "BENIGN":                     "BENIGN",
    "DDoS":                       "DDoS",
    "DoS Hulk":                   "DoS",
    "DoS GoldenEye":              "DoS",
    "DoS slowloris":              "DoS",
    "DoS Slowhttptest":           "DoS",
    "PortScan":                   "PortScan",
    "FTP-Patator":                "BruteForce",
    "SSH-Patator":                "BruteForce",
    "Bot":                        "Bot",
    "Web Attack Brute Force":     "WebAttack",
    "Web Attack XSS":             "WebAttack",
    "Web Attack Sql Injection":   "WebAttack",
    "Infiltration":               "Other",
    "Heartbleed":                 "Other",
}
FAMILY_IDS = {
    "BENIGN": 0, "DDoS": 1, "DoS": 2, "PortScan": 3,
    "BruteForce": 4, "Bot": 5, "WebAttack": 6, "Other": 7,
}
ID_TO_FAMILY = {v: k for k, v in FAMILY_IDS.items()}

# -- Federated nodes (non-IID by design) --------------------------------------
# Each simulated node is an organisation that mostly sees one kind of attack,
# plus a random share of benign traffic. This is the "data heterogeneity" case.
NUM_NODES = 4
NODE_ATTACK_FAMILIES = {
    0: ["DDoS", "DoS"],              # volumetric flood site
    1: ["PortScan", "Bot"],          # reconnaissance / botnet site
    2: ["BruteForce", "WebAttack"],  # credential + web application site
    3: ["Other"],                    # quiet site: almost only benign traffic
}
TEST_FRACTION = 0.20            # global stratified hold-out kept by the server
RANDOM_SEED   = 42

# -- Training ------------------------------------------------------------------
AE_INPUT_DIM        = len(SELECTED_COLUMNS)
AE_LOCAL_EPOCHS     = 1
AE_BATCH_SIZE       = 1024
AE_ROUNDS           = 15
XGB_ROUNDS          = 40          # federated rounds for XGBoost bagging
XGB_TREES_PER_ROUND = 3           # num_parallel_tree per client per round
XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "max_depth": 8,
    "eta": 0.1,       # higher than Phase 1: bagging adds one tree batch per node per round
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
    "num_parallel_tree": XGB_TREES_PER_ROUND,
    "nthread": 4,
}

# -- Decision rule (shared with core/predictor.py through threshold.json) -------
AE_THRESHOLD_PERCENTILE = 90      # percentile of benign reconstruction error
ANOMALY_MULTIPLIER      = 10.0    # DL flags ATTACK when mse > multiplier * threshold
ML_THRESHOLD            = 0.5

# -- Networking ------------------------------------------------------------------
SERVER_ADDRESS = "127.0.0.1:8085"
