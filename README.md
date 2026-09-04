# A-IDAPS-FL — Adaptive AI-Driven Intrusion Detection & Protection System

*B.Tech Major Project | Group G-52 | Dept. of CSE (ML & Cyber Security) | MMMUT Gorakhpur*

A real-time hybrid ML + Deep Learning intrusion detection system whose models are trained
**federatedly** across simulated network nodes with [Flower](https://flower.ai), served through a
live Streamlit dashboard with per-prediction explainability.

| Phase | Status | What it delivers |
|---|---|---|
| Phase 1 (Jan–Apr 2026) | done | Autoencoder + XGBoost trained centrally on the CIC-IDS2017 DDoS day, Scapy sniffer, Streamlit dashboard, attack simulator |
| Phase 2 (Sep 2026 →) | in progress | Federated training over 4 non-IID nodes with TLS, multi-attack data (all 8 CIC-IDS2017 days), calibrated hybrid rule, differential privacy option, centralized-vs-federated evaluation, honest dashboard metrics |

---

## Results (CIC-IDS2017, report split of the server hold-out: 283,074 flows, 55,741 attacks)

| Model | Accuracy | Precision | Recall | F1 | FPR | Raw rows sent to server |
|---|---|---|---|---|---|---|
| Phase 1 hybrid (DDoS day only) | 88.92% | 86.62% | 51.75% | 64.79% | 1.96% | 225,745 |
| Phase 2 centralized hybrid | 99.55% | 98.04% | 99.68% | 98.86% | 0.49% | 2,264,594 |
| **Phase 2 federated hybrid** | **99.23%** | 96.27% | **99.95%** | 98.07% | 0.95% | **0** |

Recall per attack family (federated hybrid): DDoS 99.97%, DoS 99.92%, PortScan 100%, BruteForce 100%,
Bot 97.5%, WebAttack 99.5%. The Phase 1 model detected 5% of port scans and 0.1% of brute-force flows
on the same data.

Communication cost of federated training (4 nodes, TLS): Autoencoder 14 MB total over 15 rounds
(117 KB per node per round); XGBoost 254 MB over 40 rounds (the global booster grows to 3.1 MB).
All numbers are produced by `python -m federated.evaluate` and stored in `federated/logs/comparison.json`.

---

## How detection works

```
flow features (30, CIC-IDS2017 units)
        │
        ├── Autoencoder (trained on BENIGN only)  → mse → dl_score  (0.5 at 10 × benign-90th-percentile)
        └── XGBoost     (BENIGN vs ATTACK)        → P(attack) → ml_score (0.5 at the calibrated cutoff)
                                                 final = max(dl_score, ml_score);  ATTACK if final > 0.5
```

A flow is an attack when **either** the supervised model recognises a known signature **or** the
reconstruction error is far above anything benign traffic produces. The two detectors are
complementary rather than averaged, so a zero-day can still trip the autoencoder while a confident
classifier is never diluted. The rule lives in `core/decision.py`; thresholds are published by the
federated server in `hybrid_model/threshold.json`.

---

## Project structure

```
app.py                      Streamlit dashboard (entry point)
main.py                     headless live IDS (prints verdicts)
core/
  prevention.py             autonomous IP blocking (dry-run default, audit, undo)
  persistence.py            SQLite event store
  decision.py               hybrid decision rule + threshold calibration (pure functions)
  predictor.py              loads models once; predict_features / predict_batch / explain_features
  simulator.py              attack simulator seeded from real per-family median flows
  state.py                  session stats, ground-truth confusion matrix, live attack-type heuristic
network/
  features.py               live flow → 30 CIC-IDS2017 features (µs times, payload lengths)
  sniffer.py                Scapy capture thread
federated/                  Phase 2
  config.py                 nodes, label map, hyper-parameters, decision constants
  prepare_data.py           8 CSVs → 4 non-IID node partitions + calibration/report hold-out
  client.py                 a node: Autoencoder (FedAvg) or XGBoost (bagging)
  server.py                 aggregation server, per-round evaluation, model export, optional DP
  run_federated.py          launches server + 4 nodes on this machine
  calibrate.py              server-side thresholds on the calibration split
  evaluate.py               Phase 1 vs centralized vs federated comparison
  export_global_model.py    installs the global model into hybrid_model/
  gen_certs.py              self-signed CA + server certificate for TLS
  make_prototypes.py        per-family median flows for the simulator
  dp_analysis.py            (epsilon, delta) budget of the DP runs
  comm_analysis.py          traffic and convergence figures
  lstm_experiment.py        temporal-model experiment
hybrid_model/               model the dashboard uses (federated export; Phase 1 backup kept aside)
tests/                      pytest: decision rule, extractor units, dataset replay, prevention, DP, LIME
docs/                       Phase 2 report, task forms, viva Q&A, presentation outline
```

---

## Setup

```bash
pip install -r requirements.txt
# Windows: install Npcap (https://npcap.com) for live capture and run the dashboard as Administrator
```

Place the CIC-IDS2017 `MachineLearningCVE` CSVs next to the repo (or set `CICIDS2017_DIR`).

## Phase 2 workflow

```bash
python -m federated.prepare_data            # 2.83M flows → node_0..3.npz + test.npz  (~1 min)
python -m federated.gen_certs               # TLS certificates (once)
python -m federated.run_federated --tls     # Autoencoder (FedAvg, 15 rounds) then XGBoost (bagging, 40 rounds)
python -m federated.evaluate                # centralized baseline + comparison table
python -m federated.export_global_model     # dashboard now uses the federated model
python -m pytest tests -q                   # 16 tests incl. dataset replay regression
streamlit run app.py
```

Options: `--dp --noise 0.3 --clip 1.0` adds server-side differential privacy (fixed clipping +
Gaussian noise) to the Autoencoder aggregation; `--out-dir` keeps a variant apart from the main
model; `--only ae|xgb` and `--rounds N` run a subset. Each node is a separate process that only
ever sends model parameters (or new trees) to `127.0.0.1:8085`.

### Differential privacy: measured, and not meaningful with 4 nodes

`--dp` wraps FedAvg in Flower's server-side fixed clipping + Gaussian noise (client-level DP).
Privacy budget from `python -m federated.dp_analysis` (Renyi accounting, delta = 1e-5):

| Setting | epsilon | Benign MSE | Usable? |
|---|---|---|---|
| no DP | inf | 2.8e-05 | reference |
| noise 0.3, clip 1.0, 15 rounds | 145 | 1.5e-01 | no, model never trains |
| noise 0.05, clip 20, 15 rounds | 3,372 | 1.9e-01 | no |
| noise 0.01, clip 1.0, 40 rounds | 203,151 | 3.3e-05 | accuracy yes, privacy no |

With four clients the noise std per weight is noise x clip / 4: strong enough for a meaningful
epsilon it destroys the model; weak enough to train it offers no privacy. DP-FedAvg needs hundreds of
clients. Privacy in this deployment rests on the architecture (raw traffic never leaves a node) and TLS.

### Communication and convergence (`python -m federated.comm_analysis`)

Autoencoder: 117 KB per node per round, 14 MB total, recall converged by round 4. XGBoost: 257 MB
total because the growing 3.1 MB booster is redistributed each round (delta-only transfer would cut
this ~20x). Centralized training would instead upload 274 MB of raw flows.

### Temporal model experiment (`python -m federated.lstm_experiment`)

An LSTM over 10-flow chronological windows reaches 88.9% accuracy / 62.7% recall (window level)
against 95.9% / 99.2% for the flow-level hybrid under the same chronological split. Negative result,
documented in `docs/Phase2_Report.md` section 7.5; not used in production.

### Protection, explainability, persistence

* `core/prevention.py`: auto-block a public source after 3 alerts scoring >= 0.75 within 60 s via
  Windows Firewall / iptables. Dry-run by default, allowlist, audit log, undo from the dashboard.
* `core/predictor.explain_lime`: LIME-style local surrogate of the final hybrid score (no extra dependency),
  next to TreeSHAP and the autoencoder error share.
* `core/persistence.py`: SQLite event store for verdicts, alerts and prevention actions (`logs/ids_events.sqlite`).
* `federated/prepare_data.py` also accepts CSE-CIC-IDS2018 CSVs (column and label mapping; untested, files not on disk).

### Federated nodes (non-IID by design)

| Node | Sees | Rows | Attacks |
|---|---|---|---|
| 0 | DDoS, DoS | 668,245 | 304,550 |
| 1 | PortScan, Bot | 492,412 | 128,717 |
| 2 | BruteForce, WebAttack | 376,508 | 12,812 |
| 3 | almost only benign | 727,429 | 38 |

Bagged XGBoost under this skew ranks flows almost perfectly (AUC 0.9998) but its raw probabilities
sit low, so the server calibrates the operating point on its own labelled calibration split (even
rows of the hold-out) and reports on the other half. Nothing tuned on one half is measured on it.

---

## Dashboard

* **Live packet feed** with measured per-flow latency and the model's two scores.
* **Attack simulator**: pick a traffic profile (median real flow of that family) and optionally
  override knobs; the confusion matrix uses the profile's true label.
* **XAI**: TreeSHAP contributions of the XGBoost decision and the autoencoder's per-feature
  reconstruction-error share, for the most recent alert or simulated flow.
* **Federated training panel**: node partitions, per-round recall/FPR curves, traffic, TLS/DP flags,
  centralized-vs-federated table.

---

## Dataset

CIC-IDS2017 (Canadian Institute for Cybersecurity), all eight `MachineLearningCVE` files:
2,830,743 flows; BENIGN 2,273,097; DoS 252,661; PortScan 158,930; DDoS 128,027; BruteForce 13,835;
WebAttack 2,180; Bot 1,966; Infiltration + Heartbleed 47 (kept as ATTACK, family "Other").
30 features selected in Phase 1 by Random Forest importance. Download: <https://www.unb.ca/cic/datasets/ids-2017.html>

## References

1. McMahan et al., *Communication-efficient learning of deep networks from decentralized data*, PMLR 2017
2. Beutel et al., *Flower: A friendly federated learning framework*, 2020
3. Buczak & Guven, *A survey of data mining and ML methods for cyber security IDS*, IEEE 2015
4. Garcia-Teodoro et al., *Anomaly-based network intrusion detection*, Computers & Security 2009
5. Lundberg & Lee, *A unified approach to interpreting model predictions (SHAP)*, NeurIPS 2017
6. Sharafaldin et al., *Toward generating a new intrusion detection dataset (CIC-IDS2017)*, ICISSP 2018

## License

MIT
