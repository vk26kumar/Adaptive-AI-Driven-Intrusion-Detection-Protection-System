# A-IDAPS-FL — Adaptive AI-Driven Intrusion Detection & Protection System

*B.Tech Major Project | Group G-52 | Dept. of CSE (ML & Cyber Security) | MMMUT Gorakhpur*

A real-time hybrid ML + Deep Learning intrusion detection system whose models are trained
**federatedly** across simulated network nodes with [Flower](https://flower.ai), served through a
live Streamlit dashboard with per-prediction explainability.

| Phase | Status | What it delivers |
|---|---|---|
| Phase 1 (Jan–Apr 2026) | done | Autoencoder + XGBoost trained centrally on the CIC-IDS2017 DDoS day, Scapy sniffer, Streamlit dashboard, attack simulator |
| Phase 2 (Sep 2026) | done | Federated training over 4 non-IID nodes with TLS, multi-attack data (all 8 CIC-IDS2017 days), calibrated hybrid rule, differential privacy option, centralized-vs-federated evaluation, honest dashboard metrics |
| Final (Nov 2026) | in progress | Full retrain, family-aware + Platt calibration, live validation with Npcap (7 live-path fixes, behavioural scan detector, local baseline), cross-dataset test on CSE-CIC-IDS2018, 29 passing tests; final report pending |

---

## Latest update (Nov 2026)

* **Full retrain from scratch.** Data preparation, federated training over TLS (Autoencoder 15
  rounds, XGBoost 40 rounds), evaluation and export were rerun on all 8 CIC-IDS2017 days. The data
  gives the same row counts per family and per node as published, and the Phase 1 model reproduces
  its published numbers exactly (88.92% accuracy, 51.75% recall). The federated hybrid now reaches
  **99.28% accuracy, 99.94% recall, 0.88% FPR** (previous run: 99.23% / 99.95% / 0.95%).
* **Family-aware calibration** (`core/decision.best_ml_threshold`). The old best-F1 cutoff rule ignores
  rare attack families. In the retrain it would have cut Bot recall to 67% and WebAttack to 62% to
  gain 0.0003 XGBoost F1. The cutoff is now the best F1 among cutoffs within 1 point of the best
  average per-family recall. Both the federated and the centralized model use this rule.
* **Corrected finding.** Phase 2 reported that federated training beat centralized training on Bot
  and WebAttack (97.5% vs 66.8%, 99.5% vs 60.0%). That gap came from the calibration flaw. With the
  same rule applied to both, the two are essentially tied (centralized 99.31% / 100% / 0.86%).
* **Simulator profiles are real flows.** Each profile is now the family *medoid*: the real captured
  flow nearest the family median, with its exact values taken from the original CSV row. The
  per-feature median, which the dashboard used before, is not a real flow, and for Bot it landed in
  benign territory, so the Bot button showed NORMAL. All 7 profiles now get their true verdict.
* **Smaller fixes.** The LSTM experiment's hybrid baseline now uses exact features (see below). The
  dashboard and `model_info.json` reported 16 Autoencoder rounds because the log includes a round-0
  baseline evaluation; they now report 15. The dashboard labels the XGBoost per-round recall as
  measured at the raw 0.5 cutoff, before server calibration.
* **Calibrated scores (Platt scaling).** Federated bagging leaves raw XGBoost probabilities far too
  low: a clear port scan got P = 0.26 and scored 0.56, the Bot demo flow 0.538, and only 0.5% of
  PortScan flows ever reached the prevention engine's 0.75 line. A Platt fit on the calibration half
  turns the score into a real probability (expected calibration error 0.101 → 0.0007 on the report
  half). It is monotone, so **every decision is identical** and all accuracy numbers are unchanged;
  median scores are now PortScan 0.97, Bot 0.99, WebAttack 0.97, and ≥ 97% of every attack family's
  flows reach 0.75 (benign flows at ≥ 0.75: 0.37% → 0.62%).
* **Exact replay data.** `prepare_data` now also writes `test_raw.npz` (exact CSV values of the
  hold-out). The replay test and the simulator prototypes use it instead of inverting the float32
  scaled data, which shifted Fwd Header Length by up to ~900 bytes.
* **Live capture validated** (Npcap 1.90, `network/live_check.py`); see *Live validation* below.
* **Cross-dataset test** on CSE-CIC-IDS2018 (16.2 M flows): the 2017-trained model detects only
  4.2% of 2018 attacks (population-weighted) — a documented negative result; see below.
* **Tests:** 29 (was 16), all passing, including the replay of 20,000 real flows.

---

## Results (CIC-IDS2017, report split of the server hold-out: 283,074 flows, 55,741 attacks)

| Model | Accuracy | Precision | Recall | F1 | FPR | Raw rows sent to server |
|---|---|---|---|---|---|---|
| Phase 1 hybrid (DDoS day only) | 88.92% | 86.62% | 51.75% | 64.79% | 1.96% | 225,745 |
| Phase 2 centralized hybrid | 99.31% | 96.61% | 100.00% | 98.27% | 0.86% | 2,264,594 |
| **Phase 2 federated hybrid** | **99.28%** | 96.53% | **99.94%** | 98.20% | 0.88% | **0** |

Recall per attack family (federated hybrid): DDoS 99.94%, DoS 99.94%, PortScan 100%, BruteForce 100%,
Bot 97.0%, WebAttack 98.6%. The Phase 1 model detected 5% of port scans and 0.1% of brute-force flows
on the same data.

Communication cost of federated training (4 nodes, TLS): Autoencoder 14 MB total over 15 rounds
(117 KB per node per round); XGBoost 256 MB over 40 rounds (the global booster grows to 3.1 MB).
All numbers are produced by `python -m federated.evaluate` and stored in `federated/logs/comparison.json`
(retrain of Nov 2026 with the family-aware calibration described under *Federated nodes* below).

---

## How detection works

```
flow features (30, CIC-IDS2017 units)
        │
        ├── Autoencoder (trained on BENIGN only)  → mse → dl_score  (0.5 at 10 × benign-90th-percentile)
        └── XGBoost     (BENIGN vs ATTACK)        → P(attack), Platt-calibrated → ml_score (0.5 at the cutoff)
                                                 final = max(dl_score, ml_score);  ATTACK if final > 0.5

live only:  behavioural port-scan detector (one source, ≥ 20 distinct ports of one host in 10 s)
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
  decision.py               hybrid decision rule, family-aware cutoff, Platt scaling (pure functions)
  predictor.py              loads models once; predict_features / predict_batch / explain_features
  simulator.py              attack simulator seeded from one real flow per family (medoid)
  state.py                  session stats, ground-truth confusion matrix, live attack-type heuristic
  scan_detector.py          behavioural port-scan detector for live traffic
network/
  features.py               live flow → 30 CIC-IDS2017 features (capture timestamps, 120 s active timeout)
  sniffer.py                Scapy capture thread (unicast scope, per-flow inference throttle)
  live_check.py             live validation: benign traffic / port scan of this machine
  baseline.py               learning mode: live autoencoder threshold for this network
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
  make_prototypes.py        per-family medoid flows for the simulator
  dp_analysis.py            (epsilon, delta) budget of the DP runs
  comm_analysis.py          traffic and convergence figures
  lstm_experiment.py        temporal-model experiment
  cross_dataset.py          deployed model on CSE-CIC-IDS2018 (no refit)
hybrid_model/               model the dashboard uses (federated export; Phase 1 backup kept aside)
tests/                      pytest: decision rule, extractor units, dataset replay, prevention, DP, LIME
```

---

## Setup

```bash
pip install -r requirements.txt
# Windows: install Npcap (https://npcap.com) for live capture. Leave "restrict to administrators"
# unchecked and no admin rights are needed (tested with Npcap 1.90).
```

Tested with Python 3.12. On Windows, if the repo sits in a long path, TensorFlow fails to install
inside a project-local venv because of the 260-character path limit. Create the venv in a short
path (e.g. `C:\Users\<you>\aidaps-venv`) or enable Windows long paths.

Place the eight CIC-IDS2017 `MachineLearningCVE` CSVs in a `MachineLearningCVE/` folder next to
the repo (or set `CICIDS2017_DIR`). The UNB download now goes through a registration form; the same
eight files are also mirrored on Hugging Face. `prepare_data` should report 2,830,743 flows.

To include the Phase 1 row in `federated.evaluate`, restore the Phase 1 artifacts once:
`git show 7939f0d:hybrid_model/<file>` for `autoencoder.keras`, `xgb_model.pkl`, `scaler.pkl`,
`selected_columns.pkl` and `threshold.json` into `hybrid_model_phase1_backup/`.

## Phase 2 workflow

```bash
python -m federated.prepare_data            # 2.83M flows → node_0..3.npz + test.npz + test_raw.npz  (~1 min)
python -m federated.gen_certs               # TLS certificates (once)
python -m federated.run_federated --tls     # Autoencoder (FedAvg, 15 rounds) then XGBoost (bagging, 40 rounds)
python -m federated.evaluate                # centralized baseline + comparison table
python -m federated.make_prototypes         # simulator profiles (family medoids)
python -m federated.export_global_model     # dashboard now uses the federated model
python -m federated.comm_analysis           # traffic and convergence summary
python -m pytest tests -q                   # 29 tests incl. dataset replay regression
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

Autoencoder: 117 KB per node per round, 14 MB total, recall still improving until round 14. XGBoost: 256 MB
total because the growing 3.1 MB booster is redistributed each round (delta-only transfer would cut
this ~20x). Centralized training would instead upload 274 MB of raw flows.

### Temporal model experiment (`python -m federated.lstm_experiment`)

An LSTM over 10-flow chronological windows reaches 88.5% accuracy / 63.2% recall / 4.95% FPR
(window level) against 96.5% / 99.9% / 4.37% for the flow-level hybrid under the same chronological
split. Per family the LSTM matches the hybrid on DDoS (99.4%) and PortScan (97.0%) but fails on DoS
(0.9%), BruteForce (3.6%) and Bot (25.6%), whose test windows contain attack variants not seen in
the training part of the day. Negative result, kept as a documented experiment
(`federated/logs/lstm_experiment.json`); not used in production. (The hybrid baseline is now computed
on the exact CSV features; the Sep 2026 version inverted float32-scaled values, which shifts Fwd
Header Length by up to ~900 bytes and understated the hybrid.)

### Protection, explainability, persistence

* `core/prevention.py`: auto-block a public source after 3 alerts scoring >= 0.75 within 60 s via
  Windows Firewall / iptables. Dry-run by default, allowlist, audit log, undo from the dashboard.
  **Verified on the real Windows Firewall** (`tests/manual_firewall_check.py`, result in
  `federated/logs/firewall_check.json`): as Administrator the 3rd alert created an inbound block rule
  for the source (confirmed with `netsh ... show rule`), a private LAN address with the same alerts
  was refused, and undo removed the rule. Without Administrator the engine records the netsh
  "requires elevation" error and marks the block inactive instead of claiming protection.
* `core/predictor.explain_lime`: LIME-style local surrogate of the final hybrid score (no extra dependency),
  next to TreeSHAP and the autoencoder error share.
* `core/persistence.py`: SQLite event store for verdicts, alerts and prevention actions (`logs/ids_events.sqlite`).
* `federated/prepare_data.py` also accepts CSE-CIC-IDS2018 CSVs; the column and label mapping is exercised by `federated/cross_dataset.py`.

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

The cutoff is the best-F1 one **among cutoffs within 1 point of the best achievable macro recall**
(mean detection rate over attack families), under a 1% FPR budget. Plain best-F1 is blind to Bot and
WebAttack (~0.4% of attacks): in the Nov 2026 retrain it picked a cutoff that cost Bot 26 and WebAttack
38 recall points (calibration half) for +0.0003 XGBoost F1. The same flaw is why the earlier centralized
baseline showed 67% / 60% on those families: rerun with the old rule it gives 66.8% / 60.0% again,
with the new rule 99.5% / 100%. Both baselines now use the family-aware rule (`core/decision.best_ml_threshold`).

The dashboard's attack simulator starts each profile from the family **medoid** (a real flow nearest
the family median), not the per-feature median, which is not a real flow and put the Bot profile in
benign territory.

## Live validation (Nov 2026, Npcap 1.90, Windows 11, office Wi-Fi)

`python -m network.live_check --scenario benign|portscan` runs the real capture pipeline
(Npcap → Scapy → `network/features.py` → model) and stores its summary in `logs/`; the runs quoted
here are kept in `federated/logs/live_check_*.json`. The first live runs exposed seven defects in the
Phase 2 live path, all fixed and unit-tested:

| # | Defect | Effect | Fix |
|---|---|---|---|
| 1 | Processing time used as packet time | a burst-processed handshake looked like 2,000,000 pkt/s | Npcap capture timestamp (`packet.time`) |
| 2 | Model run on every packet (~17 ms each) | a download (~3,000 pkt/s) starved every other flow | per-flow throttle: first 10 packets, then 1/s; statistics still updated per packet |
| 3 | No active timeout; packet lists capped at 4,096 | long flows saturated; rates wrong | CICFlowMeter 120 s active timeout; exact O(1) running statistics (sample std) |
| 4 | Flows open at capture start oriented by their first packet | a running download looked like a flood towards the client's port | non-SYN first packets (incl. SYN-ACK) oriented by port |
| 5 | Classification only after 2 *forward* packets | **port scans invisible** (a probe is SYN → RST, 1 forward packet) | classify at 2 packets in either direction |
| 6 | Alert source taken from the triggering packet | a probe is classified on the victim's RST, so the alert named the victim | flow's client → server orientation |
| 7 | LAN multicast / broadcast judged by the model | mDNS, SSDP, subnet broadcasts were most false positives | unicast scope (CIC-IDS2017 attacks are all unicast); `IDS_INCLUDE_MULTICAST=1` to override |

**Port scan of this machine** (100 ports on loopback): the per-flow model classified 98 probes and
flagged **0**. The cause is a dataset artifact, not a pipeline bug: 100% of CIC-IDS2017 PortScan flows
carry the attacker's TCP window (29200 = Linux/Kali default, or 1024 = nmap SYN scan) and a 6-byte
Ethernet padding on the RST. A Windows connect() probe over loopback scores 0.009; setting only those
two fields to the testbed values makes the identical probe score 0.94. A single probe holds no
evidence of scanning; the evidence is how many ports one source touches. The live path therefore
adds a **behavioural detector** (`core/scan_detector.py`, the idea of Snort's sfPortscan): one source,
≥ 20 distinct ports of one host within 10 s. It raised alerts at 20 and 40 ports and names the
scanner as the source; it feeds the same prevention engine.

**Benign traffic**: every remaining false positive came from the autoencoder (XGBoost gave
P(attack) ≈ 0), mostly on long-lived TLS keep-alive connections — domain shift from the 2017 lab.
`python -m network.baseline --minutes 5` learns a **live-only** autoencoder threshold on this
network (same definition as the server: 90th percentile of benign reconstruction error; it can only
raise the validated threshold). Here the local 90th percentile was 6.7× the dataset's.

| Live benign run | Packets classified | Flagged | Flows flagged |
|---|---|---|---|
| First run (Phase 2 code, during a 7 GB download) | 1,875 | 1,400 (74.7%) | 2 of 4 seen |
| Defects 1–7 fixed, no baseline | 930 | 64 (6.9%) | 20 of 72 |
| + local baseline (separate 90 s capture) | 1,359 | **15 (1.1%)** | 5 of 69 |

Per-packet latency (feature extraction + both models) median 13–17 ms. The remaining flagged flows
were this host's own outbound HTTPS; the prevention engine never blocks a private source, so they
cannot trigger a block. Not tested live: attacks other than port scans (no isolated lab network), and
federated training across physically separate machines (the nodes run as separate processes on one
host over TLS).

---

## Cross-dataset generalisation: CSE-CIC-IDS2018 (negative result)

`python -m federated.cross_dataset` runs the **deployed model unchanged** (trained and calibrated on
CIC-IDS2017 only) on CSE-CIC-IDS2018 — a different, AWS-hosted network recorded a year later with
CICFlowMeter-V3. All 10 CSVs (16,232,943 flows) are read in chunks; a fixed-seed sample of 281,918
flows (≤ 15,000 per attack label, 150,000 benign) is scored. Numbers in
`federated/logs/cross_dataset_2018.json`.

| Hybrid on 2018 | Accuracy | Precision | Recall | F1 | FPR |
|---|---|---|---|---|---|
| Weighted back to the full 16.2 M-flow population | 82.0% | 29.0% | **4.2%** | 7.4% | 2.1% |
| Balanced sample (131,918 attacks) | 57.7% | 83.4% | 12.1% | 21.1% | 2.1% |

Detection by 2018 label: DDoS (LOIC-HTTP/HOIC) 0.0%, Bot 0.1%, FTP-Patator 0.0%, DoS Hulk 2.2%,
DoS Slowhttptest 0.0%, DoS GoldenEye 18.4%, DoS slowloris 46.2%, SSH-Patator 47.9%, Web brute
force 24.2%, XSS 49.6%, SQL injection 2.3%, Infiltration 2.0%.

The feature mapping was checked on benign traffic (median packets per flow 2 vs 2, payload 37 vs 40
bytes, IATs in µs in both), so this is not a unit error: the attacks themselves differ. 2018's DDoS
tools have a median TCP window of 32,738 and 1,123 pkt/s against 256 and 1.7 for the 2017 DDoS the
model learned. Together with the live port-scan finding this is one consistent conclusion: the
supervised model learned the 2017 testbed's tools and machines, not attack behaviour in general, so
99% on a CIC-IDS2017 hold-out says little about other networks. Retraining with 2018 data as further
federated nodes (the schema mapping already exists) is the natural next step.

---

## Dashboard

* **Live packet feed** with measured per-flow latency and the model's two scores (XGBoost as a
  Platt-calibrated probability); behavioural port-scan alerts are labelled PortScan.
* **Attack simulator**: pick a traffic profile (a real flow of that family, the medoid) and optionally
  override knobs; the confusion matrix uses the profile's true label.
* **XAI**: TreeSHAP contributions of the XGBoost decision and the autoencoder's per-feature
  reconstruction-error share, for the most recent alert or simulated flow.
* **Federated training panel**: node partitions, per-round recall/FPR curves (XGBoost curve at the raw
  0.5 cutoff, before server calibration), traffic, TLS/DP flags, DP and LSTM tables,
  centralized-vs-federated table.

---

## Dataset

CIC-IDS2017 (Canadian Institute for Cybersecurity), all eight `MachineLearningCVE` files:
2,830,743 flows; BENIGN 2,273,097; DoS 252,661; PortScan 158,930; DDoS 128,027; BruteForce 13,835;
WebAttack 2,180; Bot 1,966; Infiltration + Heartbleed 47 (kept as ATTACK, family "Other").
30 features selected in Phase 1 by Random Forest importance. Download: <https://www.unb.ca/cic/datasets/ids-2017.html>

CSE-CIC-IDS2018 (cross-dataset test only, never used for training): the 10 "Processed Traffic Data for ML
Algorithms" CSVs from the public bucket `s3://cse-cic-ids2018/` (6.9 GB, 16,232,943 flows), placed in a
`CSE-CIC-IDS2018/` folder next to the repo (or set `CICIDS2018_DIR`).

## References

1. McMahan et al., *Communication-efficient learning of deep networks from decentralized data*, PMLR 2017
2. Beutel et al., *Flower: A friendly federated learning framework*, 2020
3. Buczak & Guven, *A survey of data mining and ML methods for cyber security IDS*, IEEE 2015
4. Garcia-Teodoro et al., *Anomaly-based network intrusion detection*, Computers & Security 2009
5. Lundberg & Lee, *A unified approach to interpreting model predictions (SHAP)*, NeurIPS 2017
6. Sharafaldin et al., *Toward generating a new intrusion detection dataset (CIC-IDS2017)*, ICISSP 2018

## License

MIT
