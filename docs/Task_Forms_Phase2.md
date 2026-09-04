# Phase 2 — Weekly Task Forms (one form per member)

Status reflects the repository on 2026-09-04. "Completed" means code, run logs and results exist in
the repo; see `docs/Phase2_Report.md` for the numbers behind each row.

## Form 1 — Vishal Kumar (2023011085)

| S.No. | Week | Task Assigned | Task Status |
|---|---|---|---|
| 1 | Week 1 | Study Federated Learning concept and FedAvg algorithm | Completed |
| 2 | Week 2 | Set up Flower (flwr 1.36) framework in gRPC deployment mode; server + 4 client processes | Completed |
| 3 | Week 3 | Design overall federated system architecture (`federated/config.py`, server/client/orchestrator) | Completed |
| 4 | Week 4 | Implement TLS between nodes and server (self-signed CA, server certificate, client verification) | Completed |
| 5 | Week 6 | Integrate hybrid model (XGBoost + Autoencoder) into the federated pipeline; redesign decision rule | Completed |
| 6 | Week 8 | Compare centralized vs federated accuracy and performance (`federated/evaluate.py`) | Completed |
| 7 | Week 10 | Final system testing (16 tests), Phase 1 audit fixes, project documentation | Completed |

## Form 2 — Shashwat Srivastava (2023021159)

| S.No. | Week | Task Assigned | Task Status |
|---|---|---|---|
| 1 | Week 1 | Study CIC-IDS2017 dataset structure and attack types (all 8 capture days) | Completed |
| 2 | Week 2 | Load and preprocess multiple attack datasets (DDoS, DoS, PortScan, BruteForce, Bot, WebAttack) | Completed |
| 3 | Week 3 | Distribute dataset across 4 simulated federated nodes with calibration/report hold-out | Completed |
| 4 | Week 4 | Handle data heterogeneity (non-IID nodes by attack family, per-node class weighting) | Completed |
| 5 | Week 6 | Integrate WebAttack and Infiltration data; CSE-CIC-IDS2018 schema mapping | Completed (2018 mapping untested: files not available) |
| 6 | Week 8 | Evaluate detection accuracy per attack family on multi-attack test data | Completed |
| 7 | Week 10 | Dataset documentation and report section (Chapter 4) | Completed |

## Form 3 — Tarkeshvar Mani Yadav (2023011078)

| S.No. | Week | Task Assigned | Task Status |
|---|---|---|---|
| 1 | Week 1 | Study local model training in a federated setup | Completed |
| 2 | Week 2 | Implement local XGBoost training on each node (bagging client) | Completed |
| 3 | Week 3 | Implement local Autoencoder training on each node (FedAvg client) | Completed |
| 4 | Week 5 | Add differential privacy to model aggregation (fixed clipping + Gaussian noise) | Completed; result negative with 4 nodes (Report §7.4) |
| 5 | Week 6 | Test local model performance before and after aggregation (per-round client and server metrics) | Completed |
| 6 | Week 8 | Measure communication overhead per federated round (`federated/comm_analysis.py`) | Completed |
| 7 | Week 10 | Algorithm section of the report (Chapters 5–6), presentation | Completed |

## Form 4 — Ritesh Pandey (2023021154)

| S.No. | Week | Task Assigned | Task Status |
|---|---|---|---|
| 1 | Week 1 | Study FedAvg and secure aggregation techniques | Completed |
| 2 | Week 2 | Implement FedAvg global aggregation on the central server; XGBoost bagging aggregation | Completed |
| 3 | Week 3 | Set up 4 simulated federated client nodes locally (`federated/run_federated.py`) | Completed |
| 4 | Week 5 | Test model convergence across federated rounds (per-round JSON logs, convergence analysis) | Completed |
| 5 | Week 6 | Secure model-update transmission (TLS channel; server-side threshold calibration) | Completed |
| 6 | Week 8 | Update IDS dashboard: federated node status, round curves, DP/LSTM tables, prevention panel | Completed |
| 7 | Week 10 | Final integration testing (Streamlit AppTest), submission preparation | Completed |

## Items beyond the 10-week forms (from the Phase 1 report's future-work list)

| Item | Status |
|---|---|
| Autonomous prevention (IP blocking, firewall rules) | Completed: `core/prevention.py`, dry-run default, audit + undo |
| LIME alongside SHAP | Completed: LIME-style local surrogate of the hybrid score |
| Persistent logging | Completed: SQLite event store |
| LSTM temporal model | Experiment completed; negative result, not adopted (Report §7.5) |
| CICIDS2018 support | Schema mapping implemented; not run (dataset not on disk) |
| Blockchain-verified model updates | Not attempted (out of scope for Phase 2) |
