# A-IDAPS-FL Phase 2 — Presentation Outline (12 slides, ~15 minutes)

1. **Title** — A-IDAPS-FL: Adaptive AI-Driven Intrusion Detection & Protection using Federated Learning. Group G-52, supervisor, session.
2. **Problem** — ML-IDS needs traffic data; centralising traffic is the privacy problem. Goal: one detector, no shared traffic.
3. **Phase 1 recap and audit** — hybrid AE + XGBoost, dashboard. Audit found: wrong threshold file (0% live recall), unit mismatch in the live extractor, DDoS-only training (5% port-scan recall). All fixed.
4. **Data** — CIC-IDS2017, all 8 days, 2.83 M flows, 7 attack families. Four non-IID nodes table. Calibration/report split of the hold-out.
5. **Architecture** — diagram: 4 node processes → TLS → Flower server; FedAvg for AE, bagging for XGBoost; server-side calibration; export to dashboard.
6. **Decision rule** — final = max(dl_score, ml_score); "either detector confident". Why calibration is needed under non-IID bagging (AUC 0.9998, recall 27% → 99.95%).
7. **Results** — table Phase 1 vs centralized vs federated; per-family recall chart. Federated 99.23% / 99.95% recall / 0 raw rows shared.
8. **Communication and convergence** — AE 14 MB, XGB 257 MB vs 274 MB raw; recall converges by round 4; delta-trees optimisation.
9. **Differential privacy** — ε table; honest conclusion: not meaningful with 4 clients, pipeline ready for many.
10. **Protection and explainability** — auto-block policy (dry-run, allowlist, audit, undo); TreeSHAP + AE error share + LIME.
11. **Live demo** — dashboard: simulate PortScan → alert → XAI → confusion matrix; federated panel with round curves.
12. **Conclusion and future work** — negative results stated (DP, LSTM); next: delta trees, more clients for DP, CICIDS2018, lab validation.

Demo checklist: `streamlit run app.py` as Administrator; simulator profiles Normal, PortScan, BruteForce; toggle LIME; enable Auto-block in dry-run; open "Differential privacy runs" expander.
