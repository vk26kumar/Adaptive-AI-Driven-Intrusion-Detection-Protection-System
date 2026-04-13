# A-IDAPS-FL | Phase 1 Dashboard

## Project Structure

```
MLMAJORPROJECT/
├── app.py                    ← Streamlit dashboard (run this)
├── requirements.txt
│
├── core/
│   ├── predictor.py          ← Loads models, hybrid DL+ML prediction
│   ├── simulator.py          ← Attack simulator feature builder
│   └── state.py              ← Shared session state & stats tracker
│
├── network/
│   ├── features.py           ← Real-time packet feature extraction
│   └── sniffer.py            ← Scapy packet capture thread
│
├── hybrid_model/             ← YOUR TRAINED MODEL FILES (unchanged)
│   ├── autoencoder.keras
│   ├── xgb_model.pkl
│   ├── scaler.pkl
│   ├── selected_columns.pkl
│   └── threshold.json
│
└── models/                   ← YOUR ML MODEL FILES (unchanged)
    ├── rf_model.pkl
    ├── xgb_model.pkl
    ├── scaler.pkl
    └── label_encoder.pkl
```

## Setup & Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

> **Note:** Live packet capture requires root/admin privileges.
> On Linux/Mac: `sudo streamlit run app.py`
> On Windows: Run terminal as Administrator.

## Features

| Feature | Description |
|---|---|
| Live Packet Feed | Real-time table of captured packets with ATTACK/NORMAL verdict |
| Traffic Timeline | Live chart of attacks vs normal packets over time |
| Attack Simulator | Tune 5 features via sliders → inject → model predicts |
| SHAP Explainability | Feature importance bars showing why model classified packet |
| Confusion Matrix | Live TP/TN/FP/FN counts |
| Attack Breakdown | DDoS / PortScan / BruteForce / SQLInject / Other |
| Model Metrics | Accuracy, Precision, Recall, F1, AUC-ROC |
| Autoencoder Viz | Layer architecture display |

## How Attack Simulation Works

1. Use sliders to tune: Packet Length, Destination Port, Flow Duration, Packets/sec Rate, Fwd Packet Count, Protocol
2. Click **⚡ Run Prediction**
3. The same `core/predictor.py` hybrid model (Autoencoder + XGBoost) runs on your synthetic feature vector
4. Results show ML probability, DL score, ensemble final score and verdict
5. Simulated packet appears in the live feed and updates all counters

## Tips for Demo

- **Trigger DDoS**: Set Pkt/s Rate > 1000, Flow Duration < 100ms
- **Trigger BruteForce**: Set Dst Port = 22 or 3389, moderate rate
- **Trigger SQLInject**: Set Dst Port = 3306 or 5432
- **Normal traffic**: Port 80/443, Rate < 100, Duration > 500ms
