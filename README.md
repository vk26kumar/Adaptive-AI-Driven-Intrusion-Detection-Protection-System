# A-IDAPS-FL — Adaptive AI-Driven Intrusion Detection & Protection System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange?style=for-the-badge&logo=tensorflow)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-green?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red?style=for-the-badge&logo=streamlit)
![Scapy](https://img.shields.io/badge/Scapy-2.5+-purple?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

**A real-time hybrid ML + Deep Learning Intrusion Detection System with live dashboard**

*B.Tech Major Project | Group G-52 | Dept. of CSE (ML & Cyber Security) | MMMUT Gorakhpur*

</div>

---

## Overview

**A-IDAPS-FL** (Adaptive AI-Driven Intrusion Detection & Protection System using Federated Learning) is a Phase 1 implementation of a real-time network intrusion detection system that combines:

- 🤖 **XGBoost** (Machine Learning) for signature-based attack classification
- 🧠 **Autoencoder** (Deep Learning) for anomaly/zero-day detection  
- 🔀 **Hybrid Ensemble** combining both models for 99.0% accuracy
- 📡 **Live Packet Sniffer** using Scapy for real-time traffic analysis
- 📊 **Streamlit Dashboard** with attack simulation, SHAP explainability, and confusion matrix

---

## Key Features

| Feature | Description |
|---------|-------------|
| 🔴 Live IDS | Real-time packet capture and classification |
| ⚡ Attack Simulator | Test DDoS, BruteForce, SQLInject, PortScan detection |
| 🧠 Hybrid Model | Autoencoder (DL) + XGBoost (ML) ensemble |
| 📈 SHAP Explainability | Visual feature importance for every prediction |
| 🎯 99.0% Accuracy | Trained on CIC-IDS2017 benchmark dataset |
| 📊 Live Metrics | Confusion matrix, attack breakdown, latency tracking |

---

## Project Structure

```
MLMAJORPROJECT/
├── app.py                    ← Streamlit dashboard (run this)
├── requirements.txt
│
├── core/
│   ├── predictor.py          ← Hybrid model inference (Autoencoder + XGBoost)
│   ├── simulator.py          ← Attack feature vector builder
│   └── state.py              ← Shared session state & stats tracker
│
├── network/
│   ├── features.py           ← Real-time packet feature extraction (30 features)
│   └── sniffer.py            ← Scapy packet capture thread
│
├── hybrid_model/             ← Trained model artifacts (not in repo - see below)
│   ├── autoencoder.keras
│   ├── xgb_model.pkl
│   ├── scaler.pkl
│   ├── selected_columns.pkl
│   └── threshold.json
│
└── models/                   ← ML pipeline models (not in repo - see below)
    ├── rf_model.pkl
    ├── xgb_model.pkl
    ├── scaler.pkl
    └── label_encoder.pkl
```

---

##Setup & Run

### 1. Clone the repository
```bash
git clone https://github.com/vk26kumar/Adaptive-AI-Driven-Intrusion-Detection-Protection-System.git
cd Adaptive-AI-Driven-Intrusion-Detection-Protection-System
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the models (or download pre-trained artifacts)

The model files are not included in the repo due to size. Train them yourself:

- **Deep Learning model** → Run the Colab notebook (see `/notebooks/` or Colab link)
- **ML models** → Run `A_IDAPS_FL_Phase1_ML_Final.py`

After training, place files in `hybrid_model/` directory:
```
autoencoder.keras
xgb_model.pkl
scaler.pkl
selected_columns.pkl
threshold.json
```

### 4. Run the dashboard
```bash
# Windows (run as Administrator for live packet capture)
python -m streamlit run app.py

# Linux/Mac (requires root for packet capture)
sudo streamlit run app.py
```

Dashboard opens at: **http://localhost:8501**

---

##Model Architecture

### Deep Learning — Autoencoder
```
Input (30) → Dense(128) → Dense(64) → Dense(32) → Bottleneck(8)
           → Dense(32)  → Dense(64) → Dense(128) → Output(30)

Optimizer : Adam
Loss      : Mean Squared Error (MSE)
Training  : BENIGN traffic only (97,718 samples)
Threshold : 90th percentile of reconstruction error
```

### Machine Learning — XGBoost
```
n_estimators   : 300
max_depth      : 8
learning_rate  : 0.05
subsample      : 0.8
colsample_bytree: 0.8
```

### Hybrid Ensemble
```
dl_score    = clip(dl_error / (threshold × 2),  0, 1)
final_score = clip(dl_score + 0.3 × ml_prob,    0, 1)
label       = ATTACK if final_score > 0.5 else NORMAL
```

---

## Performance Results

| Metric    | Autoencoder | XGBoost | **Hybrid Ensemble** |
|-----------|:-----------:|:-------:|:-------------------:|
| Accuracy  | 77.0%       | 99.1%   | **99.0%**           |
| Precision | 79.0%       | 98.9%   | **98.8%**           |
| Recall    | 78.0%       | 99.2%   | **99.2%**           |
| F1-Score  | 77.0%       | 99.1%   | **99.0%**           |
| AUC-ROC   | —           | 99.6%   | **99.7%**           |

*Evaluated on CIC-IDS2017 (225,745 samples)*

---

## ⚡ Attack Simulator Guide

| Attack Type | Destination Port | Packets/sec Rate | Expected Result |
|-------------|:----------------:|:----------------:|:---------------:|
| **DDoS**        | 80              | > 1000           | 🔴 ATTACK       |
| **BruteForce**  | 22 / 3389       | > 200            | 🔴 ATTACK       |
| **SQL Inject**  | 3306 / 5432     | > 200            | 🔴 ATTACK       |
| **Port Scan**   | Any             | 101 – 1000       | 🔴 ATTACK       |
| **Normal**      | 80 / 443        | < 100            | 🟢 NORMAL       |

---

## Dataset

- **Name:** CIC-IDS2017 (Canadian Institute for Cybersecurity)
- **File:** `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`
- **Samples:** 225,745 (128,027 DDoS + 97,718 BENIGN)
- **Features:** 79 raw → 30 selected via Random Forest importance
- **Download:** [UNB CIC-IDS2017](https://www.unb.ca/cic/datasets/ids-2017.html)

---

## References

1. McMahan et al., *"Communication-efficient learning of deep networks from decentralized data"*, PMLR 2017
2. Buczak & Guven, *"A survey of data mining and ML methods for cyber security IDS"*, IEEE 2015
3. Balyan et al., *"ML-based IDS for healthcare data"*, IEEE VLSI DCS 2022
4. Garcia-Teodoro et al., *"Anomaly-based network intrusion detection"*, Computers & Security 2009
5. Lundberg & Lee, *"A unified approach to interpreting model predictions (SHAP)"*, NeurIPS 2017
6. Ribeiro et al., *"Why should I trust you? (LIME)"*, ACM SIGKDD 2016

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">


</div>
