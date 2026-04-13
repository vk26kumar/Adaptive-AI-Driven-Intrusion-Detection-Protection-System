import numpy as np
import pandas as pd
import joblib
import json
from tensorflow.keras.models import load_model

# Load models
autoencoder = load_model("hybrid_model/autoencoder.keras")
xgb = joblib.load("hybrid_model/xgb_model.pkl")
scaler = joblib.load("hybrid_model/scaler.pkl")
selected_columns = joblib.load("hybrid_model/selected_columns.pkl")

with open("hybrid_model/threshold.json") as f:
    threshold = json.load(f)["threshold"]


def predict_intrusion(df):
    # Select required features
    df = df[selected_columns]

    # Clean data
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # Scale data
    data = scaler.transform(df)
     
    # Deep Learning (Autoencoder)
     
    recon = autoencoder.predict(data)
    dl_error = np.mean(np.square(data - recon), axis=1)

    # Normalize DL score
    dl_score = (dl_error - dl_error.min()) / (dl_error.max() - dl_error.min())
     
    # Machine Learning (XGBoost)
    ml_prob = xgb.predict_proba(data)[:, 1]
     
    # Hybrid Decision
    final_score = 0.5 * dl_score + 0.5 * ml_prob
    predictions = ["ATTACK" if s > 0.5 else "NORMAL" for s in final_score]
    return predictions