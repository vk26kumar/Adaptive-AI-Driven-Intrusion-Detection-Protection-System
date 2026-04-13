import joblib
import numpy as np

# Load everything once
rf_model = joblib.load("models/rf_model.pkl")
xgb_model = joblib.load("models/xgb_model.pkl")
scaler = joblib.load("models/scaler.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")


def predict_attack(features):
    """
    features = list or array of input values (no label)
    """

    # Convert to numpy
    features = np.array(features).reshape(1, -1)

    # Scale input
    features_scaled = scaler.transform(features)

    # Predict using XGBoost (main model)
    prediction = xgb_model.predict(features_scaled)

    # Convert to label (ATTACK / NORMAL)
    label = label_encoder.inverse_transform(prediction)

    return label[0]