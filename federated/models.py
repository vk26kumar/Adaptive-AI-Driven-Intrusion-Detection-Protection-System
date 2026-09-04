"""
federated/models.py
Model builders shared by every federated node and the server.
The Autoencoder architecture is identical to Phase 1 so results stay comparable.
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np


def build_autoencoder(input_dim: int):
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Dense

    inp = Input(shape=(input_dim,))
    x = Dense(128, activation="relu")(inp)
    x = Dense(64, activation="relu")(x)
    x = Dense(32, activation="relu")(x)
    x = Dense(8, activation="relu")(x)           # bottleneck
    x = Dense(32, activation="relu")(x)
    x = Dense(64, activation="relu")(x)
    x = Dense(128, activation="relu")(x)
    out = Dense(input_dim, activation="sigmoid")(x)
    model = Model(inp, out)
    model.compile(optimizer="adam", loss="mse")
    return model


def reconstruction_error(model, X: np.ndarray, batch_size: int = 8192) -> np.ndarray:
    rec = model.predict(X, batch_size=batch_size, verbose=0)
    return np.mean(np.square(X - rec), axis=1)


def anomaly_threshold(errors_benign: np.ndarray, percentile: float) -> float:
    return float(np.percentile(errors_benign, percentile))
