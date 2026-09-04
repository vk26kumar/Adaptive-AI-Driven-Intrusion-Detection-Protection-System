"""
federated/client.py
One federated node.  Trains on its own partition only; nothing but model
parameters ever leaves the process.

    python -m federated.client --node 0 --model ae  [--tls]
    python -m federated.client --node 0 --model xgb [--tls]

--model ae   Autoencoder, aggregated with FedAvg (weights are averaged).
--model xgb  XGBoost, aggregated with bagging (each node adds trees on top of
             the global booster; the server concatenates them).
"""
import argparse
import os
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import flwr as fl
from flwr.common import (
    Code, EvaluateIns, EvaluateRes, FitIns, FitRes, GetParametersIns,
    GetParametersRes, Parameters, Status,
)

from federated import config as C
from federated.data_utils import load_node
from federated.gen_certs import load_root_certificate
from federated.models import build_autoencoder, reconstruction_error


# ---------------------------------------------------------------------------
# Autoencoder node (FedAvg)
# ---------------------------------------------------------------------------
class AutoencoderClient(fl.client.NumPyClient):
    def __init__(self, node_id: int):
        self.node_id = node_id
        X, y, _ = load_node(node_id)
        self.X_benign = X[y == 0]          # the AE learns NORMAL traffic only
        self.X_attack = X[y == 1]
        self.model = build_autoencoder(C.AE_INPUT_DIM)
        print(f"[node {node_id}] AE client ready: benign={len(self.X_benign):,} attack={len(self.X_attack):,}")

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        epochs = int(config.get("local_epochs", C.AE_LOCAL_EPOCHS))
        t0 = time.perf_counter()
        hist = self.model.fit(self.X_benign, self.X_benign, epochs=epochs,
                              batch_size=C.AE_BATCH_SIZE, verbose=0, shuffle=True)
        weights = self.model.get_weights()
        metrics = {
            "node": self.node_id,
            "train_loss": float(hist.history["loss"][-1]),
            "train_seconds": time.perf_counter() - t0,
            "bytes_up": int(sum(w.nbytes for w in weights)),
        }
        return weights, len(self.X_benign), metrics

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        thr = float(config.get("ae_threshold", 0.0))
        mult = float(config.get("anomaly_multiplier", C.ANOMALY_MULTIPLIER))
        mse_b = reconstruction_error(self.model, self.X_benign)
        metrics = {"node": self.node_id, "benign_mse_mean": float(mse_b.mean())}
        if thr > 0:
            metrics["benign_fpr"] = float((mse_b > mult * thr).mean())
            if len(self.X_attack):
                mse_a = reconstruction_error(self.model, self.X_attack)
                metrics["attack_recall"] = float((mse_a > mult * thr).mean())
        return float(mse_b.mean()), len(self.X_benign), metrics


# ---------------------------------------------------------------------------
# XGBoost node (bagging)
# ---------------------------------------------------------------------------
class XgbClient(fl.client.Client):
    def __init__(self, node_id: int):
        import xgboost as xgb
        self.xgb = xgb
        self.node_id = node_id
        X, y, _ = load_node(node_id)
        self.dtrain = xgb.DMatrix(X, label=y)
        self.y = y
        self.params = dict(C.XGB_PARAMS)
        # Class balance differs per node; scale_pos_weight keeps local trees honest.
        pos = max(int(y.sum()), 1)
        self.params["scale_pos_weight"] = float((len(y) - pos) / pos)
        print(f"[node {node_id}] XGB client ready: rows={len(y):,} attacks={pos:,} "
              f"scale_pos_weight={self.params['scale_pos_weight']:.3f}")

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        return GetParametersRes(status=Status(Code.OK, "ok"),
                                parameters=Parameters(tensor_type="", tensors=[]))

    def _booster_from(self, tensors):
        bst = self.xgb.Booster(params=self.params)
        if tensors:
            bst.load_model(bytearray(tensors[0]))
        return bst

    def fit(self, ins: FitIns) -> FitRes:
        t0 = time.perf_counter()
        global_tensors = ins.parameters.tensors
        if not global_tensors:
            # Round 1: nothing to build on yet -> one boosting iteration from scratch
            bst = self.xgb.train(self.params, self.dtrain, num_boost_round=1)
        else:
            bst = self._booster_from(global_tensors)
            bst.update(self.dtrain, bst.num_boosted_rounds())
        # Send back ONLY the trees this node just added (one iteration =
        # num_parallel_tree trees); the server appends them to the global model.
        n = bst.num_boosted_rounds()
        local = bst[n - 1:n]
        raw = bytes(local.save_raw("json"))
        metrics = {"node": self.node_id, "bytes_up": len(raw),
                   "train_seconds": time.perf_counter() - t0,
                   "trees_added": int(self.params["num_parallel_tree"])}
        return FitRes(status=Status(Code.OK, "ok"),
                      parameters=Parameters(tensor_type="", tensors=[raw]),
                      num_examples=len(self.y), metrics=metrics)

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        if not ins.parameters.tensors:
            return EvaluateRes(status=Status(Code.OK, "ok"), loss=0.0,
                               num_examples=len(self.y), metrics={"node": self.node_id})
        bst = self._booster_from(ins.parameters.tensors)
        p = bst.predict(self.dtrain)
        pred = p > C.ML_THRESHOLD
        eps = 1e-7
        logloss = float(-np.mean(self.y * np.log(p + eps) + (1 - self.y) * np.log(1 - p + eps)))
        metrics = {
            "node": self.node_id,
            "accuracy": float((pred == self.y).mean()),
            "attack_recall": float(pred[self.y == 1].mean()) if (self.y == 1).any() else 0.0,
            "benign_fpr": float(pred[self.y == 0].mean()) if (self.y == 0).any() else 0.0,
        }
        return EvaluateRes(status=Status(Code.OK, "ok"), loss=logloss,
                           num_examples=len(self.y), metrics=metrics)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--node", type=int, required=True)
    ap.add_argument("--model", choices=["ae", "xgb"], required=True)
    ap.add_argument("--server", default=C.SERVER_ADDRESS)
    ap.add_argument("--tls", action="store_true", help="verify the server with federated/certs/ca.crt")
    args = ap.parse_args()

    if args.model == "ae":
        client = AutoencoderClient(args.node).to_client()
    else:
        client = XgbClient(args.node)

    kwargs = dict(server_address=args.server, client=client)
    if args.tls:
        kwargs["root_certificates"] = load_root_certificate()
    else:
        kwargs["insecure"] = True
    fl.client.start_client(**kwargs)


if __name__ == "__main__":
    main()
