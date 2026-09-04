"""
federated/server.py
Central aggregation server.  Never sees raw traffic; it only receives model
parameters, aggregates them, evaluates the global model on the server-side
hold-out set, logs every round and saves the final global model.

    python -m federated.server --model ae  --rounds 10 [--tls] [--dp --noise 0.3 --clip 1.0]
    python -m federated.server --model xgb --rounds 15 [--tls]

Aggregation
  ae   FedAvg (weighted average of Autoencoder weights, weights = benign rows)
       optional server-side differential privacy (fixed clipping + Gaussian noise)
  xgb  Bagging (FedXgbBagging): each round every node's new trees are appended
"""
import argparse
import json
import os
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import flwr as fl
from flwr.common import Parameters, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import FedAvg, FedXgbBagging, DifferentialPrivacyServerSideFixedClipping

from federated import config as C
from federated.data_utils import load_test
from federated.gen_certs import load_server_certificates
from federated.metrics_logger import RoundLogger
from federated.models import build_autoencoder, reconstruction_error, anomaly_threshold


def _binary_metrics(y, pred):
    y = y.astype(bool); pred = pred.astype(bool)
    tp = int((pred & y).sum()); fp = int((pred & ~y).sum())
    fn = int((~pred & y).sum()); tn = int((~pred & ~y).sum())
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    return {
        "accuracy": (tp + tn) / max(len(y), 1), "precision": prec, "recall": rec,
        "f1": 2 * prec * rec / max(prec + rec, 1e-9), "fpr": fp / max(fp + tn, 1),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def _weighted_avg(results):
    """fit/evaluate metrics aggregation: weighted mean of numeric client metrics."""
    total = sum(n for n, _ in results) or 1
    out = {}
    keys = {k for _, m in results for k in m if k != "node"}
    for k in keys:
        vals = [(n, m[k]) for n, m in results if k in m and isinstance(m[k], (int, float))]
        if vals:
            out[k] = float(sum(n * v for n, v in vals) / max(sum(n for n, _ in vals), 1))
    if "bytes_up" in keys:  # totals make more sense than averages for traffic
        out["bytes_up_total"] = int(sum(m.get("bytes_up", 0) for _, m in results))
    out["num_clients"] = len(results)
    return out


# ---------------------------------------------------------------------------
class CapturingFedAvg(FedAvg):
    """FedAvg that remembers the latest aggregated parameters so we can save them."""
    latest_parameters = None
    latest_fit_metrics = {}

    def aggregate_fit(self, server_round, results, failures):
        params, metrics = super().aggregate_fit(server_round, results, failures)
        if params is not None:
            self.latest_parameters = params
        self.latest_fit_metrics = metrics
        return params, metrics


def run_autoencoder(args, X_test, y_test, logger: RoundLogger):
    model = build_autoencoder(C.AE_INPUT_DIM)
    init_weights = model.get_weights()
    param_bytes = int(sum(w.nbytes for w in init_weights))
    X_test_benign = X_test[y_test == 0]
    state = {"threshold": 0.0}

    def evaluate_fn(server_round, ndarrays, config):
        model.set_weights(ndarrays)
        mse = reconstruction_error(model, X_test)
        thr = anomaly_threshold(mse[y_test == 0], C.AE_THRESHOLD_PERCENTILE)
        state["threshold"] = thr
        pred = mse > C.ANOMALY_MULTIPLIER * thr
        m = _binary_metrics(y_test, pred)
        m.update({"benign_mse_mean": float(mse[y_test == 0].mean()),
                  "attack_mse_mean": float(mse[y_test == 1].mean()),
                  "ae_threshold": thr,
                  "bytes_down_total": param_bytes * C.NUM_NODES,
                  "bytes_up_total": param_bytes * C.NUM_NODES,
                  "bytes_per_client": param_bytes})
        logger.log_round(server_round, model="ae", **m)
        print(f"[server][ae] round {server_round}: benign_mse={m['benign_mse_mean']:.3e} "
              f"thr90={thr:.3e} recall={m['recall']:.3f} fpr={m['fpr']:.4f}")
        return float(mse[y_test == 0].mean()), m

    def fit_config(server_round):
        return {"local_epochs": C.AE_LOCAL_EPOCHS, "server_round": server_round}

    def eval_config(server_round):
        return {"ae_threshold": state["threshold"], "anomaly_multiplier": C.ANOMALY_MULTIPLIER}

    strategy = CapturingFedAvg(
        fraction_fit=1.0, fraction_evaluate=1.0,
        min_fit_clients=C.NUM_NODES, min_evaluate_clients=C.NUM_NODES,
        min_available_clients=C.NUM_NODES,
        initial_parameters=ndarrays_to_parameters(init_weights),
        evaluate_fn=evaluate_fn, on_fit_config_fn=fit_config,
        on_evaluate_config_fn=eval_config,
        fit_metrics_aggregation_fn=_weighted_avg,
        evaluate_metrics_aggregation_fn=_weighted_avg,
    )
    inner = strategy
    if args.dp:
        strategy = DifferentialPrivacyServerSideFixedClipping(
            strategy, noise_multiplier=args.noise, clipping_norm=args.clip,
            num_sampled_clients=C.NUM_NODES)
        print(f"[server][ae] differential privacy ON: noise_multiplier={args.noise} clipping_norm={args.clip}")

    hist = _start(args, strategy)

    # Save the global model + the threshold derived on the server hold-out set
    os.makedirs(C.FL_MODEL_DIR, exist_ok=True)
    model.set_weights(parameters_to_ndarrays(inner.latest_parameters))
    model.save(os.path.join(C.FL_MODEL_DIR, "autoencoder.keras"))
    from federated.calibrate import calibrate
    calibrate(C.FL_MODEL_DIR)          # ae_threshold (and ml_threshold if a booster exists)
    logger.finish(rounds_completed=args.rounds, dp=bool(args.dp),
                  dp_noise_multiplier=args.noise if args.dp else None,
                  dp_clipping_norm=args.clip if args.dp else None,
                  bytes_per_round=param_bytes * C.NUM_NODES * 2,
                  bytes_total=param_bytes * C.NUM_NODES * 2 * args.rounds,
                  ae_threshold=state["threshold"], tls=bool(args.tls))
    print(f"[server][ae] saved global autoencoder + threshold to {C.FL_MODEL_DIR}")


# ---------------------------------------------------------------------------
def run_xgboost(args, X_test, y_test, logger: RoundLogger):
    import xgboost as xgb
    dtest = xgb.DMatrix(X_test, label=y_test)
    strategy_holder = {}

    def evaluate_fn(server_round, parameters: Parameters, config):
        if not parameters.tensors:
            return None
        raw = parameters.tensors[0]
        bst = xgb.Booster(params=dict(C.XGB_PARAMS))
        bst.load_model(bytearray(raw))
        p = bst.predict(dtest)
        m = _binary_metrics(y_test, p > C.ML_THRESHOLD)
        eps = 1e-7
        loss = float(-np.mean(y_test * np.log(p + eps) + (1 - y_test) * np.log(1 - p + eps)))
        try:
            from sklearn.metrics import roc_auc_score
            m["auc"] = float(roc_auc_score(y_test, p))
        except Exception:
            pass
        m.update({"logloss": loss, "num_trees": int(bst.num_boosted_rounds() * C.XGB_TREES_PER_ROUND),
                  "global_model_bytes": len(raw),
                  "bytes_down_total": len(raw) * C.NUM_NODES})
        logger.log_round(server_round, model="xgb", **m)
        print(f"[server][xgb] round {server_round}: trees={m['num_trees']} acc={m['accuracy']:.4f} "
              f"recall={m['recall']:.4f} fpr={m['fpr']:.4f} logloss={loss:.4f}")
        return loss, m

    strategy = FedXgbBagging(
        evaluate_function=evaluate_fn,
        fraction_fit=1.0, fraction_evaluate=1.0,
        min_fit_clients=C.NUM_NODES, min_evaluate_clients=C.NUM_NODES,
        min_available_clients=C.NUM_NODES,
        initial_parameters=Parameters(tensor_type="", tensors=[]),
        fit_metrics_aggregation_fn=_weighted_avg,
        evaluate_metrics_aggregation_fn=_weighted_avg,
    )
    strategy_holder["s"] = strategy
    _start(args, strategy)

    os.makedirs(C.FL_MODEL_DIR, exist_ok=True)
    raw = strategy.global_model
    with open(os.path.join(C.FL_MODEL_DIR, "xgb_model.json"), "wb") as f:
        f.write(raw)
    from federated.calibrate import calibrate
    cal = calibrate(C.FL_MODEL_DIR)    # picks ml_threshold on the calibration split
    logger.data["ml_threshold"] = cal.get("ml_threshold")
    logger.finish(rounds_completed=args.rounds, final_model_bytes=len(raw), tls=bool(args.tls),
                  trees_total=args.rounds * C.NUM_NODES * C.XGB_TREES_PER_ROUND)
    print(f"[server][xgb] saved global booster ({len(raw):,} bytes) to {C.FL_MODEL_DIR}")


# ---------------------------------------------------------------------------
def _start(args, strategy):
    kwargs = dict(server_address=args.address,
                  config=fl.server.ServerConfig(num_rounds=args.rounds),
                  strategy=strategy)
    if args.tls:
        kwargs["certificates"] = load_server_certificates()
        print(f"[server] TLS enabled, listening on {args.address}")
    else:
        print(f"[server] INSECURE (no TLS), listening on {args.address}")
    return fl.server.start_server(**kwargs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["ae", "xgb"], required=True)
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--address", default=C.SERVER_ADDRESS)
    ap.add_argument("--tls", action="store_true")
    ap.add_argument("--dp", action="store_true", help="server-side differential privacy (ae only)")
    ap.add_argument("--noise", type=float, default=0.3, help="DP noise multiplier")
    ap.add_argument("--clip", type=float, default=1.0, help="DP clipping norm")
    ap.add_argument("--out-dir", default=C.FL_MODEL_DIR, help="where to save the global model")
    args = ap.parse_args()
    C.FL_MODEL_DIR = args.out_dir          # all save paths below read this
    if args.rounds is None:
        args.rounds = C.AE_ROUNDS if args.model == "ae" else C.XGB_ROUNDS

    X_test, y_test, _ = load_test("report")   # thresholds are tuned on the 'calib' half
    run_name = f"{args.model}{'_dp' if (args.dp and args.model == 'ae') else ''}"
    logger = RoundLogger(run_name)
    t0 = time.time()
    if args.model == "ae":
        run_autoencoder(args, X_test, y_test, logger)
    else:
        if args.dp:
            print("[server] --dp is ignored for xgb: tree bagging has no gradient to clip/noise.")
        run_xgboost(args, X_test, y_test, logger)
    print(f"[server] finished {args.model} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
