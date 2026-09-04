"""
federated/run_federated.py
Launch a complete federated training run on one machine: a server process
plus NUM_NODES client processes, first for the Autoencoder (FedAvg) and then
for XGBoost (bagging).  Logs go to federated/logs/.

    python -m federated.run_federated                 # both models, no TLS
    python -m federated.run_federated --tls           # encrypted gRPC
    python -m federated.run_federated --tls --dp      # + differential privacy on the AE
    python -m federated.run_federated --only xgb --rounds 20
"""
import argparse
import os
import subprocess
import sys
import time

from federated import config as C


def _launch(args_list, log_path):
    log = open(log_path, "w", encoding="utf-8")
    env = dict(os.environ, TF_CPP_MIN_LOG_LEVEL="3", PYTHONIOENCODING="utf-8")
    return subprocess.Popen([sys.executable, "-m", *args_list], stdout=log, stderr=subprocess.STDOUT,
                            cwd=C.ROOT_DIR, env=env), log


def run_one(model: str, rounds: int, tls: bool, dp: bool, noise: float, clip: float, out_dir: str = None) -> bool:
    os.makedirs(C.LOG_DIR, exist_ok=True)
    tag = f"{model}{'_dp' if dp and model == 'ae' else ''}"
    print(f"\n=== Federated {model.upper()}  rounds={rounds}  tls={tls}  dp={dp and model == 'ae'} ===")

    srv_args = ["federated.server", "--model", model, "--rounds", str(rounds)]
    if tls:
        srv_args.append("--tls")
    if dp and model == "ae":
        srv_args += ["--dp", "--noise", str(noise), "--clip", str(clip)]
    if out_dir:
        srv_args += ["--out-dir", out_dir]
    server, srv_log = _launch(srv_args, os.path.join(C.LOG_DIR, f"server_{tag}.log"))
    time.sleep(6)  # let the server bind before the nodes connect

    clients = []
    for node in range(C.NUM_NODES):
        cl_args = ["federated.client", "--node", str(node), "--model", model]
        if tls:
            cl_args.append("--tls")
        clients.append(_launch(cl_args, os.path.join(C.LOG_DIR, f"node{node}_{tag}.log")))
        time.sleep(1)

    t0 = time.time()
    rc = server.wait()
    for p, log in clients:
        try:
            p.wait(timeout=60)
        except subprocess.TimeoutExpired:
            p.kill()
        log.close()
    srv_log.close()
    print(f"=== {model.upper()} done in {time.time() - t0:.0f}s (server exit code {rc}) ===")
    if rc != 0:
        print(f"    see {os.path.join(C.LOG_DIR, f'server_{tag}.log')}")
    return rc == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["ae", "xgb"], help="run just one model")
    ap.add_argument("--rounds", type=int, help="override rounds for the selected model(s)")
    ap.add_argument("--tls", action="store_true")
    ap.add_argument("--dp", action="store_true")
    ap.add_argument("--noise", type=float, default=0.3)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--out-dir", default=None, help="save global models here instead of hybrid_model_fl/")
    args = ap.parse_args()

    if args.tls and not os.path.exists(os.path.join(C.CERT_DIR, "ca.crt")):
        subprocess.check_call([sys.executable, "-m", "federated.gen_certs"], cwd=C.ROOT_DIR)

    ok = True
    if args.only in (None, "ae"):
        ok &= run_one("ae", args.rounds or C.AE_ROUNDS, args.tls, args.dp, args.noise, args.clip, args.out_dir)
    if args.only in (None, "xgb"):
        ok &= run_one("xgb", args.rounds or C.XGB_ROUNDS, args.tls, False, 0, 0, args.out_dir)
    if ok:
        print(f"\nGlobal models saved in {args.out_dir or C.FL_MODEL_DIR}")
        print("Next:  python -m federated.evaluate          (centralized vs federated)")
        print("       python -m federated.export_global_model (make the dashboard use them)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
