"""
federated/export_global_model.py
Make the dashboard use the federated global model.

    python -m federated.export_global_model

1. Backs up the Phase 1 artifacts once to hybrid_model_phase1_backup/.
2. Copies the federated Autoencoder, XGBoost booster, scaler, selected
   columns and threshold.json into hybrid_model/.
3. Writes hybrid_model/model_info.json describing where the model came from.
core/predictor.py reads hybrid_model/ on start-up, so restarting the
dashboard is all that is needed afterwards.
"""
import json
import os
import shutil
import time

from federated import config as C

PHASE1_FILES = ["autoencoder.keras", "xgb_model.pkl", "scaler.pkl", "selected_columns.pkl", "threshold.json"]


def main():
    src_ae = os.path.join(C.FL_MODEL_DIR, "autoencoder.keras")
    src_xgb = os.path.join(C.FL_MODEL_DIR, "xgb_model.json")
    src_thr = os.path.join(C.FL_MODEL_DIR, "threshold.json")
    for p in (src_ae, src_xgb, src_thr):
        if not os.path.exists(p):
            raise SystemExit(f"{p} missing. Run python -m federated.run_federated first.")

    backup = C.HYBRID_DIR + "_phase1_backup"
    if not os.path.exists(backup):
        os.makedirs(backup)
        for f in PHASE1_FILES:
            p = os.path.join(C.HYBRID_DIR, f)
            if os.path.exists(p):
                shutil.copy2(p, backup)
        print(f"Phase 1 artifacts backed up to {backup}")

    shutil.copy2(src_ae, os.path.join(C.HYBRID_DIR, "autoencoder.keras"))
    shutil.copy2(src_xgb, os.path.join(C.HYBRID_DIR, "xgb_model.json"))
    shutil.copy2(src_thr, os.path.join(C.HYBRID_DIR, "threshold.json"))
    shutil.copy2(os.path.join(C.DATA_DIR, "scaler.pkl"), os.path.join(C.HYBRID_DIR, "scaler.pkl"))
    shutil.copy2(os.path.join(C.DATA_DIR, "selected_columns.pkl"), os.path.join(C.HYBRID_DIR, "selected_columns.pkl"))
    protos = os.path.join(C.DATA_DIR, "family_prototypes.json")
    if os.path.exists(protos):
        shutil.copy2(protos, os.path.join(C.HYBRID_DIR, "family_prototypes.json"))
    # The old sklearn pickle would shadow the new booster; remove it from hybrid_model/
    old_pkl = os.path.join(C.HYBRID_DIR, "xgb_model.pkl")
    if os.path.exists(old_pkl):
        os.remove(old_pkl)

    comparison = {}
    cmp_path = os.path.join(C.LOG_DIR, "comparison.json")
    if os.path.exists(cmp_path):
        with open(cmp_path, encoding="utf-8") as f:
            comparison = json.load(f).get("federated_phase2", {})
    info = {
        "source": "federated (Flower) global model",
        "exported": time.strftime("%Y-%m-%d %H:%M:%S"),
        "nodes": C.NUM_NODES,
        "node_attack_families": C.NODE_ATTACK_FAMILIES,
        "training_data": "CIC-IDS2017, all 8 days, 30 selected features",
        "test_metrics": comparison,
    }
    with open(os.path.join(C.HYBRID_DIR, "model_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    print(f"Federated global model exported to {C.HYBRID_DIR}. Restart the dashboard.")


if __name__ == "__main__":
    main()
