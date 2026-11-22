"""
report/make_figures.py
Regenerates every chart and diagram used in the final report from the logged
results (federated/logs/*.json) and the saved models.

    python report/make_figures.py            (writes report/figures/*.png)

Colours: a validated categorical palette (slots blue / orange / aqua, light
mode); at most three series per chart, one y-axis per chart, value labels on
bars because aqua is below 3:1 contrast on white.
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
LOG = os.path.join(ROOT, "federated", "logs")
OUT = os.path.join(ROOT, "report", "figures")
os.makedirs(OUT, exist_ok=True)

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.family": "Times New Roman", "font.size": 11, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "legend.frameon": False, "savefig.dpi": 200, "savefig.bbox": "tight",
})


def load(name):
    with open(os.path.join(LOG, name), encoding="utf-8") as f:
        return json.load(f)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), facecolor="white")
    plt.close(fig)
    print("  ", name)


# ----------------------------------------------------------------- diagrams --
def _box(ax, x, y, w, h, text, fc="#eef4fc", ec=BLUE, fs=10, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK,
            fontweight="bold" if bold else "normal", wrap=True)


def _arrow(ax, x1, y1, x2, y2, text=None, color=INK2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12, lw=1.2, color=color))
    if text:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.12, text, ha="center", va="bottom", fontsize=8.5, color=INK2)


def fig_architecture():
    fig, ax = plt.subplots(figsize=(10, 6.2)); ax.set_xlim(0, 10); ax.set_ylim(0, 6.4); ax.axis("off")
    ax.grid(False)
    # federated training layer
    ax.text(0.1, 6.15, "Federated training (Flower, gRPC + TLS)", fontsize=11, fontweight="bold", color=INK)
    for i, t in enumerate(["Node 0\nDDoS, DoS", "Node 1\nPortScan, Bot", "Node 2\nBruteForce, Web", "Node 3\nalmost only benign"]):
        _box(ax, 0.1 + i * 1.55, 4.75, 1.4, 0.95, t, fs=9)
        _arrow(ax, 0.8 + i * 1.55, 4.75, 3.4, 3.95)
    ax.text(6.35, 4.3, "only model weights / new trees\nare sent; raw flows stay on the node", fontsize=8.5, color=INK2, ha="left")
    _box(ax, 2.2, 3.05, 2.4, 0.9, "Aggregation server\nFedAvg (AE) · bagging (XGB)", fc="#fff1ea", ec=ORANGE, fs=9)
    _box(ax, 4.95, 3.05, 2.0, 0.9, "Server calibration\ncutoff + Platt scaling", fc="#fff1ea", ec=ORANGE, fs=9)
    _arrow(ax, 4.6, 3.5, 4.95, 3.5)
    _box(ax, 7.3, 3.05, 2.55, 0.9, "Global model\nhybrid_model/", fc="#e8f7f1", ec=AQUA, fs=9, bold=True)
    _arrow(ax, 6.95, 3.5, 7.3, 3.5, )
    # detection layer
    ax.text(0.1, 2.55, "Detection and protection (each host)", fontsize=11, fontweight="bold", color=INK)
    _box(ax, 0.1, 1.15, 1.75, 1.05, "Npcap / Scapy\npacket capture", fs=9)
    _box(ax, 2.15, 1.15, 1.9, 1.05, "Flow features\n30 CIC-IDS2017\nfeatures", fs=9)
    _box(ax, 4.35, 1.15, 2.6, 1.05, "Hybrid model\nmax(AE score, XGB score)\n+ behavioural scan detector", fs=9, bold=True)
    _box(ax, 7.3, 1.6, 2.55, 0.6, "Prevention: firewall block\n(audit + undo)", fc="#fdecec", ec="#e34948", fs=8.5)
    _box(ax, 7.3, 0.75, 2.55, 0.7, "Dashboard · SQLite log\nSHAP / LIME explanations", fs=8.5)
    _arrow(ax, 1.85, 1.67, 2.15, 1.67); _arrow(ax, 4.05, 1.67, 4.35, 1.67)
    _arrow(ax, 6.95, 1.85, 7.3, 1.9); _arrow(ax, 6.95, 1.45, 7.3, 1.1)
    _arrow(ax, 8.55, 3.05, 5.9, 2.2, color=AQUA)
    ax.text(7.75, 2.62, "deployed to every host", fontsize=8.5, color=INK2)
    save(fig, "fig_architecture.png")


def fig_fl_round():
    fig, ax = plt.subplots(figsize=(10, 3.2)); ax.set_xlim(0, 10); ax.set_ylim(0, 3); ax.axis("off"); ax.grid(False)
    steps = ["1. Server sends\nglobal model", "2. Node trains on\nits own flows", "3. Node returns\nweights / trees",
             "4. Server\naggregates", "5. Server evaluates\non report half", "6. After last round:\ncalibrate + export"]
    for i, s in enumerate(steps):
        _box(ax, 0.05 + i * 1.66, 1.0, 1.45, 1.1, s, fs=9, fc="#fff1ea" if i in (0, 3, 4, 5) else "#eef4fc",
             ec=ORANGE if i in (0, 3, 4, 5) else BLUE)
        if i < 5:
            _arrow(ax, 1.5 + i * 1.66, 1.55, 1.71 + i * 1.66, 1.55)
    ax.plot([7.1, 7.1], [1.0, 0.7], color=INK2, lw=1.2)
    _arrow(ax, 7.1, 0.7, 0.75, 0.7, color=INK2)
    ax.plot([0.75, 0.75], [0.7, 0.98], color=INK2, lw=1.2)
    ax.text(4.1, 0.3, "repeat: Autoencoder 15 rounds (FedAvg) · XGBoost 40 rounds (bagging, 3 trees per node per round)",
            ha="center", fontsize=9, color=INK2)
    ax.text(0.05, 2.45, "Server side (orange) · node side (blue) · raw flows never leave a node", fontsize=9.5, color=INK)
    save(fig, "fig_fl_round.png")


# ------------------------------------------------------------------- charts --
def fig_class_distribution():
    s = load("../data/summary.json") if os.path.exists(os.path.join(ROOT, "federated", "data", "summary.json")) else None
    fam = s["families"] if s else {"BENIGN": 2273097, "DoS": 252661, "PortScan": 158930, "DDoS": 128027,
                                   "BruteForce": 13835, "WebAttack": 2180, "Bot": 1966, "Other": 47}
    names = list(fam)[::-1]; vals = [fam[n] for n in names]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.barh(names, vals, color=BLUE, height=0.55)
    ax.set_xscale("log"); ax.set_xlabel("Flows (log scale)"); ax.grid(axis="y", visible=False)
    for n, v in zip(names, vals):
        ax.text(v * 1.15, n, f"{v:,}", va="center", fontsize=9, color=INK)
    ax.set_xlim(10, 3e7)
    save(fig, "fig_class_distribution.png")


def fig_family_recall():
    c = load("comparison.json")
    fams = ["DDoS", "DoS", "PortScan", "BruteForce", "Bot", "WebAttack"]
    series = [("Phase 1 (DDoS day only)", "phase1_ddos_only", ORANGE), ("Centralized", "centralized_phase2", AQUA),
              ("Federated", "federated_phase2", BLUE)]
    y = np.arange(len(fams)); h = 0.26
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for k, (lab, key, col) in enumerate(series):
        v = [c[key]["hybrid_recall_per_family"][f] * 100 for f in fams]
        ax.barh(y + (1 - k) * h, v, height=h - 0.03, color=col, label=lab)
        for yy, vv in zip(y + (1 - k) * h, v):
            ax.text(vv + 1, yy, f"{vv:.1f}", va="center", fontsize=7.5, color=INK)
    ax.set_yticks(y, fams); ax.invert_yaxis(); ax.set_xlim(0, 112); ax.set_xlabel("Detection rate (%)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=9)
    save(fig, "fig_family_recall.png")


def fig_ae_convergence():
    r = [x for x in load("ae_rounds.json")["rounds"] if x["round"] > 0]
    rd = [x["round"] for x in r]
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.4))
    axs[0].plot(rd, [x["benign_mse_mean"] for x in r], color=BLUE, lw=2, marker="o", ms=4)
    axs[0].set_yscale("log"); axs[0].set_xlabel("Round"); axs[0].set_ylabel("Benign reconstruction MSE (log)")
    axs[0].set_title("Benign reconstruction error", fontsize=11, color=INK)
    axs[1].plot(rd, [x["recall"] * 100 for x in r], color=BLUE, lw=2, marker="o", ms=4)
    axs[1].set_xlabel("Round"); axs[1].set_ylabel("Autoencoder-only recall (%)")
    axs[1].set_title("Autoencoder-only recall", fontsize=11, color=INK)
    from matplotlib.ticker import MaxNLocator
    for a_ in axs:
        a_.xaxis.set_major_locator(MaxNLocator(integer=True))
    save(fig, "fig_ae_convergence.png")


def fig_xgb_rounds():
    r = load("xgb_rounds.json")["rounds"]
    rd = [x["round"] for x in r]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(rd, [x["auc"] for x in r], color=BLUE, lw=2, label="ROC AUC")
    ax.plot(rd, [x["recall"] for x in r], color=ORANGE, lw=2, label="Recall at the raw 0.5 cutoff")
    ax.set_ylim(0, 1.05); ax.set_xlabel("Federated round"); ax.set_ylabel("Value (0 to 1)")
    ax.text(rd[-1] + 0.5, r[-1]["auc"], "AUC", va="center", fontsize=9, color=INK)
    ax.text(rd[-1] + 0.5, r[-1]["recall"], "recall @0.5", va="center", fontsize=9, color=INK)
    ax.legend(loc="center right", fontsize=9)
    save(fig, "fig_xgb_rounds.png")


def _calib_and_report():
    import xgboost as xgb
    from tensorflow.keras.models import load_model
    from federated.data_utils import load_test
    from federated.models import reconstruction_error
    bst = xgb.Booster(); bst.load_model(os.path.join(ROOT, "hybrid_model", "xgb_model.json"))
    ae = load_model(os.path.join(ROOT, "hybrid_model", "autoencoder.keras"))
    out = {}
    for split in ("calib", "report"):
        X, y, fam = load_test(split)
        out[split] = (bst.predict(xgb.DMatrix(X)), reconstruction_error(ae, X), y, fam)
    return out


def fig_cutoff_sweep(data):
    from federated import config as C
    p, _, y, fam = data["calib"]
    cuts = np.linspace(0.05, 0.35, 61)
    fig, ax = plt.subplots(figsize=(9, 3.8))
    for name, col in (("Bot", ORANGE), ("WebAttack", AQUA)):
        m = fam == C.FAMILY_IDS[name]
        ax.plot(cuts, [(p[m] > t).mean() * 100 for t in cuts], color=col, lw=2, label=name)
    thr = load("../../hybrid_model/threshold.json")
    ax.axvline(0.253769, color=INK2, lw=1, ls="--"); ax.text(0.256, 8, "best-F1 cutoff\n0.254", fontsize=8.5, color=INK2)
    ax.axvline(thr["ml_threshold"], color=BLUE, lw=1.2); ax.text(thr["ml_threshold"] - 0.004, 8,
                                                              f"family-aware\ncutoff {thr['ml_threshold']:.3f}",
                                                              fontsize=8.5, color=INK, ha="right")
    ax.set_xlabel("XGBoost probability cutoff"); ax.set_ylabel("Detection rate (%)"); ax.set_ylim(0, 105)
    ax.legend(loc="lower left", fontsize=9)
    save(fig, "fig_cutoff_sweep.png")


def fig_platt(data):
    from federated import config as C
    from core.decision import dl_score, ml_score, final_score, load_thresholds, calibrated_prob
    cfg = load_thresholds(os.path.join(ROOT, "hybrid_model", "threshold.json"))
    p, mse, y, fam = data["report"]
    dl = dl_score(mse, cfg["ae_threshold"], cfg["anomaly_multiplier"])
    old = final_score(dl, ml_score(p, cfg["ml_threshold"]))
    new = final_score(dl, ml_score(calibrated_prob(p, cfg), float(calibrated_prob(cfg["ml_threshold"], cfg))))
    fams = ["DDoS", "DoS", "PortScan", "BruteForce", "Bot", "WebAttack"]
    y_ = np.arange(len(fams)); h = 0.36
    fig, ax = plt.subplots(figsize=(9, 3.9))
    for k, (lab, arr, col) in enumerate((("Before (raw probability)", old, ORANGE), ("After Platt scaling", new, BLUE))):
        v = [np.mean(arr[fam == C.FAMILY_IDS[f]] >= 0.75) * 100 for f in fams]
        ax.barh(y_ + (k - 0.5) * h, v, height=h - 0.04, color=col, label=lab)
        for yy, vv in zip(y_ + (k - 0.5) * h, v):
            ax.text(vv + 1, yy, f"{vv:.1f}", va="center", fontsize=8, color=INK)
    ax.set_yticks(y_, fams); ax.invert_yaxis(); ax.set_xlim(0, 112)
    ax.set_xlabel("Flows scoring at least 0.75, the prevention threshold (%)"); ax.grid(axis="y", visible=False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=9)
    save(fig, "fig_platt.png")


def fig_live_fp():
    labels = ["Phase 2 live path\n(first run)", "Seven defects fixed,\nunicast scope", "+ local\nbaseline"]
    vals = [1400 / 1875 * 100, 64 / 930 * 100, 15 / 1359 * 100]
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    ax.bar(labels, vals, color=BLUE, width=0.5)
    for i, v in enumerate(vals):
        ax.text(i, v + 1.5, f"{v:.1f}%", ha="center", fontsize=10, color=INK)
    ax.set_ylabel("Benign packets flagged (%)"); ax.set_ylim(0, 85); ax.grid(axis="x", visible=False)
    save(fig, "fig_live_fp.png")


def fig_cross_dataset():
    d = load("cross_dataset_2018.json")["per_label"]
    rows = sorted(((k, v["detection_rate"] * 100) for k, v in d.items() if k != "BENIGN"), key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.barh([r[0] for r in rows], [r[1] for r in rows], color=BLUE, height=0.55)
    for n, v in rows:
        ax.text(v + 1, n, f"{v:.1f}%", va="center", fontsize=8.5, color=INK)
    ax.set_xlim(0, 105); ax.set_xlabel("Detected on CSE-CIC-IDS2018 (%)"); ax.grid(axis="y", visible=False)
    save(fig, "fig_cross_dataset.png")


def crop_screenshots(media):
    """Dashboard screenshots taken with Playwright (report/figures/screens/*)."""
    from PIL import Image
    src = os.path.join(media, "dash_sim_tall.png")
    if os.path.exists(src):
        im = Image.open(src)
        W, H = im.size
        im.crop((0, 0, W, int(H * 0.47))).save(os.path.join(OUT, "fig_dash_alerts_xai.png"))
        im.crop((0, int(H * 0.475), W, int(H * 0.93))).save(os.path.join(OUT, "fig_dash_simulator.png"))
        print("   fig_dash_alerts_xai.png, fig_dash_simulator.png")
    src = os.path.join(media, "dash_live_full.png")
    if os.path.exists(src):
        im = Image.open(src); W, H = im.size
        im.crop((0, 0, W, int(H * 0.81))).save(os.path.join(OUT, "fig_dash_live.png"))
        print("   fig_dash_live.png")


if __name__ == "__main__":
    print("figures ->", OUT)
    fig_architecture(); fig_fl_round(); fig_class_distribution(); fig_family_recall()
    fig_ae_convergence(); fig_xgb_rounds(); fig_live_fp(); fig_cross_dataset()
    data = _calib_and_report()
    fig_cutoff_sweep(data); fig_platt(data)
    if len(sys.argv) > 1:
        crop_screenshots(sys.argv[1])
