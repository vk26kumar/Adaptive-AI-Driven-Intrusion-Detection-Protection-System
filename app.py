"""
app.py  --  A-IDAPS-FL  |  Real-Time IDS Dashboard (Phase 2)
Run:  streamlit run app.py      (Administrator / root needed for live capture)

Phase 2 changes versus Phase 1
  * Model: federated global model (Flower, 4 nodes, all CIC-IDS2017 attack days)
  * Metrics are measured, not hard-coded: latency is timed per packet, the
    confusion matrix only counts flows whose true label is known, accuracy
    comes from the server-side hold-out evaluation.
  * Explainability is per prediction (TreeSHAP + autoencoder error share),
    not a static table.
  * New "Federated Training" panel reads federated/logs/*.json.
"""
import sys, os
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import json
import threading
import time
from queue import Empty

import pandas as pd
import streamlit as st

from core.state import packet_queue, stop_event, Stats, classify_attack_type, ATTACK_FAMILIES
from core.predictor import (predict_features, explain_features, GLOBAL_IMPORTANCE,
                            THRESHOLDS, MODEL_INFO, XGB_SOURCE)
from core.simulator import simulate_packet, PROFILES
from core.predictor import explain_lime
from core.prevention import engine as prevention, BLOCK_THRESHOLD, MIN_HITS, WINDOW_S
from core.persistence import store

st.set_page_config(page_title="A-IDAPS-FL | IDS Dashboard", layout="wide",
                   initial_sidebar_state="collapsed")

# ---------------------------------------------------------------- styling ----
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap" rel="stylesheet">
<style>
html, body, [class*="css"] { background:#0A0C10 !important; color:#E2E8F0 !important; font-family:'Syne',sans-serif !important; }
[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"] { display:none !important; }
.block-container { padding: 0.5rem 1.5rem 1rem 1.5rem !important; }
[data-testid="metric-container"] { background:#141820; border:1px solid #1E2535; border-radius:10px; padding:14px 18px !important; }
[data-testid="metric-container"] label { color:#4A5568 !important; font-family:'Space Mono',monospace !important; font-size:10px !important; letter-spacing:2px !important; text-transform:uppercase; }
h1,h2,h3 { font-family:'Syne',sans-serif !important; }
.stButton > button { font-family:'Space Mono',monospace !important; font-size:11px !important; letter-spacing:1px !important; border-radius:6px !important; }
hr { border-color:#1E2535 !important; }
details { background:#141820 !important; border:1px solid #1E2535 !important; border-radius:8px !important; }
.top-banner { display:flex; align-items:center; justify-content:space-between; padding:10px 20px; background:#0F1219; border:1px solid #1E2535; border-radius:10px; margin-bottom:16px; }
.banner-title { font-family:'Syne',sans-serif; font-size:20px; font-weight:800; background:linear-gradient(90deg,#00E5FF,#7C3AED); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.banner-sub { font-family:'Space Mono',monospace; font-size:10px; color:#4A5568; letter-spacing:1.5px; }
.status-pill { font-family:'Space Mono',monospace; font-size:10px; padding:4px 12px; border-radius:20px; letter-spacing:1px; }
.pill-live { background:rgba(0,255,157,0.12); color:#00FF9D; border:1px solid rgba(0,255,157,0.3); }
.pill-stopped { background:rgba(255,59,107,0.12); color:#FF3B6B; border:1px solid rgba(255,59,107,0.3); }
.section-label { font-family:'Space Mono',monospace; font-size:10px; color:#4A5568; letter-spacing:2px; text-transform:uppercase; border-bottom:1px solid #1E2535; padding-bottom:6px; margin-bottom:10px; }
.bar-row { display:flex; align-items:center; gap:10px; margin-bottom:6px; }
.bar-label { font-family:'Space Mono',monospace; font-size:10px; color:#4A5568; width:150px; text-align:right; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.bar-outer { flex:1; height:8px; background:#1E2535; border-radius:4px; overflow:hidden; }
.bar-score { font-family:'Space Mono',monospace; font-size:10px; color:#E2E8F0; width:60px; text-align:right; }
.alert-item { padding:8px 12px; border-radius:6px; margin-bottom:6px; font-family:'Space Mono',monospace; font-size:11px; line-height:1.6; }
.alert-crit { background:rgba(255,59,107,0.08); border:1px solid rgba(255,59,107,0.25); color:#FF3B6B; }
.alert-warn { background:rgba(255,184,48,0.08); border:1px solid rgba(255,184,48,0.22); color:#FFB830; }
.matrix-cell { background:#141820; border:1px solid #1E2535; border-radius:8px; padding:12px; text-align:center; }
.matrix-val { font-family:'Space Mono',monospace; font-size:24px; font-weight:700; }
.matrix-lbl { font-family:'Space Mono',monospace; font-size:9px; color:#4A5568; letter-spacing:1px; margin-top:4px; }
.small { font-family:'Space Mono',monospace; font-size:10px; color:#4A5568; }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------ session init ----
def _init():
    defaults = {
        "running": False, "stats": Stats(), "logs": [], "alerts": [],
        "chart_attacks": [0] * 40, "chart_normal": [0] * 40, "chart_last_rotate": time.time(),
        "sim_result": None, "last_explained": None, "sniffer_thread": None, "sniffer_error": None,
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)
_init()
S = st.session_state


def _start_sniffer():
    try:
        from network.sniffer import start_sniffing
        stop_event.clear()
        t = threading.Thread(target=start_sniffing, args=(packet_queue, stop_event), daemon=True)
        t.start()
        S["sniffer_thread"] = t
        S["sniffer_error"] = None
    except Exception as e:  # missing Npcap / scapy / privileges
        S["sniffer_error"] = str(e)
        S["running"] = False
        st.error(f"Packet capture unavailable: {e}. Install Npcap and run as Administrator. "
                 "The Attack Simulator still works.", icon="🔌")


def _push_log(res, src_ip, dst_port, proto, pkt_len, source):
    S["logs"].insert(0, {
        "Time": time.strftime("%H:%M:%S"), "Source": source, "Src IP": src_ip,
        "Dst Port": dst_port, "Proto": proto, "Pkt Len": pkt_len,
        "ML Prob": f"{res.get('ml_prob', 0) * 100:.1f}%", "DL Score": f"{res.get('dl_score', 0) * 100:.1f}%",
        "Score": f"{res.get('final_score', 0):.3f}", "Latency": f"{res.get('latency_ms', 0):.1f} ms",
        "Verdict": "ATTACK" if res.get("label") == "ATTACK" else "NORMAL",
    })
    S["logs"] = S["logs"][:80]


def _push_alert(src, port, proto, score, atype):
    S["alerts"].insert(0, {"time": time.strftime("%H:%M:%S"), "src": src, "port": port,
                           "proto": proto, "score": score, "type": atype})
    S["alerts"] = S["alerts"][:15]


def _bucket(label):
    if label == "ATTACK":
        S["chart_attacks"][-1] += 1
    else:
        S["chart_normal"][-1] += 1


def _rotate_chart():
    if time.time() - S["chart_last_rotate"] >= 3.0:
        S["chart_attacks"] = (S["chart_attacks"] + [0])[-40:]
        S["chart_normal"] = (S["chart_normal"] + [0])[-40:]
        S["chart_last_rotate"] = time.time()


def _drain_queue(max_items=50):
    for _ in range(max_items):
        try:
            res = packet_queue.get_nowait()
        except Empty:
            break
        label, fs, feat = res.get("label", "NORMAL"), res.get("final_score", 0.0), res.get("features", {})
        S["stats"].record(label, fs, latency_ms=res.get("latency_ms", 0.0), truth=None)
        _bucket(label)
        atype = None
        if label == "ATTACK":
            atype = classify_attack_type(feat)
            S["stats"].attack_types[atype] = S["stats"].attack_types.get(atype, 0) + 1
            _push_alert(res.get("src_ip", "?"), res.get("dst_port", "?"), res.get("proto", "?"), fs, atype)
            S["last_explained"] = {"features": feat, "label": label, "who": f"live {res.get('src_ip','?')}:{res.get('dst_port','?')}"}
            blocked = prevention.observe(res.get("src_ip", "?"), fs, atype)
            if blocked:
                store.log_action("block", blocked.ip, {"mode": blocked.mode, "score": fs, "type": atype})
                S["alerts"].insert(0, {"time": time.strftime("%H:%M:%S"), "src": blocked.ip, "port": "-",
                                       "proto": "BLOCKED" if blocked.mode != "dry-run" else "WOULD BLOCK",
                                       "score": fs, "type": atype})
        store.log_flow(res, source="live", attack_type=atype)
        _push_log(res, res.get("src_ip", "?"), res.get("dst_port", "?"), res.get("proto", "?"),
                  res.get("pkt_len", 0), "live")
    _rotate_chart()


def _read_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _bars(items, color_fn, fmt="{:+.3f}", denom=None):
    html = ""
    denom = denom or max((abs(v) for _, v in items), default=1.0) or 1.0
    for name, v in items:
        w = int(min(abs(v) / denom, 1.0) * 100)
        html += (f'<div class="bar-row"><div class="bar-label" title="{name}">{name}</div>'
                 f'<div class="bar-outer"><div style="height:100%;width:{w}%;background:{color_fn(v)};border-radius:4px"></div></div>'
                 f'<div class="bar-score">{fmt.format(v)}</div></div>')
    return html


# =============================================================== LAYOUT =======
test_metrics = (MODEL_INFO.get("test_metrics") or {}).get("hybrid", {})
model_tag = "FEDERATED GLOBAL MODEL" if "federated" in XGB_SOURCE else "PHASE 1 MODEL"
pill = ("pill-live", "● LIVE") if S["running"] else ("pill-stopped", "■ STOPPED")
st.markdown(f"""
<div class="top-banner">
  <div><div class="banner-title">A-IDAPS-FL</div>
  <div class="banner-sub">ADAPTIVE AI INTRUSION DETECTION & PROTECTION SYSTEM | PHASE 2 | {model_tag} | GROUP G-52 | MMMUT</div></div>
  <div style="display:flex;align-items:center;gap:14px">
    <span class="banner-sub">{time.strftime('%A %d %b %Y  %H:%M:%S')}</span>
    <span class="status-pill {pill[0]}">{pill[1]}</span></div>
</div>""", unsafe_allow_html=True)

c1, c2, c3, _ = st.columns([1, 1, 1, 5])
if c1.button("▶  Start IDS", width="stretch") and not S["running"]:
    S["running"] = True
    _start_sniffer()
if c2.button("■  Stop IDS", width="stretch"):
    S["running"] = False
    stop_event.set()
if c3.button("✕  Clear", width="stretch"):
    S.update({"logs": [], "alerts": [], "stats": Stats(), "chart_attacks": [0] * 40,
              "chart_normal": [0] * 40, "sim_result": None, "last_explained": None})
st.markdown("---")

if S["running"]:
    _drain_queue()

stats: Stats = S["stats"]
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("📦 Flows Analyzed", f"{stats.total_packets:,}")
m2.metric("⚠️ Attacks Detected", f"{stats.total_attacks:,}")
m3.metric("📈 Attack Rate", f"{stats.attack_rate:.1f}%")
m4.metric("⚡ Avg Latency (measured)", f"{stats.avg_latency:.1f} ms")
m5.metric("🎯 Hold-out Accuracy", f"{test_metrics['accuracy'] * 100:.2f}%" if test_metrics else "n/a",
          help="Hybrid model accuracy on the server-side CIC-IDS2017 hold-out set (federated/evaluate.py)")
st.markdown("---")

left, right = st.columns([2, 1], gap="medium")
with left:
    st.markdown('<div class="section-label">Live Packet Feed</div>', unsafe_allow_html=True)
    if S["logs"]:
        df = pd.DataFrame(S["logs"])
        df["Verdict"] = df["Verdict"].map(lambda v: "⚠ ATTACK" if v == "ATTACK" else "✓ NORMAL")
        st.dataframe(df, width="stretch", height=240, hide_index=True)
    else:
        st.info("No flows yet. Click **▶ Start IDS** for live capture or use the Attack Simulator.", icon="📡")
    if S.get("sniffer_error"):
        st.warning(S["sniffer_error"])
    st.markdown('<div class="section-label" style="margin-top:14px">Traffic Timeline (3 s buckets)</div>', unsafe_allow_html=True)
    st.line_chart(pd.DataFrame({"Attacks": S["chart_attacks"], "Normal": S["chart_normal"]}),
                  width="stretch", height=160)

with right:
    st.markdown('<div class="section-label">Threat Alerts</div>', unsafe_allow_html=True)
    if S["alerts"]:
        for a in S["alerts"][:6]:
            cls = "alert-crit" if a["score"] > 0.75 else "alert-warn"
            st.markdown(f'<div class="alert-item {cls}">⚑ {a["src"]}:{a["port"]} [{a["proto"]}] · {a["type"]}<br>'
                        f'<span style="font-size:9px;opacity:0.7">Score {a["score"]:.3f} · {a["time"]}</span></div>',
                        unsafe_allow_html=True)
    else:
        st.markdown('<p class="small">No threats detected</p>', unsafe_allow_html=True)

    st.markdown('<div class="section-label" style="margin-top:14px">XAI — why this verdict?</div>', unsafe_allow_html=True)
    ex = S.get("last_explained")
    if ex:
        try:
            exp = explain_features(ex["features"])
            st.markdown(f'<p class="small">Explaining: {ex["who"]} → {ex["label"]}</p>', unsafe_allow_html=True)
            st.markdown('<p class="small">XGBoost SHAP (log-odds, + = attack)</p>', unsafe_allow_html=True)
            st.markdown(_bars(list(exp["ml_contrib"].items()), lambda v: "#FF3B6B" if v > 0 else "#00E5FF"),
                        unsafe_allow_html=True)
            st.markdown('<p class="small">Autoencoder: share of reconstruction error</p>', unsafe_allow_html=True)
            st.markdown(_bars(list(exp["dl_contrib"].items()), lambda v: "#7C3AED", fmt="{:.1%}", denom=1.0),
                        unsafe_allow_html=True)
            if st.toggle("LIME local surrogate of the final score", value=False, key="xai_lime"):
                lm = explain_lime(ex["features"])
                st.markdown(f'<p class="small">local linear fit R² {lm["local_fit_r2"]:.2f} · intercept {lm["intercept"]:.3f}</p>',
                            unsafe_allow_html=True)
                st.markdown(_bars(list(lm["coefficients"].items()), lambda v: "#FFB830" if v > 0 else "#00FF9D"),
                            unsafe_allow_html=True)
        except Exception as e:
            st.warning(f"explanation failed: {e}")
    else:
        st.markdown('<p class="small">Global XGBoost importance (gain). Run the simulator or detect an attack for a per-flow explanation.</p>',
                    unsafe_allow_html=True)
        st.markdown(_bars(list(GLOBAL_IMPORTANCE.items())[:6], lambda v: "#FFB830", fmt="{:.3f}"), unsafe_allow_html=True)

st.markdown("---")
sim_col, matrix_col, fed_col = st.columns(3, gap="medium")

# ------------------------------------------------------------ simulator ------
with sim_col:
    st.markdown('<div class="section-label">⚡ Attack Simulator</div>', unsafe_allow_html=True)
    st.caption("Profiles start from the median real CIC-IDS2017 flow of that family. Override knobs to probe the model.")
    profile = st.selectbox("Traffic profile", PROFILES, key="sim_profile")
    override = st.checkbox("Override flow knobs", value=False, key="sim_override")
    ka, kb = st.columns(2)
    pkt_len = ka.number_input("Fwd payload length (bytes)", 0, 9000, 40, 1, key="sim_pkt", disabled=not override)
    dst_port = kb.number_input("Destination port", 1, 65535, 80, 1, key="sim_port", disabled=not override)
    flow_dur = ka.number_input("Flow duration (ms)", 1, 120000, 500, 10, key="sim_dur", disabled=not override)
    pkt_rate = kb.number_input("Fwd packets / s", 0.0, 3_000_000.0, 50.0, 1.0, key="sim_rate", disabled=not override)
    fwd_pkts = ka.number_input("Fwd packet count", 1, 5000, 5, 1, key="sim_fpkts", disabled=not override)
    proto_lbl = kb.radio("Protocol", ["TCP (6)", "UDP (17)"], horizontal=True, key="sim_proto")
    proto = 6 if proto_lbl.startswith("TCP") else 17

    if st.button("⚡  Inject flow", width="stretch", key="sim_run"):
        kw = dict(dst_port=dst_port, pkt_rate=pkt_rate, pkt_len=pkt_len, flow_duration_ms=flow_dur,
                  fwd_pkts=fwd_pkts) if override else {}
        res = simulate_packet(profile=profile, proto=proto, **kw)
        S["sim_result"] = res
        S["stats"].record(res["label"], res["final_score"], latency_ms=res["latency_ms"], truth=res["truth"])
        _bucket(res["label"])
        if res["label"] == "ATTACK":
            atype = res["family"] if res["family"] != "BENIGN" else classify_attack_type(res["features"])
            S["stats"].attack_types[atype] = S["stats"].attack_types.get(atype, 0) + 1
            _push_alert("SIM:10.0.99.1", int(res["features"]["Destination Port"]), proto_lbl[:3], res["final_score"], atype)
        _push_log(res, "SIM:10.0.99.1", int(res["features"]["Destination Port"]), proto_lbl[:3],
                  int(res["features"]["Fwd Packet Length Mean"]), f"sim:{profile}")
        S["last_explained"] = {"features": res["features"], "label": res["label"], "who": f"simulated {profile}"}
        store.log_flow(res, source=f"sim:{profile}", truth=res["truth"],
                       attack_type=res["family"] if res["family"] != "BENIGN" else None)
        st.rerun()   # metrics above were drawn before this handler ran; redraw with the new flow

    if S["sim_result"]:
        r = S["sim_result"]
        correct = r["label"] == r["truth"]
        msg = f"{'⚠ ATTACK' if r['label'] == 'ATTACK' else '✓ NORMAL'} | confidence {r['confidence']}% | truth: {r['truth']} {'✓' if correct else '✗'}"
        (st.error if r["label"] == "ATTACK" else st.success)(msg)
        with st.expander("Score breakdown"):
            st.markdown(f"""
| Component | Value |
|---|---|
| XGBoost P(attack) | `{r['ml_prob']:.4f}` |
| Autoencoder MSE | `{r['mse']:.3e}` (threshold {THRESHOLDS['ae_threshold']:.3e} × {THRESHOLDS['anomaly_multiplier']:.0f}) |
| Autoencoder score | `{r['dl_score']:.4f}` |
| **Final = max(DL, ML)** | **`{r['final_score']:.4f}`** (attack if > {THRESHOLDS['decision_threshold']}) |
| Latency | `{r['latency_ms']:.1f} ms` |
""")

# ------------------------------------------------------ confusion matrix -----
with matrix_col:
    st.markdown('<div class="section-label">Detection Matrix (labelled flows only)</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px">
      <div class="matrix-cell"><div class="matrix-val" style="color:#00FF9D">{stats.cm_tp}</div><div class="matrix-lbl">TRUE POSITIVE</div></div>
      <div class="matrix-cell"><div class="matrix-val" style="color:#00E5FF">{stats.cm_tn}</div><div class="matrix-lbl">TRUE NEGATIVE</div></div>
      <div class="matrix-cell"><div class="matrix-val" style="color:#FFB830">{stats.cm_fp}</div><div class="matrix-lbl">FALSE POSITIVE</div></div>
      <div class="matrix-cell"><div class="matrix-val" style="color:#FF3B6B">{stats.cm_fn}</div><div class="matrix-lbl">FALSE NEGATIVE</div></div>
    </div>
    <p class="small" style="margin-top:6px">{stats.labeled} labelled (simulator) · {stats.unlabeled} live flows without ground truth</p>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label" style="margin-top:10px">Session metrics (labelled)</div>', unsafe_allow_html=True)
    live = stats.live_metrics()
    if live:
        st.markdown(_bars(list(live.items()), lambda v: "#00E5FF", fmt="{:.1f}%", denom=100.0), unsafe_allow_html=True)
    else:
        st.markdown('<p class="small">Inject simulated flows to build session metrics.</p>', unsafe_allow_html=True)

    st.markdown('<div class="section-label" style="margin-top:10px">Hold-out evaluation (CIC-IDS2017)</div>', unsafe_allow_html=True)
    if test_metrics:
        st.markdown(_bars([(k.capitalize(), v * 100) for k, v in test_metrics.items() if k != "fpr"],
                          lambda v: "#7C3AED", fmt="{:.2f}%", denom=100.0), unsafe_allow_html=True)
        st.markdown(f'<p class="small">FPR {test_metrics.get("fpr", 0) * 100:.2f}% · {MODEL_INFO.get("training_data", "")}</p>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<p class="small">Run python -m federated.evaluate then export_global_model.</p>', unsafe_allow_html=True)

    st.markdown('<div class="section-label" style="margin-top:10px">Autonomous Prevention</div>', unsafe_allow_html=True)
    pc1, pc2 = st.columns(2)
    prevention.enabled = pc1.toggle("Auto-block", value=prevention.enabled, key="prev_on",
                                    help=f"Block a public source IP after {MIN_HITS} alerts with score >= {BLOCK_THRESHOLD} within {WINDOW_S:.0f}s")
    prevention.dry_run = not pc2.toggle("Apply to firewall", value=not prevention.dry_run, key="prev_real",
                                        help="Off = dry-run (decide and log only). On = netsh / iptables rules; needs Administrator/root.")
    if prevention.blocked:
        for ip, rec in list(prevention.blocked.items()):
            b1, b2 = st.columns([3, 1])
            b1.markdown(f'<p class="small">{ip} · {rec.attack_type} · {rec.mode} · score {rec.score:.2f}'
                        f'{" · " + rec.error if rec.error else ""}</p>', unsafe_allow_html=True)
            if b2.button("unblock", key=f"unblock_{ip}"):
                prevention.unblock(ip); store.log_action("unblock", ip); st.rerun()
    else:
        st.markdown('<p class="small">No sources blocked. Private/LAN addresses and the simulator are never auto-blocked.</p>',
                    unsafe_allow_html=True)
    counts = store.counts()
    st.markdown(f'<p class="small">Event store: {counts["flows"]:,} flows · {counts["attacks"]:,} attacks · {counts["actions"]} actions (logs/ids_events.sqlite)</p>',
                unsafe_allow_html=True)

    st.markdown('<div class="section-label" style="margin-top:10px">Attack Type Breakdown</div>', unsafe_allow_html=True)
    colors = {"DDoS": "#FF3B6B", "DoS": "#FF7A45", "PortScan": "#FFB830", "BruteForce": "#7C3AED",
              "Bot": "#00FF9D", "WebAttack": "#00E5FF", "Other": "#4A5568"}
    tot = max(sum(stats.attack_types.values()), 1)
    st.markdown(_bars([(k, stats.attack_types.get(k, 0) / tot * 100) for k in ATTACK_FAMILIES],
                      lambda v: "#FFB830", fmt="{:.0f}%", denom=100.0), unsafe_allow_html=True)

# --------------------------------------------------------- federated panel ---
with fed_col:
    st.markdown('<div class="section-label">Federated Training (Flower)</div>', unsafe_allow_html=True)
    fed_dir = os.path.join(_ROOT, "federated")
    summary = _read_json(os.path.join(fed_dir, "data", "summary.json"))
    ae_log = _read_json(os.path.join(fed_dir, "logs", "ae_rounds.json"))
    xgb_log = _read_json(os.path.join(fed_dir, "logs", "xgb_rounds.json"))
    cmp = _read_json(os.path.join(fed_dir, "logs", "comparison.json"))

    if summary:
        rows = []
        for nid, n in summary["nodes"].items():
            rows.append({"Node": f"node {nid}", "Rows": f"{n['rows']:,}", "Attacks": f"{n['attacks']:,}",
                         "Sees": ", ".join(n["assigned_families"])})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", height=175)
    else:
        st.markdown('<p class="small">No partitions yet: python -m federated.prepare_data</p>', unsafe_allow_html=True)

    for name, log in (("Autoencoder · FedAvg", ae_log), ("XGBoost · bagging", xgb_log)):
        if not log:
            continue
        r = log["rounds"]
        done = "finished" if log.get("finished") else "running"
        flags = " · ".join(x for x in (
            "TLS" if log.get("tls") else "", "DP" if log.get("dp") else "") if x)
        st.markdown(f'<p class="small" style="margin-top:8px">{name}: {len(r)} rounds ({done}) {flags}</p>',
                    unsafe_allow_html=True)
        if r:
            chart = pd.DataFrame({"round": [x["round"] for x in r],
                                  "recall": [x.get("recall", 0) for x in r],
                                  "fpr": [x.get("fpr", 0) for x in r]}).set_index("round")
            st.line_chart(chart, height=120, width="stretch")
            last = r[-1]
            parts = [f"last round: recall {last.get('recall', 0):.3f}", f"FPR {last.get('fpr', 0):.4f}"]
            mb = (log.get("bytes_total") or 0) / 1e6
            if mb:
                parts.append(f"traffic {mb:.1f} MB total")
            if "num_trees" in last:
                parts.append(f"{last['num_trees']} trees")
            if log.get("final_model_bytes"):
                parts.append(f"model {log['final_model_bytes'] / 1e6:.2f} MB")
            st.markdown('<p class="small">' + " · ".join(parts) + '</p>', unsafe_allow_html=True)

    dp = _read_json(os.path.join(fed_dir, "logs", "dp_analysis.json"))
    comm = _read_json(os.path.join(fed_dir, "logs", "comm_analysis.json"))
    lstm = _read_json(os.path.join(fed_dir, "logs", "lstm_experiment.json"))
    if comm:
        c_raw = comm["centralized"]["raw_megabytes"]; f_tot = comm["federated"].get("megabytes_total", 0)
        st.markdown(f'<p class="small" style="margin-top:8px">Traffic: federated {f_tot:.0f} MB of parameters vs {c_raw:.0f} MB of raw flows for centralized training · zero raw records leave a node</p>',
                    unsafe_allow_html=True)
    if dp and dp.get("runs"):
        rows = [{"DP setting": r["setting"], "ε (δ=1e-5)": ("∞" if r["epsilon"] == float("inf") or r["epsilon"] is None else f"{r['epsilon']:.1f}"),
                 "benign MSE": f"{r['benign_mse']:.1e}", "AE recall": f"{r['ae_recall']:.2f}"} for r in dp["runs"]]
        with st.expander("Differential privacy runs"):
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    if lstm:
        with st.expander("LSTM temporal model (10-flow windows)"):
            l, h = lstm["lstm_window_level"], lstm["hybrid_anyflow_window_level"]
            st.dataframe(pd.DataFrame([{"Detector": "LSTM", **l}, {"Detector": "Hybrid, any flow", **h}]), hide_index=True, width="stretch")

    if cmp:
        st.markdown('<div class="section-label" style="margin-top:10px">Centralized vs Federated (hybrid, hold-out)</div>',
                    unsafe_allow_html=True)
        rows = []
        for key, label in (("phase1_ddos_only", "Phase 1 (DDoS day)"), ("centralized_phase2", "Centralized"),
                           ("federated_phase2", "Federated")):
            if key in cmp:
                h = cmp[key]["hybrid"]
                rows.append({"Model": label, "Acc": f"{h['accuracy']*100:.2f}%", "Recall": f"{h['recall']*100:.2f}%",
                             "FPR": f"{h['fpr']*100:.2f}%",
                             "Raw rows sent to server": f"{cmp[key].get('training_rows_moved_to_server', 'n/a'):,}"
                             if isinstance(cmp[key].get("training_rows_moved_to_server"), int) else "n/a"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

if S["running"]:
    time.sleep(0.3)
    st.rerun()
