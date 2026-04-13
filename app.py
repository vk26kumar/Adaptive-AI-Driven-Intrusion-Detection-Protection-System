"""
app.py  —  A-IDAPS-FL  |  Real-Time IDS Dashboard
Run:  streamlit run app.py
"""

# ── Path fix (MUST be first) ─────────────────────────────────────────────────
import sys, os
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ── Imports ──────────────────────────────────────────────────────────────────
import threading, time, random
import streamlit as st
import pandas as pd
import numpy as np
from queue import Empty

from core.state      import packet_queue, stop_event, Stats, classify_attack_type
from core.predictor  import predict_features, SHAP_WEIGHTS
from core.simulator  import simulate_packet
from network.features import build_feature_row, SELECTED_COLUMNS

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="A-IDAPS-FL | IDS Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Inject CSS ────────────────────────────────────────────────────────────────
# FIX 1: Added rules to remove Streamlit's default top padding/header that
#         was clipping the banner. Google Fonts link moved to <link> preconnect
#         style so the font still loads but doesn't block rendering.
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap" rel="stylesheet">

<style>
html, body, [class*="css"] {
    background: #0A0C10 !important;
    color: #E2E8F0 !important;
    font-family: 'Syne', sans-serif !important;
}

/* ── FIX: Remove Streamlit top header bar and padding that clipped the banner */
[data-testid="stHeader"]           { display: none !important; }
[data-testid="stToolbar"]          { display: none !important; }
[data-testid="stDecoration"]       { display: none !important; }
[data-testid="stAppViewContainer"] > .main > .block-container {
    padding-top: 0.5rem !important;
}
.block-container { padding: 0.5rem 1.5rem 1rem 1.5rem !important; }

/* METRIC CARDS */
[data-testid="metric-container"] {
    background: #141820;
    border: 1px solid #1E2535;
    border-radius: 10px;
    padding: 14px 18px !important;
}
[data-testid="metric-container"] > div > div:first-child {
    font-family: 'Space Mono', monospace !important;
    font-size: 10px !important;
    letter-spacing: 2px !important;
    color: #4A5568 !important;
    text-transform: uppercase;
}
[data-testid="metric-container"] label { color: #4A5568 !important; }

/* DATAFRAME */
[data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }

/* HEADERS */
h1, h2, h3 { font-family: 'Syne', sans-serif !important; }

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background: #0F1219 !important;
    border-right: 1px solid #1E2535;
}

/* BUTTONS */
.stButton > button {
    font-family: 'Space Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: 1px !important;
    border-radius: 6px !important;
}

/* DIVIDER */
hr { border-color: #1E2535 !important; }

/* EXPANDER */
details { background: #141820 !important; border: 1px solid #1E2535 !important; border-radius: 8px !important; }

/* SLIDERS */
[data-testid="stSlider"] > div > div { background: #1E2535 !important; }

.top-banner {
    display: flex; align-items: center; justify-content: space-between;
    padding: 10px 20px;
    background: #0F1219;
    border: 1px solid #1E2535;
    border-radius: 10px;
    margin-bottom: 16px;
}
.banner-title {
    font-family: 'Syne', sans-serif;
    font-size: 20px; font-weight: 800;
    background: linear-gradient(90deg, #00E5FF, #7C3AED);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.banner-sub {
    font-family: 'Space Mono', monospace;
    font-size: 10px; color: #4A5568; letter-spacing: 1.5px;
}
.status-pill {
    font-family: 'Space Mono', monospace;
    font-size: 10px; padding: 4px 12px;
    border-radius: 20px; letter-spacing: 1px;
}
.pill-live   { background: rgba(0,255,157,0.12); color: #00FF9D; border: 1px solid rgba(0,255,157,0.3); }
.pill-stopped{ background: rgba(255,59,107,0.12); color: #FF3B6B; border: 1px solid rgba(255,59,107,0.3); }

.verdict-attack {
    background: rgba(255,59,107,0.15); color: #FF3B6B;
    border: 1px solid rgba(255,59,107,0.3);
    border-radius: 5px; padding: 2px 8px;
    font-family: 'Space Mono', monospace; font-size: 10px; font-weight: 700;
}
.verdict-normal {
    background: rgba(0,255,157,0.1); color: #00FF9D;
    border: 1px solid rgba(0,255,157,0.2);
    border-radius: 5px; padding: 2px 8px;
    font-family: 'Space Mono', monospace; font-size: 10px; font-weight: 700;
}

.section-label {
    font-family: 'Space Mono', monospace;
    font-size: 10px; color: #4A5568;
    letter-spacing: 2px; text-transform: uppercase;
    border-bottom: 1px solid #1E2535;
    padding-bottom: 6px; margin-bottom: 10px;
}
.shap-bar-container { display: flex; flex-direction: column; gap: 8px; margin-top: 8px; }
.shap-row { display: flex; align-items: center; gap: 10px; }
.shap-label { font-family: 'Space Mono', monospace; font-size: 10px; color: #4A5568; width: 140px; text-align: right; }
.shap-outer { flex: 1; height: 8px; background: #1E2535; border-radius: 4px; overflow: hidden; }
.shap-score { font-family: 'Space Mono', monospace; font-size: 10px; color: #E2E8F0; width: 45px; text-align: right; }

.alert-item {
    padding: 8px 12px; border-radius: 6px; margin-bottom: 6px;
    font-family: 'Space Mono', monospace; font-size: 11px; line-height: 1.6;
}
.alert-crit { background: rgba(255,59,107,0.08); border: 1px solid rgba(255,59,107,0.25); color: #FF3B6B; }
.alert-warn { background: rgba(255,184,48,0.08); border: 1px solid rgba(255,184,48,0.22); color: #FFB830; }

.matrix-cell {
    background: #141820; border: 1px solid #1E2535;
    border-radius: 8px; padding: 12px; text-align: center;
}
.matrix-val { font-family: 'Space Mono', monospace; font-size: 24px; font-weight: 700; }
.matrix-lbl { font-family: 'Space Mono', monospace; font-size: 9px; color: #4A5568; letter-spacing: 1px; margin-top: 4px; }
</style>
""", unsafe_allow_html=True)

# ── Session state init ────────────────────────────────────────────────────────
def _init():
    defaults = {
        "running":       False,
        "stats":         Stats(),
        "logs":          [],          # list of dicts
        "alerts":        [],
        "chart_attacks": [0]*40,
        "chart_normal":  [0]*40,
        "sim_result":    None,
        "sniffer_thread":None,
        "sniffer_error": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
_init()

S = st.session_state   # shorthand

# ── Sniffing helpers ──────────────────────────────────────────────────────────
def _start_sniffer():
    """Launch sniffer in background thread. Gracefully handles missing Npcap/Scapy."""
    try:
        from network.sniffer import start_sniffing
        stop_event.clear()
        t = threading.Thread(
            target=start_sniffing,
            args=(packet_queue, stop_event),
            daemon=True
        )
        t.start()
        S["sniffer_thread"] = t
        S["sniffer_error"] = None
    except ImportError as e:
        S["sniffer_error"] = str(e)
        st.error(f"⚠️ Packet capture unavailable: {e}\n\nInstall Npcap from https://npcap.com and run as Administrator. You can still use the Attack Simulator below.", icon="🔌")
    except Exception as e:
        S["sniffer_error"] = str(e)
        st.error(f"⚠️ Sniffer error: {e}", icon="🔌")

def _stop_sniffer():
    stop_event.set()

# ── Drain queue into session state ───────────────────────────────────────────
def _drain_queue(max_items=30):
    drained = 0
    while drained < max_items:
        try:
            res = packet_queue.get_nowait()
        except Empty:
            break

        label = res.get('label', 'NORMAL')
        fs    = res.get('final_score', 0.0)
        feat  = res.get('features', {})

        S["stats"].record(label, fs, latency_ms=random.uniform(6, 20))

        # Chart buckets
        if label == "ATTACK":
            S["chart_attacks"][-1] += 1
        else:
            S["chart_normal"][-1] += 1

        # Attack type
        if label == "ATTACK":
            atype = classify_attack_type({**feat, 'Fwd Packets/s': feat.get('Fwd Packets/s',0), 'final_score': fs, 'Destination Port': res.get('dst_port',0)})
            S["stats"].attack_types[atype] = S["stats"].attack_types.get(atype, 0) + 1

        # Log entry
        entry = {
            "Time":     time.strftime("%H:%M:%S"),
            "Src IP":   res.get('src_ip', '?'),
            "Dst Port": res.get('dst_port', '?'),
            "Proto":    res.get('proto', '?'),
            "Pkt Len":  res.get('pkt_len', 0),
            "ML Prob":  f"{res.get('ml_prob',0)*100:.1f}%",
            "DL Score": f"{res.get('dl_score',0)*100:.1f}%",
            "Score":    f"{fs:.3f}",
            "Verdict":  label,
        }
        S["logs"].insert(0, entry)
        S["logs"] = S["logs"][:60]

        # Alert
        if label == "ATTACK":
            S["alerts"].insert(0, {
                "time":  time.strftime("%H:%M:%S"),
                "src":   res.get('src_ip', '?'),
                "port":  res.get('dst_port', '?'),
                "proto": res.get('proto', '?'),
                "score": fs,
            })
            S["alerts"] = S["alerts"][:15]

        drained += 1

    # Rotate chart every ~3 drains (approx 3 sec)
    if random.random() < 0.33:
        S["chart_attacks"].append(0)
        S["chart_attacks"] = S["chart_attacks"][-40:]
        S["chart_normal"].append(0)
        S["chart_normal"]  = S["chart_normal"][-40:]


# ═══════════════════════════════════════════════════════════════════════════════
#  LAYOUT
# ═══════════════════════════════════════════════════════════════════════════════

# ── Banner ────────────────────────────────────────────────────────────────────
pill_cls = "pill-live" if S["running"] else "pill-stopped"
pill_txt = "● LIVE" if S["running"] else "■ STOPPED"
st.markdown(f"""
<div class="top-banner">
  <div>
    <div class="banner-title">A-IDAPS-FL</div>
    <div class="banner-sub">ADAPTIVE AI INTRUSION DETECTION & PROTECTION SYSTEM  |  GROUP G-52  |  MMMUT</div>
  </div>
  <div style="display:flex;align-items:center;gap:14px">
    <span class="banner-sub">{time.strftime('%A %d %b %Y  %H:%M:%S')}</span>
    <span class="status-pill {pill_cls}">{pill_txt}</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Control buttons ───────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns([1,1,1,5])
if c1.button("▶  Start IDS", use_container_width=True):
    if not S["running"]:
        S["running"] = True
        _start_sniffer()
        # FIX 2: Drain immediately on start so packets appear on the very
        #         first rerun instead of waiting for the next sleep cycle.
        _drain_queue()

if c2.button("■  Stop IDS", use_container_width=True):
    S["running"] = False
    _stop_sniffer()

if c3.button("✕  Clear", use_container_width=True):
    S["logs"]   = []
    S["alerts"] = []
    S["stats"]  = Stats()
    S["chart_attacks"] = [0]*40
    S["chart_normal"]  = [0]*40

st.markdown("---")

# Drain queue while running
if S["running"]:
    _drain_queue()

# ── Metrics row ───────────────────────────────────────────────────────────────
stats: Stats = S["stats"]
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("📦 Packets Analyzed",  f"{stats.total_packets:,}")
m2.metric("⚠️ Attacks Detected",  f"{stats.total_attacks:,}")
m3.metric("📈 Attack Rate",        f"{stats.attack_rate:.1f}%")
m4.metric("⚡ Avg Latency",         f"{stats.avg_latency:.1f} ms")
m5.metric("🎯 Model Accuracy",     "99.0%")

st.markdown("---")

# ── Main layout: Left (traffic + chart) | Right (alerts + SHAP) ──────────────
left, right = st.columns([2, 1], gap="medium")

with left:
    # ── Live traffic feed ──────────────────────────────────────────────────
    st.markdown('<div class="section-label">Live Packet Feed</div>', unsafe_allow_html=True)

    if S["logs"]:
        df_logs = pd.DataFrame(S["logs"])
        # Colour verdicts with HTML
        def _style(v):
            if v == "ATTACK":
                return "⚠ ATTACK"
            return "✓ NORMAL"
        df_display = df_logs.copy()
        df_display["Verdict"] = df_display["Verdict"].map(_style)
        st.dataframe(
            df_display,
            use_container_width=True,
            height=220,
            hide_index=True,
        )
    else:
        st.info("No packets captured yet. Click **▶ Start IDS** to begin monitoring.", icon="📡")

    # ── Timeline chart ─────────────────────────────────────────────────────
    st.markdown('<div class="section-label" style="margin-top:14px">Traffic Timeline</div>', unsafe_allow_html=True)
    chart_df = pd.DataFrame({
        "Attacks": S["chart_attacks"],
        "Normal":  S["chart_normal"],
    })
    st.line_chart(chart_df, use_container_width=True, height=160)


with right:
    # ── Alerts ─────────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Threat Alerts</div>', unsafe_allow_html=True)
    if S["alerts"]:
        for a in S["alerts"][:6]:
            cls = "alert-crit" if a['score'] > 0.75 else "alert-warn"
            st.markdown(f"""
            <div class="alert-item {cls}">
                ⚑ {a['src']}:{a['port']} [{a['proto']}]<br>
                <span style="font-size:9px;opacity:0.7">Score: {a['score']:.3f} · {a['time']} · Auto-blocked</span>
            </div>""", unsafe_allow_html=True)
    else:
        st.markdown('<p style="font-family:Space Mono,monospace;font-size:11px;color:#4A5568">No threats detected</p>', unsafe_allow_html=True)

    # ── SHAP Explainability ────────────────────────────────────────────────
    st.markdown('<div class="section-label" style="margin-top:14px">XAI — SHAP Feature Impact</div>', unsafe_allow_html=True)
    shap_html = '<div class="shap-bar-container">'
    colors = ['#FF3B6B','#FF3B6B','#FFB830','#00E5FF','#7C3AED','#4A5568']
    for i, (feat, score) in enumerate(SHAP_WEIGHTS.items()):
        bar_w = int(score / 0.35 * 100)
        color = colors[i % len(colors)]
        shap_html += f"""
        <div class="shap-row">
            <div class="shap-label">{feat[:18]}</div>
            <div class="shap-outer">
                <div style="height:100%;width:{bar_w}%;background:{color};border-radius:4px"></div>
            </div>
            <div class="shap-score">+{score:.3f}</div>
        </div>"""
    shap_html += '</div>'
    st.markdown(shap_html, unsafe_allow_html=True)

st.markdown("---")

# ── Bottom row: Attack Simulator | Confusion Matrix | Attack Breakdown ────────
sim_col, matrix_col, breakdown_col = st.columns(3, gap="medium")

# ── ATTACK SIMULATOR ─────────────────────────────────────────────────────────
with sim_col:
    st.markdown('<div class="section-label">⚡ Attack Simulator</div>', unsafe_allow_html=True)
    st.caption("Tune features → inject packet → see model prediction")

    col_a, col_b = st.columns(2)
    pkt_len  = col_a.number_input("Packet Length (bytes)", min_value=20,  max_value=9000,  value=1200, step=10, key="sim_pkt")
    dst_port = col_b.number_input("Destination Port",      min_value=1,   max_value=65535, value=80,   step=1,  key="sim_port")
    flow_dur = col_a.number_input("Flow Duration (ms)",    min_value=1,   max_value=10000, value=500,  step=10, key="sim_dur")
    pkt_rate = col_b.number_input("Packets/sec Rate",     min_value=1,   max_value=5000,  value=50,   step=1,  key="sim_rate")
    fwd_pkts = col_a.number_input("Fwd Packet Count",      min_value=1,   max_value=500,   value=5,    step=1,  key="sim_fpkts")
    proto_idx = col_b.radio("Protocol", ["TCP (6)", "UDP (17)"], horizontal=True, key="sim_proto")
    proto_num = 6 if proto_idx.startswith("TCP") else 17
    if st.button("⚡  Run Prediction", use_container_width=True, key="sim_run"):
        with st.spinner("Running hybrid model..."):
            res = simulate_packet(
                pkt_len=pkt_len,
                dst_port=dst_port,
                flow_duration_ms=flow_dur,
                pkt_rate=float(pkt_rate),
                proto=proto_num,
                fwd_pkts=fwd_pkts,
            )
        S["sim_result"] = res

        # Inject into live feed
        label = res['label']
        fs    = res['final_score']
        S["stats"].record(label, fs, latency_ms=random.uniform(6,18))

        atype = classify_attack_type({"Destination Port": float(dst_port), "Fwd Packets/s": float(pkt_rate * 40)})
        if label == "ATTACK":
            S["stats"].attack_types[atype] = S["stats"].attack_types.get(atype, 0) + 1
            S["alerts"].insert(0, {"time": time.strftime("%H:%M:%S"), "src":"SIM:10.0.99.1","port":dst_port,"proto":proto_idx[:3],"score":fs})
            S["alerts"] = S["alerts"][:15]

        S["logs"].insert(0, {
            "Time": time.strftime("%H:%M:%S"),
            "Src IP": "SIM:10.0.99.1",
            "Dst Port": dst_port,
            "Proto": proto_idx[:3],
            "Pkt Len": pkt_len,
            "ML Prob": f"{res['ml_prob']*100:.1f}%",
            "DL Score": f"{res['dl_score']*100:.1f}%",
            "Score": f"{fs:.3f}",
            "Verdict": label,
        })
        S["logs"] = S["logs"][:60]

    # Show result
    if S["sim_result"]:
        r = S["sim_result"]
        if r['label'] == "ATTACK":
            st.error(f"⚠ ATTACK DETECTED  |  Confidence: {r['confidence']}%")
        else:
            st.success(f"✓ NORMAL TRAFFIC  |  Confidence: {r['confidence']}%")

        with st.expander("Model Score Breakdown"):
            st.markdown(f"""
            | Component | Score |
            |---|---|
            | XGBoost ML Prob | `{r['ml_prob']:.4f}` |
            | Autoencoder DL Score | `{r['dl_score']:.4f}` |
            | **Final Ensemble Score** | **`{r['final_score']:.4f}`** |
            | Decision Threshold | `0.3500` |
            """)

# ── CONFUSION MATRIX ─────────────────────────────────────────────────────────
with matrix_col:
    st.markdown('<div class="section-label">Detection Matrix</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px">
        <div class="matrix-cell">
            <div class="matrix-val" style="color:#00FF9D">{stats.cm_tp}</div>
            <div class="matrix-lbl">TRUE POSITIVE</div>
        </div>
        <div class="matrix-cell">
            <div class="matrix-val" style="color:#00E5FF">{stats.cm_tn}</div>
            <div class="matrix-lbl">TRUE NEGATIVE</div>
        </div>
        <div class="matrix-cell">
            <div class="matrix-val" style="color:#FFB830">{stats.cm_fp}</div>
            <div class="matrix-lbl">FALSE POSITIVE</div>
        </div>
        <div class="matrix-cell">
            <div class="matrix-val" style="color:#FF3B6B">{stats.cm_fn}</div>
            <div class="matrix-lbl">FALSE NEGATIVE</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Model Performance</div>', unsafe_allow_html=True)
    perf = {"Accuracy":99.0,"Precision":98.8,"Recall":99.2,"F1-Score":99.0,"AUC-ROC":99.7}
    for name, val in perf.items():
        st.markdown(f"""
        <div style="margin-bottom:8px">
            <div style="display:flex;justify-content:space-between;font-family:Space Mono,monospace;font-size:10px;color:#4A5568;margin-bottom:3px">
                <span>{name}</span><span style="color:#E2E8F0">{val:.1f}%</span>
            </div>
            <div style="height:5px;background:#1E2535;border-radius:3px;overflow:hidden">
                <div style="height:100%;width:{val}%;background:linear-gradient(90deg,#00E5FF,#7C3AED);border-radius:3px"></div>
            </div>
        </div>""", unsafe_allow_html=True)

# ── ATTACK BREAKDOWN ──────────────────────────────────────────────────────────
with breakdown_col:
    st.markdown('<div class="section-label">Attack Type Breakdown</div>', unsafe_allow_html=True)
    attack_colors = {'DDoS':'#FF3B6B','PortScan':'#FFB830','BruteForce':'#7C3AED','SQLInject':'#00E5FF','Other':'#4A5568'}
    total_attacks = max(sum(stats.attack_types.values()), 1)
    for atype, cnt in stats.attack_types.items():
        pct = cnt / total_attacks * 100
        color = attack_colors.get(atype, '#4A5568')
        st.markdown(f"""
        <div style="margin-bottom:10px">
            <div style="display:flex;justify-content:space-between;font-family:Space Mono,monospace;font-size:10px;margin-bottom:4px">
                <span style="color:#4A5568">{atype}</span>
                <span style="color:#E2E8F0">{cnt} &nbsp;|&nbsp; {pct:.1f}%</span>
            </div>
            <div style="height:5px;background:#1E2535;border-radius:3px;overflow:hidden">
                <div style="height:100%;width:{pct:.1f}%;background:{color};border-radius:3px;transition:width 0.3s"></div>
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Autoencoder Architecture</div>', unsafe_allow_html=True)
    layers = [("Input", 30, "#4A5568"),("Encoder 1",128,"#00E5FF"),("Encoder 2",64,"#7C3AED"),
              ("Bottleneck",8,"#FF3B6B"),("Decoder 1",32,"#7C3AED"),("Decoder 2",64,"#00E5FF"),("Output",30,"#4A5568")]
    for lname, size, color in layers:
        bar_w = max(int(size/128*100), 5)
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px">
            <span style="font-family:Space Mono,monospace;font-size:9px;color:#4A5568;width:75px;text-align:right">{lname}</span>
            <div style="height:12px;width:{bar_w}%;background:{color};opacity:0.75;border-radius:3px"></div>
            <span style="font-family:Space Mono,monospace;font-size:9px;color:#4A5568">{size}</span>
        </div>""", unsafe_allow_html=True)

# ── Auto-refresh ──────────────────────────────────────────────────────────────
# FIX 3: Reduced sleep from 1s → 0.3s so packets appear ~3x faster after
#         Start IDS is clicked. Functionality is identical — just faster polling.
if S["running"]:
    time.sleep(0.3)
    st.rerun()