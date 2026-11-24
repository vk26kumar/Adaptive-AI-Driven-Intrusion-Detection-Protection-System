"""
app.py  --  A-IDAPS-FL dashboard (simple view)
Run:  streamlit run app.py

Four tabs, plain language:
  Live Monitor      start/stop live capture, verdict banner, recent connections, "why?"
  Try an Attack     send a real recorded attack (or normal) flow through the model
  Protection        automatic blocking (practice mode or real firewall), unblock
  How It Was Trained  federated training summary and accuracy; research details collapsed
The detection, prevention and logging logic is unchanged (core/, network/).
"""
import json
import os
import sys
import threading
import time
from queue import Empty

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import pandas as pd
import streamlit as st

from core.state import packet_queue, stop_event, Stats, classify_attack_type, alert_kind
from core.predictor import explain_features, MODEL_INFO
from core.simulator import simulate_packet, PROFILES
from core.prevention import engine as prevention, BLOCK_THRESHOLD, MIN_HITS, WINDOW_S
from core.persistence import store

st.set_page_config(page_title="A-IDAPS-FL | Network Guard", layout="wide")

# ----------------------------------------------------------- plain language --
FRIENDLY = {
    "DDoS": "Flood attack (DDoS)", "DoS": "Overload attack (DoS)", "PortScan": "Port scan (checking for open doors)",
    "BruteForce": "Password guessing", "Bot": "Infected machine (bot)", "WebAttack": "Website attack",
    "Other": "Other attack", "Normal": "Normal traffic", "BENIGN": "Normal traffic",
    "Unusual": "Unusual traffic",
}
FEATURE_WORDS = {
    "Destination Port": "the port it targeted", "Init_Win_bytes_forward": "the sender's TCP window size",
    "Init_Win_bytes_backward": "the receiver's TCP window size", "Fwd Packets/s": "how fast packets were sent",
    "Total Fwd Packets": "how many packets were sent", "Total Length of Fwd Packets": "how much data was sent",
    "Subflow Fwd Bytes": "how much data was sent", "Fwd Packet Length Mean": "the average packet size sent",
    "Fwd Packet Length Max": "the largest packet sent", "Bwd Packet Length Min": "the smallest reply packet",
    "Bwd Packet Length Mean": "the average reply size", "Fwd IAT Total": "how long the sending lasted",
    "Fwd IAT Mean": "the gap between packets", "Flow IAT Std": "how irregular the timing was",
    "Min Packet Length": "the smallest packet", "Average Packet Size": "the average packet size",
    "Down/Up Ratio": "download vs upload ratio", "Fwd Header Length": "the packet header size",
    "Fwd Header Length.1": "the packet header size", "Subflow Bwd Packets": "how many replies came back",
}
TECH = "Hybrid AI (Autoencoder + XGBoost), trained with federated learning on 4 sites"

st.markdown("""
<style>
.block-container { padding-top: 1.2rem; max-width: 1200px; }
.banner { padding: 18px 22px; border-radius: 12px; font-size: 1.25rem; font-weight: 600; margin: 6px 0 14px 0; }
.ok   { background: #e8f7ef; color: #11643a; border: 1px solid #b7e4c7; }
.bad  { background: #fdecec; color: #9b1c1c; border: 1px solid #f5c2c2; }
.idle { background: #eef2f7; color: #334155; border: 1px solid #d6dde6; }
.warn { background: #fff7e0; color: #7a5300; border: 1px solid #f3dc9b; }
.sub  { font-size: 0.95rem; font-weight: 400; margin-top: 4px; }
</style>""", unsafe_allow_html=True)


# ------------------------------------------------------------------- state --
def _init():
    for k, v in {"running": False, "stats": Stats(), "rows": [], "alerts": [], "sim": None,
                 "sim_tests": 0, "sim_correct": 0, "sim_attacks": 0, "sniffer_error": None,
                 "flows_seen": set(), "flows_attack": set(), "flows_unusual": set(), "last_alert": {}}.items():
        st.session_state.setdefault(k, v)


_init()
S = st.session_state


def _start_sniffer():
    try:
        from network.sniffer import start_sniffing
        stop_event.clear()
        iface = os.environ.get("IDS_IFACE") or None          # e.g. \Device\NPF_Loopback for local tests
        threading.Thread(target=start_sniffing, args=(packet_queue, stop_event, iface), daemon=True).start()
        S["sniffer_error"] = None
    except Exception as e:  # missing Npcap / scapy
        S["sniffer_error"] = str(e)
        S["running"] = False


def _add_row(source, src, port, label, score, atype, features, kind=None):
    kind = kind or ("attack" if label == "ATTACK" else "normal")
    S["rows"].insert(0, {"time": time.strftime("%H:%M:%S"), "source": source, "from": src, "port": port,
                         "label": label, "score": score, "type": atype, "features": features, "kind": kind})
    S["rows"] = S["rows"][:60]


def _add_alert(src, port, score, atype, features, note="", kind="attack"):
    S["alerts"].insert(0, {"kind": kind, "ts": time.time(), "time": time.strftime("%H:%M:%S"), "from": src, "port": port, "score": score,
                           "type": atype, "features": features, "note": note})
    S["alerts"] = S["alerts"][:20]


def _drain_queue(max_items=60):
    for _ in range(max_items):
        try:
            r = packet_queue.get_nowait()
        except Empty:
            break
        label, score, feat = r.get("label", "NORMAL"), r.get("final_score", 0.0), r.get("features", {})
        S["stats"].record(label, score, latency_ms=r.get("latency_ms", 0.0), truth=None)
        # one connection is re-checked several times; count and alert per connection, not per check
        key = (r.get("src_ip"), r.get("dst_ip"), r.get("dst_port"), r.get("proto"), r.get("detector", ""))
        S["flows_seen"].add(key)
        kind = alert_kind(r)
        atype = None
        if kind == "attack":
            atype = r.get("attack_type") or classify_attack_type(feat)
            S["flows_attack"].add(key)
            blk = prevention.observe(r.get("src_ip", "?"), score, atype)   # only real attacks can lead to a block
            if blk:
                store.log_action("block", blk.ip, {"mode": blk.mode, "score": score, "type": atype})
        elif kind == "unusual":
            atype = "Unusual"
            S["flows_unusual"].add(key)
        if kind != "normal" and time.time() - S["last_alert"].get(key, 0) > 60:
            S["last_alert"][key] = time.time()
            if kind == "attack":
                S["stats"].attack_types[atype] = S["stats"].attack_types.get(atype, 0) + 1
            _add_alert(r.get("src_ip", "?"), r.get("dst_port", "?"), score, atype, feat, r.get("detector", ""), kind)
        store.log_flow(r, source="live", attack_type=atype if kind == "attack" else None)
        _add_row("live", r.get("src_ip", "?"), r.get("dst_port", "?"), label, score, atype, feat, kind)


def _read_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _why(features, label):
    """Top 3 reasons in plain words (XGBoost TreeSHAP contributions)."""
    try:
        exp = explain_features(features, top_k=6)
    except Exception as e:
        st.caption(f"Explanation unavailable: {e}")
        return
    items = list(exp["ml_contrib"].items())
    if label == "ATTACK":
        reasons = [k for k, v in items if v > 0][:3] or [k for k, _ in items][:3]
        intro = "Main reasons it looks like an attack:"
    else:
        reasons = [k for k, v in items if v < 0][:3] or [k for k, _ in items][:3]
        intro = "Main reasons it looks normal:"
    st.markdown(intro)
    for k in reasons:
        st.markdown(f"- {FEATURE_WORDS.get(k, k)}")
    ae = list(exp["dl_contrib"].items())[:1]
    if ae and label == "ATTACK":
        st.caption(f"The anomaly detector found {FEATURE_WORDS.get(ae[0][0], ae[0][0])} most unusual "
                   f"({ae[0][1]:.0%} of what it could not explain).")


# ================================================================== header ==
if S["running"]:
    _drain_queue()

st.title("A-IDAPS-FL Network Guard")
st.caption("Watches your network traffic, spots attacks with AI, explains why, and can block the attacker.")

b1, b2, b3, status = st.columns([1, 1, 1, 3])
if b1.button("▶ Start monitoring", type="primary", width="stretch", disabled=S["running"]):
    S["running"] = True
    _start_sniffer()
    st.rerun()
if b2.button("■ Stop", width="stretch", disabled=not S["running"]):
    S["running"] = False
    stop_event.set()
    st.rerun()
if b3.button("Clear", width="stretch"):
    S.update({"stats": Stats(), "rows": [], "alerts": [], "sim": None, "sim_tests": 0, "sim_correct": 0, "sim_attacks": 0,
              "flows_seen": set(), "flows_attack": set(), "flows_unusual": set(), "last_alert": {}})
    st.rerun()
status.markdown(f"**Monitoring:** {'ON (watching live traffic)' if S['running'] else 'OFF'}")
if S["sniffer_error"]:
    st.error(f"Live capture is not available: {S['sniffer_error']}. Install Npcap (npcap.com). "
             "The 'Try an Attack' tab still works.")

stats: Stats = S["stats"]
acc = ((MODEL_INFO.get("test_metrics") or {}).get("hybrid") or {}).get("accuracy")
k1, k2, k3, k4 = st.columns(4)
n_conn = len(S["flows_seen"]) + S["sim_tests"]
n_att = len(S["flows_attack"]) + S["sim_attacks"]
k1.metric("Connections checked", f"{n_conn:,}")
k2.metric("Attacks found", f"{n_att:,}", help="Counted once per connection, even if it is checked several times.")
k3.metric("Attackers blocked", f"{len(prevention.blocked)}")
k4.metric("Model accuracy (test data)", f"{acc * 100:.2f}%" if acc else "n/a",
          help="Measured on 283,074 recorded connections the model never saw during training.")

tab_live, tab_try, tab_protect, tab_train = st.tabs(
    ["Live Monitor", "Try an Attack", "Protection", "How It Was Trained"])

# ============================================================ live monitor ==
with tab_live:
    now = time.time()
    recent_attack = next((a for a in S["alerts"] if a.get("kind") == "attack" and now - a["ts"] < 30), None)
    recent_unusual = next((a for a in S["alerts"] if a.get("kind") == "unusual" and now - a["ts"] < 30), None)
    if not S["running"] and not S["rows"]:
        st.markdown('<div class="banner idle">Monitoring is off.<div class="sub">Press <b>▶ Start monitoring</b> '
                    'to watch this computer\'s live traffic, or use the <b>Try an Attack</b> tab.</div></div>',
                    unsafe_allow_html=True)
    elif recent_attack:
        a = recent_attack
        st.markdown(f'<div class="banner bad">Attack detected: {FRIENDLY.get(a["type"], a["type"])}'
                    f'<div class="sub">From {a["from"]} to port {a["port"]} · '
                    f'{a["score"]:.0%} sure · {a["time"]}</div></div>', unsafe_allow_html=True)
    elif recent_unusual:
        st.markdown(f'<div class="banner warn">No attacks · some unusual traffic'
                    f'<div class="sub">{n_conn:,} connections checked. A few connections look different from the '
                    f'traffic the AI was trained on, but it does not recognise them as attacks. They are not '
                    f'blocked.</div></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="banner ok">All clear<div class="sub">{n_conn:,} connections checked, '
                    f'no attack in the last 30 seconds.</div></div>', unsafe_allow_html=True)

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.subheader("Recent connections")
        if S["rows"]:
            df = pd.DataFrame([{"Time": r["time"], "From": r["from"], "To port": r["port"],
                                "Verdict": {"attack": "Attack", "unusual": "Unusual"}.get(r.get("kind"), "Normal"),
                                "How sure": f"{(r['score'] if r['label'] == 'ATTACK' else 1 - r['score']):.0%}",
                                "Source": "live" if r["source"] == "live" else "test"} for r in S["rows"][:25]])
            st.dataframe(df, hide_index=True, width="stretch", height=380)
        else:
            st.caption("Nothing yet.")
    with right:
        st.subheader("Alerts")
        if S["alerts"]:
            for i, a in enumerate(S["alerts"][:5]):
                # stable key per alert: avoids "ghost" copies while the page auto-refreshes
                with st.container(border=True, key=f"alert_{a['ts']:.6f}"):
                    if a.get("kind") == "unusual":
                        st.markdown(f"**Unusual traffic (not an attack)**  \n"
                                    f"from `{a['from']}` → port {a['port']} · {a['time']}")
                        with st.expander("What does this mean?"):
                            st.markdown("Only the anomaly detector flagged this connection: it looks different "
                                        "from the 2017 traffic the AI learned from. The attack classifier does "
                                        "**not** recognise it as any known attack, so it is not blocked. This is "
                                        "usually a newer kind of normal traffic (for example a long streaming "
                                        "connection, Windows update sharing or DNS).")
                        continue
                    st.markdown(f"**{FRIENDLY.get(a['type'], a['type'])}**  \n"
                                f"from `{a['from']}` → port {a['port']} · {a['score']:.0%} sure · {a['time']}")
                    if a["note"]:
                        st.caption(a["note"].replace("behaviour:", "Seen by the port-scan rule:"))
                    with st.expander("Why?"):
                        if a["note"]:
                            st.markdown("One source tried many different ports in a few seconds, "
                                        "which is how port scanners behave.")
                        else:
                            _why(a["features"], "ATTACK")
            if S["flows_unusual"]:
                st.caption(f"{len(S['flows_unusual'])} connection(s) flagged as unusual so far (not counted as attacks).")
        else:
            st.caption("No attacks detected.")
    if S["running"]:
        st.caption(f"Average check time: {stats.avg_latency:.0f} ms per connection.")

# ============================================================ try an attack ==
with tab_try:
    st.markdown("Send one **real recorded connection** from the training dataset through the system and see "
                "whether it is caught. Nothing is sent over your network.")
    c1, c2 = st.columns([2, 3], gap="large")
    with c1:
        choice = st.radio("Type of traffic", PROFILES, format_func=lambda p: FRIENDLY.get(p, p), key="try_profile")
        with st.expander("Advanced: change the connection"):
            override = st.checkbox("Use my values", key="try_override")
            port = st.number_input("Target port", 1, 65535, 80, key="try_port", disabled=not override)
            rate = st.number_input("Packets per second", 0.0, 3_000_000.0, 50.0, key="try_rate", disabled=not override)
            size = st.number_input("Packet size (bytes)", 0, 9000, 40, key="try_size", disabled=not override)
        if st.button("Send test traffic", type="primary", width="stretch"):
            kw = dict(dst_port=port, pkt_rate=rate, pkt_len=size) if override else {}
            res = simulate_packet(profile=choice, **kw)
            S["sim"] = res
            S["stats"].record(res["label"], res["final_score"], latency_ms=res["latency_ms"], truth=res["truth"])
            S["sim_tests"] += 1
            S["sim_correct"] += int(res["label"] == res["truth"])
            S["sim_attacks"] += int(res["label"] == "ATTACK")
            dport = int(res["features"]["Destination Port"])
            atype = res["family"] if res["family"] != "BENIGN" else None
            if res["label"] == "ATTACK":
                atype = atype or classify_attack_type(res["features"])
                S["stats"].attack_types[atype] = S["stats"].attack_types.get(atype, 0) + 1
                _add_alert("test traffic", dport, res["final_score"], atype, res["features"])
            _add_row("test", "test traffic", dport, res["label"], res["final_score"], atype, res["features"])
            store.log_flow(res, source=f"sim:{choice}", truth=res["truth"], attack_type=atype)
            st.rerun()
    with c2:
        r = S["sim"]
        if r:
            caught = r["label"] == "ATTACK"
            correct = r["label"] == r["truth"]
            sure = r["final_score"] if caught else 1 - r["final_score"]
            what = FRIENDLY.get(r["profile"], r["profile"])
            css = "bad" if caught else "ok"
            title = f"Detected as ATTACK ({sure:.0%} sure)" if caught else f"Judged NORMAL ({sure:.0%} sure)"
            st.markdown(f'<div class="banner {css}">{title}<div class="sub">You sent: {what} · '
                        f'{"the system was right" if correct else "the system was wrong"}</div></div>',
                        unsafe_allow_html=True)
            _why(r["features"], r["label"])
        else:
            st.info("Pick a type of traffic and press **Send test traffic**.")
        if S["sim_tests"]:
            st.metric("Test score this session", f"{S['sim_correct']} / {S['sim_tests']} correct")

# ============================================================== protection ==
with tab_protect:
    st.markdown(f"When the same **outside** address causes **{MIN_HITS} strong alerts** (at least "
                f"{BLOCK_THRESHOLD:.0%} sure) within **{WINDOW_S:.0f} seconds**, the system can block it. "
                "Local / home-network addresses are never blocked.")
    p1, p2 = st.columns(2)
    prevention.enabled = p1.toggle("Automatic blocking", value=prevention.enabled, key="prot_on")
    real = p2.toggle("Real firewall (off = practice mode, only records what it would block)",
                     value=not prevention.dry_run, key="prot_real")
    prevention.dry_run = not real
    if real:
        st.warning("Real blocking adds Windows Firewall rules and needs the dashboard to run as Administrator.")
    st.subheader("Blocked addresses")
    if prevention.blocked:
        for ip, rec in list(prevention.blocked.items()):
            q1, q2 = st.columns([4, 1])
            mode = "practice mode (not really blocked)" if rec.mode == "dry-run" else "blocked in firewall"
            q1.markdown(f"`{ip}` · {FRIENDLY.get(rec.attack_type, rec.attack_type)} · {mode}"
                        f"{' · ' + rec.error if rec.error else ''}")
            if q2.button("Unblock", key=f"unblock_{ip}"):
                prevention.unblock(ip); store.log_action("unblock", ip); st.rerun()
    else:
        st.caption("Nothing blocked.")
    counts = store.counts()
    st.caption(f"History saved: {counts['flows']:,} connections, {counts['attacks']:,} attacks, "
               f"{counts['actions']} block/unblock actions (logs/ids_events.sqlite).")

# ======================================================= how it was trained ==
with tab_train:
    st.markdown("**How the AI learned, without anyone sharing their data:** 4 sites (organisations) each trained "
                "the model on their own traffic and sent only the learned settings, encrypted, to a server that "
                "combined them. No traffic records left any site. This computer then joined as a 5th site and "
                "taught the model what its own normal traffic looks like (local adaptation), again without its "
                "traffic leaving the computer.")
    local = _read_json(os.path.join(_ROOT, "federated", "logs", "local_adaptation.json"))
    cmp = _read_json(os.path.join(_ROOT, "federated", "logs", "comparison.json"))
    if cmp:
        rows = []
        for key, name in (("phase1_ddos_only", "Old version (Phase 1)"), ("centralized_phase2", "All data in one place"),
                          ("federated_phase2", "Our federated model (no data shared)")):
            h = cmp[key]["hybrid"]
            sent = cmp[key].get("training_rows_moved_to_server")
            rows.append({"Model": name, "Accuracy": f"{h['accuracy']:.2%}", "Attacks caught": f"{h['recall']:.2%}",
                         "Normal traffic wrongly flagged": f"{h['fpr']:.2%}",
                         "Data records shared": f"{sent:,}" if isinstance(sent, int) else "-"})
        if local:
            h = local["report_half_after"]
            rows.append({"Model": "+ this computer as 5th site (in use now)", "Accuracy": f"{h['accuracy']:.2%}",
                         "Attacks caught": f"{h['recall']:.2%}", "Normal traffic wrongly flagged": f"{h['fpr']:.2%}",
                         "Data records shared": "0"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption(f"Tested on {cmp['test_rows']:,} recorded connections the model never saw during training.")
        fam = (local or {}).get("report_half_after", {}).get("per_family") or cmp["federated_phase2"]["hybrid_recall_per_family"]
        st.subheader("Attacks caught, by type (model in use)")
        for k, v in fam.items():
            if k != "Other":
                st.progress(min(v, 1.0), text=f"{FRIENDLY.get(k, k)}: **{v:.1%}** caught")
    with st.expander("Technical details (for developers)"):
        st.markdown(f"- **Model:** {TECH}.\n"
                    "- **Autoencoder:** 30-128-64-32-8-32-64-128-30 dense network trained on normal traffic only; "
                    "flags connections it cannot rebuild.\n"
                    "- **XGBoost:** 480 gradient-boosted trees (depth 8); recognises known attacks.\n"
                    "- **Decision:** attack if either model is confident; scores are calibrated (Platt scaling).\n"
                    "- **Training:** Flower framework, TLS, FedAvg (Autoencoder, 15 rounds) and tree bagging "
                    "(XGBoost, 40 rounds).")
        for name, f in (("Autoencoder training rounds", "ae_rounds.json"), ("XGBoost training rounds", "xgb_rounds.json")):
            log = _read_json(os.path.join(_ROOT, "federated", "logs", f))
            if log:
                r = [x for x in log["rounds"] if x.get("round", 1) > 0]
                st.caption(f"{name}: recall per round" + (" (raw 0.5 cut-off, before calibration)" if "xgb" in f else ""))
                st.line_chart(pd.DataFrame({"recall": [x["recall"] for x in r]}, index=[x["round"] for x in r]), height=140)
        comm = _read_json(os.path.join(_ROOT, "federated", "logs", "comm_analysis.json"))
        if local:
            lb, la = local["live_test_before"], local["live_test_after"]
            st.caption(f"Local adaptation: on a separate capture of this computer's normal traffic, normal connections "
                       f"called an attack fell from {lb['classifier_flags_pct']}% to {la['classifier_flags_pct']}%.")
        if comm:
            st.caption(f"Network traffic of training: {comm['federated']['megabytes_total']:.0f} MB of model updates vs "
                       f"{comm['centralized']['raw_megabytes']:.0f} MB to upload the raw data.")
        cross = _read_json(os.path.join(_ROOT, "federated", "logs", "cross_dataset_2018.json"))
        if cross:
            st.caption(f"Known limit: on a different dataset (CSE-CIC-IDS2018) it catches only "
                       f"{cross['hybrid_population_weighted']['recall']:.1%} of attacks.")

if S["running"]:
    time.sleep(1.0)
    st.rerun()
