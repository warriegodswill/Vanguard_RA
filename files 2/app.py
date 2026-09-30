"""
Vanguard RA - Streamlit Dashboard
----------------------------------
Run with:
    streamlit run app.py

Lets a merchant upload their own Bank / POS / Inventory CSVs, or load the
bundled simulated demo data, then shows matched/mismatch/leak/anomaly
breakdowns, risk score, and downloadable reports.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

from reconciliation_engine import ReconciliationEngine
from recovery_tracker import RecoveryStatus, RecoveryTracker

st.set_page_config(
    page_title="Vanguard RA | Revenue Assurance Dashboard",
    page_icon="🛡️",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Sidebar — data input
# ---------------------------------------------------------------------------
st.sidebar.title("🛡️ Vanguard RA")
st.sidebar.caption("Protecting Every Transaction. Building Financial Trust.")

data_source = st.sidebar.radio(
    "Data source",
    ["Use simulated demo data", "Upload my own CSVs"],
)

pos_df = bank_df = inventory_df = None

if data_source == "Use simulated demo data":
    try:
        pos_df = pd.read_csv("pos_records.csv")
        bank_df = pd.read_csv("bank_records.csv")
        inventory_df = pd.read_csv("inventory_records.csv")
        st.sidebar.success("Loaded simulated demo dataset.")
    except FileNotFoundError:
        st.sidebar.error("Demo CSVs not found. Run generate_sample_data.py first.")
else:
    pos_file = st.sidebar.file_uploader("POS Records CSV", type="csv")
    bank_file = st.sidebar.file_uploader("Bank Records CSV", type="csv")
    inv_file = st.sidebar.file_uploader("Inventory Records CSV", type="csv")
    if pos_file and bank_file and inv_file:
        pos_df = pd.read_csv(pos_file)
        bank_df = pd.read_csv(bank_file)
        inventory_df = pd.read_csv(inv_file)

st.sidebar.divider()
st.sidebar.markdown(
    "**Business model**\n"
    "- Free dashboard\n"
    "- Premium monitoring\n"
    "- 0.1% success fee on recovered leakages"
)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
st.title("AI-Powered Revenue Assurance Dashboard")

if pos_df is None or bank_df is None or inventory_df is None:
    st.info("⬅️ Load demo data or upload your Bank, POS, and Inventory CSVs from the sidebar to begin.")
    st.stop()

data_signature = (len(pos_df), len(bank_df), len(inventory_df))
if "data_signature" not in st.session_state or st.session_state.data_signature != data_signature:
    with st.spinner("Running reconciliation + ML anomaly detection..."):
        engine = ReconciliationEngine(pos_df, bank_df, inventory_df)
        result = engine.run(run_ml_detection=True)
    st.session_state.result = result
    st.session_state.tracker = result.recovery_tracker
    st.session_state.data_signature = data_signature

result = st.session_state.result
tracker = st.session_state.tracker
s = result.summary
recovery = tracker.summary()

# --- KPI row ---
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total POS Revenue", f"₦{s['total_pos_revenue']:,.0f}")
col2.metric("Leaked Revenue (detected)", f"₦{s['total_leaked_amount']:,.0f}", f"{s['leak_percentage']}% of revenue")
col3.metric("Actually Recovered", f"₦{recovery['total_recovered']:,.0f}", f"{recovery['recovery_rate_pct']}% recovery rate")
col4.metric("Risk Score", f"{s['risk_score']} / 100", s["risk_band"])
col5.metric("Billable Success Fee (0.1%)", f"₦{recovery['total_success_fees']:,.2f}",
            help="Charged ONLY on money confirmed recovered — not on detected leaks. "
                 "This is deliberately different from 'detected leaks × 0.1%'.")

st.caption(
    f"⚠️ Detected ≠ recovered. Of ₦{s['total_leaked_amount']:,.0f} in detected leaks, "
    f"only ₦{recovery['total_recovered']:,.0f} has been confirmed recovered so far "
    f"({recovery['by_status']['detected']['count']} leaks still sitting undisputed, "
    f"{recovery['by_status']['written_off']['count']} written off as unrecoverable). "
    f"The success fee is billed on the recovered figure only."
)

st.divider()

# --- Alert banner ---
if s["risk_band"] == "High":
    st.error(f"⚠️ High risk detected — ₦{s['total_leaked_amount']:,.0f} in leakages found. Immediate review recommended.")
elif s["risk_band"] == "Medium":
    st.warning(f"⚠️ Medium risk — ₦{s['total_leaked_amount']:,.0f} in leakages detected across {s['n_leaks']} transactions.")
else:
    st.success("✅ Low risk — reconciliation looks healthy.")

# --- Breakdown chart ---
left, right = st.columns([1, 1])

with left:
    st.subheader("Reconciliation Breakdown")
    breakdown = pd.DataFrame({
        "Category": ["Matched", "Mismatches", "Leaks", "Anomalies"],
        "Count": [s["n_matched"], s["n_mismatches"], s["n_leaks"], s["n_anomalies"]],
    })
    fig = px.pie(
        breakdown, names="Category", values="Count", hole=0.5,
        color="Category",
        color_discrete_map={
            "Matched": "#16a34a",
            "Mismatches": "#f59e0b",
            "Leaks": "#dc2626",
            "Anomalies": "#7c3aed",
        },
    )
    fig.update_traces(textinfo="value+percent")
    st.plotly_chart(fig, width='stretch')

with right:
    st.subheader("Leakage Over Time")
    if not result.leaks.empty and "timestamp" in result.leaks.columns:
        leak_ts = result.leaks.copy()
        leak_ts["timestamp"] = pd.to_datetime(leak_ts["timestamp"], errors="coerce")
        leak_ts["date"] = leak_ts["timestamp"].dt.date
        daily_leak = leak_ts.groupby("date")["leaked_amount"].sum().reset_index()
        fig2 = px.bar(daily_leak, x="date", y="leaked_amount", labels={"leaked_amount": "Leaked ₦"})
        fig2.update_traces(marker_color="#dc2626")
        st.plotly_chart(fig2, width='stretch')
    else:
        st.caption("No time-stamped leak data available.")

st.divider()

# --- Detail tabs ---
tab_leaks, tab_mismatch, tab_anomaly, tab_ml, tab_recovery, tab_matched = st.tabs([
    f"🔴 Leaks ({s['n_leaks']})", f"🟠 Mismatches ({s['n_mismatches']})",
    f"🟣 Rule Anomalies ({s['n_anomalies']})", f"🧠 ML Anomalies ({s['n_ml_anomalies']})",
    f"💰 Recovery Tracking ({recovery['n_leaks']})", f"🟢 Matched ({s['n_matched']})",
])

with tab_leaks:
    st.dataframe(result.leaks, width='stretch')
    if not result.leaks.empty:
        st.download_button(
            "Download leaks report (CSV)",
            result.leaks.to_csv(index=False),
            file_name=f"vanguard_ra_leaks_{datetime.now():%Y%m%d}.csv",
        )

with tab_mismatch:
    st.dataframe(result.mismatches, width='stretch')

with tab_anomaly:
    st.caption("Fixed-rule anomalies (e.g. settlement delayed beyond a set threshold).")
    st.dataframe(result.anomalies, width='stretch')

with tab_ml:
    st.caption(
        "Statistical anomalies from an Isolation Forest model fit on THIS merchant's own "
        "settlement history — flags what's unusual for them, not against a generic fixed rule. "
        "This is the actual 'AI' in AI-powered; the tabs above are deterministic bookkeeping rules."
    )
    if result.ml_anomalies.empty:
        st.info("No statistical anomalies flagged (or insufficient transaction history to train on).")
    else:
        st.dataframe(result.ml_anomalies, width='stretch')

with tab_recovery:
    st.caption(
        "Every detected leak starts here as 'detected' and is worth ₦0 in fees until it's "
        "actually disputed and recovered. Update statuses below to reflect real-world outcomes."
    )

    rc1, rc2, rc3, rc4 = st.columns(4)
    rc1.metric("Detected", f"₦{recovery['by_status']['detected']['amount']:,.0f}", f"{recovery['by_status']['detected']['count']} leaks")
    rc2.metric("Disputed", f"₦{recovery['by_status']['disputed']['amount']:,.0f}", f"{recovery['by_status']['disputed']['count']} leaks")
    rc3.metric("Recovered", f"₦{recovery['total_recovered']:,.0f}", f"{recovery['by_status']['recovered']['count'] + recovery['by_status']['partial']['count']} leaks")
    rc4.metric("Written Off", f"₦{recovery['total_written_off']:,.0f}", f"{recovery['by_status']['written_off']['count']} leaks")

    st.divider()
    st.subheader("Update a leak's recovery status")

    open_leaks = {
        lid: r for lid, r in tracker.records.items()
        if RecoveryTracker.VALID_TRANSITIONS.get(r.status)
    }
    if not open_leaks:
        st.success("All leaks have reached a terminal status (recovered or written off).")
    else:
        selected = st.selectbox(
            "Select a leak",
            options=list(open_leaks.keys()),
            format_func=lambda lid: f"{lid} — ₦{open_leaks[lid].detected_amount:,.0f} — currently '{open_leaks[lid].status.value}'",
        )
        current_record = open_leaks[selected]
        valid_next = sorted([s.value for s in RecoveryTracker.VALID_TRANSITIONS[current_record.status]])

        colA, colB, colC = st.columns([1, 1, 2])
        next_status = colA.selectbox("New status", options=valid_next)
        recovered_amt = None
        if next_status in ("recovered", "partial"):
            recovered_amt = colB.number_input(
                "Recovered amount (₦)", min_value=0.0,
                max_value=float(current_record.detected_amount),
                value=float(current_record.detected_amount) if next_status == "recovered" else 0.0,
            )
        notes = colC.text_input("Notes (optional)")

        if st.button("Apply status update"):
            from recovery_tracker import RecoveryStatus as RS
            try:
                tracker.update_status(selected, RS(next_status), recovered_amount=recovered_amt, notes=notes)
                st.success(f"{selected} moved to '{next_status}'.")
                st.rerun()
            except ValueError as e:
                st.error(str(e))

    st.divider()
    st.dataframe(tracker.to_dataframe(), width='stretch')
    st.download_button(
        "Download recovery ledger (JSON)",
        tracker.export_json(),
        file_name=f"vanguard_ra_recovery_{datetime.now():%Y%m%d}.json",
    )

with tab_matched:
    st.dataframe(result.matched, width='stretch')

st.divider()
st.caption(
    "Vanguard RA — Prototype dashboard. Pipeline: Open Banking API/CSV → "
    "Transaction Processing → Reconciliation Engine → Matched/Mismatch/Leak/Anomaly → Dashboard, Alerts & Reports."
)
