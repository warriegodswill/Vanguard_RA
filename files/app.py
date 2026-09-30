"""
Vanguard Revenue Assurance Platform.
------------------------------------
Streamlit Dashboard
Run with: streamlit run app.py
------------------------------------
Lets a merchant upload their business Bank,POS,Inventory CSVs files or load the
bundled simulated demo data, then it displaced the matched,mismatch,leak, anomaly
breakdowns, risk score, and a downloadable table reports.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
from column_mapper import map_columns_ui, validate_schema
from reconciliation_engine import ReconciliationEngine

st.set_page_config(
    page_title="Vanguard Revenue Assurance Platform Dashboard",
    page_icon="🛡️",
    layout="wide",
)
st.markdown(
    """
    <style>
    /1. /
    .stApp {
        background-color: #6C7C59 !important;
        color: black !important;
    }
    
    /* 2. Sidebar*/
    [data-testid="stSidebar"] {
        background-color: #004225 !important;
        border-right: 1px solid #334155 !important;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploadDropzone"] {
        background-color: #334155 !important; 
        border: 1px dashed #64748B !important;  
        border-radius: 8px !important;
    }
    /* Text and icons inside dropzone */
    [data-testid="stSidebar"] [data-testid="stFileUploadDropzone"] [data-testid="stWidgetLabel"] p,
    [data-testid="stSidebar"] [data-testid="stFileUploadDropzone"] span,
    [data-testid="stSidebar"] [data-testid="stFileUploadDropzone"] div {
        color: #CBD5E1 !important;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploadDropzone"] small {
        color: #94A3B8 !important;
    }
    /* upload button */
    [data-testid="stSidebar"] [data-testid="stFileUploadDropzone"] button {
        background-color: #475569 !important;
        color: #F8FAFC !important;
        border: 1px solid #64748B !important;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploadDropzone"] button svg {
        fill: #F8FAFC !important;
    }

    /* 3. Fonts */
    h3, h6, p, label {
        font-family: 'Segoe UI', Roboto, Helvetica, sans-serif !important;
    }
    h3 {
        color: #FFFFFF!important;
    }
    h6 {
        color: #1B4D3E!important;
    }
    [data-testid="stHeader"] {
        background-color: rgba(0,0,0,0) !important;
    }
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] label {
        color: #F1F5F9 !important;
    }

    /* 4. Metric Card Structure */
    div[data-testid="stMetric"] {
        background: #FFFFFF !important;
        border: 1px solid #334155 !important; 
        border-radius: 12px !important;
        padding: 1.5rem !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
        height:125px !important;
        margin-buttom:30px,
    }
    [data-testid="stMetricLabel"] p {
        font-size: 1.1rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.08em !important;
        color: #00401A !important;
        font-weight: 900 !important;
        visibility: visible !important;
    }
    div[data-testid="stMetricDelta"] {
        color: #EF4444 !important;
        background-color: transparent !important;
        text-align: right;
    }
    div[data-testid="stMetricDelta"] svg {
        fill: #EF4444 !important;
        text-align: right;
    }
    div[data-testid="stMetricValue"] {
        font-size: 2.2rem !important;
        font-weight: 500 !important;
        font-family: monospace !important; 
        color: #000000 !important;
        text-align: right;
        margin-right:20px;
    }

    /* 5. title Layout*/
    .box {
        border: 1px solid #FF6B00;
        border-radius: 12px;
        padding: 20px;
        background-color: #1B4D3E !important;
        box-shadow: 0 4px 20px rgba(255, 107, 0, 0.1);
        font-weight: bold !important;
        font-size: 2.2rem !important;
        text-color: #FFFFFF;
        margin-bottom: 15px;
        text-align:center;
    }
    
    /* 7. Clean Sidebar File Upload Dropzones */
    [data-testid="stSidebar"] [data-testid="stfile_uploader"] {
        background-color: #334155 !important; 
        border: 1px dashed #64748B !important;  
        border-radius: 8px !important;
    }

    /* Target text strings and icons inside dropzone */
    [data-testid="stSidebar"] [data-testid="stfile_uploaderDropzone"] [data-testid="stWidgetLabel"] p,
    [data-testid="stSidebar"] [data-testid="stfile_uploaderDropzone"] span,
    [data-testid="stSidebar"] [data-testid="stfile_uploaderDropzone"] div {
        color: #CBD5E1 !important;
    }
    
    [data-testid="stSidebar"] [data-testid="stfile_uploaderDropzone"] small {
        color: #94A3B8 !important;
    }
    
    /* Browser upload button adjustment */
    [data-testid="stSidebar"] [data-testid="stfile_uploaderDropzone"] button {
        background-color: #475569 !important;
        color: #F8FAFC !important;
        border: 1px solid #64748B !important;
    }
    
    [data-testid="stSidebar"] [data-testid="stfile_uploaderDropzone"] button svg {
        fill: #F8FAFC !important;
    }
    .sttab{
    color: #1B4D3E!important;
    }
    div[data-testid="stPlotlyChart"] {
        border: 1px solid #E0E0E0 !important;
        background-color: #FFFFFF !important;
        box-shadow:0px 40px 10px rgba(0, 0, 0, 0.03)!important;
        border-radius: 20px !important;
        padding: 20px !important;
        overflow: hidden !important;
    }
    /* 9. Clean White Download Button Styling */
    div.stDownloadButton > button {
        background-color: #FFFFFF !important;
        color: #000000 !important;
        border: 1px solid #000000 !important;
        border-radius: 8px !important;
        padding: 0.5rem 1rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }

    /* Subtle hover effect */
    div.stDownloadButton > button:hover {
        background-color: #F1F5F9 !important;
        border-color: #00401A !important;
        color: #00401A !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1) !important;
    }
    /* Style security text */
    .security-note {
        font-size: 0.99rem;
        color: #fffff !important;
        margin-top: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# ---------------------------------------------------------------------------
# Sidebar — data input
# ---------------------------------------------------------------------------
if "mapped_data" not in st.session_state:
    st.session_state.mapped_data = {}

if "mapping_complete" not in st.session_state:
    st.session_state.mapping_complete = False

st.sidebar.title("🛡️ Vanguard RA")
st.sidebar.caption("Protecting Every Transaction. Building Financial Trust.")

data_source = st.sidebar.radio(
    "Data source",
    ["Use simulated demo data", "Upload my own CSVs"],
)
pos_df = bank_df = inventory_df = None

if data_source == "Use simulated demo data":
    # Reset custom mapping state if user switches back to demo data
    st.session_state.mapping_complete = False
    st.session_state.mapped_data = {}
    
    try:
        pos_df = pd.read_csv("pos_records.csv")
        bank_df = pd.read_csv("bank_records.csv")
        inventory_df = pd.read_csv("inventory_records.csv")
        st.sidebar.success("Loaded simulated demo dataset.")
    except FileNotFoundError:
        st.sidebar.error("Demo CSVs not found. Run generate_sample_data.py first.")

else:
    # --- IF ALREADY MAPPED: Hide the forms & pull data straight from Session State ---
    if st.session_state.mapping_complete and st.session_state.mapped_data:
        st.sidebar.success("✅ Data successfully mapped and loaded!")
        
        # Load data directly from state
        pos_df = st.session_state.mapped_data["pos_df"]
        bank_df = st.session_state.mapped_data["bank_df"]
        inventory_df = st.session_state.mapped_data["inventory_df"]
        
        # Give user an option to reset if they want to upload different files
        if st.sidebar.button("🔄 Re-upload / Remap Data"):
            st.session_state.mapping_complete = False
            st.session_state.mapped_data = {}
            st.rerun()

    # --- IF NOT MAPPED: Show file uploaders and column mapping forms ---
    else:
        pos_file = st.sidebar.file_uploader("POS Records CSV", type="csv")
        bank_file = st.sidebar.file_uploader("Bank Records CSV", type="csv")
        inv_file = st.sidebar.file_uploader("Inventory Records CSV", type="csv")

        if pos_file and bank_file and inv_file:
            raw_pos_df = pd.read_csv(pos_file)
            raw_bank_df = pd.read_csv(bank_file)
            raw_inventory_df = pd.read_csv(inv_file)

            # Display the column mapper forms
            pos_df_mapped = map_columns_ui(raw_pos_df, "pos")
            bank_df_mapped = map_columns_ui(raw_bank_df, "bank")
            inventory_df_mapped = map_columns_ui(raw_inventory_df, "inventory")

            if pos_df_mapped is None or bank_df_mapped is None or inventory_df_mapped is None:
                st.info("Finish mapping all three files above to continue.")
                st.stop()

            all_errors = (
                validate_schema(pos_df_mapped, "pos")
                + validate_schema(bank_df_mapped, "bank")
                + validate_schema(inventory_df_mapped, "inventory")
            )
            
            if all_errors:
                st.error("Some issues need fixing before reconciliation can run:")
                for e in all_errors:
                    st.write(f"- {e}")
                st.stop()

            # Store finished DataFrames in Session State
            st.session_state.mapped_data = {
                "pos_df": pos_df_mapped,
                "bank_df": bank_df_mapped,
                "inventory_df": inventory_df_mapped,
            }
            st.session_state.mapping_complete = True
            st.rerun()

st.sidebar.divider()
st.sidebar.markdown(
        """
        <div class="security-note">
            🔒 <b>Bank-Grade Security:</b><br>
            Data processed in-memory. Files are encrypted & never stored or shared.
        </div>
        """, 
        unsafe_allow_html=True
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
st.markdown("<div class='box'>🛡️ Vanguard Revenue Assurance Platform</div>",unsafe_allow_html=True,)
st.markdown(
    "<h3>Automated Financial Reconciliation & Revenue Assurance Dashboard</h3>",
    unsafe_allow_html=True)

if pos_df is None or bank_df is None or inventory_df is None:
    st.markdown("<h6>⬅️ Load demo data or upload your Bank, POS, and Inventory CSVs from the sidebar to begin.</h6>",unsafe_allow_html=True)
    st.stop()

engine = ReconciliationEngine(pos_df, bank_df, inventory_df)
result = engine.run()
s = result.summary
total_transactions = s["n_matched"] + s["n_mismatches"] + s["n_leaks"] + s["n_anomalies"]
# --- KPI row ---
col1, col2, col3, col4 = st.columns(4)
with col1: st.metric(label="Leaked Revenue",value= f"₦{s['total_leaked_amount']:,.0f}",delta= f"{s['leak_percentage']}% of revenue")
col2.metric("Total POS Revenue", f"₦{s['total_pos_revenue']:,.0f}")
col3.metric("Matched Transactions", f"{s['n_matched']:,} / {total_transactions:,}")
col4.metric("Risk Score", f"{s['risk_score']} / 100", s["risk_band"])
st.markdown('')
# --- Alert banner ---
if s["risk_band"] == "High":
    st.error(f"⚠️ High risk detected — ₦{s['total_leaked_amount']:,.0f} in leakages found. Immediate review recommended.")
elif s["risk_band"] == "Medium":
    st.warning(f"⚠️ Medium risk — ₦{s['total_leaked_amount']:,.0f} in leakages detected across {s['n_leaks']} transactions.")
else:
    st.success("✅ Low risk — reconciliation looks healthy.")

# --- AI Insights Banner ---
if result.ai_summary:
    st.info(result.ai_summary)

# --- Breakdown chart ---
left, right = st.columns([1, 1])

with left:
    st.markdown("<h3 style='text-align:center;'>Reconciliation Breakdown</h3>",unsafe_allow_html=True,)
    breakdown = pd.DataFrame({
        "Category": ["Matched", "Mismatches", "Leaks", "Anomalies"],
        "Count": [s["n_matched"], s["n_mismatches"], s["n_leaks"], s["n_anomalies"]],
    })
    fig = px.pie(
        breakdown, names="Category", values="Count",
        color="Category",
        color_discrete_map={
            "Matched": "#16a34a",
            "Mismatches": "#f59e0b",
            "Leaks": "#dc2626",
            "Anomalies": "#000000",
        },
    )
    fig.update_traces(textinfo="value+percent")
    fig.update_layout(
                template='plotly_white',
                plot_bgcolor='rgba(255,255,255,1)',
                paper_bgcolor='rgba(255,255,255,1)',
                font=dict(color='#000000'),
                margin=dict(t=50, b=10, l=20, r=160),
                legend=dict(orientation="v",yanchor="middle", y=0.5,xanchor="left",x=1.02,font=dict(size=14,color='#000000'))
            )
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.markdown("<h3 style='text-align:center;'>Leakage Over Time</h3>",unsafe_allow_html=True,)
    if not result.leaks.empty and "timestamp" in result.leaks.columns:
        leak_ts = result.leaks.copy()
        leak_ts["timestamp"] = pd.to_datetime(leak_ts["timestamp"], errors="coerce")
        leak_ts["date"] = leak_ts["timestamp"].dt.date
        daily_leak = leak_ts.groupby("date")["leaked_amount"].sum().reset_index()
        fig2 = px.bar(daily_leak, x="date", y="leaked_amount", labels={"leaked_amount": "Leaked ₦"})
        fig2.update_traces(marker_color="#dc2626")
        fig2.update_layout(
                template='plotly_white',
                plot_bgcolor='rgba(255,255,255,1)',
                paper_bgcolor='rgba(255,255,255,1)',
                font=dict(color='#000000'),
                margin=dict(t=40, b=30, l=20, r=20),
                legend=dict(orientation="h", y=1.02,font=dict(size=14,color='#000000')),
                xaxis=dict(tickfont=dict(color='#222222',size=12),title_font=dict(color='#222222')),
                yaxis=dict(tickfont=dict(color='#222222',size=12),title_font=dict(color='#222222')),
                coloraxis_showscale=False
            )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.caption("No time-stamped leak data available.")

st.markdown('')
# --- Detail tabs ---
tab_leaks, tab_mismatch, tab_anomaly, tab_ml_anomalies, tab_matched = st.tabs(
    [
        f"🔴 Leaks ({s['n_leaks']})",
        f"🟠 Mismatches ({s['n_mismatches']})",
        f"🟣 Rule Anomalies ({s['n_anomalies']})",
        f"🤖 ML Outliers ({s.get('n_ml_anomalies', 0)})",
        f"🟢 Matched ({s['n_matched']})"
    ]
)
table_style= {"background-color": "#FFFFFF","color":"#000000"}
grid_line_style=[{"selector":"th,td","props":[("border","1px solid #000000 !important")]},{"selector":"table","props":[("border-collapse","collapse !important")]}]
with tab_leaks:
    st.dataframe(result.leaks.style.set_properties(**table_style).set_table_styles(grid_line_style), use_container_width=True)
    if not result.leaks.empty:
        st.download_button(
            "Download leaks report (CSV)",
            result.leaks.to_csv(index=False),
            file_name=f"vanguard_ra_leaks_{datetime.now():%Y%m%d}.csv",
        )

with tab_mismatch:
    st.dataframe(result.mismatches.style.set_properties(**table_style).set_table_styles(grid_line_style), use_container_width=True)

with tab_anomaly:
    st.dataframe(result.anomalies.style.set_properties(**table_style).set_table_styles(grid_line_style), use_container_width=True)

with tab_ml_anomalies:
    st.markdown("**Unsupervised Machine Learning Flags (Isolation Forest)**")
    st.caption("Identifies statistical anomalies across transaction volumes and ratios beyond simple threshold rules.")
    if hasattr(result, "ml_anomalies") and not result.ml_anomalies.empty:
        st.dataframe(result.ml_anomalies.style.set_properties(**table_style).set_table_styles(grid_line_style), use_container_width=True)
    else:
        st.info("No statistical ML anomalies detected in this dataset.")

with tab_matched:
    st.dataframe(result.matched.style.set_properties(**table_style).set_table_styles(grid_line_style), use_container_width=True)

st.divider()
st.caption(
    "Vanguard RA — Prototype dashboard. Pipeline: CSV's → "
    "Transaction Processing → Reconciliation Engine → Matched/Mismatch/Leak/Anomaly → Dashboard, Alerts & Reports."
)
