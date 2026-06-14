import sys
from pathlib import Path
import warnings

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.inference import analyze_single_listing

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CarScope AI",
    page_icon="none",  
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Modern UI Refined Style Mappings ──────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"]  { background:#0a0a0e; color:#e2e8f0; font-family:-apple-system,BlinkMacSystemFont,sans-serif; }
[data-testid="stSidebar"]           { background:#0f0f15; border-right:1px solid #1e1e2f; }
[data-testid="stSidebar"] *         { color:#94a3b8 !important; }
[data-testid="stSidebar"] .stRadio > label { color:#64748b !important; font-size:12px !important; text-transform:uppercase; letter-spacing:1px; }

h1, h2, h3, h4, h5 { color:#f8fafc; font-weight:700; letter-spacing:-0.02em; }

.stForm { background:#11111b; border:1px solid #1e1e2f; border-radius:16px; padding:24px; box-shadow: 0 4px 20px rgba(0,0,0,0.3); }
.stSelectbox > div > div, .stNumberInput > div > div > input,
.stTextInput > div > div > input {
    background:#161622 !important; color:#f8fafc !important;
    border:1px solid #27273a !important; border-radius:8px !important; height:42px !important;
}

.stButton>button {
    background: linear-gradient(135deg, #2563eb, #1d4ed8);
    color:#ffffff !important; border:none; border-radius:8px; font-weight:600;
    padding:10px 24px; transition: all 0.25s ease;
}
.stButton>button:hover { background: linear-gradient(135deg, #3b82f6, #2563eb); transform: translateY(-1px); }

.score-card { background: linear-gradient(145deg, #131322 0%, #17172e 100%); border: 1px solid #252542; border-radius: 20px; padding: 32px 24px; text-align: center; box-shadow: 0 8px 32px rgba(0,0,0,0.4); }
.score-number { font-size:96px; font-weight:900; line-height:1; letter-spacing:-4px; margin-bottom:4px; }
.score-label  { font-size:12px; text-transform:uppercase; letter-spacing:3px; color:#64748b; font-weight:600; }

.badge { display:inline-block; padding:6px 18px; border-radius:100px; font-weight:700; font-size:12px; letter-spacing:1px; text-transform:uppercase; }
.badge-Low       { background: rgba(16,185,129,0.1); color:#34d399; border:1px solid rgba(52,211,153,0.3); }
.badge-Medium    { background: rgba(245,158,11,0.1); color:#fbbf24; border:1px solid rgba(251,191,36,0.3); }
.badge-High      { background: rgba(249,115,22,0.1); color:#fb923c; border:1px solid rgba(251,146,60,0.3); }
.badge-Very-High { background: rgba(239,68,68,0.1); color:#f87171; border:1px solid rgba(248,113,113,0.3); }

.metric-tile { background:#11111b; border:1px solid #1e1e2f; border-radius:14px; padding:22px 16px; text-align:center; margin-bottom: 15px;}
.metric-val   { font-size:32px; font-weight:800; color:#f8fafc; letter-spacing:-1px; }
.metric-lbl   { font-size:12px; color:#64748b; text-transform:uppercase; letter-spacing:1.5px; margin-top:6px; font-weight:500; }

.sec-hdr { font-size:12px; text-transform:uppercase; letter-spacing:3px; color:#475569; margin:28px 0 14px; padding-bottom:8px; border-bottom:1px solid #1e1e2f; font-weight:600; }
.sidebar-logo { font-size:24px; font-weight:900; letter-spacing:-1px; color:#3b82f6; }
.sidebar-sub  { font-size:11px; color:#475569; letter-spacing:1.5px; margin-top:4px; font-weight:600; }

.step-box { background:#11111b; border-left:4px solid #3b82f6; padding:20px; border-radius:0 12px 12px 0; margin-bottom:20px; }
.step-title { font-size:18px; font-weight:700; color:#f8fafc; margin-bottom:10px; }
.step-text { font-size:14px; color:#cbd5e1; line-height:1.6; }
</style>
""", unsafe_allow_html=True)

# ── Data Loaders ──────────────────────────────────────────────────────────────
@st.cache_data
def load_raw_dataset():
    raw_path = ROOT_DIR / "data" / "Car Sell Dataset.csv"
    return pd.read_csv(raw_path) if raw_path.exists() else pd.DataFrame()

@st.cache_data
def load_engineered_dataset():
    eng_path = ROOT_DIR / "data" / "X_train_engineered.csv"
    return pd.read_csv(eng_path) if eng_path.exists() else pd.DataFrame()

@st.cache_data
def load_pca_components():
    pca_path = ROOT_DIR / "data" / "pca_results.csv"
    return pd.read_csv(pca_path) if pca_path.exists() else None

def fetch_risk_hex_color(score):
    if score < 35: return "#34d399"
    if score < 60: return "#fbbf24"
    if score < 80: return "#fb923c"
    return "#f87171"

# ── Sidebar Navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-logo">CarScope AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-sub">INDIA · USED CAR FRAUD DETECTOR</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    pages = ["Project Journey", "Model Comparison & PCA", "Analyze a Listing", "Market Overview"]
    page = st.radio("Navigation Mapping", pages, label_visibility="collapsed")
    
    st.markdown("---")
    st.markdown("<small style='color:#475569'>Ensemble Model Framework<br>Backend: XGBoost, Isolation Forest, LOF</small>", unsafe_allow_html=True)

# ── Page 1: Project Journey (Landing Page) ────────────────────────────────────
if page == "Project Journey":
    st.markdown("## Project Lifecycle & Architecture")
    st.markdown("<p style='color:#94a3b8;font-size:14px'>An end-to-end overview of the data engineering, exploratory analysis, and anomaly detection modeling utilized in CarScope AI.</p>", unsafe_allow_html=True)
    
    raw_df = load_raw_dataset()
    eng_df = load_engineered_dataset()
    
    st.markdown('<div class="sec-hdr">1. Exploratory Data Analysis (EDA)</div>', unsafe_allow_html=True)
    if not raw_df.empty:
        col1, col2 = st.columns(2)
        with col1:
            fig_hist = px.histogram(raw_df, x="Price", nbins=50, title="Target Distribution: Vehicle Prices", color_discrete_sequence=["#3b82f6"])
            fig_hist.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#cbd5e1"))
            st.plotly_chart(fig_hist, use_container_width=True)
        with col2:
            fig_scatter = px.scatter(raw_df, x="Kilometers", y="Price", color="Fuel Type", title="Depreciation: Price vs Kilometers", opacity=0.6)
            fig_scatter.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#cbd5e1"))
            st.plotly_chart(fig_scatter, use_container_width=True)
            
        fig_box = px.box(raw_df, x="Brand", y="Price", title="Market Valuation by Brand Identity", color_discrete_sequence=["#10b981"])
        fig_box.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#cbd5e1"))
        st.plotly_chart(fig_box, use_container_width=True)
    else:
        st.warning("Raw dataset not found. Please place 'Car Sell Dataset.csv' in the data directory.")

    st.markdown('<div class="sec-hdr">2. Feature Engineering Protocol</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="step-box">
        <div class="step-title">Transformations Applied</div>
        <div class="step-text">
        - <b>Temporal Features:</b> Extracted <code>Car_Age</code> from the production <code>Year</code> and calculated structural wear via <code>KM_Per_Year</code>.<br>
        - <b>Ordinal Encodings:</b> Mapped qualitative variables like <code>Owner</code>, <code>Accidental</code>, and <code>Transmission</code> to binary/ordinal matrices.<br>
        - <b>Target Encoding & Scaling:</b> High-cardinality features (Brand, Model) underwent smoothed Target Encoding. The entire matrix was subsequently mapped via StandardScaler for distance-based estimators.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if not eng_df.empty:
        st.dataframe(eng_df.head(5), use_container_width=True)

    st.markdown('<div class="sec-hdr">3. Anomaly Detection Architecture</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1: st.markdown('<div class="metric-tile"><div class="metric-val" style="font-size:20px;color:#3b82f6">Targeted XGBoost</div><div class="metric-lbl">Baseline Price Estimation</div></div>', unsafe_allow_html=True)
    with c2: st.markdown('<div class="metric-tile"><div class="metric-val" style="font-size:20px;color:#10b981">Isolation Forest</div><div class="metric-lbl">Global Cluster Deviance</div></div>', unsafe_allow_html=True)
    with c3: st.markdown('<div class="metric-tile"><div class="metric-val" style="font-size:20px;color:#f59e0b">Local Outlier Factor</div><div class="metric-lbl">Local Density Deviance</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="sec-hdr">4. Anomaly Resolution & Analysis</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="step-box">
        <div class="step-title">Consensus Voting Logic</div>
        <div class="step-text">
        The pipeline does not rely on a single algorithm. Anomalies are classified using a composite scoring mechanism combining Unsupervised Global distance (Isolation Forest), Unsupervised Local density (LOF), and Supervised Target Residuals (XGBoost Price Delta). Records flagged by multiple independent matrices are marked as "Consensus Fraud Deviance".
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Page 2: Model Comparison & PCA ────────────────────────────────────────────
elif page == "Model Comparison & PCA":
    st.markdown("## Multi-Model Dimensionality Analysis")
    st.markdown("<p style='color:#94a3b8;font-size:14px'>Interactive 3D Principal Component Analysis (PCA) showcasing how different algorithms partition normal parameters versus structural anomalies.</p>", unsafe_allow_html=True)
    
    pca_df = load_pca_components()
    
    if pca_df is not None:
        samp_pca = pca_df.sample(min(4000, len(pca_df)), random_state=42)
        
        # Calculate high-level summary metrics
        total_pts = len(pca_df)
        iso_count = pca_df["iso_anomaly"].sum()
        lof_count = pca_df["lof_anomaly"].sum()
        cons_count = pca_df["consensus_anomaly"].sum()
        
        col1, col2, col3 = st.columns(3)
        with col1: st.markdown(f'<div class="metric-tile"><div class="metric-val" style="color:#10b981">{total_pts - iso_count} / {iso_count}</div><div class="metric-lbl">Iso Forest (Normal / Anomaly)</div></div>', unsafe_allow_html=True)
        with col2: st.markdown(f'<div class="metric-tile"><div class="metric-val" style="color:#f59e0b">{total_pts - lof_count} / {lof_count}</div><div class="metric-lbl">LOF (Normal / Anomaly)</div></div>', unsafe_allow_html=True)
        with col3: st.markdown(f'<div class="metric-tile"><div class="metric-val" style="color:#ef4444">{total_pts - cons_count} / {cons_count}</div><div class="metric-lbl">Consensus (Normal / Anomaly)</div></div>', unsafe_allow_html=True)

        tab1, tab2, tab3 = st.tabs(["Isolation Forest Map", "Local Outlier Factor Map", "Consensus Voting Map"])
        
        def render_3d_scatter(df, color_col, title, colorscale):
            fig = px.scatter_3d(df, x="PC1", y="PC2", z="PC3", color=color_col, opacity=0.7, color_continuous_scale=colorscale, title=title)
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", scene={"bgcolor": "#0a0a0e", "xaxis": {"gridcolor": "#1e1e2f"}, "yaxis": {"gridcolor": "#1e1e2f"}, "zaxis": {"gridcolor": "#1e1e2f"}}, height=600)
            return fig

        with tab1:
            st.plotly_chart(render_3d_scatter(samp_pca, "iso_anomaly", "Isolation Forest: Global Outliers", ["#1f77b4", "#10b981"]), use_container_width=True)
        with tab2:
            st.plotly_chart(render_3d_scatter(samp_pca, "lof_anomaly", "Local Outlier Factor: Density Outliers", ["#1f77b4", "#f59e0b"]), use_container_width=True)
        with tab3:
            st.plotly_chart(render_3d_scatter(samp_pca, "consensus_anomaly", "Consensus Matrix: High Confidence Fraud", ["#1f77b4", "#ef4444"]), use_container_width=True)

        st.markdown('<div class="sec-hdr">Algorithmic Overlap Matrix</div>', unsafe_allow_html=True)
        overlap_data = pd.DataFrame({
            "Metric": ["Total Evaluated Records", "Flagged exclusively by Iso Forest", "Flagged exclusively by LOF", "Flagged by Both (Strict Consensus)"],
            "Count": [total_pts, 
                      len(pca_df[(pca_df["iso_anomaly"]==1) & (pca_df["lof_anomaly"]==0)]),
                      len(pca_df[(pca_df["iso_anomaly"]==0) & (pca_df["lof_anomaly"]==1)]),
                      len(pca_df[(pca_df["iso_anomaly"]==1) & (pca_df["lof_anomaly"]==1)])]
        })
        st.dataframe(overlap_data, use_container_width=True, hide_index=True)
        
    else:
        st.warning("PCA component matrix missing. Please execute the core pipeline to generate visualization artifacts.")

# ── Page 3: Analyze a Listing (Original Tool) ─────────────────────────────────
elif page == "Analyze a Listing":
    st.markdown("## Real-Time Listing Evaluation")
    st.markdown("<p style='color:#94a3b8;font-size:14px'>Input evaluation profile attributes to determine pricing abnormalities and transaction risks instantly.</p>", unsafe_allow_html=True)

    brands, models, states, variants = ["Maruti Suzuki", "Hyundai", "Honda", "Toyota", "Tata"], ["Swift", "City", "Innova", "Nexon"], ["Delhi", "Maharashtra", "Karnataka"], ["Base", "VXI", "ZXI"]

    with st.form("evaluation_form"):
        st.markdown('<div class="sec-hdr">Vehicle Parameters</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            brand = st.selectbox("Brand Identity", brands)
            model_name = st.selectbox("Model Classification", models)
            model_variant = st.selectbox("Trim Tier Variant", variants)
        with c2:
            year = st.selectbox("Production Year", list(range(2026, 2005, -1)))
            fuel_type = st.selectbox("Propulsion System Type", ["Petrol", "Diesel", "CNG"])
            car_type = st.selectbox("Body Architecture", ["Hatchback", "Sedan", "SUV"])
        with c3:
            transmission = st.selectbox("Gearbox Configuration", ["Manual", "Automatic"])
            owner = st.selectbox("Prior Ownership Count", ["1st", "2nd", "3rd+"])
            accidental = st.selectbox("Damage History", ["No", "Yes"])

        st.markdown('<div class="sec-hdr">Financial & Utilization Baselines</div>', unsafe_allow_html=True)
        p1, p2, p3 = st.columns(3)
        with p1: listed_price = st.number_input("Target Evaluation Price (INR)", min_value=30000, value=650000)
        with p2: kilometers = st.number_input("Accumulated Kilometers", min_value=100, value=42000)
        with p3: state = st.selectbox("Registration Jurisdiction", states)

        submitted = st.form_submit_button("Execute Integrity Analysis", use_container_width=True)

    if submitted:
        package = {"Brand": brand, "Model Name": model_name, "Model Variant": model_variant, "State": state, "Fuel Type": fuel_type, "Car Type": car_type, "Transmission": transmission, "Owner": owner, "Accidental": accidental, "Year": year, "Kilometers": kilometers, "Price": listed_price}
        metrics_report = analyze_single_listing(package)
        
        hex_color = fetch_risk_hex_color(metrics_report["fraud_score"])
        col_left, col_right = st.columns([1, 2], gap="large")
        
        with col_left:
            st.markdown(f'<div class="score-card"><div class="score-number" style="color:{hex_color}">{metrics_report["fraud_score"]:.0f}</div><div class="score-label">Anomaly Index Rating</div><br><span class="badge badge-{metrics_report["risk_level"].replace(" ", "-")}">{metrics_report["risk_level"]} Risk State</span><br><br><div style="font-size:11px;color:#475569;text-transform:uppercase;letter-spacing:2px">Identified Pattern</div><div style="font-size:14px;color:#cbd5e1;margin-top:6px;font-weight:600">{metrics_report["anomaly_type"]}</div></div>', unsafe_allow_html=True)

        with col_right:
            m1, m2, m3 = st.columns(3)
            with m1: st.markdown(f'<div class="metric-tile"><div class="metric-val">INR {metrics_report["listed_price"]/100000:.2f}L</div><div class="metric-lbl">Listed Value</div></div>', unsafe_allow_html=True)
            with m2: st.markdown(f'<div class="metric-tile"><div class="metric-val">INR {metrics_report["predicted_price"]/100000:.2f}L</div><div class="metric-lbl">Fair Market Value</div></div>', unsafe_allow_html=True)
            with m3: st.markdown(f'<div class="metric-tile"><div class="metric-val" style="color:{"#f87171" if metrics_report["price_deviation_pct"] > 0 else "#34d399"}">{metrics_report["price_deviation_pct"]:+.1f}%</div><div class="metric-lbl">Value Delta Deviation</div></div>', unsafe_allow_html=True)
            
            gauge_fig = go.Figure(go.Indicator(mode="gauge+number", value=metrics_report["fraud_score"], domain={"x": [0, 1], "y": [0, 1]}, number={"font": {"color": hex_color, "size": 48}}, gauge={"axis": {"range": [0, 100], "tickcolor": "#27273a"}, "bar": {"color": hex_color}, "bgcolor": "#11111b", "bordercolor": "#27273a"}))
            gauge_fig.update_layout(height=220, paper_bgcolor="rgba(0,0,0,0)", margin={"t": 20, "b": 0})
            st.plotly_chart(gauge_fig, use_container_width=True)
            
            st.markdown('<div class="sec-hdr">Automated Integrity Observations</div>', unsafe_allow_html=True)
            for insight in metrics_report["insights"]: st.markdown(f'<div style="background:#141421; border-left:4px solid #2563eb; padding:16px; margin:10px 0; font-size:14px; color:#cbd5e1;">{insight}</div>', unsafe_allow_html=True)

# ── Page 4: Market Overview (Original Metrics) ────────────────────────────────
elif page == "Market Overview":
    st.markdown("## Aggregate Market Health")
    pca_df = load_pca_components()
    
    k1, k2, k3 = st.columns(3)
    with k1: st.markdown('<div class="metric-tile"><div class="metric-val" style="color:#3b82f6">140,904</div><div class="metric-lbl">Total Monitored Units</div></div>', unsafe_allow_html=True)
    with k2: st.markdown('<div class="metric-tile"><div class="metric-val" style="color:#fb923c">4,227</div><div class="metric-lbl">Flagged Structural Anomalies</div></div>', unsafe_allow_html=True)
    with k3: st.markdown('<div class="metric-tile"><div class="metric-val" style="color:#34d399">3.00%</div><div class="metric-lbl">Baseline Anomaly Frequency</div></div>', unsafe_allow_html=True)