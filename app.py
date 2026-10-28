"""
WaterWise: Large-Scale Water Consumption Pattern Analysis and Anomaly Detection Using PySpark.
Streamlit Interactive Dashboard.
"""

import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Configure system path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.spark_session import get_spark_session
from pyspark.sql import functions as F
from src.data_loader import locate_dataset_file, load_raw_data, compute_missing_value_stats
from src.preprocessing import preprocess_and_clean
from src.analytics import (
    apply_filters,
    compute_kpis,
    get_hourly_trends,
    get_daily_trends,
    get_day_of_week_trends,
    get_monthly_trends,
    get_fixture_distribution,
    get_water_type_distribution,
    get_apartment_comparison,
    identify_peak_periods,
    export_processed_outputs
)
from src.anomaly_detection import detect_fixture_anomalies, EXPLANATION_DISCLAIMER
from src.forecasting import train_daily_forecast_model


# ==========================================
# PAGE CONFIGURATION & THEME
# ==========================================
st.set_page_config(
    page_title="WaterWise — PySpark Analytics",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main background & high-contrast base text color */
    .stApp {
        background-color: #F8FAFC !important;
        color: #0F172A !important;
    }

    /* Global Headings */
    h1, h2, h3, h4, h5, h6,
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4,
    [data-testid="stHeading"] {
        color: #0B2545 !important;
        font-weight: 700 !important;
    }

    /* Paragraphs, descriptions & labels */
    p, span, label, .stMarkdown p, .stCaption {
        color: #1E293B !important;
    }

    /* Tab navigation text */
    button[data-baseweb="tab"] {
        color: #334155 !important;
        font-weight: 600 !important;
        background-color: transparent !important;
    }
    button[data-baseweb="tab"] p,
    button[data-baseweb="tab"] div,
    button[data-baseweb="tab"] span {
        color: #334155 !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #028090 !important;
        border-bottom-color: #028090 !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] p,
    button[data-baseweb="tab"][aria-selected="true"] div,
    button[data-baseweb="tab"][aria-selected="true"] span {
        color: #028090 !important;
        font-weight: 700 !important;
    }

    /* Streamlit Alert Boxes: info, success, warning */
    .stAlert, [data-testid="stAlert"] {
        border-radius: 10px !important;
    }
    .stAlert p, .stAlert span, .stAlert div,
    [data-testid="stAlert"] p, [data-testid="stAlert"] span, [data-testid="stAlert"] div {
        color: #0F172A !important;
        font-size: 0.92rem !important;
    }
    .stAlert strong, [data-testid="stAlert"] strong {
        color: #0B2545 !important;
    }

    /* Sidebar Dark Hydrodynamic Styling */
    [data-testid="stSidebar"] {
        background-color: #0B1E36 !important;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #FFFFFF !important;
    }
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] label {
        color: #E2E8F0 !important;
    }
    [data-testid="stSidebar"] .stCaption {
        color: #94A3B8 !important;
    }
    [data-testid="stSidebar"] hr {
        border-color: #1E3A5F !important;
    }

    /* Metric Card Styling - MUST REMAIN CRISP WHITE TEXT */
    .metric-card {
        background: linear-gradient(135deg, #0B2545 0%, #134074 100%);
        border-radius: 12px;
        padding: 20px 24px;
        box-shadow: 0 4px 12px rgba(11, 37, 69, 0.12);
        margin-bottom: 12px;
        border-left: 5px solid #00A896;
    }
    .metric-card.teal {
        background: linear-gradient(135deg, #028090 0%, #05668D 100%);
        border-left: 5px solid #E2E8F0;
    }
    .metric-card.coral {
        background: linear-gradient(135deg, #D64045 0%, #B82A2E 100%);
        border-left: 5px solid #FFD166;
    }
    .metric-card, .metric-card * {
        color: #FFFFFF !important;
    }
    .metric-title {
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        opacity: 0.90;
        margin-bottom: 6px;
        color: #E2E8F0 !important;
    }
    .metric-value {
        font-size: 1.85rem;
        font-weight: 700;
        line-height: 1.1;
        margin-bottom: 4px;
        color: #FFFFFF !important;
    }
    .metric-subtitle {
        font-size: 0.78rem;
        opacity: 0.85;
        color: #E2E8F0 !important;
    }

    /* Section Cards */
    .section-card {
        background: #FFFFFF !important;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        border: 1px solid #E2E8F0;
    }
    .section-card * {
        color: #1E293B !important;
    }

    /* Viva Box */
    .viva-box {
        background: #F0F7FF !important;
        border-left: 4px solid #134074 !important;
        padding: 16px 20px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 16px;
    }
    .viva-box * {
        color: #0F172A !important;
    }

    /* Disclaimer Box */
    .disclaimer-card {
        background: #FEF3C7 !important;
        border-left: 4px solid #F59E0B !important;
        padding: 14px 18px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 16px;
        color: #92400E !important;
        font-size: 0.88rem;
    }
    .disclaimer-card * {
        color: #92400E !important;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def apply_chart_theme(fig):
    """Enforce crisp, high-contrast light theme with white canvas and dark labels."""
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#F8FAFC",
        font=dict(family="Inter, sans-serif", color="#0B2545", size=12),
        title_font=dict(family="Inter, sans-serif", color="#0B2545", size=15),
        legend=dict(font=dict(color="#0B2545"))
    )
    if hasattr(fig, "update_xaxes"):
        fig.update_xaxes(showgrid=True, gridcolor="#E2E8F0", title_font=dict(color="#0B2545"), tickfont=dict(color="#1E293B"))
        fig.update_yaxes(showgrid=True, gridcolor="#E2E8F0", title_font=dict(color="#0B2545"), tickfont=dict(color="#1E293B"))
    return fig


# ==========================================
# CACHED DATA PIPELINE FUNCTIONS
# ==========================================
@st.cache_resource(show_spinner="Initializing Local Apache PySpark Session...")
def init_spark():
    return get_spark_session(app_name="WaterWise-Dashboard", memory="4g")


@st.cache_resource(show_spinner="Running PySpark Extraction & Cleaning Pipeline...")
def process_data(data_dir: str, sample_size: str):
    spark = init_spark()
    max_rows = None
    if sample_size == "Demo Sample (50,000 records)":
        max_rows = 50000
    elif sample_size == "Medium Sample (150,000 records)":
        max_rows = 150000

    raw_df, raw_meta = load_raw_data(spark, data_dir=data_dir, max_rows=max_rows)
    missing_stats = compute_missing_value_stats(raw_df)
    cleaned_df, clean_summary = preprocess_and_clean(raw_df, output_dir="outputs")
    cleaned_df.cache()
    # Force evaluation to materialize cache
    _ = cleaned_df.count()

    return cleaned_df, raw_meta, missing_stats, clean_summary


# ==========================================
# SIDEBAR CONTROLS & DATA LOADING
# ==========================================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/water.png", width=64)
    st.title("WaterWise")
    st.caption("PySpark Water Analytics & Anomaly Detection")
    st.markdown("---")

    st.subheader("⚙️ Data Configuration")
    data_dir_input = st.text_input("Data Directory", value="data", help="Directory where dataset CSV is stored")
    
    sample_option = st.selectbox(
        "Dataset Volume",
        options=[
            "Demo Sample (50,000 records)",
            "Medium Sample (150,000 records)",
            "Full Dataset (All ~652k records)"
        ],
        index=0,
        help="Use Demo Sample for fast college presentations or Full Dataset for exhaustive analysis"
    )

    reload_btn = st.button("🔄 Reload & Re-Execute Pipeline", use_container_width=True)
    if reload_btn:
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

    st.markdown("---")
    st.subheader("🔎 Filters & Parameters")
    iqr_multiplier = st.slider("Anomaly Sensitivity (IQR Multiplier)", min_value=1.0, max_value=3.0, value=1.5, step=0.25, help="Standard IQR threshold: Q3 + (multiplier * IQR)")


# ==========================================
# DATA LOADING OR EXPLICIT ERROR HANDLING
# ==========================================
spark = init_spark()

try:
    csv_file = locate_dataset_file(data_dir_input)
except FileNotFoundError as fnf_err:
    st.error("### ⚠️ Dataset Not Found")
    st.error(str(fnf_err))
    st.info("""
    **Where to download:**
    - Download the HSB Living Lab Dataset from Zenodo: [https://zenodo.org/records/18971107](https://zenodo.org/records/18971107)
    - Place `HSB_Living_Lab_Water_Consumption_Anonymized.csv` directly into the `data/` folder.
    """)
    st.stop()
except Exception as e:
    st.error(f"Error checking dataset: {e}")
    st.stop()

# Run or retrieve cached PySpark pipeline
with st.spinner("Processing household water consumption dataset with Apache PySpark..."):
    cleaned_df, raw_meta, missing_stats, clean_summary = process_data(data_dir_input, sample_option)

# Populate interactive sidebar filters from cleaned PySpark DataFrame
available_fixtures = [row["fixture"] for row in cleaned_df.select("fixture").distinct().collect() if row["fixture"] is not None]
available_water_types = [row["water_type"] for row in cleaned_df.select("water_type").distinct().collect() if row["water_type"] is not None]
available_apartments = [row["apartment"] for row in cleaned_df.select("apartment").distinct().collect() if row["apartment"] is not None]
min_max_dates = cleaned_df.select(F.min("date").alias("min_d"), F.max("date").alias("max_d")).collect()[0]

min_d_val = min_max_dates["min_d"] if min_max_dates["min_d"] is not None else datetime.now().date()
max_d_val = min_max_dates["max_d"] if min_max_dates["max_d"] is not None else datetime.now().date()
if min_d_val > max_d_val:
    min_d_val, max_d_val = max_d_val, min_d_val

with st.sidebar:
    selected_fixtures = st.multiselect("Filter Fixtures", options=["All"] + sorted(available_fixtures), default=["All"])
    selected_water_types = st.multiselect("Filter Water Type", options=["All"] + sorted(available_water_types), default=["All"])
    selected_apartments = st.multiselect("Filter Apartment / Unit", options=["All"] + sorted(available_apartments), default=["All"])
    
    date_range = st.date_input(
        "Date Range",
        value=(min_d_val, max_d_val),
        min_value=min_d_val,
        max_value=max_d_val
    )

# Apply filters on PySpark DataFrame
start_str, end_str = None, None
if isinstance(date_range, (tuple, list)):
    if len(date_range) == 2:
        d0, d1 = sorted(date_range)
        start_str = d0.strftime("%Y-%m-%d")
        end_str = d1.strftime("%Y-%m-%d")
    elif len(date_range) == 1:
        start_str = date_range[0].strftime("%Y-%m-%d")
        end_str = date_range[0].strftime("%Y-%m-%d")
elif hasattr(date_range, "strftime"):
    start_str = date_range.strftime("%Y-%m-%d")
    end_str = date_range.strftime("%Y-%m-%d")

active_df = apply_filters(
    cleaned_df,
    start_date=start_str,
    end_date=end_str,
    fixtures=None if (not selected_fixtures or "All" in selected_fixtures) else selected_fixtures,
    water_types=None if (not selected_water_types or "All" in selected_water_types) else selected_water_types,
    apartments=None if (not selected_apartments or "All" in selected_apartments) else selected_apartments
)


# ==========================================
# RUN CORE ANALYTICS & ANOMALY DETECTION
# ==========================================
with st.spinner("Aggregating metrics in PySpark..."):
    kpis = compute_kpis(spark, active_df)
    hourly_trends = get_hourly_trends(spark, active_df)
    daily_trends = get_daily_trends(spark, active_df)
    dow_trends = get_day_of_week_trends(spark, active_df)
    fixture_dist = get_fixture_distribution(spark, active_df)
    water_type_dist = get_water_type_distribution(spark, active_df)
    apt_comp = get_apartment_comparison(spark, active_df, top_n=10)
    peaks = identify_peak_periods(spark, active_df)

    anomalies_spark_df, thresholds_pdf, anomaly_meta = detect_fixture_anomalies(
        spark, active_df, iqr_multiplier=iqr_multiplier
    )
    # Collect top anomalies for inspection table
    anomalies_top_pdf = anomalies_spark_df.limit(500).toPandas()


# ==========================================
# DASHBOARD HEADER
# ==========================================
st.markdown("## 💧 WaterWise: Water Consumption Analysis & Anomaly Detection")
st.markdown(
    "Large-Scale Big Data Analytics on real telemetry from the **HSB Living Lab** (Gothenburg, Sweden) "
    "powered by **Apache PySpark**, **Spark SQL**, and **Plotly**."
)
st.markdown("---")


# ==========================================
# TOP KPI ROW
# ==========================================
kpi_c1, kpi_c2, kpi_c3, kpi_c4, kpi_c5 = st.columns(5)

with kpi_c1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Cleaned Events</div>
        <div class="metric-value">{kpis['total_events']:,}</div>
        <div class="metric-subtitle">Valid sensor readings</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_c2:
    st.markdown(f"""
    <div class="metric-card teal">
        <div class="metric-title">Total Consumption</div>
        <div class="metric-value">{kpis['total_volume_liters'] / 1000.0:,.1f} m³</div>
        <div class="metric-subtitle">{kpis['total_volume_liters']:,} Liters</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_c3:
    st.markdown(f"""
    <div class="metric-card teal">
        <div class="metric-title">Daily Average</div>
        <div class="metric-value">{kpis['daily_avg_liters']:,.1f} L</div>
        <div class="metric-subtitle">Across {kpis['total_days']} active days</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_c4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Active Meters</div>
        <div class="metric-value">{kpis['active_sensors']}</div>
        <div class="metric-subtitle">{kpis['active_apartments']} Residential Units</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_c5:
    st.markdown(f"""
    <div class="metric-card coral">
        <div class="metric-title">Flagged Anomalies</div>
        <div class="metric-value">{anomaly_meta.get('total_anomalies', 0):,}</div>
        <div class="metric-subtitle">{anomaly_meta.get('overall_anomaly_pct', 0.0)}% of valid events ({iqr_multiplier}x IQR)</div>
    </div>
    """, unsafe_allow_html=True)


# ==========================================
# DASHBOARD TABS
# ==========================================
tab_overview, tab_trends, tab_fixtures, tab_anomalies, tab_quality, tab_forecast, tab_viva = st.tabs([
    "📊 Executive Summary",
    "📈 Consumption Trends",
    "🚿 Fixtures & Categories",
    "🚨 Anomaly Detection (IQR)",
    "🧹 Data Quality & Pipeline",
    "🔮 Forecasting (Enhancement)",
    "🎓 College Viva & Architecture"
])


# ----------------------------------------------------
# TAB 1: EXECUTIVE SUMMARY & KEY INSIGHTS
# ----------------------------------------------------
with tab_overview:
    st.subheader("Executive Insights & Behavioral Patterns")

    col_ins1, col_ins2, col_ins3 = st.columns(3)
    with col_ins1:
        peak_hour_str = f"{peaks['peak_hour']}:00 UTC" if peaks.get('peak_hour') is not None else "N/A"
        st.info(f"""
        **🕒 Diurnal Peak Usage**
        - Peak Consumption Hour: **{peak_hour_str}**
        - Hourly Peak Volume: **{peaks.get('peak_hour_volume_liters', 0.0):,.1f} Liters**
        - Primary Driver: Morning & midday hygiene routines and kitchen meal preparation.
        """)
    with col_ins2:
        st.success(f"""
        **📅 Weekly Rhythm**
        - Peak Day of Week: **{peaks.get('peak_day_of_week') or 'N/A'}**
        - Daily Peak Total: **{peaks.get('peak_day_volume_liters', 0.0):,.1f} Liters**
        - Pattern: Weekend water consumption spikes due to higher home occupancy and laundry cycles.
        """)
    with col_ins3:
        st.warning(f"""
        **🚰 Dominant End-Use Fixture**
        - Highest Volume: **{peaks.get('top_fixture') or 'N/A'}**
        - Share of Total Water: **{peaks.get('top_fixture_pct', 0.0)}%**
        - Secondary Driver: Kitchen tap and toilet flushes form the majority of frequency events.
        """)

    st.markdown("### 24-Hour Diurnal Profile vs. Daily Time Series")
    c_chart1, c_chart2 = st.columns([1, 1])

    with c_chart1:
        if not hourly_trends.empty:
            fig_h = px.area(
                hourly_trends,
                x="hour",
                y="total_liters",
                title="Diurnal Hourly Water Demand (Liters)",
                labels={"hour": "Hour of Day (UTC)", "total_liters": "Total Liters"},
                color_discrete_sequence=["#028090"]
            )
            fig_h.update_traces(line=dict(width=3))
            fig_h = apply_chart_theme(fig_h)
            fig_h.update_layout(margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_h, use_container_width=True)
        else:
            st.info("No hourly trend data available under current filters.")

    with c_chart2:
        if not daily_trends.empty:
            fig_d = go.Figure()
            fig_d.add_trace(go.Scatter(
                x=daily_trends["date"],
                y=daily_trends["total_liters"],
                mode="lines",
                name="Daily Total (L)",
                line=dict(color="#B0C4DE", width=1)
            ))
            fig_d.add_trace(go.Scatter(
                x=daily_trends["date"],
                y=daily_trends["rolling_7d_avg"],
                mode="lines",
                name="7-Day Moving Avg",
                line=dict(color="#0B2545", width=2.5)
            ))
            fig_d = apply_chart_theme(fig_d)
            fig_d.update_layout(
                title="Daily Water Consumption Over Time (Liters)",
                xaxis_title="Date",
                yaxis_title="Liters",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_d, use_container_width=True)
        else:
            st.info("No daily time-series data available under current filters.")


# ----------------------------------------------------
# TAB 2: CONSUMPTION TRENDS
# ----------------------------------------------------
with tab_trends:
    st.subheader("Temporal Consumption Dynamics")
    
    t_col1, t_col2 = st.columns(2)
    with t_col1:
        if not dow_trends.empty:
            fig_dow = px.bar(
                dow_trends,
                x="day_name",
                y="total_liters",
                title="Total Water Demand by Day of Week",
                labels={"day_name": "Day of Week", "total_liters": "Total Liters"},
                color="total_liters",
                color_continuous_scale="Blues"
            )
            fig_dow = apply_chart_theme(fig_dow)
            fig_dow.update_layout(margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_dow, use_container_width=True)
        else:
            st.info("No day-of-week data available.")

    with t_col2:
        if not hourly_trends.empty:
            fig_havg = px.line(
                hourly_trends,
                x="hour",
                y="avg_liters_per_event",
                title="Average Intensity per Water Use Event by Hour",
                labels={"hour": "Hour of Day (UTC)", "avg_liters_per_event": "Avg Liters / Event"},
                color_discrete_sequence=["#D64045"],
                markers=True
            )
            fig_havg = apply_chart_theme(fig_havg)
            fig_havg.update_layout(margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_havg, use_container_width=True)
        else:
            st.info("No hourly intensity data available.")

    st.markdown("### Monthly Aggregate View")
    monthly_pdf = get_monthly_trends(spark, active_df)
    if not monthly_pdf.empty:
        monthly_pdf["year_month"] = monthly_pdf["year"].astype(str) + "-" + monthly_pdf["month"].astype(str).str.zfill(2)
        fig_m = px.bar(
            monthly_pdf,
            x="year_month",
            y="monthly_m3",
            title="Monthly Water Consumption (m³)",
            labels={"year_month": "Month", "monthly_m3": "Volume (m³)"},
            color_discrete_sequence=["#05668D"]
        )
        fig_m = apply_chart_theme(fig_m)
        st.plotly_chart(fig_m, use_container_width=True)
    else:
        st.info("No monthly data available under current filters.")


# ----------------------------------------------------
# TAB 3: FIXTURES & CATEGORIES
# ----------------------------------------------------
with tab_fixtures:
    st.subheader("Fixture-Level and Category Breakdown")

    f_col1, f_col2 = st.columns([1, 1])
    with f_col1:
        if not fixture_dist.empty:
            fig_pie = px.pie(
                fixture_dist,
                names="fixture",
                values="total_liters",
                title="Water Consumption Share by Fixture Type",
                hole=0.45,
                color_discrete_sequence=px.colors.qualitative.Prism
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            fig_pie = apply_chart_theme(fig_pie)
            fig_pie.update_layout(margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No fixture distribution data available.")

    with f_col2:
        if not water_type_dist.empty:
            fig_wt = px.pie(
                water_type_dist,
                names="water_type",
                values="total_liters",
                title="Thermal Breakdown: Cold vs. Hot Water",
                hole=0.45,
                color_discrete_map={
                    "cold water consumption": "#05668D",
                    "hot water consumption": "#D64045",
                    "Standard": "#00A896"
                }
            )
            fig_wt.update_traces(textposition='inside', textinfo='percent+label')
            fig_wt = apply_chart_theme(fig_wt)
            fig_wt.update_layout(margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_wt, use_container_width=True)
        else:
            st.info("No thermal water type data available.")

    st.markdown("### Top Consuming Apartments / Residential Units")
    if not apt_comp.empty:
        fig_apt = px.bar(
            apt_comp,
            x="apartment",
            y="total_liters",
            color="avg_event_liters",
            title="Water Consumption by Apartment Unit (Top 10)",
            labels={"apartment": "Apartment Identifier", "total_liters": "Total Liters", "avg_event_liters": "Avg Liters / Event"},
            color_continuous_scale="Viridis"
        )
        fig_apt = apply_chart_theme(fig_apt)
        st.plotly_chart(fig_apt, use_container_width=True)
    else:
        st.info("No apartment consumption data available.")


# ----------------------------------------------------
# TAB 4: STATISTICAL ANOMALY DETECTION (IQR)
# ----------------------------------------------------
with tab_anomalies:
    st.subheader("Interquartile Range (IQR) Statistical Anomaly Detection")
    st.markdown(f'<div class="disclaimer-card">{EXPLANATION_DISCLAIMER}</div>', unsafe_allow_html=True)

    st.markdown("#### Fixture-Stratified Dynamic Thresholds")
    st.caption(
        "Each fixture type has unique hydrodynamic characteristics (e.g. standard shower vs toilet flush). "
        "Thresholds are computed separately per fixture using PySpark percentiles to avoid unfair single-threshold bias."
    )

    if not thresholds_pdf.empty:
        display_th = thresholds_pdf[[
            "fixture", "group_count", "q1_liters", "median_liters", "q3_liters", "iqr", "threshold_liters", "anomaly_count", "anomaly_pct"
        ]].rename(columns={
            "fixture": "Fixture",
            "group_count": "Total Readings",
            "q1_liters": "Q1 (25%) L",
            "median_liters": "Median L",
            "q3_liters": "Q3 (75%) L",
            "iqr": "IQR L",
            "threshold_liters": f"Upper Cutoff ({iqr_multiplier}x IQR) L",
            "anomaly_count": "Flagged Anomalies",
            "anomaly_pct": "Anomaly %"
        })
        st.dataframe(display_th, use_container_width=True)
    else:
        st.info("No fixture readings available for threshold calculation.")

    st.markdown("#### Flagged Anomalous Records Inspection")
    if not anomalies_top_pdf.empty:
        fig_anom = px.scatter(
            anomalies_top_pdf,
            x="timestamp",
            y="volume_liters",
            color="fixture",
            size="excess_liters",
            hover_data=["apartment", "threshold_liters", "anomaly_ratio"],
            title=f"Flagged Outlier Readings Over Time (Sensitivity: {iqr_multiplier}x IQR)",
            labels={"timestamp": "Timestamp (UTC)", "volume_liters": "Recorded Volume (Liters)"}
        )
        fig_anom = apply_chart_theme(fig_anom)
        st.plotly_chart(fig_anom, use_container_width=True)

        st.dataframe(
            anomalies_top_pdf[[
                "timestamp", "apartment", "fixture", "water_type", "volume_liters", "threshold_liters", "excess_liters", "anomaly_ratio"
            ]].rename(columns={
                "timestamp": "Timestamp (UTC)",
                "apartment": "Apartment",
                "fixture": "Fixture",
                "water_type": "Water Type",
                "volume_liters": "Observed (L)",
                "threshold_liters": "Cutoff (L)",
                "excess_liters": "Excess (L)",
                "anomaly_ratio": "Ratio (x Cutoff)"
            }),
            use_container_width=True
        )

        csv_download = anomalies_top_pdf.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Flagged Anomalies CSV",
            data=csv_download,
            file_name="waterwise_statistical_anomalies.csv",
            mime="text/csv"
        )
    else:
        st.success("No readings exceeded the configured IQR threshold under current filters.")


# ----------------------------------------------------
# TAB 5: DATA QUALITY & PIPELINE SUMMARY
# ----------------------------------------------------
with tab_quality:
    st.subheader("Data Cleaning, Validation & Schema Traceability")

    dq_col1, dq_col2 = st.columns(2)
    with dq_col1:
        st.markdown("#### Pipeline Data Retention")
        retention_df = pd.DataFrame([
            {"Stage": "Raw Ingested Records", "Count": clean_summary["raw_record_count"]},
            {"Stage": "Invalid Timestamps Dropped", "Count": clean_summary["invalid_timestamps_dropped"]},
            {"Stage": "Zero or Negative Values Dropped", "Count": clean_summary["zero_negative_dropped"]},
            {"Stage": "Physical Telemetry Outliers (>0.3 m³)", "Count": clean_summary["physical_outliers_dropped"]},
            {"Stage": "Duplicate Composite Keys (time + sensor)", "Count": clean_summary["duplicates_dropped"]},
            {"Stage": "Cleaned Records Retained", "Count": clean_summary["cleaned_record_count"]},
        ])
        st.dataframe(retention_df, use_container_width=True)
        st.metric("Overall Retention Rate", f"{clean_summary['retention_percentage']}%")

    with dq_col2:
        st.markdown("#### Raw Column Missing Values (Null / NaN / Blank)")
        missing_df = pd.DataFrame([
            {"Column": k, "Missing Count": v} for k, v in missing_stats.items()
        ])
        st.dataframe(missing_df, use_container_width=True)

    st.markdown("#### Real CSV vs. PySpark Application Schema Mapping")
    schema_map_df = pd.DataFrame([
        {"Raw CSV Column": "timestamp", "Raw Type": "String (UTC)", "Spark Target Type": "TimestampType", "Description": "10-minute rounded UTC telemetry timestamp"},
        {"Raw CSV Column": "date", "Raw Type": "String (YYYY-MM-DD)", "Spark Target Type": "DateType", "Description": "Extracted calendar date"},
        {"Raw CSV Column": "hour", "Raw Type": "Integer (0-23)", "Spark Target Type": "IntegerType", "Description": "Extracted diurnal hour"},
        {"Raw CSV Column": "day_of_week", "Raw Type": "Integer (0-6)", "Spark Target Type": "IntegerType", "Description": "Day of week (1=Sun, 2=Mon...)"},
        {"Raw CSV Column": "sensor_id", "Raw Type": "String (S0001)", "Spark Target Type": "StringType", "Description": "Anonymized 1-to-1 pulse meter ID"},
        {"Raw CSV Column": "apartment", "Raw Type": "String (APT_001)", "Spark Target Type": "StringType", "Description": "Anonymized unit (Imputed 'Shared Facility' if Null)"},
        {"Raw CSV Column": "type", "Raw Type": "String", "Spark Target Type": "StringType", "Description": "Cold water or Hot water consumption"},
        {"Raw CSV Column": "attached_to", "Raw Type": "String", "Spark Target Type": "StringType", "Description": "End-use fixture (Shower, Toilet, Tap, etc.)"},
        {"Raw CSV Column": "value", "Raw Type": "Float (m³)", "Spark Target Type": "DoubleType", "Description": "Volume in m³; converted to volume_liters (x1000)"},
        {"Raw CSV Column": "aggregated_value", "Raw Type": "Float (m³)", "Spark Target Type": "DoubleType", "Description": "Cumulative volume meter index"}
    ])
    st.dataframe(schema_map_df, use_container_width=True)

    st.markdown("#### Export Pipeline Outputs")
    if st.button("💾 Export Parquet Aggregates & Audit Reports", use_container_width=True):
        with st.spinner("Persisting analytical Parquet and summary reports..."):
            artifacts = export_processed_outputs(
                spark,
                active_df,
                output_dir="outputs",
                summary_meta=clean_summary,
                anomalies_df=anomalies_spark_df
            )
            st.success(f"Successfully generated {len(artifacts)} analytical artifacts in `outputs/`!")


# ----------------------------------------------------
# TAB 6: FORECASTING (OPTIONAL ENHANCEMENT)
# ----------------------------------------------------
with tab_forecast:
    st.subheader("Daily Water Consumption Forecasting (Optional Enhancement)")
    st.caption(
        "Demonstrates out-of-sample time-series forecasting. "
        "Trained using a strictly chronological train/test split to eliminate future data leakage."
    )

    fc_days = st.slider("Forecast Horizon (Days)", min_value=7, max_value=30, value=14, step=7)
    
    with st.spinner("Training chronological time-series regressor on daily series..."):
        test_eval_df, forecast_df, fc_metrics = train_daily_forecast_model(
            daily_trends, forecast_horizon_days=fc_days, test_split_ratio=0.20
        )

    if "error" in fc_metrics:
        st.warning(f"Forecasting note: {fc_metrics['error']}")
    else:
        st.markdown(f'<div class="disclaimer-card">ℹ️ {fc_metrics["disclaimer"]}</div>', unsafe_allow_html=True)

        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("Model Architecture", "Ridge + Autoregressive Lags")
        m_col2.metric("Test MAE", f"{fc_metrics['test_mae_liters']:,} L")
        m_col3.metric("Test RMSE", f"{fc_metrics['test_rmse_liters']:,} L")
        m_col4.metric("Test R² Score", f"{fc_metrics['test_r2_score']}")

        st.markdown("#### Out-of-Sample Predictions vs. Actual Observations")
        fig_eval = go.Figure()
        fig_eval.add_trace(go.Scatter(
            x=test_eval_df["date"],
            y=test_eval_df["actual_liters"],
            mode="lines+markers",
            name="Actual Water Demand (L)",
            line=dict(color="#0B2545", width=2)
        ))
        fig_eval.add_trace(go.Scatter(
            x=test_eval_df["date"],
            y=test_eval_df["predicted_liters"],
            mode="lines+markers",
            name="Model Prediction (L)",
            line=dict(color="#D64045", dash="dash", width=2)
        ))
        fig_eval = apply_chart_theme(fig_eval)
        fig_eval.update_layout(
            title="Chronological Test Evaluation (Held-Out Observations)",
            xaxis_title="Date",
            yaxis_title="Liters"
        )
        st.plotly_chart(fig_eval, use_container_width=True)

        st.markdown(f"#### {fc_days}-Day Forward Projection")
        fig_fc = go.Figure()
        fig_fc.add_trace(go.Scatter(
            x=daily_trends["date"].iloc[-45:],
            y=daily_trends["total_liters"].iloc[-45:],
            mode="lines",
            name="Recent Historical Daily Total (L)",
            line=dict(color="#028090", width=2)
        ))
        fig_fc.add_trace(go.Scatter(
            x=forecast_df["date"],
            y=forecast_df["forecast_estimated_liters"],
            mode="lines+markers",
            name=f"{fc_days}-Day Forecast Estimate (L)",
            line=dict(color="#FF9F1C", dash="dot", width=2.5)
        ))
        fig_fc = apply_chart_theme(fig_fc)
        fig_fc.update_layout(
            title=f"Historical Baseline and {fc_days}-Day Forward Estimate",
            xaxis_title="Date",
            yaxis_title="Liters"
        )
        st.plotly_chart(fig_fc, use_container_width=True)

        st.dataframe(forecast_df.rename(columns={
            "date": "Projected Date",
            "forecast_estimated_liters": "Estimated Liters",
            "forecast_estimated_m3": "Estimated m³"
        }), use_container_width=True)


# ----------------------------------------------------
# TAB 7: COLLEGE VIVA & ARCHITECTURE GUIDE
# ----------------------------------------------------
with tab_viva:
    st.subheader("🎓 College Viva Presentation & Technical Defense Guide")
    st.markdown("""
    Use this section to walk through your architecture and answer examination questions during your college evaluation.
    """)

    st.markdown("### 1. PySpark Architecture & End-to-End Pipeline")
    st.markdown("""
    ```
    +-----------------------------------------------------------------------+
    | 1. RAW INGESTION: Read HSB Living Lab CSV (652,315 rows, 10-min pulse) |
    +-----------------------------------+-----------------------------------+
                                        |
                                        v
    +-----------------------------------------------------------------------+
    | 2. DATA QUALITY & CLEANING (PySpark DataFrame API)                   |
    |    - Timestamp Validation & UTC Parsing                                |
    |    - Value Filtering (0 < volume <= 0.3 m³)                          |
    |    - De-duplication on Composite Key: (timestamp, sensor_id)          |
    |    - Imputation: Null Apartment -> 'Shared Facility'                  |
    +-----------------------------------+-----------------------------------+
                                        |
                                        v
    +-----------------------------------------------------------------------+
    | 3. FEATURE ENGINEERING & PARQUET PERSISTENCE                         |
    |    - Derived Temporal Fields: date, hour, day_of_week, day_name, is_wknd|
    |    - Unit Conversion: volume_liters = volume_m3 * 1000.0              |
    |    - Partitioned Snappy Parquet Export for High-Speed Analytical I/O  |
    +-----------------------------------+-----------------------------------+
                                        |
                                        v
    +-----------------------------------------------------------------------+
    | 4. SPARK SQL DISTRIBUTED ANALYTICS & IQR ANOMALY DETECTION            |
    |    - 24-hr Diurnal Aggregations, Weekly Trends, Fixture Shares        |
    |    - Group-stratified percentiles (approxQuantile / percentile_approx) |
    |    - Upper Threshold = Q3 + (1.5 * IQR) computed per fixture category |
    |    - Distributed Join: Flag outliers with excess volume & ratio      |
    +-----------------------------------+-----------------------------------+
                                        |
                                        v
    +-----------------------------------------------------------------------+
    | 5. PRESENTATION LAYER: Streamlit + Plotly Interactive Dashboard      |
    +-----------------------------------------------------------------------+
    ```
    """)

    st.markdown("### 2. High-Yield Viva Questions and Accurate Answers")

    viva_qas = [
        (
            "Q1: Why did you choose Apache PySpark instead of pure Pandas for this project?",
            "Answer: The HSB Living Lab dataset contains over 652,000 high-frequency 10-minute sensor readings across 214 meters over 3 years. Pure Pandas loads the entire dataset into driver RAM in an eager single-threaded manner, which does not scale when aggregating city-wide or multi-building IoT telemetry. PySpark uses a lazy-evaluated, distributed Directed Acyclic Graph (DAG) query engine with Catalyst optimization, columnar batch processing, and memory-managed partition execution. We only convert compact aggregated results to Pandas for rendering interactive Plotly charts."
        ),
        (
            "Q2: How does Spark's Lazy Evaluation benefit your data pipeline?",
            "Answer: When we apply timestamp parsing, value range filters, and composite de-duplication in `preprocessing.py`, Spark does not immediately execute those operations row-by-row. Instead, Catalyst builds a logical execution plan. It optimizes predicate pushdowns and projection pruning so that when an action (such as `.count()` or writing Parquet) is called, Spark executes the minimal necessary physical stages efficiently."
        ),
        (
            "Q3: Why did you stratify the IQR Anomaly Detection by Fixture instead of applying a single threshold?",
            "Answer: Water consumption end-use fixtures have fundamentally distinct hydrodynamic profiles. A standard toilet flush draws 3 to 6 Liters in a quick burst, whereas an extended shower legitimately consumes 40 to 90 Liters. If we applied a single global building-wide threshold, either almost every shower would be falsely flagged as an anomaly (high false-positive rate), or extreme washbasin/toilet leaks would go completely undetected (high false-negative rate). Grouping by fixture guarantees explainable, fair statistical thresholds."
        ),
        (
            "Q4: How did you handle true duplicates versus concurrent legitimate readings?",
            "Answer: In IoT building metering, multiple meters transmit readings in the exact same 10-minute interval (e.g. APT_001 toilet and APT_002 kitchen tap both reporting at 08:10:00). Blindly dropping rows with matching timestamps would delete legitimate consumption events. We defined a composite primary key consisting of `(timestamp, sensor_id)`. Readings sharing both the identical timestamp AND the identical sensor meter are true duplicates and were safely removed."
        ),
        (
            "Q5: Why did you use Parquet format for intermediate output storage?",
            "Answer: Parquet is a columnar storage format with Snappy compression. Unlike row-based CSVs that require parsing strings and scanning all columns sequentially, Parquet stores columns independently with embedded dictionary encoding and statistics (min/max). When analytical queries filter on `fixture` or aggregate `volume_liters`, Spark reads only the required byte offsets from disk, reducing I/O bandwidth and speeding up query performance by 5x-10x."
        ),
        (
            "Q6: Does a flagged anomaly definitively prove that a water pipe burst or leak occurred?",
            "Answer: No. In statistical quality control, IQR flags observations exceeding Q3 + 1.5 × IQR as statistical outliers relative to typical fixture behavior. An outlier could represent an extended shower, filling a large container or aquarium, heavy deep-cleaning, or multiple back-to-back usages. We explicitly document this distinction in the dashboard to avoid unwarranted alarm."
        )
    ]

    for q, a in viva_qas:
        st.markdown(f"""
        <div class="viva-box">
            <b>{q}</b><br>
            <p style="margin-top: 6px; margin-bottom: 0;">{a}</p>
        </div>
        """, unsafe_allow_html=True)
