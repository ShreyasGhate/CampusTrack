"""Streamlit web application for Campus Energy Usage Profiling."""

from __future__ import annotations

import hashlib
from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data_generator import (
    DEFAULT_DATA_PATH,
    HOUR_COLUMNS,
    load_or_generate_dataset,
    validate_campus_csv,
)
from ml_model import generate_recommendations, run_clustering, run_pca

st.set_page_config(
    page_title="Campus Energy Usage Profiling",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inline SVG icons for KPI cards
ICONS = {
    "building": '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#15803D" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="2" width="16" height="20" rx="2"/><path d="M9 22v-4h6v4"/><path d="M8 6h.01"/><path d="M16 6h.01"/><path d="M12 6h.01"/><path d="M12 10h.01"/><path d="M12 14h.01"/><path d="M16 10h.01"/><path d="M16 14h.01"/><path d="M8 10h.01"/><path d="M8 14h.01"/></svg>',
    "bolt": '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#15803D" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>',
    "activity": '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#0D9488" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>',
    "layers": '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#D97706" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/></svg>',
    "peak": '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#DC2626" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/></svg>',
    "low": '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#0D9488" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 18 13.5 8.5 8.5 13.5 1 6"/><polyline points="17 18 23 18 23 12"/></svg>',
    "check": '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#15803D" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>',
}

st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] {
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
            color: #111827;
        }

        .stApp {
            background-color: #FAF7F2;
            color: #111827;
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 2.5rem;
            max-width: 1320px;
        }

        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header[data-testid="stHeader"] {
            background-color: rgba(250, 247, 242, 0.9);
            backdrop-filter: blur(8px);
        }

        /* Sidebar */
        [data-testid="stSidebar"] {
            background-color: #FFFDF9;
            border-right: 1px solid #E7E0D4;
        }
        .sidebar-brand {
            background: #F0FDF4;
            border: 1px solid #BBF7D0;
            border-radius: 14px;
            padding: 16px;
            margin-bottom: 18px;
        }
        .sidebar-brand-title {
            font-size: 1.08rem;
            font-weight: 800;
            color: #052E16;
            margin: 0;
            letter-spacing: -0.01em;
        }
        .sidebar-brand-sub {
            font-size: 0.78rem;
            color: #166534;
            margin-top: 4px;
            line-height: 1.35;
        }
        .sidebar-section-label {
            font-size: 0.7rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #57534E;
            margin: 12px 0 6px 2px;
        }
        .dataset-pill {
            background-color: #FAF7F2;
            border: 1px solid #E7E0D4;
            border-radius: 10px;
            padding: 10px 12px;
            font-size: 0.8rem;
            color: #111827;
            margin-top: 10px;
        }

        /* Radio navigation polish */
        [data-testid="stSidebar"] [data-testid="stRadio"] label {
            padding: 7px 10px;
            border-radius: 8px;
            transition: background-color 0.15s ease;
            font-weight: 500;
        }
        [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
            background-color: #F3EFE6;
        }

        /* Page Hero Banner */
        .page-hero {
            background-color: #FFFDF9;
            border: 1px solid #E7E0D4;
            border-radius: 16px;
            padding: 22px 26px;
            margin-bottom: 22px;
            box-shadow: 0 2px 8px rgba(17, 24, 39, 0.04);
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 16px;
        }
        .page-eyebrow {
            display: inline-block;
            font-size: 0.7rem;
            font-weight: 700;
            letter-spacing: 0.09em;
            text-transform: uppercase;
            color: #15803D;
            background-color: #ECFDF5;
            border: 1px solid #BBF7D0;
            padding: 3px 10px;
            border-radius: 999px;
            margin-bottom: 8px;
        }
        .page-title {
            font-size: 1.8rem;
            font-weight: 800;
            color: #111827;
            margin: 0;
            letter-spacing: -0.02em;
            line-height: 1.15;
        }
        .page-subtitle {
            font-size: 0.92rem;
            color: #374151;
            margin-top: 6px;
            margin-bottom: 0;
        }

        /* KPI Cards */
        .kpi-card {
            background-color: #FFFDF9;
            border: 1px solid #E7E0D4;
            border-radius: 14px;
            padding: 18px 20px;
            box-shadow: 0 2px 6px rgba(17, 24, 39, 0.03);
            transition: transform 0.15s ease, border-color 0.15s ease, box-shadow 0.15s ease;
            height: 100%;
        }
        .kpi-card:hover {
            transform: translateY(-2px);
            border-color: #86EFAC;
            box-shadow: 0 6px 14px rgba(21, 128, 61, 0.08);
        }
        .kpi-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }
        .kpi-label {
            font-size: 0.82rem;
            color: #374151;
            font-weight: 600;
        }
        .kpi-icon-box {
            width: 34px;
            height: 34px;
            border-radius: 9px;
            background-color: #F0FDF4;
            border: 1px solid #DCFCE7;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .kpi-value {
            font-size: 1.7rem;
            color: #111827;
            font-weight: 800;
            line-height: 1.15;
            letter-spacing: -0.02em;
            font-variant-numeric: tabular-nums;
        }
        .kpi-meta {
            font-size: 0.76rem;
            color: #4B5563;
            margin-top: 6px;
        }

        /* Building Info Strip */
        .info-strip {
            background-color: #FFFDF9;
            border: 1px solid #E7E0D4;
            border-radius: 12px;
            padding: 14px 18px;
            display: flex;
            flex-wrap: wrap;
            gap: 24px;
            align-items: center;
            margin-top: 4px;
        }
        .info-item-label {
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: #57534E;
            font-weight: 700;
        }
        .info-item-val {
            font-size: 0.96rem;
            color: #111827;
            font-weight: 700;
            margin-top: 2px;
        }

        /* Cluster Summary Card */
        .cluster-card {
            background-color: #FFFDF9;
            border: 1px solid #E7E0D4;
            border-radius: 12px;
            padding: 14px 16px;
            margin-bottom: 10px;
        }
        .cluster-card-top {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 6px;
        }
        .cluster-name {
            font-size: 0.94rem;
            font-weight: 700;
            color: #111827;
        }
        .cluster-count {
            font-size: 0.78rem;
            font-weight: 700;
            color: #111827;
            background-color: #F3EFE6;
            padding: 2px 9px;
            border-radius: 999px;
        }
        .cluster-bar-bg {
            width: 100%;
            height: 7px;
            background-color: #EDE8DF;
            border-radius: 999px;
            overflow: hidden;
            margin: 8px 0 6px 0;
        }
        .cluster-bar-fill {
            height: 100%;
            border-radius: 999px;
        }
        .cluster-avg {
            font-size: 0.8rem;
            color: #374151;
            font-weight: 500;
        }

        /* Note Callout */
        .section-note {
            background-color: #FFFDF9;
            border: 1px solid #E7E0D4;
            border-left: 4px solid #15803D;
            padding: 14px 18px;
            border-radius: 10px;
            color: #111827;
            font-size: 0.92rem;
            margin-bottom: 18px;
            box-shadow: 0 1px 3px rgba(17, 24, 39, 0.03);
        }

        /* Recommendation Cards */
        .rec-card {
            background-color: #FFFDF9;
            border: 1px solid #E7E0D4;
            border-radius: 14px;
            padding: 18px 20px;
            margin-bottom: 14px;
            box-shadow: 0 2px 6px rgba(17, 24, 39, 0.03);
            transition: border-color 0.15s ease;
        }
        .rec-card:hover {
            border-color: #CBD5E1;
        }
        .rec-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            gap: 10px;
            margin-bottom: 10px;
        }
        .rec-title {
            font-size: 1.02rem;
            font-weight: 700;
            color: #111827;
            margin: 0;
        }
        .rec-sub {
            font-size: 0.8rem;
            color: #4B5563;
            margin-top: 3px;
        }
        .rec-badge {
            font-size: 0.72rem;
            font-weight: 700;
            padding: 4px 11px;
            border-radius: 999px;
            white-space: nowrap;
        }
        .badge-high { background-color: #FEE2E2; color: #991B1B; border: 1px solid #FCA5A5; }
        .badge-night { background-color: #FEF3C7; color: #92400E; border: 1px solid #FCD34D; }
        .badge-peak { background-color: #FFEDD5; color: #9A3412; border: 1px solid #FDBA74; }
        .badge-low { background-color: #DCFCE7; color: #166534; border: 1px solid #86EFAC; }
        .badge-balanced { background-color: #F3EFE6; color: #1F2937; border: 1px solid #D6CFC2; }

        .rec-pattern {
            font-size: 0.85rem;
            font-weight: 700;
            color: #111827;
            margin-bottom: 5px;
        }
        .rec-body {
            font-size: 0.87rem;
            color: #374151;
            line-height: 1.5;
            margin: 0;
        }

        /* Primary Button */
        div.stButton > button:first-child {
            background-color: #15803D;
            color: #FFFFFF;
            border: none;
            border-radius: 10px;
            padding: 0.6rem 1.4rem;
            font-weight: 700;
            font-size: 0.9rem;
            box-shadow: 0 2px 6px rgba(21, 128, 61, 0.22);
            transition: background-color 0.15s ease, transform 0.1s ease;
        }
        div.stButton > button:first-child:hover {
            background-color: #166534;
            color: #FFFFFF;
            transform: translateY(-1px);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

CHART_PALETTE = [
    "#15803D",
    "#0D9488",
    "#0284C7",
    "#D97706",
    "#DC2626",
    "#7C3AED",
    "#475569",
    "#65A30D",
]

CLUSTER_COLOR_MAP = {
    "Very Low Consumption": "#6EE7B7",
    "Low Consumption": "#16A34A",
    "Medium Consumption": "#D97706",
    "High Consumption": "#DC2626",
    "Very High Consumption": "#7C3AED",
}


def apply_chart_theme(fig: go.Figure, height: Optional[int] = None) -> go.Figure:
    """Apply consistent warm-white and black-text styling to all Plotly figures."""
    layout_kwargs = dict(
        plot_bgcolor="#FFFDF9",
        paper_bgcolor="#FFFDF9",
        font=dict(family="Plus Jakarta Sans, sans-serif", color="#111827", size=12),
        title_font=dict(size=15, color="#111827"),
        xaxis=dict(
            showgrid=False,
            linecolor="#D6CFC2",
            tickfont=dict(color="#111827"),
            title_font=dict(color="#111827"),
        ),
        yaxis=dict(
            gridcolor="#EDE8DF",
            zerolinecolor="#D6CFC2",
            tickfont=dict(color="#111827"),
            title_font=dict(color="#111827"),
        ),
        legend=dict(
            bgcolor="#FFFDF9",
            bordercolor="#E7E0D4",
            borderwidth=1,
            font=dict(size=11, color="#111827"),
        ),
    )
    if height is not None:
        layout_kwargs["height"] = height
    fig.update_layout(**layout_kwargs)
    return fig


def render_page_hero(eyebrow: str, title: str, subtitle: str) -> None:
    """Render a clean editorial hero banner."""
    st.markdown(
        f"""
        <div class="page-hero">
            <div>
                <div class="page-eyebrow">{eyebrow}</div>
                <h1 class="page-title">{title}</h1>
                <p class="page-subtitle">{subtitle}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_card(label: str, value: str, meta: str = "", icon_key: str = "bolt") -> None:
    """Render a KPI card with an inline SVG icon badge."""
    icon_svg = ICONS.get(icon_key, ICONS["bolt"])
    meta_html = f'<div class="kpi-meta">{meta}</div>' if meta else ""
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-header">
                <span class="kpi-label">{label}</span>
                <span class="kpi-icon-box">{icon_svg}</span>
            </div>
            <div class="kpi-value">{value}</div>
            {meta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def get_default_dataset() -> pd.DataFrame:
    """Load or generate the default campus energy dataset."""
    return load_or_generate_dataset(DEFAULT_DATA_PATH)


@st.cache_data
def cached_clustering(
    df_input: pd.DataFrame,
    algorithm: str,
    n_clusters: int,
) -> dict:
    """Cache clustering results for a given dataset and configuration."""
    return run_clustering(df_input, algorithm=algorithm, n_clusters=n_clusters)


def load_active_dataset(uploaded_file: Optional[object]) -> tuple[pd.DataFrame, str, str]:
    """Return active DataFrame, signature, and source description."""
    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)
            is_valid, error_msg = validate_campus_csv(uploaded_df)
            if is_valid:
                for col in ["floor_area"] + HOUR_COLUMNS:
                    uploaded_df[col] = pd.to_numeric(uploaded_df[col], errors="coerce")
                sig = hashlib.md5(uploaded_df.to_csv(index=False).encode()).hexdigest()
                return uploaded_df, sig, f"Uploaded CSV ({len(uploaded_df)} buildings)"
            st.sidebar.error(f"Invalid CSV: {error_msg} Using default dataset instead.")
        except Exception as exc:
            st.sidebar.error(f"Could not read CSV ({exc}). Using default dataset instead.")

    default_df = get_default_dataset()
    sig = hashlib.md5(default_df.to_csv(index=False).encode()).hexdigest()
    return default_df, sig, f"Default Dataset ({len(default_df)} buildings)"


def ensure_clustering_state(df: pd.DataFrame, dataset_sig: str) -> dict:
    """Ensure clustering results exist in session state for the active dataset."""
    if (
        "clustering_result" not in st.session_state
        or st.session_state.get("dataset_sig") != dataset_sig
    ):
        st.session_state["dataset_sig"] = dataset_sig
        st.session_state["selected_algorithm"] = st.session_state.get("selected_algorithm", "K-Means")
        st.session_state["selected_clusters"] = st.session_state.get("selected_clusters", 3)
        st.session_state["clustering_result"] = cached_clustering(
            df,
            st.session_state["selected_algorithm"],
            st.session_state["selected_clusters"],
        )
    return st.session_state["clustering_result"]


def render_dashboard_overview(df: pd.DataFrame, clustering_result: dict) -> None:
    """PAGE 1: Dashboard Overview."""
    render_page_hero(
        "Campus Energy Analytics",
        "Dashboard Overview",
        "Monitor campus-wide electricity demand, hourly load curves, and category breakdowns.",
    )

    daily_totals = df[HOUR_COLUMNS].sum(axis=1)
    total_buildings = len(df)
    total_categories = df["building_type"].nunique()
    total_daily_kwh = float(daily_totals.sum())
    avg_daily_kwh = float(daily_totals.mean())
    num_clusters = int(clustering_result["n_clusters_found"])

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_kpi_card("Total Buildings", f"{total_buildings}", f"Across {total_categories} categories", "building")
    with col2:
        render_kpi_card("Total Daily Consumption", f"{total_daily_kwh:,.1f} kWh", "24-hour campus aggregate", "bolt")
    with col3:
        render_kpi_card("Avg Daily / Building", f"{avg_daily_kwh:,.1f} kWh", "Mean daily load per facility", "activity")
    with col4:
        render_kpi_card("Identified Clusters", f"{num_clusters}", f"Active: {clustering_result['algorithm']}", "layers")

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    chart_df = df[["building_name", "building_type"]].copy()
    chart_df["Daily Consumption (kWh)"] = daily_totals.round(2)
    chart_df = chart_df.sort_values("Daily Consumption (kWh)", ascending=False)

    with st.container(border=True):
        fig_bar = px.bar(
            chart_df,
            x="building_name",
            y="Daily Consumption (kWh)",
            color="building_type",
            title="Total Daily Energy Consumption by Building",
            labels={"building_name": "Building", "building_type": "Building Type"},
            color_discrete_sequence=CHART_PALETTE,
        )
        apply_chart_theme(fig_bar, height=410)
        fig_bar.update_layout(xaxis_tickangle=-38, margin=dict(t=50, b=95, l=45, r=20))
        st.plotly_chart(fig_bar, width="stretch")

    left_col, right_col = st.columns(2)

    with left_col:
        with st.container(border=True):
            hourly_avg = df[HOUR_COLUMNS].mean(axis=0)
            hourly_df = pd.DataFrame({
                "Hour": HOUR_COLUMNS,
                "Average Consumption (kWh)": hourly_avg.values.round(2),
            })
            fig_line = px.area(
                hourly_df,
                x="Hour",
                y="Average Consumption (kWh)",
                markers=True,
                line_shape="spline",
                title="Average Hourly Campus Energy Consumption",
            )
            fig_line.update_traces(
                line=dict(color="#15803D", width=3),
                marker=dict(color="#052E16", size=6),
                fillcolor="rgba(21, 128, 61, 0.12)",
            )
            apply_chart_theme(fig_line, height=370)
            fig_line.update_layout(xaxis_tickangle=-45, margin=dict(t=50, b=65, l=45, r=20))
            st.plotly_chart(fig_line, width="stretch")

    with right_col:
        with st.container(border=True):
            type_totals = (
                chart_df.groupby("building_type", as_index=False)["Daily Consumption (kWh)"]
                .sum()
                .sort_values("Daily Consumption (kWh)", ascending=False)
            )
            fig_pie = px.pie(
                type_totals,
                names="building_type",
                values="Daily Consumption (kWh)",
                hole=0.55,
                title="Energy Distribution by Building Type",
                color_discrete_sequence=CHART_PALETTE,
            )
            fig_pie.update_traces(
                textposition="inside",
                textinfo="percent",
                marker=dict(line=dict(color="#FFFDF9", width=2)),
            )
            apply_chart_theme(fig_pie, height=370)
            fig_pie.add_annotation(
                text=f"<b>{total_daily_kwh:,.0f}</b><br>kWh",
                x=0.5,
                y=0.5,
                font=dict(size=14, color="#111827"),
                showarrow=False,
            )
            fig_pie.update_layout(margin=dict(t=50, b=20, l=15, r=15))
            st.plotly_chart(fig_pie, width="stretch")


def render_consumption_analysis(df: pd.DataFrame) -> None:
    """PAGE 2: Energy Consumption Analysis."""
    render_page_hero(
        "Building Deep Dive",
        "Energy Consumption Analysis",
        "Inspect a single building's 24-hour demand curve and benchmark it against the rest of campus.",
    )

    building_names = df["building_name"].tolist()
    sel_col1, sel_col2 = st.columns([1.3, 2.2])
    with sel_col1:
        selected_building = st.selectbox("Select a Building", building_names)

    row = df[df["building_name"] == selected_building].iloc[0]
    hourly_series = row[HOUR_COLUMNS].astype(float)

    daily_total = float(hourly_series.sum())
    peak_hour = str(hourly_series.idxmax())
    peak_val = float(hourly_series.max())
    low_hour = str(hourly_series.idxmin())
    low_val = float(hourly_series.min())
    avg_hourly = float(hourly_series.mean())

    with sel_col2:
        st.markdown(
            f"""
            <div class="info-strip">
                <div>
                    <div class="info-item-label">Category</div>
                    <div class="info-item-val">{row['building_type']}</div>
                </div>
                <div>
                    <div class="info-item-label">Floor Area</div>
                    <div class="info-item-val">{int(row['floor_area']):,} sq. m.</div>
                </div>
                <div>
                    <div class="info-item-label">Building ID</div>
                    <div class="info-item-val">{row['building_id']}</div>
                </div>
                <div>
                    <div class="info-item-label">Peak / Avg Ratio</div>
                    <div class="info-item-val">{(peak_val / avg_hourly):.2f}x</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_kpi_card("Daily Consumption", f"{daily_total:,.1f} kWh", "Total 24-hour usage", "bolt")
    with c2:
        render_kpi_card("Peak Hour", peak_hour, f"{peak_val:.1f} kWh maximum load", "peak")
    with c3:
        render_kpi_card("Lowest Hour", low_hour, f"{low_val:.1f} kWh minimum load", "low")
    with c4:
        render_kpi_card("Average Hourly", f"{avg_hourly:.1f} kWh", "Mean hourly demand", "activity")

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    with st.container(border=True):
        building_hourly_df = pd.DataFrame({
            "Hour": HOUR_COLUMNS,
            "Energy Consumption (kWh)": hourly_series.values,
        })
        fig_hourly = px.area(
            building_hourly_df,
            x="Hour",
            y="Energy Consumption (kWh)",
            markers=True,
            line_shape="spline",
            title=f"Hourly Electricity Consumption — {selected_building}",
        )
        fig_hourly.update_traces(
            line=dict(color="#15803D", width=3),
            marker=dict(color="#052E16", size=7),
            fillcolor="rgba(21, 128, 61, 0.13)",
        )
        fig_hourly.add_hline(
            y=avg_hourly,
            line_dash="dash",
            line_color="#D97706",
            annotation_text=f"Avg: {avg_hourly:.1f} kWh",
            annotation_position="top right",
            annotation_font=dict(color="#92400E", size=11),
        )
        apply_chart_theme(fig_hourly, height=390)
        fig_hourly.update_layout(xaxis_tickangle=-45, margin=dict(t=55, b=65, l=45, r=20))
        st.plotly_chart(fig_hourly, width="stretch")

    with st.container(border=True):
        comp_df = pd.DataFrame({
            "Building": df["building_name"],
            "Building Type": df["building_type"],
            "Daily Consumption (kWh)": df[HOUR_COLUMNS].sum(axis=1).round(2),
        }).sort_values("Daily Consumption (kWh)", ascending=True)

        colors = [
            "#15803D" if b == selected_building else "#D6CFC2"
            for b in comp_df["Building"]
        ]
        fig_comp = go.Figure(
            go.Bar(
                x=comp_df["Daily Consumption (kWh)"],
                y=comp_df["Building"],
                orientation="h",
                marker_color=colors,
                text=comp_df["Daily Consumption (kWh)"].map(lambda v: f"{v:,.0f} kWh"),
                textposition="outside",
                textfont=dict(color="#111827", size=11),
                hovertemplate="<b>%{y}</b><br>Daily Consumption: %{x:,.2f} kWh<extra></extra>",
            )
        )
        apply_chart_theme(fig_comp, height=max(560, len(comp_df) * 23))
        fig_comp.update_layout(
            title=f"Campus-Wide Comparison (Highlighted: {selected_building})",
            xaxis_title="Daily Consumption (kWh)",
            yaxis_title="",
            margin=dict(l=205, r=70, t=55, b=40),
        )
        st.plotly_chart(fig_comp, width="stretch")


def render_ml_clustering(df: pd.DataFrame, dataset_sig: str) -> None:
    """PAGE 3: Machine Learning Clustering."""
    render_page_hero(
        "Unsupervised Machine Learning",
        "Machine Learning Clustering",
        "Group campus buildings into consumption profiles using K-Means or Gaussian Mixture Models.",
    )

    with st.container(border=True):
        ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([2, 2, 1.1])
        with ctrl_col1:
            algorithm = st.selectbox(
                "Clustering Algorithm",
                ["K-Means", "Gaussian Mixture Model"],
                index=0 if st.session_state.get("selected_algorithm", "K-Means") == "K-Means" else 1,
            )
        with ctrl_col2:
            n_clusters = st.slider(
                "Number of Clusters",
                min_value=2,
                max_value=5,
                value=int(st.session_state.get("selected_clusters", 3)),
                step=1,
            )
        with ctrl_col3:
            st.markdown("<div style='height: 26px;'></div>", unsafe_allow_html=True)
            run_clicked = st.button("Run Clustering", width="stretch")

    if run_clicked:
        st.session_state["selected_algorithm"] = algorithm
        st.session_state["selected_clusters"] = n_clusters
        st.session_state["dataset_sig"] = dataset_sig
        st.session_state["clustering_result"] = cached_clustering(
            df,
            algorithm,
            n_clusters,
        )

    result = st.session_state["clustering_result"]
    sil_score = result["silhouette_score"]

    m1, m2, m3 = st.columns(3)
    with m1:
        render_kpi_card("Active Algorithm", result["algorithm"], "Standardized 24-hour features", "activity")
    with m2:
        render_kpi_card("Clusters Identified", str(result["n_clusters_found"]), "Named by mean consumption", "layers")
    with m3:
        sil_display = f"{sil_score:.3f}" if sil_score is not None else "N/A"
        render_kpi_card("Silhouette Score", sil_display, "Cluster separation quality (-1 to 1)", "check")

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    left_col, right_col = st.columns([1.05, 1.95])
    with left_col:
        st.subheader("Cluster Summary")
        summary_df = result["cluster_summary"]
        total_b = max(int(summary_df["Number of Buildings"].sum()), 1)
        for _, srow in summary_df.iterrows():
            cname = srow["Cluster"]
            bcount = int(srow["Number of Buildings"])
            avg_c = float(srow["Average Daily Consumption (kWh)"])
            pct = int(round((bcount / total_b) * 100))
            bar_color = CLUSTER_COLOR_MAP.get(cname, "#15803D")
            st.markdown(
                f"""
                <div class="cluster-card">
                    <div class="cluster-card-top">
                        <span class="cluster-name">{cname}</span>
                        <span class="cluster-count">{bcount} buildings ({pct}%)</span>
                    </div>
                    <div class="cluster-bar-bg">
                        <div class="cluster-bar-fill" style="width:{pct}%; background-color:{bar_color};"></div>
                    </div>
                    <div class="cluster-avg">Avg Daily Consumption: <strong>{avg_c:,.2f} kWh</strong></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with right_col:
        st.subheader("Building Cluster Assignments")
        display_table = result["result_df"][
            ["building_name", "building_type", "daily_consumption", "cluster_name"]
        ].rename(
            columns={
                "building_name": "Building",
                "building_type": "Building Type",
                "daily_consumption": "Daily Consumption (kWh)",
                "cluster_name": "Cluster",
            }
        )
        st.dataframe(display_table, width="stretch", hide_index=True, height=500)


def render_cluster_visualization(clustering_result: dict) -> None:
    """PAGE 4: Cluster Visualization using PCA."""
    render_page_hero(
        "Dimensionality Reduction",
        "Cluster Visualization (PCA)",
        "Explore how buildings group together in two dimensions based on their 24-hour energy profiles.",
    )

    st.markdown(
        '<div class="section-note">'
        "PCA reduces the original 24 hourly consumption features into two dimensions so that "
        "buildings with similar energy usage patterns can be visualized together."
        "</div>",
        unsafe_allow_html=True,
    )

    components, variance_pct = run_pca(clustering_result["X_scaled"])
    total_variance = float(variance_pct.sum())

    v1, v2, v3 = st.columns(3)
    with v1:
        render_kpi_card("PC1 Variance Explained", f"{variance_pct[0]:.2f}%", "Primary variation axis", "activity")
    with v2:
        render_kpi_card("PC2 Variance Explained", f"{variance_pct[1]:.2f}%", "Secondary variation axis", "layers")
    with v3:
        render_kpi_card("Total Variance Captured", f"{total_variance:.2f}%", "Combined 2D information retained", "check")

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    with st.container(border=True):
        plot_df = clustering_result["result_df"].copy()
        plot_df["Principal Component 1"] = components[:, 0]
        plot_df["Principal Component 2"] = components[:, 1]

        fig_pca = px.scatter(
            plot_df,
            x="Principal Component 1",
            y="Principal Component 2",
            color="cluster_name",
            hover_name="building_name",
            hover_data={
                "building_type": True,
                "daily_consumption": ":.2f",
                "cluster_name": True,
                "Principal Component 1": ":.2f",
                "Principal Component 2": ":.2f",
            },
            color_discrete_map=CLUSTER_COLOR_MAP,
            title=f"2D PCA Projection of Campus Buildings ({clustering_result['algorithm']})",
            labels={"cluster_name": "Cluster"},
        )
        fig_pca.update_traces(marker=dict(size=15, opacity=0.92, line=dict(width=2, color="#FFFDF9")))
        apply_chart_theme(fig_pca, height=560)
        fig_pca.update_xaxes(zeroline=True, zerolinecolor="#D6CFC2", showgrid=True, gridcolor="#EDE8DF")
        fig_pca.update_yaxes(zeroline=True, zerolinecolor="#D6CFC2", showgrid=True, gridcolor="#EDE8DF")
        st.plotly_chart(fig_pca, width="stretch")


def _badge_for_pattern(pattern_text: str) -> tuple[str, str, str]:
    """Return CSS class, short label, and left accent border color for a recommendation pattern."""
    if "High Energy Consumption" in pattern_text:
        return "badge-high", "High Load", "#DC2626"
    if "Nighttime" in pattern_text:
        return "badge-night", "Nighttime Load", "#D97706"
    if "Sharp Consumption Peak" in pattern_text:
        return "badge-peak", "Peak Demand", "#EA580C"
    if "Relatively Low" in pattern_text:
        return "badge-low", "Efficient", "#15803D"
    return "badge-balanced", "Balanced", "#78716C"


def render_recommendations(df: pd.DataFrame) -> None:
    """PAGE 5: Energy Saving Recommendations."""
    render_page_hero(
        "Actionable Insights",
        "Energy Saving Recommendations",
        "Rule-based efficiency recommendations derived directly from each building's hourly consumption profile.",
    )

    rec_df = generate_recommendations(df)

    high_count = int(rec_df["Identified Consumption Pattern"].str.contains("High Energy Consumption").sum())
    night_count = int(rec_df["Identified Consumption Pattern"].str.contains("Nighttime").sum())
    peak_count = int(rec_df["Identified Consumption Pattern"].str.contains("Peak").sum())
    low_count = int(rec_df["Identified Consumption Pattern"].str.contains("Relatively Low").sum())

    s1, s2, s3, s4 = st.columns(4)
    with s1:
        render_kpi_card("High Consumption Flags", str(high_count), "Lighting / HVAC audit targets", "peak")
    with s2:
        render_kpi_card("Nighttime Load Flags", str(night_count), "After-hours equipment checks", "bolt")
    with s3:
        render_kpi_card("Peak Demand Flags", str(peak_count), "Load-shifting candidates", "activity")
    with s4:
        render_kpi_card("Efficient Buildings", str(low_count), "Well below campus average", "check")

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    f1, f2 = st.columns([1, 1])
    with f1:
        building_types = ["All"] + sorted(rec_df["Building Type"].unique().tolist())
        selected_type = st.selectbox("Filter by Building Type", building_types)
    with f2:
        pattern_filter = st.selectbox(
            "Filter by Priority / Pattern",
            ["All", "High Load", "Nighttime Load", "Peak Demand", "Efficient"],
        )

    filtered_df = rec_df if selected_type == "All" else rec_df[rec_df["Building Type"] == selected_type]
    if pattern_filter == "High Load":
        filtered_df = filtered_df[filtered_df["Identified Consumption Pattern"].str.contains("High Energy Consumption")]
    elif pattern_filter == "Nighttime Load":
        filtered_df = filtered_df[filtered_df["Identified Consumption Pattern"].str.contains("Nighttime")]
    elif pattern_filter == "Peak Demand":
        filtered_df = filtered_df[filtered_df["Identified Consumption Pattern"].str.contains("Peak")]
    elif pattern_filter == "Efficient":
        filtered_df = filtered_df[filtered_df["Identified Consumption Pattern"].str.contains("Relatively Low")]

    rows = filtered_df.to_dict(orient="records")
    if not rows:
        st.info("No buildings match the selected filters.")
    else:
        for idx in range(0, len(rows), 2):
            card_cols = st.columns(2)
            for offset in range(2):
                if idx + offset < len(rows):
                    item = rows[idx + offset]
                    badge_cls, badge_label, accent_color = _badge_for_pattern(
                        item["Identified Consumption Pattern"]
                    )
                    with card_cols[offset]:
                        st.markdown(
                            f"""
                            <div class="rec-card" style="border-left: 4px solid {accent_color};">
                                <div class="rec-header">
                                    <div>
                                        <div class="rec-title">{item['Building']}</div>
                                        <div class="rec-sub">{item['Building Type']} &middot; {item['Daily Consumption (kWh)']:,.1f} kWh/day &middot; Peak {item['Peak Hour']}</div>
                                    </div>
                                    <span class="rec-badge {badge_cls}">{badge_label}</span>
                                </div>
                                <div class="rec-pattern">{item['Identified Consumption Pattern']}</div>
                                <p class="rec-body">{item['Suggested Improvement']}</p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

    with st.expander("View Full Recommendations Table"):
        st.dataframe(
            filtered_df[
                [
                    "Building",
                    "Building Type",
                    "Daily Consumption (kWh)",
                    "Identified Consumption Pattern",
                    "Suggested Improvement",
                ]
            ],
            width="stretch",
            hide_index=True,
        )


def main() -> None:
    """Main Streamlit application entry point."""
    st.sidebar.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-brand-title">⚡ Campus Energy Profiling</div>
            <div class="sidebar-brand-sub">ML-powered campus electricity analytics &amp; clustering</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown('<div class="sidebar-section-label">Workspace</div>', unsafe_allow_html=True)
    page = st.sidebar.radio(
        "Navigation",
        [
            "Dashboard Overview",
            "Energy Consumption Analysis",
            "Machine Learning Clustering",
            "Cluster Visualization",
            "Energy Saving Recommendations",
        ],
        label_visibility="collapsed",
    )

    st.sidebar.markdown('<div class="sidebar-section-label">Dataset Source</div>', unsafe_allow_html=True)
    uploaded_file = st.sidebar.file_uploader(
        "Upload Campus Energy CSV (Optional)",
        type=["csv"],
        help="CSV must include building_id, building_name, building_type, floor_area, and 00:00 to 23:00 columns.",
    )

    df, dataset_sig, source_label = load_active_dataset(uploaded_file)
    st.sidebar.markdown(
        f'<div class="dataset-pill"><strong>Active:</strong> {source_label}</div>',
        unsafe_allow_html=True,
    )

    clustering_result = ensure_clustering_state(df, dataset_sig)

    if page == "Dashboard Overview":
        render_dashboard_overview(df, clustering_result)
    elif page == "Energy Consumption Analysis":
        render_consumption_analysis(df)
    elif page == "Machine Learning Clustering":
        render_ml_clustering(df, dataset_sig)
    elif page == "Cluster Visualization":
        render_cluster_visualization(clustering_result)
    elif page == "Energy Saving Recommendations":
        render_recommendations(df)


if __name__ == "__main__":
    main()
