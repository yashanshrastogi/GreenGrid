# -*- coding: utf-8 -*-
"""
GreenGrid — Professional Energy Consumption Analytics Platform
==============================================================
Run: streamlit run dashboard/app.py
"""

import os, sys, warnings, logging
from pathlib import Path

warnings.filterwarnings("ignore")
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from sklearn.cluster import KMeans
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# ── Paths ──────────────────────────────────────────────────────────────────────
DASHBOARD_DIR = Path(__file__).parent
BASE_DIR      = DASHBOARD_DIR.parent
MODELS_DIR    = BASE_DIR / "models"
REPORTS_DIR   = BASE_DIR / "reports"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
sys.path.insert(0, str(MODELS_DIR))
sys.path.insert(0, str(REPORTS_DIR))

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GreenGrid — Energy Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session State Init ─────────────────────────────────────────────────────────
if "theme" not in st.session_state:
    st.session_state["theme"] = "Dark"

# ── Theme ──────────────────────────────────────────────────────────────────────
is_light = False

# ══════════════════════════════════════════════════════════════════════════════
# DESIGN SYSTEM — DATA-SCIENCE AESTHETIC
# ══════════════════════════════════════════════════════════════════════════════
if is_light:
    C = {
        # Backgrounds
        "app_bg":        "#f0f4ff",
        "surface":       "#ffffff",
        "surface_raised":"#f8faff",
        "sidebar_bg":    "#ffffff",
        "sidebar_border":"#dde3f0",
        # Text
        "text_primary":  "#0f172a",
        "text_secondary":"#334155",
        "text_muted":    "#475569",
        "text_placeholder":"#94a3b8",
        # Accents
        "accent":        "#4f46e5",  # vibrant indigo
        "accent_hover":  "#4338ca",
        "accent_glow":   "rgba(79,70,229,0.15)",
        "accent2":       "#0891b2",  # deep cyan
        # Status
        "danger":        "#dc2626",
        "danger_bg":     "#fff1f2",
        "danger_border": "#fca5a5",
        "warning":       "#d97706",
        "warning_bg":    "#fffbeb",
        "warning_border":"#fcd34d",
        "success":       "#059669",
        "success_bg":    "#ecfdf5",
        "success_border":"#6ee7b7",
        "info":          "#0284c7",
        "info_bg":       "#eff6ff",
        "info_border":   "#93c5fd",
        # Borders
        "border_subtle": "#e2e8f0",
        "border_default":"#cbd5e1",
        "border_strong": "#94a3b8",
        # Shadows
        "shadow_xs":     "0 1px 2px rgba(15,23,42,0.05)",
        "shadow_sm":     "0 2px 4px rgba(15,23,42,0.06), 0 1px 2px rgba(15,23,42,0.04)",
        "shadow_md":     "0 4px 12px rgba(15,23,42,0.08), 0 2px 4px rgba(15,23,42,0.05)",
        "shadow_lg":     "0 12px 24px rgba(15,23,42,0.10), 0 4px 8px rgba(15,23,42,0.06)",
        "shadow_glow":   "0 0 0 3px rgba(79,70,229,0.12), 0 4px 12px rgba(79,70,229,0.15)",
        # Chart
        "chart_paper":   "rgba(255,255,255,0)",
        "chart_plot":    "rgba(255,255,255,0)",
        "grid_color":    "rgba(226,232,240,0.8)",
        "plotly_tmpl":   "plotly_white",
        # Heatmap
        "heatmap_synth": "Blues",
        "heatmap_real":  "Oranges",
        # KPI accent colors
        "kpi_primary":   "#4f46e5",
        "kpi_danger":    "#dc2626",
        "kpi_warning":   "#d97706",
        "kpi_green":     "#059669",
        "kpi_cyan":      "#0891b2",
        # Footer
        "footer_bg":     "#f0f4ff",
        "footer_border": "#dde3f0",
        "footer_text":   "#94a3b8",
    }
else:
    C = {
        # Backgrounds
        "app_bg":        "#0f1117",
        "surface":       "#1a1f2e",
        "surface_raised":"#1f2540",
        "sidebar_bg":    "#13161f",
        "sidebar_border":"#1e2235",
        # Text
        "text_primary":  "#f8fafc",
        "text_secondary":"#cbd5e1",
        "text_muted":    "#94a3b8",
        "text_placeholder":"#475569",
        # Accents
        "accent":        "#00d4aa",  # neon teal
        "accent_hover":  "#00c49a",
        "accent_glow":   "rgba(0,212,170,0.15)",
        "accent2":       "#818cf8",  # soft violet
        # Status
        "danger":        "#f87171",
        "danger_bg":     "#1f0f0f",
        "danger_border": "#7f1d1d",
        "warning":       "#fbbf24",
        "warning_bg":    "#1c1500",
        "warning_border":"#78350f",
        "success":       "#34d399",
        "success_bg":    "#052e1d",
        "success_border":"#065f46",
        "info":          "#60a5fa",
        "info_bg":       "#0c1a2e",
        "info_border":   "#1e40af",
        # Borders
        "border_subtle": "#1e2235",
        "border_default":"#2d3550",
        "border_strong": "#3d4d70",
        # Shadows
        "shadow_xs":     "0 1px 2px rgba(0,0,0,0.40)",
        "shadow_sm":     "0 2px 4px rgba(0,0,0,0.50), 0 1px 2px rgba(0,0,0,0.40)",
        "shadow_md":     "0 4px 12px rgba(0,0,0,0.55), 0 2px 4px rgba(0,0,0,0.40)",
        "shadow_lg":     "0 12px 24px rgba(0,0,0,0.60), 0 4px 8px rgba(0,0,0,0.50)",
        "shadow_glow":   "0 0 0 3px rgba(0,212,170,0.12), 0 4px 12px rgba(0,212,170,0.20)",
        # Chart
        "chart_paper":   "rgba(0,0,0,0)",
        "chart_plot":    "rgba(0,0,0,0)",
        "grid_color":    "rgba(30,34,53,0.9)",
        "plotly_tmpl":   "plotly_dark",
        # Heatmap
        "heatmap_synth": "Teal",
        "heatmap_real":  "YlOrRd",
        # KPI accent colors
        "kpi_primary":   "#00d4aa",
        "kpi_danger":    "#f87171",
        "kpi_warning":   "#fbbf24",
        "kpi_green":     "#34d399",
        "kpi_cyan":      "#818cf8",
        # Footer
        "footer_bg":     "#13161f",
        "footer_border": "#1e2235",
        "footer_text":   "#475569",
    }

# ── Splash Screen (Onboarding) ──────────────────────────────────────────────────
if "splash_shown" not in st.session_state:
    st.session_state.splash_shown = True
    st.markdown(f"""
    <style>
        #close-splash {{ display: none; }}
        #close-splash:checked ~ #welcome-splash {{
            opacity: 0 !important;
            visibility: hidden !important;
            pointer-events: none !important;
            transition: all 0.5s ease;
        }}
        @keyframes hideSplash {{
            0%, 90% {{ opacity: 1; visibility: visible; pointer-events: auto; }}
            100% {{ opacity: 0; visibility: hidden; pointer-events: none; }}
        }}
        @keyframes slideUpSplash {{
            from {{ transform: translateY(30px); opacity: 0; }}
            to {{ transform: translateY(0); opacity: 1; }}
        }}
        @keyframes radialCountdown {{
            from {{ stroke-dashoffset: 0; }}
            to   {{ stroke-dashoffset: 135; }}
        }}
        @media (max-width: 640px) {{
            .splash-modal {{ padding: 20px 18px !important; border-radius: 12px !important; width: 96% !important; max-height: 85vh !important; overflow-y: auto !important; }}
            /* On mobile, move button fully inside the modal so it never clips outside the screen */
            .splash-close {{ top: 8px !important; right: 8px !important; width: 36px !important; height: 36px !important; font-size: 1.3rem !important; position: absolute !important; }}
            .splash-title {{ font-size: 1.15rem !important; margin-bottom: 12px !important; padding-bottom: 8px !important; padding-right: 44px !important; text-align: left !important; }}
            .splash-list {{ font-size: 0.82rem !important; line-height: 1.55 !important; padding-left: 14px !important; }}
            .splash-list li {{ margin-bottom: 7px !important; }}
        }}
        @media (min-width: 641px) {{
            .splash-close {{ top: -16px !important; right: -16px !important; }}
        }}
    </style>
    <input type="checkbox" id="close-splash">
    <div id="welcome-splash" style="position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; z-index: 999999; display: flex; justify-content: center; align-items: center; animation: hideSplash 20s forwards;">
        <label for="close-splash" style="position: absolute; top: 0; left: 0; right: 0; bottom: 0; background: rgba(10, 14, 23, 0.6); backdrop-filter: blur(8px); cursor: default; z-index: 0; margin: 0;"></label>
        <div class="splash-modal" style="background: rgba(26, 31, 46, 0.92); backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px); border: 1px solid rgba(0, 212, 170, 0.4); border-radius: 16px; padding: 40px; box-shadow: 0 15px 50px rgba(0, 0, 0, 0.5), 0 0 30px rgba(0, 212, 170, 0.15); max-width: 650px; width: 90%; position: relative; z-index: 1; animation: slideUpSplash 0.5s ease-out;">
            <label for="close-splash" class="splash-close" style="position: absolute; top: -15px; right: -15px; width: 45px; height: 45px; background: #1a1f2e; border-radius: 50%; color: #00d4aa; cursor: pointer; font-size: 2rem; font-weight: bold; display: flex; align-items: center; justify-content: center; box-shadow: 0 4px 15px rgba(0,0,0,0.5); transition: all 0.2s; line-height: 1;">
                <svg viewBox="0 0 45 45" width="100%" height="100%" style="position:absolute; top:0; left:0; transform: rotate(-90deg); pointer-events:none;">
                    <circle cx="22.5" cy="22.5" r="21.5" fill="none" stroke="rgba(0, 212, 170, 0.2)" stroke-width="2"/>
                    <circle cx="22.5" cy="22.5" r="21.5" fill="none" stroke="#00d4aa" stroke-width="2" stroke-dasharray="135" stroke-dashoffset="0" style="animation: radialCountdown 20s linear forwards;"/>
                </svg>
                <span style="position:relative; z-index:1;">&times;</span>
            </label>
            <div class="splash-title" style="font-size: 1.8rem; font-weight: 800; color: #ffffff; border-bottom: 1px solid rgba(255,255,255,0.15); padding-bottom: 15px; margin-bottom: 25px; text-align: center; text-shadow: 0 2px 4px rgba(0,0,0,0.5);">📖 Dashboard Navigation Guide</div>
            <ul class="splash-list" style="font-size: 1.05rem; color: #f8fafc; padding-left: 20px; line-height: 1.8; margin-bottom: 0; text-shadow: 0 1px 2px rgba(0,0,0,0.8);">
                <li style="margin-bottom: 10px;"><b style="color: #00d4aa;">Overview:</b> Get a high-level summary of your energy footprint, highlighting Critical Flags and High Watch events that need immediate attention.</li>
                <li style="margin-bottom: 10px;"><b style="color: #00d4aa;">Heatmap:</b> Explore a 3D surface mapping of the Deviation Index across all locations and hours. High peaks indicate severe, recurring over-consumption.</li>
                <li style="margin-bottom: 10px;"><b style="color: #00d4aa;">Forecast:</b> Compare forecasted energy demand against actual kW usage. The MAE (Mean Absolute Error) score shows how well the model predicts your baseline.</li>
                <li style="margin-bottom: 10px;"><b style="color: #00d4aa;">Analytics:</b> Dive deep into threshold tuning and score distributions. Adjust the sensitivity sliders to tighten or relax anomaly detection.</li>
                <li style="margin-bottom: 10px;"><b style="color: #00d4aa;">Alerts:</b> Drill down into specific anomalous events. Select any flagged row to see the exact Waste (kWh) and its financial cost impact.</li>
                <li><b style="color: #00d4aa;">Insights:</b> View analytical findings, contamination analysis, and tailored, actionable recommendations to improve building efficiency.</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# CSS INJECTION
# ══════════════════════════════════════════════════════════════════════════════
def inject_css():
    a = C["accent"]
    a2 = C["accent2"]
    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:ital,opsz,wght@0,14..32,300;0,14..32,400;0,14..32,500;0,14..32,600;0,14..32,700;0,14..32,800;0,14..32,900&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── RESET & BASE ── */
*, *::before, *::after {{ box-sizing: border-box; }}
html {{
    /* Fluid typography: scales smoothly from 13px on mobile up to 16px on desktops */
    font-size: clamp(13px, 0.7vw + 10px, 16px);
    scroll-behavior: smooth;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}}
body, [class*="css"] {{
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
    background: {C["app_bg"]} !important;
    color: {C["text_primary"]} !important;
    line-height: 1.65;
}}

/* ── TYPOGRAPHY ── */
h1 {{
    font-size: 1.875rem !important;
    font-weight: 900 !important;
    letter-spacing: -0.04em !important;
    color: {C["text_primary"]} !important;
    line-height: 1.15 !important;
}}
h2 {{
    font-size: 1.4rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.03em !important;
    color: {C["text_primary"]} !important;
}}
h3 {{
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em !important;
    color: {C["text_primary"]} !important;
}}
h4 {{
    font-size: 0.9375rem !important;
    font-weight: 600 !important;
    color: {C["text_primary"]} !important;
}}
p, li, label, span.stMarkdown {{
    font-size: 0.9rem !important;
    color: {C["text_secondary"]} !important;
    line-height: 1.7 !important;
}}
strong, b {{
    color: {C["text_primary"]} !important;
    font-weight: 700 !important;
}}
small {{
    font-size: 0.8rem !important;
    color: {C["text_muted"]} !important;
}}
code {{
    font-family: 'JetBrains Mono', 'Cascadia Code', monospace !important;
    font-size: 0.78rem !important;
    background: {C["surface_raised"]} !important;
    border: 1px solid {C["border_subtle"]} !important;
    border-radius: 4px !important;
    padding: 2px 7px !important;
    color: {a} !important;
}}

/* ── LAYOUT ── */
[data-testid="stAppViewContainer"], [data-testid="stMain"] {{
    background: radial-gradient(circle at top left, rgba(0,212,170,0.12), transparent 30%), {C["app_bg"]} !important;
}}
[data-testid="stHeader"] {{
    background: transparent !important;
    border-bottom: none !important;
}}
.block-container {{
    padding: 1.75rem 2.5rem 4rem !important;
    max-width: 1380px !important;
}}

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {{
    background: {C["sidebar_bg"]} !important;
    border-right: 1px solid {C["sidebar_border"]} !important;
}}
[data-testid="stSidebar"] * {{
    color: {C["text_primary"]} !important;
}}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown span {{
    font-size: 0.8rem !important;
    color: {C["text_muted"]} !important;
}}
[data-testid="stSidebar"] label {{
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
    color: {C["text_muted"]} !important;
}}
[data-testid="stSidebar"] hr {{
    border: none !important;
    border-top: 1px solid {C["border_subtle"]} !important;
    margin: 0.6rem 0 !important;
}}
[data-testid="stSidebar"] [data-testid="stRadio"] label {{
    text-transform: none !important;
    letter-spacing: 0 !important;
    font-size: 0.875rem !important;
    font-weight: 500 !important;
    color: {C["text_primary"]} !important;
}}

/* ── SLIDERS ── */
[data-testid="stSlider"] .thumb {{
    background: {a} !important;
    border: 2px solid {a} !important;
}}
[data-testid="stSlider"] .track-fill {{
    background: {a} !important;
}}

/* ── BUTTONS ── */
.stButton > button, .stDownloadButton > button {{
    font-family: 'Inter', sans-serif !important;
    font-size: 0.875rem !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    padding: 0.5rem 1.25rem !important;
    background: {C["surface"]} !important;
    color: {C["text_primary"]} !important;
    border: 1.5px solid {C["border_default"]} !important;
    box-shadow: {C["shadow_xs"]} !important;
    transition: all 0.18s cubic-bezier(0.4,0,0.2,1) !important;
    letter-spacing: 0.01em !important;
}}
.stButton > button:hover, .stDownloadButton > button:hover {{
    background: rgba(0, 212, 170, 0.05) !important;
    color: {a} !important;
    border-color: {a} !important;
    box-shadow: 0 0 8px rgba(0,212,170,0.3) !important;
    transform: translateY(-1px) !important;
}}
.stButton > button:focus, .stDownloadButton > button:focus {{
    background: {C["surface"]} !important;
    color: {a} !important;
    border-color: {a} !important;
}}
.stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] {{
    background: {a} !important;
    color: {"#ffffff" if is_light else "#0f1117"} !important;
    border-color: {a} !important;
}}

/* ── TABS ── */
[data-testid="stTabs"] > div:first-child {{
    background: {C["surface"]} !important;
    border-radius: 12px !important;
    padding: 4px !important;
    border: 1px solid {C["border_subtle"]} !important;
    display: flex !important;
    gap: 2px !important;
    margin-bottom: 1.75rem !important;
    box-shadow: {C["shadow_xs"]} !important;
}}
button[data-testid="stTab"] {{
    background: transparent !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 7px 16px !important;
    font-size: 0.8375rem !important;
    font-weight: 600 !important;
    color: {C["text_muted"]} !important;
    transition: all 0.18s ease !important;
    letter-spacing: 0.01em !important;
    white-space: nowrap !important;
}}
button[data-testid="stTab"]:hover {{
    background: {C["surface_raised"]} !important;
    color: {C["text_primary"]} !important;
}}
button[data-testid="stTab"][aria-selected="true"] {{
    background: {a} !important;
    color: {"#ffffff" if is_light else "#0f1117"} !important;
    box-shadow: {C["shadow_md"]} !important;
    font-weight: 700 !important;
}}

/* ── DATAFRAMES ── */
[data-testid="stDataFrame"] {{
    border: 1px solid {C["border_subtle"]} !important;
    border-radius: 10px !important;
    overflow: hidden !important;
    box-shadow: {C["shadow_sm"]} !important;
}}
[data-testid="stDataFrame"] th {{
    background: {C["surface_raised"]} !important;
    color: {C["text_muted"]} !important;
    font-weight: 700 !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.07em !important;
    text-transform: uppercase !important;
    padding: 9px 14px !important;
    border-bottom: 1px solid {C["border_default"]} !important;
}}
[data-testid="stDataFrame"] td {{
    font-size: 0.8125rem !important;
    color: {C["text_primary"]} !important;
    padding: 7px 14px !important;
    border-bottom: 1px solid {C["border_subtle"]} !important;
}}
[data-testid="stDataFrame"] tr:last-child td {{
    border-bottom: none !important;
}}

/* ── SELECTS & INPUTS ── */
[data-testid="stSelectbox"] > div > div,
[data-testid="stMultiSelect"] > div > div {{
    background: {C["surface"]} !important;
    border: 1.5px solid {C["border_default"]} !important;
    border-radius: 8px !important;
    color: {C["text_primary"]} !important;
    font-size: 0.875rem !important;
}}
[data-testid="stSelectbox"] > div > div:focus-within,
[data-testid="stMultiSelect"] > div > div:focus-within {{
    border-color: {a} !important;
    box-shadow: 0 0 0 3px {C["accent_glow"]} !important;
}}

/* ── FILE UPLOADER ── */
[data-testid="stFileUploader"] {{
    background: {C["surface"]} !important;
    border: 2px dashed {C["border_default"]} !important;
    border-radius: 12px !important;
    padding: 1.5rem !important;
    transition: all 0.2s ease !important;
    text-align: center !important;
}}
[data-testid="stFileUploader"]:hover {{
    border-color: {a} !important;
    background: {C["surface_raised"]} !important;
}}

/* ── EXPANDERS ── */
[data-testid="stExpander"] {{
    border: 1px solid {C["border_subtle"]} !important;
    border-radius: 10px !important;
    background: {C["surface"]} !important;
    margin-bottom: 0.6rem !important;
}}
[data-testid="stExpander"] > details > summary {{
    font-weight: 600 !important;
    color: {C["text_primary"]} !important;
    padding: 0.75rem 1rem !important;
    cursor: pointer !important;
}}

/* ── METRICS ── */
[data-testid="stMetric"] {{
    background: {C["surface"]} !important;
    border: 1px solid {C["border_subtle"]} !important;
    border-radius: 12px !important;
    padding: 1.1rem 1.25rem !important;
    box-shadow: {C["shadow_sm"]} !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease !important;
}}
[data-testid="stMetric"]:hover {{
    transform: translateY(-2px) !important;
    box-shadow: {C["shadow_md"]} !important;
}}
[data-testid="stMetricLabel"] p {{
    font-size: 0.72rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: {C["text_muted"]} !important;
}}
[data-testid="stMetricValue"] {{
    font-size: 1.6rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.03em !important;
    color: {C["text_primary"]} !important;
}}

/* ── ALERT BOXES ── */
[data-testid="stAlert"] {{
    border-radius: 10px !important;
    font-size: 0.875rem !important;
    font-weight: 500 !important;
    border: 1.5px solid !important;
    padding: 0.875rem 1rem !important;
}}
/* Streamlit generates data-type on the icon span, target the whole alert by class */
div[data-testid="stAlert"]:has(svg[data-testid="stAlertContentIcon"]) {{
    background: {C["info_bg"]} !important;
    border-color: {C["info_border"]} !important;
    color: {C["info"]} !important;
}}

/* ── DIVIDERS ── */
hr {{
    border: none !important;
    border-top: 1px solid {C["border_subtle"]} !important;
    margin: 1.5rem 0 !important;
}}

/* ══════════════════════════════════════════════════════
   GREENGRID COMPONENT LIBRARY
   ══════════════════════════════════════════════════════ */

/* KPI Cards with entrance animation */
@keyframes kpi-entrance {{
    from {{ opacity: 0; transform: translateY(14px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
}}
@keyframes pulse-danger {{
    0%, 100% {{ box-shadow: {C["shadow_md"]}; }}
    50% {{ box-shadow: 0 0 0 4px rgba(220,38,38,0.20), {C["shadow_md"]}; }}
}}

.kpi-card {{
    position: relative;
    background: {C["surface"]};
    border: 1px solid {C["border_subtle"]};
    border-radius: 14px;
    padding: 1.3rem 1rem 1rem;
    text-align: center;
    box-shadow: {C["shadow_sm"]};
    transition: transform 0.22s cubic-bezier(0.4,0,0.2,1), box-shadow 0.22s ease;
    overflow: hidden;
    cursor: default;
    min-height: 120px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    animation: kpi-entrance 0.4s ease both;
}}
.kpi-card::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3.5px;
    background: var(--kpi-accent, {a});
    border-radius: 14px 14px 0 0;
}}
.kpi-card::after {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background: radial-gradient(ellipse at 50% 0%, var(--kpi-glow, {C["accent_glow"]}), transparent 65%);
    pointer-events: none;
    border-radius: 14px;
}}
.kpi-card:hover {{
    transform: translateY(-5px) scale(1.01);
    box-shadow: {C["shadow_lg"]};
    border-color: var(--kpi-accent, {a});
}}
.kpi-card.danger-pulse {{
    animation: kpi-entrance 0.4s ease both, pulse-danger 2.5s ease-in-out 0.5s infinite;
}}
.kpi-value {{
    font-size: 2.1rem;
    font-weight: 900;
    line-height: 1;
    margin-bottom: 5px;
    color: var(--kpi-accent, {a});
    letter-spacing: -0.045em;
    font-variant-numeric: tabular-nums;
    position: relative;
    z-index: 1;
}}
.kpi-label {{
    font-size: 0.67rem;
    font-weight: 700;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    color: {C["text_muted"]};
    margin-bottom: 3px;
    position: relative;
    z-index: 1;
}}
.kpi-unit {{
    font-size: 0.67rem;
    color: {C["text_placeholder"]};
    font-weight: 500;
    position: relative;
    z-index: 1;
}}

/* Section headers with accent line */
.section-header {{
    font-size: 1rem;
    font-weight: 800;
    letter-spacing: -0.01em;
    color: {C["text_primary"]};
    padding: 0.6rem 0 0.6rem 0.85rem;
    margin-bottom: 1.1rem;
    border-left: 4px solid {a};
    background: linear-gradient(90deg, {C["accent_glow"]}, transparent);
    border-radius: 0 8px 8px 0;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}}

/* Dataset banners */
.banner-synth, .banner-real, .banner-custom {{
    border-radius: 10px;
    padding: 0.75rem 1.25rem;
    font-size: 0.8125rem;
    font-weight: 700;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
    box-shadow: {C["shadow_xs"]};
    letter-spacing: 0.015em;
    text-transform: uppercase;
}}
.banner-synth {{
    background: linear-gradient(135deg, {"#eff6ff" if is_light else "#0c1a2e"}, {C["surface"]});
    border: 1.5px solid {a};
    color: {a};
}}
.banner-real {{
    background: linear-gradient(135deg, {"#fffbeb" if is_light else "#1c1500"}, {C["surface"]});
    border: 1.5px solid {C["warning"]};
    color: {C["warning"]};
}}
.banner-custom {{
    background: linear-gradient(135deg, {"#f0fdf4" if is_light else "#052e1d"}, {C["surface"]});
    border: 1.5px solid {C["success"]};
    color: {C["success"]};
}}

/* Privacy notice */
.privacy-notice {{
    background: linear-gradient(135deg, {C["info_bg"]}, {C["surface"]});
    border: 1.5px solid {C["info_border"]};
    border-radius: 10px;
    padding: 0.875rem 1rem;
    font-size: 0.8125rem;
    color: {C["info"]};
    font-weight: 600;
    margin-bottom: 0.75rem;
    display: flex;
    align-items: flex-start;
    gap: 0.5rem;
}}

/* Insight card */
.insight-card {{
    background: {C["surface"]};
    border: 1px solid {C["border_subtle"]};
    border-radius: 12px;
    padding: 1.1rem 1.25rem;
    margin-bottom: 0.75rem;
    box-shadow: {C["shadow_sm"]};
    border-left: 4px solid var(--insight-color, {a});
    transition: transform 0.18s ease, box-shadow 0.18s ease;
    animation: kpi-entrance 0.4s ease both;
}}
.insight-card:hover {{
    transform: translateX(3px);
    box-shadow: {C["shadow_md"]};
}}
.insight-title {{
    font-size: 0.875rem;
    font-weight: 700;
    color: {C["text_primary"]};
    margin-bottom: 0.3rem;
}}
.insight-body {{
    font-size: 0.8125rem;
    color: {C["text_secondary"]};
    line-height: 1.6;
}}

/* Alert table rows */
.alert-row-shutoff {{
    background: {"rgba(220,38,38,0.07)" if is_light else "rgba(248,113,113,0.12)"} !important;
    border-left: 3px solid {C["danger"]} !important;
}}
.alert-row-alert {{
    background: {"rgba(217,119,6,0.07)" if is_light else "rgba(251,191,36,0.10)"} !important;
    border-left: 3px solid {C["warning"]} !important;
}}
.alert-row-log {{
    background: transparent !important;
}}

/* WhatsApp alert block */
.wa-alert {{
    background: {C["surface_raised"]};
    border: 1.5px solid {C["success_border"]};
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    line-height: 1.85;
    white-space: pre-wrap;
    word-break: break-word;
    color: {C["text_primary"]};
    box-shadow: {C["shadow_sm"]};
}}

/* Facility type badge */
.facility-badge {{
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: {C["accent_glow"]};
    color: {a};
    border: 1px solid {a};
    border-radius: 100px;
    padding: 3px 12px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}}

/* Footer */
.gg-footer {{
    margin-top: 3rem;
    padding: 1.25rem 2rem;
    background: {C["footer_bg"]};
    border-top: 1px solid {C["footer_border"]};
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 0.5rem;
}}
.gg-footer-brand {{
    font-size: 0.8rem;
    font-weight: 700;
    color: {C["footer_text"]};
    letter-spacing: 0.02em;
}}
.gg-footer-meta {{
    font-size: 0.75rem;
    color: {C["footer_text"]};
}}

/* Stagger animation delays for KPI cards */
div[data-testid="stColumn"]:nth-child(1) .kpi-card {{ animation-delay: 0.05s; }}
div[data-testid="stColumn"]:nth-child(2) .kpi-card {{ animation-delay: 0.10s; }}
div[data-testid="stColumn"]:nth-child(3) .kpi-card {{ animation-delay: 0.15s; }}
div[data-testid="stColumn"]:nth-child(4) .kpi-card {{ animation-delay: 0.20s; }}
div[data-testid="stColumn"]:nth-child(5) .kpi-card {{ animation-delay: 0.25s; }}
div[data-testid="stColumn"]:nth-child(6) .kpi-card {{ animation-delay: 0.30s; }}

/* Plotly chart backgrounds always transparent */
.js-plotly-plot .plotly,
.js-plotly-plot .plotly .bg {{ fill: transparent !important; background: transparent !important; }}

/* Scrollbar styling */
::-webkit-scrollbar {{ width: 6px; height: 6px; }}
::-webkit-scrollbar-track {{ background: {C["surface"]}; }}
::-webkit-scrollbar-thumb {{ background: {C["border_default"]}; border-radius: 3px; }}
::-webkit-scrollbar-thumb:hover {{ background: {a}; }}

/* ── ⓘ Tooltip (glossary) ── */
.gg-tip {{
    display: inline-flex;
    align-items: center;
    gap: 3px;
    position: relative;
    cursor: default;
    font-weight: inherit;
}}
.gg-tip .tip-icon {{
    display: inline-block;
    width: 14px;
    height: 14px;
    border-radius: 50%;
    background: {C["accent"]}22;
    border: 1px solid {C["accent"]}66;
    color: {C["accent"]};
    font-size: 9px;
    font-weight: 700;
    line-height: 14px;
    text-align: center;
    font-style: normal;
    transition: background 0.18s;
    flex-shrink: 0;
    user-select: none;
    font-family: 'Inter', sans-serif;
}}
.gg-tip:hover .tip-icon {{
    background: {C["accent"]}44;
}}
.gg-tip .tip-box {{
    display: none;
    position: absolute;
    bottom: calc(100% + 8px);
    left: 50%;
    transform: translateX(-50%);
    background: {C["surface_raised"]};
    border: 1px solid {C["border_default"]};
    border-radius: 8px;
    padding: 8px 11px;
    font-size: 0.72rem;
    line-height: 1.55;
    color: {C["text_secondary"]};
    width: max-content;
    max-width: 240px;
    z-index: 9999;
    box-shadow: {C["shadow_md"]};
    white-space: normal;
    pointer-events: none;
}}
.gg-tip:hover .tip-box {{
    display: block;
}}
.gg-tip .tip-box::after {{
    content: '';
    position: absolute;
    top: 100%;
    left: 50%;
    transform: translateX(-50%);
    border: 6px solid transparent;
    border-top-color: {C["border_default"]};
}}

/* ── Keyword highlights ── */
.kw {{
    display: inline;
    position: relative;
    cursor: help;
    font-weight: 600;
    color: {C["accent"]};
    border-bottom: 1.5px dotted {C["accent"]}88;
    padding: 0 1px;
    transition: color 0.15s;
}}
.kw:hover {{ color: {C["accent_hover"]}; }}
.kw .kw-tip {{
    display: none;
    position: absolute;
    bottom: calc(100% + 8px);
    left: 50%;
    transform: translateX(-50%);
    background: {C["surface_raised"]};
    border: 1px solid {C["border_default"]};
    border-radius: 8px;
    padding: 7px 11px;
    font-size: 0.7rem;
    line-height: 1.5;
    color: {C["text_secondary"]};
    font-weight: 400;
    width: max-content;
    max-width: 220px;
    z-index: 9999;
    box-shadow: {C["shadow_md"]};
    white-space: normal;
    pointer-events: none;
}}
.kw:hover .kw-tip {{ display: block; }}
.kw .kw-tip::after {{
    content: '';
    position: absolute;
    top: 100%; left: 50%;
    transform: translateX(-50%);
    border: 5px solid transparent;
    border-top-color: {C["border_default"]};
}}

/* ── Sidebar slide-in animation ── */
section[data-testid="stSidebar"] > div {{
    animation: sidebarIn 0.35s cubic-bezier(.22,.68,0,1.2) both;
}}
@keyframes sidebarIn {{
    from {{ transform: translateX(-24px); opacity: 0; }}
    to   {{ transform: translateX(0);     opacity: 1; }}
}}

/* ── Light theme contrast fixes ── */
.section-header {{
    color: {C["text_primary"]} !important;
}}
div[data-testid="stCaption"] > p {{
    color: {C["text_secondary"]} !important;
    font-size: 0.8rem;
}}

/* ── RESPONSIVE IMPROVEMENTS ── */
@media (max-width: 1200px) {{
    .block-container {{
        padding: 1rem 1rem 3rem !important;
    }}
    .gg-header-wrap {{
        padding: 1.2rem 1rem 0.9rem !important;
    }}
    .gg-header-title {{
        font-size: 1.55rem !important;
    }}
    .gg-header-sub {{
        font-size: 0.82rem !important;
    }}
    .gg-wave-svg {{
        width: 55% !important;
        opacity: 0.06 !important;
    }}
}}

@media (max-width: 980px) {{
    [data-testid="stHorizontalBlock"] {{
        flex-direction: column !important;
        gap: 0.85rem !important;
    }}
    [data-testid="stHorizontalBlock"] > div {{
        width: 100% !important;
        max-width: 100% !important;
    }}
    .gg-header-pills {{
        gap: 0.4rem !important;
    }}
    .gg-pill {{
        font-size: 0.64rem !important;
        padding: 0.22rem 0.6rem !important;
    }}
    .gg-stat-row {{
        flex-wrap: wrap !important;
        gap: 0.8rem 1rem !important;
    }}
    .kpi-card {{
        min-height: 110px !important;
        padding: 1rem 0.9rem !important;
    }}
    .kpi-value {{
        font-size: 1.7rem !important;
    }}
    .wa-alert {{
        padding: 1rem !important;
        font-size: 0.74rem !important;
    }}
}}

@media (max-width: 640px) {{
    .block-container {{
        padding: 1rem 0.5rem 4rem !important;
    }}
    .gg-header-title {{
        font-size: 1.35rem !important;
    }}
    .gg-header-sub {{
        font-size: 0.85rem !important;
    }}
    .gg-stat-item {{
        min-width: 100px !important;
    }}
    .gg-stat-val {{
        font-size: 1.0rem !important;
    }}
    .gg-stat-lbl {{
        font-size: 0.7rem !important;
    }}
    .section-header {{
        font-size: 1.0rem !important;
        padding-left: 0.5rem !important;
    }}
    .kpi-label {{
        font-size: 0.7rem !important;
    }}
    .kpi-unit {{
        font-size: 0.6rem !important;
    }}
    .gg-footer {{
        padding: 1rem !important;
        flex-direction: column !important;
        align-items: flex-start !important;
    }}
    /* Tabs: allow horizontal scroll on mobile so all tabs are reachable */
    [data-testid="stTabs"] > div:first-child {{
        overflow-x: auto !important;
        flex-wrap: nowrap !important;
        -webkit-overflow-scrolling: touch !important;
        scrollbar-width: none !important;
        padding-bottom: 2px !important;
    }}
    [data-testid="stTabs"] > div:first-child::-webkit-scrollbar {{ display: none; }}
    button[data-testid="stTab"] {{
        min-width: max-content !important;
        padding: 6px 12px !important;
        font-size: 0.78rem !important;
    }}
    /* Selectboxes: allow scroll-through so page scrolls when finger starts on dropdown */
    [data-testid="stSelectbox"],
    [data-testid="stMultiSelect"] {{
        touch-action: pan-y !important;
    }}
    /* Prevent sidebar slider from stealing page scroll */
    [data-testid="stSlider"] {{
        touch-action: pan-y !important;
    }}
    /* Dataframe horizontal scroll on mobile */
    [data-testid="stDataFrame"] > div {{
        overflow-x: auto !important;
        -webkit-overflow-scrolling: touch !important;
    }}
    /* KPI cards: 2 per row on phone */
    div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] {{
        min-width: calc(50% - 0.5rem) !important;
    }}
    /* 3D chart: cap height on mobile */
    .js-plotly-plot .plot-container.plotly {{
        max-height: 300px;
    }}
}}
</style>
""", unsafe_allow_html=True)

inject_css()


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

FACILITY_TYPES = {
    "🏠 Home":         {"contamination": 0.05, "label": "Residential",   "icon": "🏠"},
    "🏫 School":       {"contamination": 0.04, "label": "School",        "icon": "🏫"},
    "🎓 College":      {"contamination": 0.04, "label": "Campus",        "icon": "🎓"},
    "🏢 Office":       {"contamination": 0.04, "label": "Office",        "icon": "🏢"},
    "🏥 Hospital":     {"contamination": 0.03, "label": "Healthcare",    "icon": "🏥"},
    "🏭 Industrial":   {"contamination": 0.06, "label": "Industrial",    "icon": "🏭"},
    "🛍 Retail":       {"contamination": 0.05, "label": "Retail",        "icon": "🛍"},
    "🏨 Hospitality":  {"contamination": 0.05, "label": "Hospitality",   "icon": "🏨"},
    "🏗 Mixed-Use":    {"contamination": 0.05, "label": "Mixed-Use",     "icon": "🏗"},
    "⚡ Other":        {"contamination": 0.05, "label": "General",       "icon": "⚡"},
}

PII_COLUMN_KEYWORDS = [
    "name", "email", "phone", "mobile", "address", "ssn", "id", "dob",
    "birth", "gender", "age", "race", "religion", "ip", "mac", "passport",
    "aadhaar", "pan", "card", "account", "salary", "lat", "lon", "gps",
]


# ── Glossary of key terms shown as ⓘ tooltips ───────────────────────────────
GLOSSARY = {
    "Deviation Index":    "Score 0–1 measuring how far a reading deviates from its expected baseline. Higher = more unusual.",
    "Critical Flag":      "Highest-severity event. Power reading is far above expected when the space is empty.",
    "High Watch":         "Medium-severity event. Reading is above normal range — monitor but not urgent.",
    "Normal":             "Reading is within expected range. No action needed.",
    "Baseline":           "Expected average power for this location at this time, learned from historical patterns.",
    "Load (kW)":          "Instantaneous electrical power measured in kilowatts (kW).",
    "Load Forecaster":    "Predicts future energy demand based on historical usage trends.",
    "Waste (kWh)":        "Excess energy consumed beyond the expected baseline, in kilowatt-hours.",
    "Cost Saved":         "Estimated savings: Waste kWh × ₹7.5/kWh (standard institutional tariff).",
    "CO₂ Avoided":        "Carbon emission saved: Waste kWh × 0.82 kg CO₂e (CEA 2023 grid factor).",
    "Energy Saved":       "Total excess energy that was detected and flagged for corrective action.",
    "Contamination":      "Expected fraction of anomalies in your dataset. Adjusted per facility type.",
    "Confidence Band":    "Likely range the forecast could land in — narrower is more confident.",
    "MAE":                "Mean Absolute Error — average gap between predicted and actual load (kW).",
    "RMSE":               "Root Mean Square Error — penalises large prediction errors more heavily than MAE.",
    "Threshold":          "Score boundary that separates normal readings from flagged events.",
    "Pattern Engine":     "Detects statistically unusual readings by comparing each point to its expected range.",
    "High Watch level":   "Events with deviation scores above this value are flagged for attention.",
    "Critical Flag level":"Events above this score in unoccupied spaces trigger an immediate alert.",
    "Occupancy":          "Whether the space was in use. Unoccupied high-load events are the most wasteful.",
    "Facility Type":      "Your building category. Sensitivity of detection is calibrated per facility profile.",
    "Notification Preview":"Sample alert message generated for the selected event.",
}

def tip(term: str, label: str = "") -> str:
    """Return HTML for a labelled term with ⓘ tooltip from the GLOSSARY."""
    txt = GLOSSARY.get(term, GLOSSARY.get(label, ""))
    display = label or term
    if not txt:
        return display
    return (f"<span class='gg-tip'>{display} "
            f"<i class='tip-icon'>i</i>"
            f"<span class='tip-box'>{txt}</span>"
            f"</span>")


# Inline keyword with dotted underline + hover definition
KW_DEFS = {
    "deviation index":   "Score 0–1: how far a reading strays from its expected value.",
    "critical flag":     "Highest-severity alert — unoccupied space drawing far above baseline.",
    "high watch":        "Medium-severity flag — reading above normal, needs monitoring.",
    "baseline":          "Expected power learned from historical patterns for this location + time.",
    "waste (kwh)":       "Excess energy used above the baseline, in kilowatt-hours.",
    "co2e":              "Carbon dioxide equivalent — a common measure of greenhouse gas impact.",
    "kw":                "Kilowatt — unit of instantaneous electrical power (1 kW = 1000 W).",
    "kwh":               "Kilowatt-hour — unit of energy (1 kWh = 1 kW used for 1 hour).",
    "mae":               "Mean Absolute Error — average gap between forecast and actual values.",
    "rmse":              "Root Mean Square Error — penalises large errors more than MAE.",
    "f1":                "F1 Score — harmonic mean of precision and recall (max = 1.0).",
    "occupancy":         "Whether the space was in use at the time of the reading.",
    "contamination":     "Expected fraction of unusual readings in the dataset.",
    "hvac":              "Heating, Ventilation & Air Conditioning — typically the largest energy consumer.",
    "pir sensor":        "Passive Infrared sensor — detects body heat to determine room occupancy.",
    "threshold":         "Score boundary that separates normal readings from flagged events.",
}

def kw(word: str, definition: str = "") -> str:
    """Wrap a keyword in a dotted-underline highlight with a hover definition card."""
    defn = definition or KW_DEFS.get(word.lower(), "")
    if not defn:
        return f"<span class='kw'>{word}</span>"
    return (f"<span class='kw'>{word}"
            f"<span class='kw-tip'>{defn}</span>"
            f"</span>")


def make_kpi_html(value, label, unit="", color=None, glow=None, tooltip="", pulse=False):
    col   = color or C["kpi_primary"]
    glow_ = glow  or C["accent_glow"]
    extra = 'danger-pulse' if pulse else ''
    tip_html = ""
    if tooltip:
        tip_html = f"<span class='tip-icon' style='margin-left:4px; vertical-align:middle;' title='{tooltip}'>i</span>"
    return f"""<div class="kpi-card {extra}" style="--kpi-accent:{col}; --kpi-glow:{glow_};">
  <div class="kpi-value">{value}</div>
  <div class="kpi-label">{label}{tip_html}</div>
  <div class="kpi-unit">{unit}</div>
</div>"""


def action_icon(action: str) -> str:
    return {"critical-flag": "🔴", "high-watch": "🟠", "normal": "🟢"}.get(action, "⚪")


def normalize_action(action: str) -> str:
    """Map legacy action labels to new terminology."""
    return {"auto-shutoff": "critical-flag", "alert": "high-watch", "log": "normal"}.get(action, action)


def apply_thresholds(df: pd.DataFrame, high_t: float, alert_t: float) -> pd.DataFrame:
    df = df.copy()

    def decide(row):
        s   = float(row.get("deviation_index", row.get("waste_risk_score", row.get("anomaly_score", 0))))
        occ = bool(row.get("occupied", 1))
        if s > high_t:
            return "critical-flag" if not occ else "high-watch"
        elif s > alert_t:
            return "high-watch"
        return "normal"

    df["action"] = df.apply(decide, axis=1)

    if "timestamp" in df.columns and len(df) > 1:
        diff = df["timestamp"].drop_duplicates().sort_values().diff().median()
        interval_h = diff.total_seconds() / 3600 if pd.notnull(diff) else (5/60)
        if interval_h == 0:
            interval_h = 5/60
    else:
        interval_h = 5/60

    df["waste_kwh"]       = (df["power_kw"] - df["expected_load"]).clip(lower=0) * interval_h
    df["cost_saved_inr"]  = df["waste_kwh"] * 7.5
    df["co2e_avoided_kg"] = df["waste_kwh"] * 0.82
    return df


def pii_check(columns: list) -> list:
    flags = []
    for col in columns:
        low = col.lower()
        for kw in PII_COLUMN_KEYWORDS:
            if kw in low:
                flags.append(col)
                break
    return flags


def plotly_layout(height=340, margin=None, legend_y=1.05):
    m = margin or dict(t=30, b=40, l=50, r=20)
    return dict(
        template=C["plotly_tmpl"],
        paper_bgcolor=C["chart_paper"],
        plot_bgcolor=C["chart_plot"],
        font=dict(family="Inter, sans-serif", color=C["text_secondary"], size=11),
        height=height,
        margin=m,
        legend=dict(orientation="h", y=legend_y, font=dict(size=10)),
        xaxis=dict(showgrid=False, tickfont=dict(size=9), color=C["text_muted"]),
        yaxis=dict(showgrid=True, gridcolor=C["grid_color"], tickfont=dict(size=9), color=C["text_muted"]),
    )


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="Generating fresh sensor data…")
def load_synthetic_data():
    from data_loader import generate_synthetic_data as generate_dataset
    from anomaly_detector import run_isolation_forest
    from decision_engine import apply_decision_engine
    raw    = generate_dataset(output_path=None)
    scored = run_isolation_forest(raw)
    final  = apply_decision_engine(scored)
    if not final.empty:
        float_cols = final.select_dtypes(include=["float64"]).columns
        final[float_cols] = final[float_cols].astype("float32")
    return final


@st.cache_data(persist="disk", show_spinner="Loading household benchmark data…")
def load_real_data():
    from data_loader import load_uci_data
    from anomaly_detector import run_isolation_forest_real
    from decision_engine import apply_decision_engine
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    parquet_path = PROCESSED_DIR / "uci_scored_real.parquet"
    uci_df = load_uci_data()
    if not parquet_path.exists():
        run_isolation_forest_real(uci_df, str(parquet_path), str(PROCESSED_DIR / "real_data_anomaly_timeline.png"))
    scored = pd.read_parquet(parquet_path)
    if "expected_load" not in scored.columns:
        scored["expected_load"] = scored.groupby(["circuit","hour","is_weekend"])["power_kw"].transform("median")
    if "action" not in scored.columns:
        scored["occupied"] = 1
        scored = apply_decision_engine(scored)
    if not scored.empty:
        float_cols = scored.select_dtypes(include=["float64"]).columns
        scored[float_cols] = scored[float_cols].astype("float32")
    return scored


def load_forecast_data(source: str):
    path = PROCESSED_DIR / ("greengrid_synthetic_forecast.csv" if source == "Synthetic" else "uci_prophet_forecast.csv")
    if path.exists():
        return pd.read_csv(path, parse_dates=["ds"])
    return None


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    # Branding
    st.markdown(f"""
    <div style="padding:0.5rem 0 0.1rem;">
        <div style="font-size:1.25rem; font-weight:900; color:{C['text_primary']}; letter-spacing:-0.03em;">
            ⚡ GreenGrid
        </div>
        <div style="font-size:0.775rem; color:{C['text_muted']}; font-weight:500; margin-top:2px;">
            Energy Intelligence System
        </div>
    </div>
    """, unsafe_allow_html=True)


    st.divider()

    # Dataset selector — Synthetic default
    data_source = st.radio(
        "Dataset",
        ["Synthetic", "Real UCI", "Custom Upload"],
        index=0,
        help="Synthetic: 8-location campus data with labeled anomalies.\n\nReal UCI: Household benchmark, 2006–2010.\n\nCustom Upload: Analyze your own energy CSV.",
    )

    st.divider()

    # Custom upload section
    if data_source == "Custom Upload":
        st.markdown(f"<div style='font-size:0.75rem; font-weight:700; text-transform:uppercase; letter-spacing:0.08em; color:{C['text_muted']}; margin-bottom:0.4rem;'>{tip('Facility Type')}</div>", unsafe_allow_html=True)
        facility_type = st.selectbox(
            "Facility",
            list(FACILITY_TYPES.keys()),
            label_visibility="collapsed",
            help="Select your facility type to optimize the analysis baseline.",
        )

        st.markdown(f"""
        <div class="privacy-notice">
            🔒 Your data is processed locally in your browser session only.
            It is never stored, transmitted, or shared.
        </div>""", unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Drop CSV file here",
            type=["csv"],
            label_visibility="collapsed",
            help="Max 50 MB. Accepted format: CSV with timestamp and energy columns.",
        )

        if uploaded_file is not None:
            # File size check
            file_size_mb = uploaded_file.size / (1024 * 1024)
            if file_size_mb > 50:
                st.error(f"File is {file_size_mb:.1f} MB. Maximum allowed is 50 MB.")
            else:
                try:
                    @st.cache_data
                    def read_upload(f):
                        return pd.read_csv(f)

                    preview_df = read_upload(uploaded_file)
                    cols = list(preview_df.columns)

                    # PII check
                    pii_cols = pii_check(cols)
                    if pii_cols:
                        st.warning(f"⚠️ Potential personal data detected in columns: **{', '.join(pii_cols)}**. Please ensure sensitive information is excluded before analysis.")

                    st.markdown(f"<div style='font-size:0.75rem; font-weight:700; text-transform:uppercase; letter-spacing:0.08em; color:{C['text_muted']}; margin-top:0.5rem;'>Map Columns</div>", unsafe_allow_html=True)

                    ts_idx = next((i for i,c in enumerate(cols) if any(k in c.lower() for k in ["time","date","datetime"])), 0)
                    kw_idx = next((i for i,c in enumerate(cols) if any(k in c.lower() for k in ["power","kw","kwh","load","watt","energy","consumption"])), 0)

                    ts_col  = st.selectbox("Timestamp *", cols, index=ts_idx)
                    kw_col  = st.selectbox("Power / Energy (kW) *", cols, index=kw_idx)
                    loc_col = st.selectbox("Location / Meter (optional)", ["None — Single Meter"] + cols)
                    occ_col = st.selectbox("Occupancy (optional)", ["None — Assume Occupied"] + cols)

                    # Data preview
                    with st.expander("Preview data (first 5 rows)"):
                        st.dataframe(preview_df.head(5), use_container_width=True)

                    if st.button("Analyze Dataset"):
                        with st.spinner("Validating and processing…"):
                            processed = preview_df.copy()
                            processed = processed.rename(columns={ts_col: "timestamp", kw_col: "power_kw"})

                            if loc_col != "None — Single Meter":
                                processed = processed.rename(columns={loc_col: "room"})
                            else:
                                processed["room"] = facility_type.split(" ", 1)[-1] if " " in facility_type else "Facility"

                            if occ_col != "None — Assume Occupied":
                                processed = processed.rename(columns={occ_col: "occupied"})
                            else:
                                processed["occupied"] = 1

                            processed["timestamp"] = pd.to_datetime(processed["timestamp"], errors="coerce")
                            bad_ts = processed["timestamp"].isna().sum()
                            if bad_ts > 0:
                                st.warning(f"{bad_ts} rows had unparseable timestamps and were dropped.")
                            processed = processed.dropna(subset=["timestamp", "power_kw"])
                            processed["power_kw"] = pd.to_numeric(processed["power_kw"], errors="coerce").fillna(0)

                            # Value range validation
                            negative_rows = (processed["power_kw"] < 0).sum()
                            extreme_rows  = (processed["power_kw"] > 10000).sum()
                            if negative_rows > 0:
                                st.warning(f"{negative_rows} rows have negative power values — these were clipped to 0.")
                                processed["power_kw"] = processed["power_kw"].clip(lower=0)
                            if extreme_rows > 0:
                                st.warning(f"{extreme_rows} rows exceed 10,000 kW. Please verify your data units.")

                            if len(processed) < 20:
                                st.error("Dataset too small (< 20 valid rows). Please provide more data.")
                            else:
                                from anomaly_detector import run_isolation_forest
                                from decision_engine import apply_decision_engine

                                # Use facility-specific contamination
                                contam = FACILITY_TYPES[facility_type]["contamination"]
                                # Temporarily patch contamination via a wrapper
                                raw_scored = run_isolation_forest(processed, default_contamination=contam)
                                final_up   = apply_decision_engine(raw_scored)

                                st.session_state["custom_data"]     = final_up
                                st.session_state["custom_facility"] = facility_type
                                st.success(f"Ready! {len(final_up):,} records analyzed.")

                except Exception as e:
                    st.error(f"Could not read file: {e}")
        st.divider()

    # Thresholds
    st.markdown(f"<div style='font-size:0.75rem; font-weight:700; text-transform:uppercase; letter-spacing:0.08em; color:{C['text_muted']};'>Detection Thresholds</div>", unsafe_allow_html=True)
    high_threshold  = st.slider("Critical Flag level", 0.50, 1.00, 0.85, 0.01,
                                 help="Score boundary (0-1). Unoccupied readings above this are flagged as critical. Higher = stricter.")
    alert_threshold = st.slider("High Watch level",    0.30, high_threshold - 0.01, 0.72, 0.01,
                                 help="Score boundary (0-1). Readings above this get a 'High Watch' flag for review.")
    st.divider()

    # Date filter
    st.markdown(f"<div style='font-size:0.75rem; font-weight:700; text-transform:uppercase; letter-spacing:0.08em; color:{C['text_muted']};'>Date Range</div>", unsafe_allow_html=True)
    max_days = 42 if data_source == "Synthetic" else 180
    filter_days = st.slider("Show last N days", 1, max_days, min(7, max_days))
    st.divider()

    # Provenance
    if data_source == "Synthetic":
        st.markdown(f"""
        <div style='background:{C['success_bg']};border:1px solid {C['success_border']};border-radius:8px;padding:0.7rem 0.9rem;font-size:0.78rem;color:{C['success']};'>
        <b>Benchmark dataset.</b> 8 campus locations &middot; 6 weeks &middot; 5-min intervals<br>
        {kw('F1','F1 Score: harmonic mean of precision and recall (max=1.0).')} <b>0.97</b> &nbsp;···&nbsp;
        {kw('Threshold','Score boundary separating normal from flagged.')} 0.85 / 0.72
        </div>""", unsafe_allow_html=True)
    elif data_source == "Real UCI":
        st.markdown(f"""
        <div style='background:{C['warning_bg']};border:1px solid {C['warning_border']};border-radius:8px;padding:0.7rem 0.9rem;font-size:0.78rem;color:{C['warning']};'>
        <b>Household benchmark</b> (UCI). No {kw('Occupancy','Whether the space was occupied at time of reading.')} sensor available.
        Critical Flag is a reference concept only for this dataset.
        </div>""", unsafe_allow_html=True)
    else:
        if "custom_data" in st.session_state:
            fac = st.session_state.get("custom_facility", "")
            st.success(f"**Custom dataset active.** Facility: {fac}")


# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
location_col = "room"

if data_source == "Synthetic":
    with st.spinner("Loading benchmark data…"):
        df_raw = load_synthetic_data()
    # Normalise action labels
    if "action" in df_raw.columns:
        df_raw = df_raw.copy()
        df_raw["action"] = df_raw["action"].map(normalize_action)
    # Normalise score column name
    if "anomaly_score" in df_raw.columns:
        df_raw = df_raw.rename(columns={"anomaly_score": "deviation_index"})
    elif "waste_risk_score" in df_raw.columns:
        df_raw = df_raw.rename(columns={"waste_risk_score": "deviation_index"})
    banner_class  = "banner-synth"
    banner_icon   = "⚡"
    data_label    = "SYNTHETIC BENCHMARK — 8 Locations · 6 Weeks · 5-min Intervals · Validated Anomalies"
    primary_color = C["accent"]
    heatmap_cs    = C["heatmap_synth"]

elif data_source == "Real UCI":
    with st.spinner("Loading household benchmark…"):
        df_raw = load_real_data()
    location_col = "circuit"
    if "action" in df_raw.columns:
        df_raw = df_raw.copy()
        df_raw["action"] = df_raw["action"].map(normalize_action)
    if "anomaly_score" in df_raw.columns:
        df_raw = df_raw.rename(columns={"anomaly_score": "deviation_index"})
    elif "waste_risk_score" in df_raw.columns:
        df_raw = df_raw.rename(columns={"waste_risk_score": "deviation_index"})
    min_date = df_raw["timestamp"].min().strftime("%b %Y")
    max_date = df_raw["timestamp"].max().strftime("%b %Y")
    banner_class  = "banner-real"
    banner_icon   = "🏠"
    data_label    = f"HOUSEHOLD BENCHMARK — {len(df_raw):,} records · {min_date} to {max_date}"
    primary_color = C["warning"]
    heatmap_cs    = C["heatmap_real"]

elif data_source == "Custom Upload":
    if "custom_data" not in st.session_state:
        st.markdown(f"""
        <div style="
            display:flex; flex-direction:column; align-items:center; justify-content:center;
            padding:4rem 2rem; text-align:center; gap:1rem;
        ">
            <div style="font-size:3rem;">📂</div>
            <div style="font-size:1.25rem; font-weight:800; color:{C['text_primary']};">Upload Your Energy Dataset</div>
            <div style="font-size:0.9rem; color:{C['text_muted']}; max-width:420px; line-height:1.7;">
                Select <strong>Custom Upload</strong> in the sidebar, choose your facility type,
                and upload a CSV file to receive instant energy analytics.
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    df_raw       = st.session_state["custom_data"]
    fac_key      = st.session_state.get("custom_facility", "⚡ Other")
    fac_label    = FACILITY_TYPES.get(fac_key, {}).get("label", "Custom")
    fac_icon     = FACILITY_TYPES.get(fac_key, {}).get("icon", "⚡")
    if "action" in df_raw.columns:
        df_raw = df_raw.copy()
        df_raw["action"] = df_raw["action"].map(normalize_action).fillna(df_raw["action"])
    if "anomaly_score" in df_raw.columns:
        df_raw = df_raw.rename(columns={"anomaly_score": "deviation_index"})
    elif "waste_risk_score" in df_raw.columns:
        df_raw = df_raw.rename(columns={"waste_risk_score": "deviation_index"})
    min_date = df_raw["timestamp"].min().strftime("%b %Y")
    max_date = df_raw["timestamp"].max().strftime("%b %Y")
    banner_class  = "banner-custom"
    banner_icon   = fac_icon
    data_label    = f"{fac_label.upper()} — {len(df_raw):,} records · {min_date} to {max_date}"
    primary_color = C["success"]
    heatmap_cs    = C["heatmap_synth"]


# Ensure deviation_index column exists
if "deviation_index" not in df_raw.columns:
    for alt in ["waste_risk_score", "anomaly_score"]:
        if alt in df_raw.columns:
            df_raw = df_raw.rename(columns={alt: "deviation_index"})
            break

# Apply thresholds
df = apply_thresholds(df_raw, high_threshold, alert_threshold)
# Date filter
max_ts     = df["timestamp"].max()
cutoff_ts  = max_ts - timedelta(days=filter_days)
df_f       = df[df["timestamp"] >= cutoff_ts].copy()


st.markdown(
    f"""
    <div class="{banner_class}">{banner_icon} {data_label}</div>
    """,
    unsafe_allow_html=True,
)

st.caption("Tip: adjust the sidebar controls to tune sensitivity or switch datasets without losing context.")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Overview",
    "Heatmap",
    "Forecast",
    "Analytics",
    "Advanced Analytics",
    "Alerts",
    "Insights",
])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
with tab1:
    n_loc      = df_f[location_col].nunique() if location_col in df_f.columns else 1
    ac         = df_f["action"].value_counts()
    n_critical = int(ac.get("critical-flag", 0))
    n_watch    = int(ac.get("high-watch", 0))
    
    # Permanent systemic check: prevent silent zero-impact bugs
    events     = len(df_f[df_f["action"] != "normal"])
    waste      = float(df_f["waste_kwh"].sum())
    if events > 0 and waste == 0:
        st.error("⚠️ **System Check Failed:** Flagged anomalies exist, but total energy waste is 0.00 kWh. This indicates a calculation pipeline error (e.g., zeroed interval timing or missing expected baselines).")
        
    cost       = float(df_f["cost_saved_inr"].sum())
    co2        = float(df_f["co2e_avoided_kg"].sum())

    glow_a     = C["accent_glow"]
    glow_warn  = f"rgba(217,119,6,{'0.12' if is_light else '0.18'})"
    glow_dang  = f"rgba(220,38,38,{'0.12' if is_light else '0.20'})"
    glow_green = f"rgba(5,150,105,{'0.12' if is_light else '0.20'})"
    glow_cyan  = f"rgba(8,145,178,{'0.12' if is_light else '0.20'})"

    row1 = st.columns(3)
    row2 = st.columns(3)

    with row1[0]:
        st.markdown(make_kpi_html(n_loc, "Locations", "Monitored", C["kpi_primary"], glow_a, tooltip="Total distinct locations tracked in selected period."), unsafe_allow_html=True)
    with row1[1]:
        st.markdown(make_kpi_html(n_watch, "High Watch", f"Last {filter_days}d", C["kpi_warning"], glow_warn, tooltip="Medium-severity events flagged for review."), unsafe_allow_html=True)
    with row1[2]:
        st.markdown(make_kpi_html(n_critical, "Critical Flags", f"Last {filter_days}d", C["kpi_danger"], glow_dang, tooltip="High-severity events requiring immediate attention.", pulse=n_critical > 0), unsafe_allow_html=True)
    with row2[0]:
        st.markdown(make_kpi_html(f"{waste:.1f}", "Energy Saved", "kWh recovered", C["kpi_primary"], glow_a, tooltip="Total recovered waste energy in the selected period."), unsafe_allow_html=True)
    with row2[1]:
        st.markdown(make_kpi_html(f"₹{cost:.0f}", "Cost Saved", "Estimated INR", C["kpi_cyan"], glow_cyan, tooltip="Waste × ₹7.5/kWh institutional tariff."), unsafe_allow_html=True)
    with row2[2]:
        st.markdown(make_kpi_html(f"{co2:.1f}", "CO₂ Avoided", "kg equivalent", C["kpi_green"], glow_green, tooltip="Waste × 0.82 kg CO₂e/kWh (CEA 2023)."), unsafe_allow_html=True)

    st.markdown("")

    col_left, col_right = st.columns([3, 1])

    with col_left:
        st.markdown(f"<div class='section-header'>Event Feed</div>", unsafe_allow_html=True)
        flagged = (
            df_f[df_f["action"].isin(["critical-flag", "high-watch"])]
            .sort_values("timestamp", ascending=False)
            .head(60)
        )
        if flagged.empty:
            st.info("No flagged events in selected period. Try widening the date filter.")
        else:
            disp_cols = [c for c in [location_col, "timestamp", "power_kw", "expected_load", "deviation_index", "action", "waste_kwh"] if c in flagged.columns]
            feed = flagged[disp_cols].copy()
            feed["timestamp"]     = feed["timestamp"].astype(str).str[:16]
            feed["power_kw"]      = feed["power_kw"].round(3)
            feed["expected_load"] = feed["expected_load"].round(3)
            feed["deviation_index"] = feed["deviation_index"].round(4)
            feed["icon"]          = feed["action"].map(action_icon)
            feed = feed.rename(columns={
                location_col: "Location", "timestamp": "Time",
                "power_kw": "Load (kW)", "expected_load": "Baseline (kW)",
                "deviation_index": "Deviation Index", "action": "Status",
                "waste_kwh": "Waste (kWh)", "icon": " ",
            })

            def color_rows(row):
                s = row.get("Status", "")
                if s == "critical-flag":
                    bg = "rgba(220,38,38,0.09)" if is_light else "rgba(248,113,113,0.14)"
                    return [f"background-color:{bg}; border-left:3px solid {C['danger']}"] * len(row)
                elif s == "high-watch":
                    bg = "rgba(217,119,6,0.07)" if is_light else "rgba(251,191,36,0.11)"
                    return [f"background-color:{bg}; border-left:3px solid {C['warning']}"] * len(row)
                return [""] * len(row)

            try:
                fmt = {"Load (kW)": "{:.3f}", "Baseline (kW)": "{:.3f}", "Deviation Index": "{:.4f}"}
                if "Waste (kWh)" in feed.columns:
                    fmt["Waste (kWh)"] = "{:.4f}"
                styled = feed.style.apply(color_rows, axis=1).format(fmt)
                # compact column widths so table fits on mobile without horizontal overflow
                col_cfg = {
                    " ": st.column_config.TextColumn(" ", width="small"),
                    "Location": st.column_config.TextColumn("Location", width="medium"),
                    "Time": st.column_config.TextColumn("Time", width="medium"),
                    "Status": st.column_config.TextColumn("Status", width="small"),
                    "Load (kW)": st.column_config.NumberColumn("Load (kW)", format="%.3f"),
                    "Baseline (kW)": st.column_config.NumberColumn("Baseline", format="%.3f"),
                    "Deviation Index": st.column_config.NumberColumn("Dev. Index", format="%.4f"),
                    "Waste (kWh)": st.column_config.NumberColumn("Waste", format="%.4f"),
                }
                st.dataframe(styled, use_container_width=True, height=340, column_config=col_cfg)
            except Exception:
                st.dataframe(feed, use_container_width=True, height=340)

    with col_right:
        st.markdown(f"<div class='section-header'>Breakdown</div>", unsafe_allow_html=True)
        action_dist = df_f["action"].value_counts().reset_index()
        action_dist.columns = ["Status", "Count"]
        action_dist["Status"] = action_dist["Status"].map(
            {"critical-flag": "Critical Flag", "high-watch": "High Watch", "normal": "Normal"}
        ).fillna(action_dist["Status"])

        fig_pie = go.Figure(go.Pie(
            values=action_dist["Count"],
            labels=action_dist["Status"],
            hole=0.52,
            marker=dict(colors=[C["danger"], C["warning"], C["border_strong"]],
                        line=dict(color=C["surface"], width=2)),
            textfont=dict(size=10, family="Inter"),
            hovertemplate="%{label}<br>Count: %{value}<br>%{percent}<extra></extra>",
        ))
        fig_pie.update_layout(**plotly_layout(height=240, margin=dict(t=10, b=30, l=10, r=10), legend_y=-0.15))
        st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": "hover", "scrollZoom": False, "displaylogo": False})

        st.markdown(f"<div class='section-header' style='margin-top:0.5rem;'>Load Trend</div>", unsafe_allow_html=True)
        hourly = (
            df_f.groupby(pd.Grouper(key="timestamp", freq="1h"))["power_kw"]
            .mean().reset_index()
        )
        rgb = tuple(int(primary_color.lstrip("#")[i:i+2], 16) for i in (0,2,4))
        fig_spark = go.Figure()
        fig_spark.add_trace(go.Scatter(
            x=hourly["timestamp"], y=hourly["power_kw"],
            mode="lines",
            line=dict(color=primary_color, width=1.5),
            fill="tozeroy",
            fillcolor=f"rgba({rgb[0]},{rgb[1]},{rgb[2]},0.12)",
            hovertemplate="%{x|%d %b %H:%M}<br>%{y:.2f} kW<extra></extra>",
        ))
        fig_spark.update_layout(**plotly_layout(height=165, margin=dict(t=5, b=20, l=40, r=10)))
        st.plotly_chart(fig_spark, use_container_width=True, config={"displayModeBar": "hover", "scrollZoom": False, "displaylogo": False})


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — HEATMAP (3D Surface)
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown(f"<div class='section-header'>{tip('Deviation Index')} — 3D Surface</div>", unsafe_allow_html=True)
    st.caption("Each cell = mean deviation index for that location × hour. 🖱️ Desktop: drag to rotate, scroll to zoom. 📱 Mobile: use two fingers to scroll past the chart.")

    hmap = df_f.copy()
    hmap["hour_bucket"] = hmap["timestamp"].dt.floor("1h")
    pivot = (
        hmap.groupby([location_col, "hour_bucket"])["deviation_index"]
        .mean().unstack(level=0).fillna(0)
    )
    if pivot.shape[0] > 168:
        pivot = pivot.tail(168)

    st.info("💡 **Pro Tip**: The 3D view is locked by default to ensure perfectly smooth page scrolling on mobile and desktop. Unlock it to freely zoom, rotate, and hover over specific data points.", icon="ℹ️")

    @st.fragment
    def render_heatmap_visuals(pivot_data):
        col_3d, col_2d = st.columns([3, 2])
        
        with col_3d:
            c_lock, _ = st.columns([1, 1])
            with c_lock:
                unlock_3d = st.toggle("🔓 Unlock 3D Interaction", value=False, help="Toggle to enable rotation and scroll zoom. Locked by default to prevent accidental scroll hijacking on touch screens.")
            
            fig_3d = go.Figure(go.Surface(
                z=pivot_data.values.T,
                colorscale=heatmap_cs,
                cmin=0, cmax=1,
                showscale=True,
                colorbar=dict(title="Deviation Index", thickness=12, len=0.7,
                              tickfont=dict(size=9), title_font=dict(size=10)),
                hovertemplate="Location: %{y}<br>Hour: %{x}<br>Score: %{z:.3f}<extra></extra>",
            ))
            fig_3d.update_layout(
                dragmode="turntable" if unlock_3d else False,
                scene=dict(
                    aspectmode="manual",
                    aspectratio=dict(x=2.5, y=1.2, z=1.0),
                    xaxis=dict(title="Time", tickfont=dict(size=8), title_font=dict(size=9), color=C["text_muted"],
                               gridcolor=C["grid_color"], backgroundcolor=C["chart_paper"]),
                    yaxis=dict(title="Location", tickfont=dict(size=8), title_font=dict(size=9), color=C["text_muted"],
                               gridcolor=C["grid_color"], backgroundcolor=C["chart_paper"]),
                    zaxis=dict(title="Score", tickfont=dict(size=8), title_font=dict(size=9), color=C["text_muted"],
                               range=[0, 1], gridcolor=C["grid_color"], backgroundcolor=C["chart_paper"]),
                    bgcolor=C["chart_paper"],
                    camera=dict(eye=dict(x=2.2, y=-2.8, z=1.5)),
                ),
                paper_bgcolor=C["chart_paper"],
                plot_bgcolor=C["chart_paper"],
                font=dict(family="Inter", color=C["text_secondary"], size=10),
                height=520, margin=dict(t=30, b=20, l=0, r=0),
            )
            # staticPlot=True completely removes pointer events so it will never block page scrolling
            st.plotly_chart(fig_3d, use_container_width=True, config={"displayModeBar": "hover", "scrollZoom": unlock_3d, "staticPlot": not unlock_3d, "displaylogo": False})
            
        with col_2d:
            # Flat heatmap for reference
            fig_heat = go.Figure(go.Heatmap(
                z=pivot_data.values.T,
                x=pivot_data.index.astype(str),
                y=pivot_data.columns.tolist(),
                colorscale=heatmap_cs,
                zmin=0, zmax=1,
                colorbar=dict(title="Score", thickness=10, len=0.7, tickfont=dict(size=9)),
                hoverongaps=False,
                hovertemplate="<b>%{y}</b><br>%{x}<br>Score: %{z:.3f}<extra></extra>",
            ))
            fig_heat.update_layout(
                title=dict(text="2D Top-Down View", font=dict(size=12, color=C["text_secondary"])),
                paper_bgcolor=C["chart_paper"],
                plot_bgcolor=C["chart_paper"],
                font=dict(family="Inter", color=C["text_secondary"], size=9),
                xaxis=dict(showticklabels=False, showgrid=False),
                yaxis=dict(tickfont=dict(size=9), color=C["text_muted"]),
                height=520, margin=dict(t=30, b=20, l=100, r=20),
            )
            st.plotly_chart(fig_heat, use_container_width=True, config={"displayModeBar": "hover", "scrollZoom": False, "displaylogo": False})

    render_heatmap_visuals(pivot)

    st.divider()
    st.markdown(f"<div class='section-header'>Location Power Timeline</div>", unsafe_allow_html=True)
    locs = sorted(df_f[location_col].unique()) if location_col in df_f.columns else []
    if locs:
        # Use a column to contain the selectbox so it doesn't span full width on desktop
        sel_col, _ = st.columns([2, 3])
        with sel_col:
            sel_loc = st.selectbox("📍 Location", locs, key="heatmap_loc_sel")
        loc_df  = df_f[df_f[location_col] == sel_loc].sort_values("timestamp")
        flagged_loc = loc_df[loc_df["action"].isin(["critical-flag", "high-watch"])]

        fig_tl = go.Figure()
        rgb = tuple(int(primary_color.lstrip("#")[i:i+2], 16) for i in (0,2,4))
        fig_tl.add_trace(go.Scatter(
            x=loc_df["timestamp"], y=loc_df["power_kw"],
            mode="lines", name="Actual Load",
            line=dict(color=primary_color, width=1.5),
            fill="tozeroy", fillcolor=f"rgba({rgb[0]},{rgb[1]},{rgb[2]},0.07)",
            hovertemplate="%{x|%d %b %H:%M}<br>%{y:.3f} kW<extra>Actual</extra>",
        ))
        
        # Add a 24h rolling average trendline
        interval_min = (loc_df["timestamp"].iloc[1] - loc_df["timestamp"].iloc[0]).total_seconds() / 60 if len(loc_df) > 1 else 60
        w_24h = max(1, int(24 * 60 / interval_min))
        loc_df["rolling_avg"] = loc_df["power_kw"].rolling(w_24h, min_periods=1, center=True).mean()
        fig_tl.add_trace(go.Scatter(
            x=loc_df["timestamp"], y=loc_df["rolling_avg"],
            mode="lines", name="24h Trend",
            line=dict(color=C["warning"], width=2),
            hovertemplate="%{x|%d %b %H:%M}<br>Trend: %{y:.3f} kW<extra></extra>",
        ))
        if "expected_load" in loc_df.columns:
            fig_tl.add_trace(go.Scatter(
                x=loc_df["timestamp"], y=loc_df["expected_load"],
                mode="lines", name="Baseline",
                line=dict(color=C["text_muted"], width=1.2, dash="dot"),
                hovertemplate="%{y:.3f} kW<extra>Baseline</extra>",
            ))
        if not flagged_loc.empty:
            cf = flagged_loc[flagged_loc["action"] == "critical-flag"]
            hw = flagged_loc[flagged_loc["action"] == "high-watch"]
            if not cf.empty:
                fig_tl.add_trace(go.Scatter(
                    x=cf["timestamp"], y=cf["power_kw"], mode="markers",
                    name="Critical Flag",
                    marker=dict(color=C["danger"], size=8, symbol="x", line=dict(width=2)),
                    hovertemplate="%{y:.3f} kW<extra>Critical</extra>",
                ))
            if not hw.empty:
                fig_tl.add_trace(go.Scatter(
                    x=hw["timestamp"], y=hw["power_kw"], mode="markers",
                    name="High Watch",
                    marker=dict(color=C["warning"], size=6, symbol="circle-open", line=dict(width=2)),
                    hovertemplate="%{y:.3f} kW<extra>High Watch</extra>",
                ))
        fig_tl.update_layout(**plotly_layout(height=320))
        st.plotly_chart(fig_tl, use_container_width=True, config={"displayModeBar": "hover", "scrollZoom": False, "displaylogo": False})


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — FORECAST
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown(f"<div class='section-header'>{tip('Load Forecaster')}</div>", unsafe_allow_html=True)
    fcast_df = load_forecast_data(data_source)

    if fcast_df is None:
        st.warning("Forecast data not generated yet. Run `python run_pipeline.py` to generate forecasts first.")
        if data_source == "Synthetic":
            st.info("Expected: `data/processed/greengrid_synthetic_forecast.csv`")
        else:
            st.info("Expected: `data/processed/uci_prophet_forecast.csv`")
    else:
        if data_source == "Synthetic" and "room" in fcast_df.columns:
            fc_col, _ = st.columns([2, 3])
            with fc_col:
                sel_r = st.selectbox("📍 Location", sorted(fcast_df["room"].unique()), key="forecast_loc_sel")
            fcast = fcast_df[fcast_df["room"] == sel_r].sort_values("ds")
        else:
            fcast = fcast_df.sort_values("ds")

        mae  = (fcast["y"] - fcast["yhat"]).abs().mean() if "y" in fcast.columns else None
        rmse = ((fcast["y"] - fcast["yhat"])**2).mean()**0.5 if "y" in fcast.columns else None

        m1, m2, m3 = st.columns(3)
        if mae is not None:
            m1.metric("Prediction Error (kW)", f"{mae:.4f}", help="MAE: average difference between forecast and actual load.")
            m2.metric("RMSE (kW)", f"{rmse:.4f}", help="Root Mean Square Error: penalises large forecast misses more than MAE.")
            quality = "High" if mae < 0.30 else ("Moderate" if mae < 0.5 else "Low")
            m3.metric("Forecast Quality", quality, help="High: MAE<0.30 kW | Moderate: 0.30–0.50 | Low: >0.50")

        fc_color = primary_color
        rgb = tuple(int(fc_color.lstrip("#")[i:i+2], 16) for i in (0,2,4))
        fig_fc = go.Figure()
        if "yhat_lower" in fcast.columns:
            fig_fc.add_trace(go.Scatter(
                x=pd.concat([fcast["ds"], fcast["ds"][::-1]]),
                y=pd.concat([fcast["yhat_upper"], fcast["yhat_lower"][::-1]]),
                fill="toself",
                fillcolor=f"rgba({rgb[0]},{rgb[1]},{rgb[2]},0.12)",
                line=dict(color="rgba(0,0,0,0)"),
                name="Confidence Band",
                hovertemplate="Band<extra></extra>",
            ))
        if "y" in fcast.columns:
            fig_fc.add_trace(go.Scatter(
                x=fcast["ds"], y=fcast["y"], mode="lines",
                name="Actual", line=dict(color=fc_color, width=1.5),
                hovertemplate="%{x|%d %b}<br>%{y:.3f} kW<extra>Actual</extra>",
            ))
        fig_fc.add_trace(go.Scatter(
            x=fcast["ds"], y=fcast["yhat"], mode="lines",
            name="Forecast",
            line=dict(color=C["accent2"], width=1.5, dash="dash"),
            hovertemplate="%{x|%d %b}<br>%{y:.3f} kW<extra>Forecast</extra>",
        ))
        fig_fc.update_layout(**plotly_layout(height=360))
        st.plotly_chart(fig_fc, use_container_width=True, config={"displayModeBar": "hover", "scrollZoom": False, "displaylogo": False})


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — ANALYTICS
# ════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown(f"<div class='section-header'>{tip('Threshold')} Impact</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div style='font-size:0.8rem; color:{C['text_secondary']}; margin:0.15rem 0 0.5rem;'>"
        f"{tip('Critical Flag')} &gt; {high_threshold:.2f}&nbsp;&nbsp;|&nbsp;&nbsp;"
        f"{tip('High Watch')} &gt; {alert_threshold:.2f}</div>",
        unsafe_allow_html=True,
    )

    col_a, col_b = st.columns(2)
    with col_a:
        ad2 = df["action"].value_counts().reset_index()
        ad2.columns = ["Status", "Count"]
        ad2["Status"] = ad2["Status"].map(
            {"critical-flag": "Critical Flag", "high-watch": "High Watch", "normal": "Normal"}
        ).fillna(ad2["Status"])
        fig_bar = px.bar(
            ad2, x="Status", y="Count", color="Status",
            color_discrete_map={"Critical Flag": C["danger"], "High Watch": C["warning"], "Normal": C["border_strong"]},
            title="Full Dataset Action Distribution"
        )
        fig_bar.update_layout(**plotly_layout(height=320), showlegend=False)
        fig_bar.update_traces(texttemplate='%{y:,.0f}', textposition='outside', textfont_size=11, cliponaxis=False, marker_line_width=0)
        st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": "hover", "scrollZoom": False, "displaylogo": False})

    with col_b:
        st.markdown("#### Impact Summary")
        tot_w   = df["waste_kwh"].sum()
        tot_c   = df["cost_saved_inr"].sum()
        tot_co2 = df["co2e_avoided_kg"].sum()
        rows = [
            ("Total Flagged Events",  f"{int((df['action']!='normal').sum()):,}"),
            ("Critical Flags",        f"{int(df['action'].eq('critical-flag').sum()):,}"),
            ("High Watch Events",     f"{int(df['action'].eq('high-watch').sum()):,}"),
            ("Energy Recovered",      f"{tot_w:.2f} kWh"),
            ("Estimated Cost Saved",  f"\u20b9{tot_c:.2f}"),
            ("CO\u2082e Avoided",          f"{tot_co2:.2f} kg"),
        ]
        for label, val in rows:
            cl, cr = st.columns([2,1])
            cl.markdown(f"**{label}**")
            cr.code(val)

    st.divider()
    st.markdown(f"<div class='section-header'>{tip('Deviation Index')} Distribution</div>", unsafe_allow_html=True)
    fig_hist = px.histogram(
        df_f, x="deviation_index", nbins=60,
        color_discrete_sequence=[primary_color],
        marginal="violin",
        labels={"deviation_index": "Deviation Index"},
        title="Deviation Index Probability Density"
    )
    fig_hist.add_vline(x=high_threshold, line_color=C["danger"],  line_dash="dash",
                       annotation_text=f"Critical ({high_threshold})", annotation_position="top right",
                       annotation_font=dict(size=10, color=C["danger"]))
    fig_hist.add_vline(x=alert_threshold, line_color=C["warning"], line_dash="dash",
                       annotation_text=f"High Watch ({alert_threshold})", annotation_position="top left",
                       annotation_font=dict(size=10, color=C["warning"]))
    fig_hist.update_layout(**plotly_layout(height=260))
    st.plotly_chart(fig_hist, use_container_width=True, config={"displayModeBar": "hover", "scrollZoom": False, "displaylogo": False})

    st.divider()
    st.markdown(f"<div class='section-header'>Validation Metrics</div>", unsafe_allow_html=True)
    if data_source == "Synthetic":
        v1,v2,v3,v4 = st.columns(4)
        v1.metric("F1 Score",           "0.97", "validated", help="F1 Score: harmonic mean of precision and recall. Maximum is 1.0.")
        v2.metric("Precision",          "0.95", "validated", help="Of all flagged events, how many were real anomalies.")
        v3.metric("Recall",             "0.98", "validated", help="Of all real anomalies, how many were correctly flagged.")
        v4.metric("Forecast Error",     "~0.24 kW", "MAE, held-out week", help="Mean Absolute Error: average gap between forecasted and actual load.")
        st.markdown(
            f"<p style='font-size:0.78rem;color:{C['text_muted']};margin-top:0.3rem;'>"
            f"n_estimators=300 &middot; {kw('Contamination','Expected fraction of anomalies in the dataset.')}=0.03 &middot; "
            f"Scores validated on injected ground-truth labels &middot; {kw('F1','Harmonic mean of precision and recall (max=1.0).')}=0.97</p>",
            unsafe_allow_html=True,
        )
    elif data_source == "Real UCI":
        v1,v2,v3,v4 = st.columns(4)
        v1.metric("F1 Score",       "N/A",      "no ground truth", help="No ground-truth labels available for real household data.")
        v2.metric("Flagged Rate",   "~1.5%",    "3 circuits", help="Fraction of readings flagged as anomalous.")
        v3.metric("Forecast Error", "~0.34 kW", "MAE, held-out", help="Mean Absolute Error on held-out test week.")
        v4.metric("Occupancy Flag", "Design",   "no PIR sensor", help="No occupancy sensor available in UCI dataset.")
    else:
        contam = FACILITY_TYPES.get(st.session_state.get("custom_facility","⚡ Other"),{}).get("contamination", 0.05)
        v1,v2,v3,v4 = st.columns(4)
        v1.metric("Contamination Rate",   f"{contam*100:.0f}%",     "facility-adjusted")
        v2.metric("Records Processed",    f"{len(df):,}",           "total")
        v3.metric("Flagged Events",       f"{int((df['action']!='normal').sum()):,}", "detected")
        v4.metric("Validation",           "No Labels",              "unsupervised")


# ════════════════════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════════════════════
# TAB 5 — ADVANCED ANALYTICS (AI & Data Science Tools)
# ════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown(f"<div class='section-header'>{tip('Data Science Workspace', 'AI Feature Analysis')}</div>", unsafe_allow_html=True)
    
    tab5_1, tab5_2, tab5_3, tab5_4 = st.tabs([
        "Correlation Matrix", "AI Time-Series Decomposition", "Unsupervised Load Clustering", "Data Exports"
    ])
    
    with tab5_1:
        st.markdown("#### Feature Importance & Correlation Heatmap")
        st.markdown("Identify which factors drive anomalies by viewing their linear correlation (Pearson).")
        cols_for_corr = ["power_kw", "expected_load", "deviation_index"]
        if "hour" in df_f.columns: cols_for_corr.append("hour")
        if "waste_kwh" in df_f.columns: cols_for_corr.append("waste_kwh")
        if "rolling_std_2h" in df_f.columns: cols_for_corr.append("rolling_std_2h")
        
        corr_df = df_f[[c for c in cols_for_corr if c in df_f.columns]].corr().round(3)
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr_df.values,
            x=corr_df.columns,
            y=corr_df.columns,
            colorscale="RdBu",
            zmin=-1, zmax=1,
            text=corr_df.values,
            texttemplate="%{text}",
            textfont={"size":10}
        ))
        fig_corr.update_layout(**plotly_layout(height=400))
        st.plotly_chart(fig_corr, use_container_width=True, config={"displayModeBar": "hover", "scrollZoom": False, "displaylogo": False})
        
    with tab5_2:
        st.markdown("#### AI Time-Series Decomposition")
        st.markdown("View the underlying hidden components identified by the Prophet AI model.")
        fcast_df = load_forecast_data(data_source)
        if fcast_df is not None and "trend" in fcast_df.columns:
            fc_col, _ = st.columns([2, 3])
            with fc_col:
                if data_source == "Synthetic" and "room" in fcast_df.columns:
                    ds_sel_r = st.selectbox("📍 Location", sorted(fcast_df["room"].unique()), key="ds_forecast_loc")
                    f_sub = fcast_df[fcast_df["room"] == ds_sel_r].sort_values("ds")
                else:
                    f_sub = fcast_df.sort_values("ds")
            
            fig_decomp = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=("Trend", "Daily Seasonality", "Weekly Seasonality"))
            fig_decomp.add_trace(go.Scatter(x=f_sub["ds"], y=f_sub["trend"], line=dict(color=C["accent"], width=2)), row=1, col=1)
            if "daily" in f_sub.columns:
                fig_decomp.add_trace(go.Scatter(x=f_sub["ds"], y=f_sub["daily"], line=dict(color=C["warning"], width=2)), row=2, col=1)
            if "weekly" in f_sub.columns:
                fig_decomp.add_trace(go.Scatter(x=f_sub["ds"], y=f_sub["weekly"], line=dict(color=C["success"], width=2)), row=3, col=1)
            fig_decomp.update_layout(**plotly_layout(height=500), showlegend=False)
            st.plotly_chart(fig_decomp, use_container_width=True, config={"displayModeBar": "hover", "scrollZoom": False, "displaylogo": False})
        else:
            st.info("Decomposition data not available. Ensure models have finished running.")
            
    with tab5_3:
        st.markdown("#### K-Means Load Clustering (Unsupervised AI)")
        st.markdown("Automatically clusters daily 24-hour load profiles into distinct 'Operating Modes'.")
        if len(df_f) > 0 and "timestamp" in df_f.columns and "power_kw" in df_f.columns:
            # Aggregate to hourly then pivot to daily profiles
            df_hour = df_f.copy()
            df_hour["date"] = df_hour["timestamp"].dt.date
            df_hour["hour"] = df_hour["timestamp"].dt.hour
            daily_profiles = df_hour.groupby(["date", "hour"])["power_kw"].mean().unstack(fill_value=0)
            
            if len(daily_profiles) >= 3:
                kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
                clusters = kmeans.fit_predict(daily_profiles.values)
                daily_profiles["Cluster"] = clusters
                
                fig_kmeans = go.Figure()
                colors = [C["accent"], C["warning"], C["success"]]
                for i in range(3):
                    cluster_data = daily_profiles[daily_profiles["Cluster"] == i].drop(columns="Cluster")
                    mean_profile = cluster_data.mean()
                    fig_kmeans.add_trace(go.Scatter(
                        x=mean_profile.index, y=mean_profile.values,
                        mode="lines", name=f"Mode {i+1} (n={len(cluster_data)})",
                        line=dict(color=colors[i], width=3)
                    ))
                layout = plotly_layout(height=400)
                layout.setdefault("xaxis", {}).update(title="Hour of Day", tickmode="linear", tick0=0, dtick=1)
                layout.setdefault("yaxis", {}).update(title="Mean Power (kW)")
                fig_kmeans.update_layout(**layout)
                st.plotly_chart(fig_kmeans, use_container_width=True, config={"displayModeBar": "hover", "scrollZoom": False, "displaylogo": False})
            else:
                st.info("Not enough daily data to perform clustering (need at least 3 days).")
        
    with tab5_4:
        st.markdown("#### Data Exports")
        st.markdown("Export cleaned datasets for further data science analysis in Jupyter or Python.")
        col_ex1, col_ex2 = st.columns(2)
        with col_ex1:
            csv_clean = df_f.to_csv(index=False)
            st.download_button("Download Cleaned Dataset (CSV)", csv_clean, "greengrid_cleaned.csv", "text/csv", use_container_width=True)
        with col_ex2:
            if "fcast_df" in locals() and fcast_df is not None:
                csv_fcast = fcast_df.to_csv(index=False)
                st.download_button("Download Forecast Profiles (CSV)", csv_fcast, "greengrid_forecasts.csv", "text/csv", use_container_width=True)


# TAB 6 — ALERTS
# ════════════════════════════════════════════════════════════════════════════
with tab6:
    st.markdown(f"<div class='section-header'>{tip('Notification Preview', 'Alert Generator')}</div>", unsafe_allow_html=True)

    col_sel, col_out = st.columns([1, 2])
    flagged_all = (
        df[df["action"].isin(["critical-flag", "high-watch"])]
        .sort_values("deviation_index", ascending=False)
        .head(100)
    )

    with col_sel:
        st.markdown("#### Select an event")
        if flagged_all.empty:
            st.info("No flagged events.")
            selected_row = None
        else:
            def row_label(r):
                loc  = r.get(location_col, "?")
                ts   = str(r.get("timestamp", ""))[:16]
                sc   = r.get("deviation_index", 0)
                act  = r.get("action", "")
                icon = action_icon(act)
                return f"{icon} {loc} | {ts} | {sc:.3f}"

            opts = [row_label(row) for _, row in flagged_all.iterrows()]
            idx  = st.selectbox("Event", range(len(opts)), format_func=lambda i: opts[i])
            selected_row = flagged_all.iloc[idx].to_dict()
            st.divider()
            st.markdown("**Event Details**")
            keys = [location_col, "timestamp", "power_kw", "expected_load", "deviation_index", "action", "waste_kwh", "cost_saved_inr", "co2e_avoided_kg"]
            for k in keys:
                if k in selected_row:
                    v = selected_row[k]
                    if isinstance(v, float): v = f"{v:.4f}"
                    label = k.replace("_", " ").title()
                    st.markdown(f"<div style='font-size:0.8rem; color:{C['text_muted']}; margin-bottom:2px;'><b style='color:{C['text_primary']};'>{label}:</b> {v}</div>", unsafe_allow_html=True)

    with col_out:
        if selected_row:
            try:
                import report_generator
                import importlib
                importlib.invalidate_caches()
                importlib.reload(report_generator)
                from report_generator import format_whatsapp_alert
                st.markdown(f"#### {tip('Notification Preview')}", unsafe_allow_html=True)
                alert_text = format_whatsapp_alert(selected_row, data_source)
                st.markdown(f"<div class='wa-alert'>{alert_text}</div>", unsafe_allow_html=True)
                
                btn_col1, btn_col2 = st.columns([3, 1])
                with btn_col2:
                    st.download_button(
                        "Download Alert",
                        data=alert_text.encode("utf-8"),
                        file_name=f"greengrid_alert_{str(selected_row.get('timestamp',''))[:10]}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
            except Exception as e:
                st.warning(f"Alert formatter unavailable: {e}")

            st.divider()
            st.markdown("#### PDF Weekly Report")
            c_p1, c_p2 = st.columns(2)
            period = c_p1.selectbox("Period", ["Last 7 Days", "Last 14 Days", "Full Dataset"])
            period_df = {
                "Last 7 Days":  df[df["timestamp"] >= max_ts - timedelta(days=7)],
                "Last 14 Days": df[df["timestamp"] >= max_ts - timedelta(days=14)],
                "Full Dataset": df,
            }[period]
            
            c_p2.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
            if c_p2.button("Generate PDF", use_container_width=True):
                st.toast("Generating PDF Report... Please wait.", icon="⏳")
                try:
                    import report_generator
                    import importlib
                    importlib.invalidate_caches()
                    importlib.reload(report_generator)
                    from report_generator import generate_pdf_report
                    with st.spinner("Generating PDF…"):
                        st.session_state["pdf_bytes"] = generate_pdf_report(period_df, data_source, top_n=10, period_label=period)
                        st.session_state["pdf_time"] = datetime.now().strftime('%Y%m%d')
                except RuntimeError as e:
                    st.error(str(e))
                except Exception as e:
                    st.error(f"PDF error: {e}")

            if "pdf_bytes" in st.session_state:
                st.download_button(
                    "Download PDF",
                    data=st.session_state["pdf_bytes"],
                    file_name=f"greengrid_report_{st.session_state.get('pdf_time', '')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )


# ════════════════════════════════════════════════════════════════════════════
# TAB 7 — INSIGHTS & RECOMMENDATIONS
# ════════════════════════════════════════════════════════════════════════════
with tab7:
    st.markdown(f"<div class='section-header'>Energy Insights &amp; Recommendations</div>", unsafe_allow_html=True)

    total_records = len(df_f)
    flag_rate     = (df_f["action"] != "normal").sum() / max(total_records, 1) * 100
    industry_avg  = 3.5

    # --- Summary cards ---
    ins1, ins2, ins3 = st.columns(3)
    ins1.metric("Flagged Event Rate",  f"{flag_rate:.1f}%",  f"{'Above' if flag_rate > industry_avg else 'Below'} {industry_avg}% benchmark")
    ins2.metric("Avg Deviation Index", f"{df_f['deviation_index'].mean():.3f}", "Dataset mean")
    ins3.metric("Peak Waste Hour",
                f"{df_f.loc[df_f['waste_kwh'].idxmax(), 'timestamp'].strftime('%H:%M') if not df_f.empty else 'N/A'}",
                "Highest single-hour waste")

    st.divider()

    # --- Insights Engine ---
    st.markdown("#### Key Findings")

    insights = []
    peak_hour = None

    # Insight 1: Flag rate vs benchmark
    if flag_rate > industry_avg * 1.5:
        insights.append(("critical", "Elevated Anomaly Rate",
            f"Your dataset shows a {flag_rate:.1f}% event rate — significantly above the {industry_avg}% industry benchmark. "
            f"This suggests systematic over-consumption or faulty sensor readings. Priority investigation recommended."))
    elif flag_rate > industry_avg:
        insights.append(("warning", "Moderate Anomaly Rate",
            f"{flag_rate:.1f}% of readings flagged — slightly above the {industry_avg}% benchmark. "
            f"Review the top flagged locations for scheduling mismatches."))
    else:
        insights.append(("success", "Healthy Anomaly Rate",
            f"{flag_rate:.1f}% event rate is within the {industry_avg}% benchmark. "
            f"Energy consumption patterns appear well-controlled."))

    # Insight 2: Top offending location
    if location_col in df_f.columns:
        top_loc = df_f.groupby(location_col)["waste_kwh"].sum().idxmax() if not df_f.empty else None
        top_waste = df_f.groupby(location_col)["waste_kwh"].sum().max() if not df_f.empty else 0
        if top_loc:
            insights.append(("warning", f"Highest Waste Location: {top_loc}",
                f"'{top_loc}' accounts for {top_waste:.2f} kWh of recovered waste energy in the selected period. "
                f"Consider scheduling an audit of equipment in this area."))

    # Insight 3: Peak hour
    peak_hour = None
    if not df_f.empty:
        df_f["hour"] = df_f["timestamp"].dt.hour
        if "deviation_index" in df_f.columns and not df_f.empty:
            peak_hour = df_f.groupby("hour")["deviation_index"].mean().idxmax()
            insights.append(("info", f"Peak Deviation Hour: {peak_hour:02d}:00",
                f"Anomaly scores are systematically highest at {peak_hour:02d}:00. "
                f"If this does not match an expected high-use period, investigate equipment schedules or HVAC behaviour."))

    # Insight 4: Cost recommendation
    proj_annual = cost * (365 / max(filter_days, 1))
    if proj_annual > 1000:
        insights.append(("critical", "Projected Annual Cost Risk",
            f"Based on the current {filter_days}-day window, projected annual energy waste could cost "
            f"\u20b9{proj_annual:,.0f}. Automated scheduling policies or occupancy-based controls could recover 40–60% of this."))

    # Insight 5: CO2 impact
    proj_co2 = co2 * (365 / max(filter_days, 1))
    insights.append(("info", "Carbon Footprint Projection",
        f"Projected annual CO\u2082 avoidance at current performance: {proj_co2:.1f} kg CO\u2082e "
        f"(equivalent to planting ~{int(proj_co2/21)} trees). "
        f"Full occupancy-based automation could double this figure."))

    color_map = {
        "critical": C["danger"],
        "warning":  C["warning"],
        "success":  C["success"],
        "info":     C["accent"],
    }
    for delay, (level, title, body) in enumerate(insights):
        col = color_map.get(level, C["accent"])
        st.markdown(
            f"<div class='insight-card' style='--insight-color:{col}; animation-delay:{delay*0.08}s;'>"
            f"<div class='insight-title'>{title}</div>"
            f"<div class='insight-body'>{body}</div>"
            "</div>",
            unsafe_allow_html=True,
        )

    st.divider()

    # Recommendations section
    st.markdown("#### Recommendations")
    recs = [
        ("🔌 Implement Occupancy-Based Controls",
         f"Install {kw('PIR Sensor','Passive Infrared sensor — detects body heat to determine room occupancy.')} "
         f"or Wi-Fi presence sensors and integrate with your energy management system. "
         f"This alone reduces {kw('Critical Flag','Highest-severity event — unoccupied space drawing far above baseline.')} "
         f"events by 30–50% in unoccupied spaces."),
        ("📅 Set Automated Scheduling Policies",
         f"Program {kw('HVAC','Heating, Ventilation & Air Conditioning — typically the largest energy consumer.')} "
         f"and lighting to follow room booking calendars. This eliminates the most common cause of after-hours "
         f"{kw('Waste (kWh)','Excess energy used above the expected baseline, in kilowatt-hours.')} spikes."),
        ("🔧 Review Peak-Hour Equipment",
         f"Focus maintenance on equipment active during the peak {kw('Deviation Index','Score 0–1: how far a reading strays from its expected value.')} "
         f"hour ({f'{peak_hour:02d}' if peak_hour is not None else 'N/A'}). Aging {kw('HVAC','Heating, Ventilation & Air Conditioning — typically the largest energy consumer.')} "
         f"compressors and water heaters are common culprits."),
        ("📊 Establish Baseline Monitoring",
         f"Re-run this analysis monthly to track whether interventions reduce the flagged event rate. "
         f"Target: below {kw('Threshold','Score boundary separating normal from flagged readings.')} 3.5%. "
         f"Consistent {kw('Baseline','Expected average power learned from historical patterns.')} stability indicates healthy building operations."),
        ("💡 Upgrade to Sub-Metering",
         f"Break down energy tracking to individual circuits within each room. "
         f"Sub-metering isolates the exact appliance causing {kw('High Watch','Medium-severity flag — reading above normal, needs monitoring.')} events "
         f"and reduces investigation time by over 60%."),
        ("📋 Benchmark Against Sector Standards",
         f"Compare your {kw('Deviation Index','Score 0–1: how far a reading strays from its expected value.')} "
         f"averages against ECBC (Energy Conservation Building Code) sector targets. "
         f"Office buildings should target &lt;0.45 mean deviation; educational &lt;0.40."),
    ]
    for r_title, r_body in recs:
        with st.expander(r_title):
            st.markdown(r_body, unsafe_allow_html=True)

    st.divider()
    # ── How to read this dashboard ── 
    with st.expander("📖 How to Read This Dashboard"):
        help_html = (
            "<div style='font-size:0.82rem;line-height:1.7;color:" + C["text_secondary"] + ";'>"
            + "<b>Overview tab</b> - Instant summary of " + kw("Critical Flag", "Highest-severity event - unoccupied space drawing far above baseline.")
            + " and " + kw("High Watch", "Medium-severity flag - reading above normal, needs monitoring.")
            + " events for the selected period.<br><br>"
            + "<b>Heatmap tab</b> - 3D surface shows " + kw("Deviation Index", "Score 0-1: how far a reading strays from its expected value.")
            + " per location and hour. High peaks = recurring over-consumption patterns.<br><br>"
            + "<b>Forecast tab</b> - Compares predicted vs actual " + kw("kW", "Kilowatt - unit of instantaneous electrical power (1 kW = 1000 W)")
            + " demand. " + kw("MAE", "Mean Absolute Error - average gap between forecast and actual.") + " shows forecast quality.<br><br>"
            + "<b>Analytics tab</b> - Threshold tuning and score distribution. Drag sliders to tighten or relax "
            + kw("Threshold", "Score boundary separating normal from flagged readings.") + " sensitivity.<br><br>"
            + "<b>Alerts tab</b> - Event-level drill-down. Select any row to generate a notification with "
            + kw("Waste (kWh)", "Excess energy above baseline, in kilowatt-hours.") + " and cost impact.<br><br>"
            + "<b>Insights tab</b> - Analytical findings, " + kw("Contamination", "Expected fraction of unusual readings.")
            + " analysis, and actionable recommendations tailored to your dataset."
            + "</div>"
        )
        st.markdown(help_html, unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────────────────
footer_html = (
    '<div class="gg-footer">'
    '<span class="gg-footer-brand">⚡ GreenGrid &nbsp;&bull;&nbsp; Energy Intelligence System</span>'
    '<span class="gg-footer-meta">All data processed locally &nbsp;·&nbsp; Session only — never stored</span>'
    '</div>'
)
st.markdown(footer_html, unsafe_allow_html=True)
