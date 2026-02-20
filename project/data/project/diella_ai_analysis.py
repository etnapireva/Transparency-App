from pathlib import Path

import streamlit as st
import pandas as pd
from config import *
from utils.data_loader import load_data
from utils.vector_store import build_vector_store
from utils.visualization import *
from tabs import (
    render_dashboard,
    render_sentiment,
    render_topics,
    render_style_metrics,
    render_speaker_comparison,
    render_qa,
    render_evaluation,
    render_methodology,
)

st.set_page_config(page_title=PAGE_TITLE, layout=PAGE_LAYOUT)

st.markdown(
    """
    <style>
    /* Sfondi i faqes së përgjithshme – që header-i të shkrihet me të */
    .stApp, section.main .block-container {
        background-color: #010409;
    }
    section[data-testid="stSidebar"] {
        background: radial-gradient(ellipse 100% 120% at 0% 0%, #0f172a 0%, #020617 50%, #010409 100%);
        border-right: none;
    }
    section[data-testid="stSidebar"] > div {
        background: transparent;
    }
    section[data-testid="stSidebar"] .stMarkdown, 
    section[data-testid="stSidebar"] label, 
    section[data-testid="stSidebar"] p {
        color: #e5e7eb !important;
    }
    section[data-testid="stSidebar"] .stCaption {
        color: #9ca3af !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header HTML
header_html = """
<div style="
    position: sticky;
    top: 0;
    z-index: 1000;
    display: flex; 
    align-items: center; 
    padding: 10px 24px; 
    background: radial-gradient(ellipse 120% 100% at 50% 0%, #0f172a 0%, #020617 40%, #010409 100%);
">
    <div style="position: relative; margin-right: 16px;">
        <!-- Ikonë dielli -->
        <div style="
            width: 42px; 
            height: 42px; 
            border-radius: 999px; 
            background: #facc15; 
            box-shadow: 0 0 20px rgba(250, 204, 21, 0.6);
        "></div>
    </div>
    <div>
        <h1 style="
            margin: 0; 
            font-size: 26px; 
            font-weight: 700; 
            letter-spacing: 0.08em;
            color: #e5e7eb;
        ">DIELLA AI</h1>
        <p style="
            margin: 0; 
            margin-top: 2px;
            color: #9ca3af;
            font-size: 14px;
        ">
            Sistemi i transparencës së politikave publike
        </p>
    </div>
</div>
"""

st.markdown(header_html, unsafe_allow_html=True)

# ==========================================
# LOAD DATA
# ==========================================

@st.cache_data
def init_data():
    df, err = load_data(DATA_PATH)
    if err:
        st.error(f"Error loading data: {err}")
        st.stop()
    return df


@st.cache_resource
def init_vector_store(df):
    model, index = build_vector_store(df)
    if model is None or index is None:
        st.warning(
            "Vector store could not be initialized. Q&A feature may not work."
        )
    return model, index


df = init_data()
# Vector store ngarkohet vetëm kur përdoruesi përdor Q&A (lazy load), që faqja të ngarkojë shpejt
model = st.session_state.get("qa_model")
index = st.session_state.get("qa_index")

# Pjesa nën header: titull + përshkrim (në linjë me dark theme)
st.markdown(
    """
    <div style="
        padding: 1rem 0 1.25rem 0;
        border-bottom: 1px solid #1f2937;
        margin-bottom: 0.5rem;
    ">
        <h2 style="
            margin: 0 0 0.35rem 0;
            font-size: 1.5rem;
            font-weight: 600;
            color: #e5e7eb;
            letter-spacing: 0.02em;
        ">Analiza e Deklaratave Publike</h2>
        <p style="
            margin: 0;
            font-size: 0.95rem;
            color: #9ca3af;
            line-height: 1.45;
        ">Analiza interaktive me sentiment, tematikë, stil, krahasim folësish dhe Q&A si asistent AI.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==========================================
# SIDEBAR - FILTERS (të thjeshta: folës + datë)
# ==========================================

st.sidebar.header("Filtrimi i Deklaratave")

speaker_list_raw = (
    sorted(df["Speaker"].dropna().unique().tolist()) if not df.empty else []
)
if len(speaker_list_raw) == 0:
    speaker_list_raw = ["Unknown"]
speaker_list = ["Të gjithë"] + speaker_list_raw

speaker = st.sidebar.selectbox("Zgjidh folësin", speaker_list, index=0)

date_from = st.sidebar.date_input(
    "Data nga",
    value=(
        df["Date"].min()
        if not df.empty and df["Date"].notna().any()
        else pd.to_datetime("2020-01-01")
    ),
)
date_to = st.sidebar.date_input(
    "Data deri",
    value=(
        df["Date"].max()
        if not df.empty and df["Date"].notna().any()
        else pd.to_datetime("2025-12-31")
    ),
)

# Apply filters
if not df.empty:
    df_filtered = df.copy()
    if speaker != "Të gjithë":
        df_filtered = df_filtered[df_filtered["Speaker"] == speaker]
    if pd.api.types.is_datetime64_any_dtype(df_filtered["Date"]):
        df_filtered = df_filtered[
            (df_filtered["Date"] >= pd.to_datetime(date_from))
            & (df_filtered["Date"] <= pd.to_datetime(date_to))
        ]
else:
    df_filtered = df.copy()

st.sidebar.markdown("---")
st.sidebar.caption(f"**{len(df_filtered)}** deklaratë(a)")
st.sidebar.markdown("© 2025 Etna Pireva")
st.sidebar.caption("Punim diplome — Mentor Msc. Alma Novobërdaliu — UBT 2025–2026")

# ==========================================
# TABS
# ==========================================

tab_dashboard, tab1, tab2, tab3, tab4, tab5, tab_eval, tab_methodology = st.tabs(
    ["Dashboard", "Sentiment", "Topics", "Style Metrics", "Krahasim Folësish", "Q&A", "Vlerësim", "Metodologji"]
)

# Tab content (delegated to tabs package)
with tab_dashboard:
    render_dashboard(df_filtered)
with tab1:
    render_sentiment(df_filtered)
with tab2:
    render_topics(df_filtered)
with tab3:
    render_style_metrics(df_filtered)
with tab4:
    render_speaker_comparison(df, speaker_list_raw)
with tab5:
    render_qa(df, init_vector_store)
with tab_eval:
    base_dir = Path(__file__).resolve().parent
    render_evaluation(base_dir, base_dir / DATA_PATH)
with tab_methodology:
    render_methodology()
