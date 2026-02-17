# ==========================================
# DIELLA AI - MAIN APPLICATION
# ==========================================

import streamlit as st
import pandas as pd
from config import *
from utils.data_loader import load_data
from utils.vector_store import build_vector_store
from utils.ollama_integration import build_qa_context
from utils.visualization import *

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(page_title=PAGE_TITLE, layout=PAGE_LAYOUT)

# Header HTML
header_html = """
<div style="
     position:sticky;
    display: flex; 
    align-items: center; 
    padding: 10px 20px; 
    border-bottom: 1px solid #ddd;
    background: #fff;
        background-color: #0f172a;

">
    <div style="position: relative; margin-right: 15px;">
        <!-- Ikonë dielli -->
        <div style="
            width: 40px; 
            height: 40px; 
            border-radius: 50%; 
            background: #facc15; 
            animation: pulse 2s infinite;
        "></div>
    </div>
    <div>
        <h1 style="
            margin: 0; 
            font-size: 28px; 
            font-weight: bold; 
            background: linear-gradient(to right, #6366f1, #ec4899); 
            -webkit-background-clip: text; 
            color: transparent;
        ">DIELLA AI</h1>
        <p style="margin: 0; color: #6b7280;">Sistemi i Transparencës së Politikave</p>
    </div>
</div>

<style>
@keyframes pulse {
    0% { box-shadow: 0 0 0px #facc15; }
    50% { box-shadow: 0 0 15px #facc15; }
    100% { box-shadow: 0 0 0px #facc15; }
}
</style>
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
model, index = init_vector_store(df)

st.title("DIELLA AI - Analiza e Deklaratave Publike")
st.markdown(
    "Analiza interaktive me sentiment, tematikë, stil, krahasim folësish dhe Q&A si asistent AI."
)

# ==========================================
# SIDEBAR - FILTERS
# ==========================================

st.sidebar.header("Filtrimi i Deklaratave")

speaker_list = (
    df["Speaker"].dropna().unique().tolist() if not df.empty else []
)
if len(speaker_list) == 0:
    speaker_list = ["Unknown"]

speaker = st.sidebar.selectbox("Zgjidh speaker", speaker_list, index=0)

date_from = st.sidebar.date_input(
    "Data nga:",
    value=(
        df["Date"].min()
        if not df.empty and df["Date"].notna().any()
        else pd.to_datetime("2020-01-01")
    ),
)
date_to = st.sidebar.date_input(
    "Data deri:",
    value=(
        df["Date"].max()
        if not df.empty and df["Date"].notna().any()
        else pd.to_datetime("2025-12-31")
    ),
)

st.sidebar.markdown("---")
st.sidebar.markdown("© 2025 Etna Pireva")

# Apply filters
if not df.empty:
    df_filtered = df.copy()
    if pd.api.types.is_datetime64_any_dtype(df_filtered["Date"]):
        df_filtered = df_filtered[
            (df_filtered["Speaker"] == speaker)
            & (df_filtered["Date"] >= pd.to_datetime(date_from))
            & (df_filtered["Date"] <= pd.to_datetime(date_to))
        ]
    else:
        df_filtered = df_filtered[df_filtered["Speaker"] == speaker]
else:
    df_filtered = df.copy()

# ==========================================
# TABS
# ==========================================

tab_dashboard, tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Dashboard", "Sentiment", "Topics", "Style Metrics", "Krahasim Folësish", "Q&A"]
)

# ==========================
# DASHBOARD TAB
# ==========================

with tab_dashboard:
    colA, colB, colC, colD = st.columns(4)

    with colA:
        st.metric("Total Deklarata", len(df_filtered))

    with colB:
        avg_sent = (
            round(df_filtered["SentimentScore"].mean(), 3)
            if not df_filtered.empty
            else 0.0
        )
        st.metric("Sentimenti Mesatar", avg_sent)

    with colC:
        active_speaker = (
            df_filtered["Speaker"].value_counts().idxmax()
            if not df_filtered.empty
            else "-"
        )
        st.metric("Folësi më Aktiv", active_speaker)

    with colD:
        diella_sent = (
            round(
                df_filtered[df_filtered["Speaker"] == "Diella"][
                    "SentimentScore"
                ].mean()
            )
            if "Diella" in df_filtered["Speaker"].values
            else 0
        )
        st.metric("Sentimenti i Dielës", diella_sent)

    # Row 2: Charts
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Shpërndarja e Sentimenteve")
        fig_pie = create_sentiment_pie_chart(df_filtered)
        if fig_pie:
            st.plotly_chart(fig_pie, use_container_width=True, key="pie_chart_dashboard")
        else:
            st.info("Nuk ka të dhëna për filtrimet aktuale.")

    with col2:
        st.subheader("Trendi i Sentimentit Mesatar Ditor")
        fig_trend = create_sentiment_trend_chart(df_filtered)
        if fig_trend:
            st.plotly_chart(fig_trend, use_container_width=True, key="trend_chart_dashboard")
        else:
            st.info("Nuk ka trend ditore.")

    # Row 3: Latest statements
    st.subheader("Deklaratat e Fundit")
    last_statements = (
        df_filtered.sort_values("Date", ascending=False).head(5)
        if not df_filtered.empty
        else pd.DataFrame()
    )
    for _, row in last_statements.iterrows():
        sentiment = row.get("SentimentLabel", "Neutral")
        color = (
            "green"
            if sentiment == "Pozitiv"
            else ("red" if sentiment == "Negativ" else "blue")
        )
        arrow = (
            "↑"
            if sentiment == "Pozitiv"
            else ("↓" if sentiment == "Negativ" else "→")
        )
        st.markdown(
            f"""
        <div style="padding:10px; margin-bottom:5px; 
            border-left:4px solid {color}; 
            background-color:#1e293b; color:#e2e8f0;">
            <b>{row.get('Speaker', '-')} ({row.get('Date') if pd.notna(row.get('Date')) else '-'}):</b> {str(row.get('Speech', ''))[:200]}...<br>
            <span style="color:{color}; font-weight:bold;">
                {arrow} SentimentScore: {round(row.get('SentimentScore', 0), 2)}, 
                TTR: {round(row.get('TTR', 0), 3)}
            </span>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Row 4: About section
    st.subheader("Rreth Sistemit")
    st.markdown(
        """
    Ky sistem analizon deklaratat publike të ministres Diella duke përdorur:
    - Analizë Sentimenti (VADER NLP)
    - Matje të transparencës (TTR)
    - Statistikë dhe vizualizime

    Qëllimi është të vlerësohet qartësia, toni dhe trendet e komunikimit publik.
    """
    )

# ==========================
# TAB 1: SENTIMENT
# ==========================

with tab1:
    st.subheader("Analiza e Sentimentit")
    if not df_filtered.empty:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Shpërndarja Totale")
            fig = create_sentiment_bar_chart(df_filtered)
            if fig:
                st.plotly_chart(fig, use_container_width=True, key="bar_chart_sentiment")

        with col2:
            st.markdown("### Trendi i Sentimenti Mesatar Ditor")
            fig = create_sentiment_trend_chart(df_filtered)
            if fig:
                st.plotly_chart(fig, use_container_width=True, key="trend_chart_sentiment")
            else:
                st.info("Nuk ka trend ditore të mjaftueshëm.")

        st.markdown("### Deklaratat dhe Sentimenti")
        for idx, row in df_filtered.iterrows():
            text = row.get("Speech", "")
            sentiment = row.get("SentimentLabel", "Neutral")
            score = row.get("SentimentScore", 0.0)

            if sentiment == "Pozitiv":
                color = "green"
                arrow = "&#9650;"
            elif sentiment == "Negativ":
                color = "red"
                arrow = "&#9660;"
            else:
                color = "blue"
                arrow = "&#8594;"

            percent = int(abs(score) * 100)

            st.markdown(
                f"""
            <div style="padding:5px; margin-bottom:4px; border-left:4px solid {color}; background-color:#000000;">
                <b>{row.get('Speaker', '-')} ({row.get('Date') if pd.notna(row.get('Date')) else '-'}):</b> {text}<br>
                <span style="color:{color}; font-weight:bold;">{arrow} {percent}%</span>
            </div>
            """,
                unsafe_allow_html=True,
            )
    else:
        st.info("Nuk ka të dhëna për këta filtra.")

# ==========================
# TAB 2: TOPICS
# ==========================

with tab2:
    st.subheader("Modelimi i Temave dhe Filtrimi i Deklaratave")

    if not df_filtered.empty:
        fig = create_topics_bar_chart(df_filtered)
        if fig:
            st.plotly_chart(fig, use_container_width=True, key="bar_chart_topics")

        topic_data = (
            df_filtered.groupby("Topic")
            .agg({"TopKeywords": "first", "Speech": "count"})
            .reset_index()
            .rename(columns={"Speech": "Vlera"})
        )

        st.dataframe(
            topic_data[["Topic", "TopKeywords", "Vlera"]].rename(
                columns={
                    "Topic": "ID Temë",
                    "TopKeywords": "Fjalëkyçet Kryesore",
                    "Vlera": "Numri",
                }
            ),
            height=200,
            use_container_width=True,
        )

        st.markdown("---")
        st.subheader("Shiko Deklaratat sipas Temës")

        topic_map = topic_data.set_index("Topic")["TopKeywords"].to_dict()
        topic_options = [
            f"Tema {int(tid)}: {keywords}"
            for tid, keywords in topic_map.items()
        ]
        topic_options.insert(0, "— Zgjidh Temën për të parë deklaratat —")

        selected_topic_label = st.selectbox(
            "Përzgjedhja e Temës",
            topic_options,
            index=0,
        )

        if (
            selected_topic_label
            != "— Zgjidh Temën për të parë deklaratat —"
        ):
            selected_topic_id = int(
                selected_topic_label.split(":")[0]
                .replace("Tema ", "")
                .strip()
            )
            speeches_in_topic = df_filtered[
                df_filtered["Topic"] == selected_topic_id
            ]

            st.info(
                f"Duke shfaqur **{len(speeches_in_topic)}** deklarata në Temën **{selected_topic_id}** ({topic_map.get(selected_topic_id, '')})"
            )

            st.dataframe(
                speeches_in_topic[["Date", "Speaker", "Speech_SQ"]],
                height=400,
                use_container_width=True,
            )
    else:
        st.info("Nuk ka të dhëna për këta filtra.")

# ==========================
# TAB 3: STYLE METRICS
# ==========================

with tab3:
    st.subheader("Metrikat e Stilit")
    if not df_filtered.empty:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Gjatësia e Deklaratave")
            fig = create_wordcount_histogram(df_filtered)
            if fig:
                st.altair_chart(fig, use_container_width=True, key="hist_wordcount_style")

        with col2:
            st.markdown("### Tema Kryesore")
            df_copy = df_filtered.copy()
            style_data2 = (
                df_copy.groupby("TopKeywords")
                .size()
                .reset_index(name="Numri")
                .sort_values("Numri", ascending=False)
                .head(10)
            )

            fig = (
                alt.Chart(style_data2)
                .mark_bar()
                .encode(
                    x=alt.X(
                        "TopKeywords",
                        title="Temat kryesore",
                        sort="-y",
                    ),
                    y=alt.Y("Numri", title="Numri i deklaratave"),
                    tooltip=["TopKeywords", "Numri"],
                    color=alt.Color(
                        "Numri",
                        scale=alt.Scale(scheme="oranges"),
                    ),
                )
                .properties(width=350, height=300)
            )
            st.altair_chart(fig, use_container_width=True, key="bar_keywords_style")
    else:
        st.info("Nuk ka të dhëna për këta filtra.")

# ==========================
# TAB 4: SPEAKER COMPARISON
# ==========================

with tab4:
    st.subheader("Kuadratet e të Dhënave Statistikore: Krahasimi i Folësve")

    speaker_stats = df.groupby("Speaker").agg(
        Count=("Speech_SQ", "size"),
        Avg_Words=("WordCount", "mean"),
        Avg_TTR=("TTR", "mean"),
        Avg_Sentiment=("SentimentScore", "mean"),
    ).reset_index()

    speaker_stats["Avg_Words"] = speaker_stats["Avg_Words"].round(1)
    speaker_stats["Avg_TTR"] = speaker_stats["Avg_TTR"].round(3)
    speaker_stats["Avg_Sentiment"] = speaker_stats["Avg_Sentiment"].round(2)

    st.dataframe(speaker_stats, hide_index=True)

    st.markdown("---")

    speakers_to_compare = st.multiselect(
        "Zgjidh folës për grafikët e krahasimit",
        speaker_list,
        default=speaker_list[:2] if len(speaker_list) >= 2 else speaker_list,
    )

    if len(speakers_to_compare) >= 2:
        col_ttr, col_boxplot = st.columns(2)

        with col_ttr:
            st.subheader("Krahasimi i Pasurisë Leksikore (TTR)")
            fig = create_speaker_comparison_chart(df, speakers_to_compare)
            if fig:
                st.altair_chart(fig, use_container_width=True, key="chart_ttr_comparison")

        with col_boxplot:
            st.subheader("Shpërndarja e Sentimenti (Boxplot)")
            fig = create_speaker_sentiment_boxplot(df, speakers_to_compare)
            if fig:
                st.plotly_chart(fig, use_container_width=True, key="boxplot_sentiment_comparison")
    else:
        st.info("Zgjidh të paktën dy folës për krahasim të detajuar.")

# ==========================
# TAB 5: Q&A
# ==========================

with tab5:
    st.subheader("Bisedo me DIELLA AI (Shqip në Shqip)")
    st.info(
        "Duke përdorur Groq API për përgjigje të shpejta dhe të sakta në shqip."
    )

    query_shqip = st.text_input(
        "Shkruaj pyetjen tënde në shqip...",
        key="qa_input_v5",
    )

    if query_shqip:
        with st.spinner("Po kërkoj në deklaratat në shqip..."):
            if model is None or index is None:
                st.error(
                    "Baza vektoriale nuk është gati (mungon modeli FAISS/SentenceTransformer ose nuk ka dokumente)."
                )
            else:
                context_text, sources = build_qa_context(
                    query_shqip,
                    model,
                    index,
                    df,
                    max_docs=MAX_QA_DOCS,
                    max_chars=MAX_CHARS_CONTEXT,
                )

                if not sources:
                    st.warning(
                        "Nuk u gjetën burime të përshtatshme për pyetjen tënde. Provoni një fjalëkyç tjetër ose hiq kufizimet e filtrit."
                    )
                else:
                    with st.spinner("DIELLA AI po arsyeton..."):
                        # Import Groq integration
                        from utils.groq_integration import generate_qa_response_groq
                        
                        response_text, sources = generate_qa_response_groq(
                            query_shqip,
                            context_text,
                            sources,
                            GROQ_API_KEY,
                            GROQ_MODEL,
                        )

                        if response_text.startswith("Gabim"):
                            st.error(response_text)
                        else:
                            st.markdown(f"💬 **DIELLA AI:**\n\n{response_text}")

                        with st.expander("Burimet e Gjetura (në shqip):"):
                            for s in sources:
                                st.markdown(
                                    f"[{s['id']}] **{s['speaker']}** ({s['date']}):\n\n{s['text']}\n\n---"
                                )

                                