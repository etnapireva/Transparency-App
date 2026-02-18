# ==========================================
# DIELLA AI - MAIN APPLICATION
# ==========================================

import json
from pathlib import Path

import streamlit as st
import pandas as pd
from config import *
from utils.data_loader import load_data
from utils.vector_store import build_vector_store
from utils.ollama_integration import build_qa_context
from utils.visualization import *
import run_evaluation

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(page_title=PAGE_TITLE, layout=PAGE_LAYOUT)

# Sidebar: gradient si header (dark theme)
st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] {
        background: radial-gradient(circle at top left, #1e293b, #020617);
        border-right: 1px solid #1f2937;
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
    border-bottom: 1px solid #1f2937;
    background: radial-gradient(circle at top left, #1e293b, #020617);
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
    sorted(df["Speaker"].dropna().unique().tolist()) if not df.empty else []
)
if len(speaker_list) == 0:
    speaker_list = ["Unknown"]

speakers_selected = st.sidebar.multiselect(
    "Folës(i)",
    options=speaker_list,
    default=speaker_list,
    help="Zgjidhni një ose më shumë folës. Zbrazni për të gjithë.",
    key="sidebar_speakers",
)

# Nëse përdoruesi hoqi të gjitha, konsiderojmë "të gjithë"
if not speakers_selected:
    speakers_selected = speaker_list

date_from = st.sidebar.date_input(
    "Data nga",
    value=(
        df["Date"].min()
        if not df.empty and df["Date"].notna().any()
        else pd.to_datetime("2020-01-01")
    ),
    key="sidebar_date_from",
)
date_to = st.sidebar.date_input(
    "Data deri",
    value=(
        df["Date"].max()
        if not df.empty and df["Date"].notna().any()
        else pd.to_datetime("2025-12-31")
    ),
    key="sidebar_date_to",
)

sentiment_options = ["Pozitiv", "Neutral", "Negativ"]
sentiments_selected = st.sidebar.multiselect(
    "Sentiment",
    options=sentiment_options,
    default=sentiment_options,
    help="Filtro sipas etiketës së sentimentit.",
    key="sidebar_sentiments",
)
if not sentiments_selected:
    sentiments_selected = sentiment_options

if st.sidebar.button("Pastro filtrat", key="sidebar_reset"):
    for key in ("sidebar_speakers", "sidebar_sentiments", "sidebar_date_from", "sidebar_date_to"):
        st.session_state.pop(key, None)
    st.rerun()

# Apply filters
if not df.empty:
    df_filtered = df.copy()
    df_filtered = df_filtered[df_filtered["Speaker"].isin(speakers_selected)]
    df_filtered = df_filtered[df_filtered["SentimentLabel"].isin(sentiments_selected)]
    if pd.api.types.is_datetime64_any_dtype(df_filtered["Date"]):
        df_filtered = df_filtered[
            (df_filtered["Date"] >= pd.to_datetime(date_from))
            & (df_filtered["Date"] <= pd.to_datetime(date_to))
        ]
else:
    df_filtered = df.copy()

st.sidebar.markdown("---")
st.sidebar.caption(f"**{len(df_filtered)}** deklaratë(a) sipas filtrit")
st.sidebar.markdown("© 2025 Etna Pireva")

# ==========================================
# TABS
# ==========================================

tab_dashboard, tab1, tab2, tab3, tab4, tab5, tab_eval = st.tabs(
    ["Dashboard", "Sentiment", "Topics", "Style Metrics", "Krahasim Folësish", "Q&A", "Vlerësim"]
)

# ==========================
# DASHBOARD TAB
# ==========================

with tab_dashboard:
    with st.expander("Metodologjia e treguesve"):
        st.markdown(
            "Treguesit e faqes kryesore pasqyrojnë grupin e deklaratave të përzgjedhura sipas filtrit (folës dhe datë): "
            "numri total i deklaratave, mesatarja e pikës së sentimentit (shkallë nga -1 në +1, sipas VADER) dhe folësi me më shumë deklarata. "
            "Grafikët ilustrojnë shpërndarjen e sentimenteve (pozitiv, neutral, negativ) dhe ndryshimin e sentimentit mesatar me kalimin e kohës."
        )
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

    # with colD:
    #     diella_sent = (
    #         round(
    #             df_filtered[df_filtered["Speaker"] == "Diella"][
    #                 "SentimentScore"
    #             ].mean()
    #         )
    #         if "Diella" in df_filtered["Speaker"].values
    #         else 0
    #     )
    #     st.metric("Sentimenti i Dielës", diella_sent)

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

    # Row 4: About section (inside Dashboard tab)
    st.subheader("Rreth Sistemit")
    about_html = """
<div style="
    background-color:#1e293b; 
    color:#e2e8f0; 
    padding:20px; 
    border-radius:10px; 
    line-height:1.6;
    border-left:5px solid #6366f1;
    box-shadow: 2px 2px 10px rgba(0,0,0,0.3);
">
    <p>
    Ky sistem është krijuar për të analizuar në mënyrë të thelluar deklaratat publike të ministres Diella, duke përdorur teknika të avancuara të përpunimit të gjuhës natyrore dhe analiza statistikore. Ai përfshin disa komponentë kryesorë:
    </p>
    <ul>
        <li><b>Analizë Sentimenti (VADER NLP)</b> – Vlerëson tonin emocional të deklaratave dhe identifikon nëse ato janë kryesisht pozitive, negative apo neutrale.</li>
        <li><b>Matje të transparencës dhe pasurisë leksikore (TTR)</b> – Përdor Type-Token Ratio për të matur variacionin dhe pasurinë e fjalëve të përdorura.</li>
        <li><b>Statistikë dhe vizualizime interaktive</b> – Gjeneron tabela dhe grafika për shpërndarjen e sentimentit, temat kryesore, aktivitetin e folësve dhe krahasime midis tyre.</li>
    </ul>
    <p>
    Qëllimi i sistemit është të ofrojë një pasqyrë të qartë dhe të kuptueshme të komunikimit publik, duke ndihmuar përdoruesit të analizojnë:
    </p>
    <ul>
        <li>Qartësinë e mesazheve</li>
        <li>Tonin emocional</li>
        <li>Frekuencën dhe aktivitetin e deklaratave</li>
        <li>Trendët dhe temat më të diskutueshme</li>
    </ul>
    <p>
    Ky sistem mund të përdoret si një mjet asistues për studiues, gazetarë, apo qytetarë që duan të kuptojnë më mirë komunikimin publik dhe transparencën e deklaratave.
    </p>
</div>
"""
    st.markdown(about_html, unsafe_allow_html=True)


# ==========================
# TAB 1: SENTIMENT
# ==========================

with tab1:
    st.subheader("Analiza e Sentimentit")
    with st.expander("Metodologjia e analizës së sentimentit"):
        st.markdown(
            "Analiza e sentimentit vlerëson tonin emocional të tekstit dhe e klasifikon atë si pozitiv, neutral ose negativ. "
            "Sistemi përdor VADER (Valence Aware Dictionary and sEntiment Reasoner), i cili prodhon një pikë compound nga -1 (shumë negativ) deri në +1 (shumë pozitiv). "
            "Kufijtë e përdorur për klasifikim janë 0,05 dhe -0,05: mbi 0,05 konsiderohet pozitiv, nën -0,05 negativ, ndërmjet tyre neutral."
        )
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
        n_total = len(df_filtered)
        show_limit = st.selectbox(
            "Shfaq deri në",
            options=[10, 25, 50, 100, 200],
            index=0,
            key="sentiment_limit",
        )
        show_limit = min(show_limit, n_total)
        df_to_show = df_filtered.head(show_limit)
        st.caption(f"Duke shfaqur {len(df_to_show)} nga {n_total} deklarata.")

        for idx, row in df_to_show.iterrows():
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
            title = f"{row.get('Speaker', '-')} ({row.get('Date') if pd.notna(row.get('Date')) else '-'}) — {arrow} {sentiment} ({percent}%)"
            with st.expander(title):
                st.markdown(
                    f"""<span style="border-left:4px solid {color}; padding-left:8px;">{text}</span>""",
                    unsafe_allow_html=True,
                )
                st.caption(f"SentimentScore: {round(score, 3)}  |  TTR: {round(row.get('TTR', 0), 3)}")
    else:
        st.info("Nuk ka të dhëna për këta filtra.")

# ==========================
# TAB 2: TOPICS
# ==========================

with tab2:
    st.subheader("Modelimi i Temave dhe Filtrimi i Deklaratave")
    with st.expander("Metodologjia e modelimit të temave"):
        st.markdown(
            "Temat nxirren automatikisht nga korpusi i deklaratave duke përdorur NMF (Non-negative Matrix Factorization) mbi paraqitjen TF-IDF të tekstit. "
            "Çdo deklaratë përshkruhet sipas rëndësisë së fjalëve, dhe NMF identifikon grupe fjalësh që shfaqen së bashku, duke formuar tema. "
            "Fjalëkyçet e secilës temë janë fjalët me peshën më të lartë në atë komponentë; numri i deklaratave tregon sa tekste u caktuan secilës temë."
        )
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
                    "TopKeywords": "Fjalëkyçe (etiketa e temës)",
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
            f"{str(keywords)[:50]}{'…' if len(str(keywords)) > 50 else ''} (Tema {int(tid)})"
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
            # Parse "keywords... (Tema N)" or "Tema N: keywords"
            if " (Tema " in selected_topic_label:
                selected_topic_id = int(
                    selected_topic_label.split(" (Tema ")[-1].rstrip(")")
                )
            else:
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
    with st.expander("Metodologjia e metrikave të stilit"):
        st.markdown(
            "Kjo faqe ofron metrika të thjeshta të stilit: gjatësia e deklaratave në numër fjalësh dhe TTR (Type-Token Ratio), pra raporti midis fjalëve unike dhe numrit total të fjalëve. "
            "TTR përdoret si tregues i pasurisë leksikore — vlera më e lartë tregon një fjalor më të larmishëm. "
            "Grafiku i temave kryesore përsërit fjalëkyçet e nxirra nga modelimi NMF dhe frekuencën e tyre në grupin e filtruar."
        )
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
    with st.expander("Metodologjia e krahasimit të folësve"):
        st.markdown(
            "Tabela përmbledh, për çdo folës, numrin e deklaratave, gjatësinë mesatare të deklaratave në fjalë, TTR mesatar dhe sentimentin mesatar. "
            "Grafikët e krahasimit ilustrojnë ndryshimet në pasurinë leksikore (TTR) dhe në shpërndarjen e sentimentit midis folësve të përzgjedhur, duke lejuar një krahasim vizual të stilit dhe tonit."
        )
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
    st.subheader("Bisedo me DIELLA AI (Shqip)")
    with st.expander("Funksionimi i modulit Q&A"):
        st.markdown(
            "Pyetja së pari kërkohet në korpusin e deklaratave (në shqip) përmes kërkimit vektorial (SentenceTransformer dhe FAISS). "
            "Deklaratat më të ngjashme me pyetjen përzgjidhen si kontekst dhe dërgohen te një model gjuhës (Groq), i cili formulon përgjigjen vetëm në bazë të atij konteksti (RAG: Retrieval-Augmented Generation), pa përdorur informacion nga burime të jashtme."
        )
    if not GROQ_API_KEY or not str(GROQ_API_KEY).strip():
        st.warning(
            "**Q&A nuk është i konfiguruar.** Vendosni **GROQ_API_KEY** në skedarin `.env` në dosjen e projektit për të përdorur bisedën me AI. [Groq](https://console.groq.com/) ofron çelësa falas."
        )
        query_shqip = None
    else:
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


# ==========================
# TAB: VLERËSIM (EVALUATION)
# ==========================

with tab_eval:
    st.subheader("Vlerësim i sistemit (për tezë)")
    with st.expander("Metodologjia e vlerësimit", expanded=True):
        st.markdown("""
        Vlerësimi përfshin dy pjesë. E para është vlerësimi i sentimentit: një grup deklarata me etiketa të caktuara manualisht (Pozitiv / Neutral / Negativ) në skedarin evaluation_sentiment_gold.csv krahasohet me parashikimet e VADER; saktësia dhe F1 matin pajtueshmërinë e modelit me këto etiketa. E dyta është koherenca e temave (NPMI): për temat e nxirra nga NMF matet nëse fjalëkyçet e tyre shfaqen së bashku në të njëjtat dokumente; vlera më e lartë NPMI tregon tema më koherente.
        """)
    st.markdown(
        "Ekzekutoni vlerësimin më poshtë. Rezultatet ruhen edhe në `evaluation_results.json`."
    )
    if st.button("Ekzekuto vlerësimin", type="primary", key="run_eval_btn"):
        eval_results = {}
        base_dir = Path(__file__).resolve().parent
        gold_path = base_dir / "evaluation_sentiment_gold.csv"
        data_path = base_dir / DATA_PATH

        with st.spinner("Po ekzekutohet vlerësimi i sentimentit..."):
            sent = run_evaluation.evaluate_sentiment(gold_path)
            if sent:
                eval_results.update(sent)

        with st.spinner("Po llogaritet koherenca e temave (NPMI)..."):
            coh = run_evaluation.run_topic_coherence(data_path)
            if coh:
                eval_results.setdefault("topic_coherence", {}).update(coh["topic_coherence"])

        if eval_results:
            # Save to file and session state so results stay visible
            results_file = base_dir / "evaluation_results.json"
            with open(results_file, "w", encoding="utf-8") as f:
                json.dump(eval_results, f, indent=2, ensure_ascii=False)
            st.session_state["eval_results"] = eval_results
            st.success(f"Rezultatet u ruajtën në `{results_file.name}`.")
        else:
            st.session_state["eval_results"] = None
            st.warning(
                "Nuk u gjenden rezultate. Kontrolloni që ekziston "
                "`evaluation_sentiment_gold.csv` (kolona: Speech, GoldLabel) dhe që të dhënat kryesore janë të ngarkuara."
            )
        st.rerun()

    # Show last evaluation results if available
    if st.session_state.get("eval_results"):
        eval_results = st.session_state["eval_results"]
        # ----- Sentiment -----
        if "sentiment" in eval_results:
            s = eval_results["sentiment"]
            st.markdown("### Sentiment (VADER vs etiketa të arta)")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Saktësia", f"{s['accuracy']:.2%}")
            with c2:
                st.metric("F1 (macro)", f"{s['f1_macro']:.4f}")
            with c3:
                st.metric("Mostra", s["n_samples"])
            st.code(s["classification_report"], language=None)
            st.markdown("**Matrica e konfuzionit:**")
            st.dataframe(
                pd.DataFrame(
                    s["confusion_matrix"],
                    index=s["labels"],
                    columns=s["labels"],
                ),
                use_container_width=False,
            )

        # ----- Topic coherence -----
        if "topic_coherence" in eval_results:
            tc = eval_results["topic_coherence"]
            st.markdown("### Koherenca e temave (NPMI)")
            st.metric("NPMI (mesatare)", f"{tc.get('npmi_mean', 0):.4f}")
            if "n_docs_used" in tc:
                st.caption(f"Dokumente të përdorura: {tc['n_docs_used']}")

                                