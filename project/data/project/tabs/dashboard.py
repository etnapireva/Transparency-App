# Dashboard tab

import pandas as pd
import streamlit as st
from utils.visualization import create_sentiment_pie_chart, create_sentiment_trend_chart


def render(df_filtered):
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
        avg_sent = round(df_filtered["SentimentScore"].mean(), 3) if not df_filtered.empty else 0.0
        st.metric("Sentimenti Mesatar", avg_sent)
    with colC:
        active_speaker = df_filtered["Speaker"].value_counts().idxmax() if not df_filtered.empty else "-"
        st.metric("Folësi më Aktiv", active_speaker)

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

    st.subheader("Deklaratat e Fundit")
    n_last = st.selectbox("Numri i deklaratave", options=[5, 10, 20], index=0, key="dashboard_n_last")
    lang_last = st.radio("Gjuha e tekstit", options=["Shqip", "English"], index=0, horizontal=True, key="dashboard_lang")
    text_col = "Speech_SQ" if lang_last == "Shqip" else "Speech"
    last_statements = df_filtered.sort_values("Date", ascending=False).head(n_last) if not df_filtered.empty else pd.DataFrame()
    for _, row in last_statements.iterrows():
        sentiment = row.get("SentimentLabel", "Neutral")
        color = "green" if sentiment == "Pozitiv" else ("red" if sentiment == "Negativ" else "blue")
        arrow = "↑" if sentiment == "Pozitiv" else ("↓" if sentiment == "Negativ" else "→")
        text = str(row.get(text_col, ""))[:200]
        if len(str(row.get(text_col, ""))) > 200:
            text += "..."
        st.markdown(
            f'<div style="padding:10px; margin-bottom:5px; border-left:4px solid {color}; background-color:#1e293b; color:#e2e8f0;">'
            f'<b>{row.get("Speaker", "-")} ({row.get("Date") if pd.notna(row.get("Date")) else "-"}):</b> {text}<br>'
            f'<span style="color:{color}; font-weight:bold;">{arrow} SentimentScore: {round(row.get("SentimentScore", 0), 2)}, TTR: {round(row.get("TTR", 0), 3)}</span></div>',
            unsafe_allow_html=True,
        )

    st.subheader("Rreth Sistemit")
    st.markdown("""
<div style="background-color:#1e293b; color:#e2e8f0; padding:20px; border-radius:10px; line-height:1.6; border-left:5px solid #6366f1;">
<p>Ky sistem është krijuar për të analizuar në mënyrë të thelluar deklaratat publike të ministres Diella, duke përdorur teknika të avancuara të përpunimit të gjuhës natyrore dhe analiza statistikore.</p>
<ul>
<li><b>Analizë Sentimenti (VADER NLP)</b> – Vlerëson tonin emocional të deklaratave.</li>
<li><b>Matje të transparencës (TTR)</b> – Type-Token Ratio për pasurinë leksikore.</li>
<li><b>Statistikë dhe vizualizime interaktive</b> – Grafikë për sentiment, tema, krahasime.</li>
</ul>
<p>Qëllimi është të ofrojë një pasqyrë të qartë të komunikimit publik dhe transparencës.</p>
</div>
""", unsafe_allow_html=True)
