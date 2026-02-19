# Sentiment tab

import pandas as pd
import streamlit as st
from utils.visualization import create_sentiment_bar_chart, create_sentiment_trend_chart


def render(df_filtered):
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
