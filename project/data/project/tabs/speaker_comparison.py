# Speaker comparison tab

import streamlit as st
from utils.visualization import create_speaker_comparison_chart, create_speaker_sentiment_boxplot


def render(df, speaker_list_raw):
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
        speaker_list_raw,
        default=speaker_list_raw[:2] if len(speaker_list_raw) >= 2 else speaker_list_raw,
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
