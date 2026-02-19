# Topics tab
import streamlit as st
from utils.visualization import create_topics_bar_chart


def render(df_filtered):
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
        selected_topic_label = st.selectbox("Përzgjedhja e Temës", topic_options, index=0)
        if selected_topic_label != "— Zgjidh Temën për të parë deklaratat —":
            if " (Tema " in selected_topic_label:
                selected_topic_id = int(selected_topic_label.split(" (Tema ")[-1].rstrip(")"))
            else:
                selected_topic_id = int(
                    selected_topic_label.split(":")[0].replace("Tema ", "").strip()
                )
            speeches_in_topic = df_filtered[df_filtered["Topic"] == selected_topic_id]
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
