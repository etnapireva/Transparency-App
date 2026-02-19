# Style Metrics tab

import altair as alt
import streamlit as st
from utils.visualization import create_wordcount_histogram

DARK_TEXT = "#e5e7eb"
DARK_GRID = "#1f2937"


def render(df_filtered):
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
            style_data2 = (
                df_filtered.copy()
                .groupby("TopKeywords")
                .size()
                .reset_index(name="Numri")
                .sort_values("Numri", ascending=False)
                .head(10)
            )

            fig = (
                alt.Chart(style_data2)
                .mark_bar(cornerRadius=4)
                .encode(
                    x=alt.X(
                        "TopKeywords",
                        title="Temat kryesore",
                        sort="-y",
                        axis=alt.Axis(labelColor=DARK_TEXT, titleColor=DARK_TEXT, gridColor=DARK_GRID, labelAngle=-45),
                    ),
                    y=alt.Y("Numri", title="Numri i deklaratave", axis=alt.Axis(labelColor=DARK_TEXT, titleColor=DARK_TEXT, gridColor=DARK_GRID)),
                    tooltip=[
                        alt.Tooltip("TopKeywords", title="Tema"),
                        alt.Tooltip("Numri", title="Numri i deklaratave", format=".0f"),
                    ],
                    color=alt.Color("Numri", scale=alt.Scale(scheme="viridis"), legend=None),
                )
                .properties(width=350, height=300)
                .configure_view(strokeWidth=0, fill="#0f172a")
                .configure_axis(domainColor=DARK_GRID, tickColor=DARK_GRID)
                .configure_text(color=DARK_TEXT)
            )
            st.altair_chart(fig, use_container_width=True, key="bar_keywords_style")
    else:
        st.info("Nuk ka të dhëna për këta filtra.")
