# Dashboard tab

import html
from datetime import datetime

import pandas as pd
import streamlit as st
from utils.visualization import create_sentiment_pie_chart, create_sentiment_trend_chart


def _build_report_html(df_filtered):
    """Gjeneron HTML për raportin e shkarkueshëm."""
    n = len(df_filtered)
    avg_sent = round(df_filtered["SentimentScore"].mean(), 3) if not df_filtered.empty else 0.0
    avg_ttr = round(df_filtered["TTR"].mean(), 3) if not df_filtered.empty else 0.0
    active = df_filtered["Speaker"].value_counts().idxmax() if not df_filtered.empty else "-"
    last_10 = df_filtered.sort_values("Date", ascending=False).head(10) if not df_filtered.empty else pd.DataFrame()

    rows_html = ""
    for _, row in last_10.iterrows():
        date_val = row.get("Date")
        date_str = str(date_val)[:10] if pd.notna(date_val) else "-"
        speaker = html.escape(str(row.get("Speaker", "-")))
        label = html.escape(str(row.get("SentimentLabel", "")))
        text = html.escape(str(row.get("Speech_SQ", row.get("Speech", ""))[:150]))
        if len(str(row.get("Speech_SQ", row.get("Speech", "")))) > 150:
            text += "..."
        rows_html += f"<tr><td>{date_str}</td><td>{speaker}</td><td>{label}</td><td>{text}</td></tr>"

    generated = datetime.now().strftime("%d.%m.%Y %H:%M")
    return f"""<!DOCTYPE html>
<html lang="sq">
<head>
<meta charset="UTF-8">
<title>Raport DIELLA AI</title>
<style>
body {{ font-family: 'Segoe UI', sans-serif; margin: 24px; background: #0f172a; color: #e2e8f0; }}
h1 {{ color: #facc15; border-bottom: 2px solid #6366f1; padding-bottom: 8px; }}
h2 {{ color: #94a3b8; margin-top: 24px; }}
table {{ border-collapse: collapse; width: 100%; margin-top: 12px; }}
th, td {{ border: 1px solid #334155; padding: 10px; text-align: left; }}
th {{ background: #1e293b; color: #facc15; }}
tr:nth-child(even) {{ background: #1e293b; }}
.metric {{ display: inline-block; background: #1e293b; padding: 12px 20px; border-radius: 8px; margin: 8px 8px 8px 0; border-left: 4px solid #6366f1; }}
.metric span {{ color: #94a3b8; font-size: 0.9em; }}
.footer {{ margin-top: 32px; font-size: 0.85em; color: #64748b; }}
</style>
</head>
<body>
<h1>DIELLA AI — Raport i shkurtër</h1>
<p><strong>Data e gjenerimit:</strong> {generated}</p>

<h2>Përmbledhje</h2>
<div class="metric"><span>Total deklarata</span><br><strong>{n}</strong></div>
<div class="metric"><span>Sentimenti mesatar</span><br><strong>{avg_sent}</strong></div>
<div class="metric"><span>TTR mesatar</span><br><strong>{avg_ttr}</strong></div>
<div class="metric"><span>Folësi më aktiv</span><br><strong>{html.escape(str(active))}</strong></div>

<h2>10 deklaratat e fundit</h2>
<table>
<thead><tr><th>Data</th><th>Folësi</th><th>Sentiment</th><th>Deklarata (fragment)</th></tr></thead>
<tbody>
{rows_html}
</tbody>
</table>

<p class="footer">Raport i gjeneruar nga DIELLA AI. Punim diplome — UBT 2025–2026.</p>
</body>
</html>"""


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

    st.subheader("Shkarko raport")
    st.caption("Gjenero një raport HTML me përmbledhje dhe 10 deklaratat e fundit për të dhënat e filtruara.")
    report_html = _build_report_html(df_filtered)
    st.download_button(
        label="Gjenero dhe shkarko raportin (HTML)",
        data=report_html,
        file_name=f"raport_diella_ai_{datetime.now().strftime('%Y%m%d_%H%M')}.html",
        mime="text/html",
        key="download_report_html",
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
