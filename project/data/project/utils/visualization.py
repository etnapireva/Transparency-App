# ==========================================
# VISUALIZATION MODULE - DIELLA AI
# ==========================================

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import altair as alt
from config import (
    COLOR_POSITIVE,
    COLOR_NEGATIVE,
    COLOR_NEUTRAL,
)

# Dark theme colors
DARK_BG = "#010409"
DARK_PAPER = "#0f172a"
DARK_TEXT = "#e5e7eb"
DARK_TEXT_SECONDARY = "#9ca3af"
DARK_GRID = "#1f2937"

# Plotly dark template
PLOTLY_DARK_TEMPLATE = {
    "layout": {
        "paper_bgcolor": DARK_PAPER,
        "plot_bgcolor": DARK_PAPER,
        "font": {"color": DARK_TEXT, "family": "Inter, sans-serif"},
        "xaxis": {
            "gridcolor": DARK_GRID,
            "linecolor": DARK_GRID,
            "zerolinecolor": DARK_GRID,
        },
        "yaxis": {
            "gridcolor": DARK_GRID,
            "linecolor": DARK_GRID,
            "zerolinecolor": DARK_GRID,
        },
    }
}


def create_sentiment_pie_chart(df_filtered):
    """
    Create sentiment distribution pie chart.
    
    Args:
        df_filtered (pd.DataFrame): Filtered dataframe
        
    Returns:
        plotly.graph_objects.Figure: Pie chart
    """
    if df_filtered.empty:
        return None

    sentiment_counts = df_filtered["SentimentLabel"].value_counts()
    pie_data = pd.DataFrame({
        "Sentiment": sentiment_counts.index,
        "Count": sentiment_counts.values,
    })

    fig = px.pie(
        pie_data,
        names="Sentiment",
        values="Count",
        color="Sentiment",
        color_discrete_map={
            "Pozitiv": "#22c55e",
            "Negativ": "#ef4444",
            "Neutral": "#3b82f6",
        },
        title="Përqindja e Sentimenteve",
        hole=0.3,
    )
    fig.update_layout(
        template=PLOTLY_DARK_TEMPLATE,
        title_font_size=16,
        title_font_color=DARK_TEXT,
        font_color=DARK_TEXT,
        legend=dict(
            bgcolor=DARK_PAPER,
            bordercolor=DARK_GRID,
            borderwidth=1,
            font_color=DARK_TEXT,
        ),
    )
    fig.update_traces(
        textposition="inside",
        textinfo="percent+label",
        hovertemplate="<b>%{label}</b><br>Numri: %{value}<br>Përqindja: %{percent}<extra></extra>",
    )
    return fig


def create_sentiment_trend_chart(df_filtered):
    """
    Create daily sentiment trend line chart.
    
    Args:
        df_filtered (pd.DataFrame): Filtered dataframe with date column
        
    Returns:
        plotly.graph_objects.Figure or None: Line chart
    """
    if df_filtered.empty or not pd.api.types.is_datetime64_any_dtype(
        df_filtered["Date"]
    ):
        return None

    trend = (
        df_filtered.set_index("Date")["SentimentScore"]
        .resample("D")
        .mean()
        .fillna(0)
        .reset_index()
    )
    trend = trend[trend["SentimentScore"] != 0]

    if trend.empty:
        return None

    fig = px.line(
        trend,
        x="Date",
        y="SentimentScore",
        markers=True,
        title="Sentimenti Mesatar Ditor",
        color_discrete_sequence=["#6366f1"],
    )
    fig.update_traces(
        line=dict(width=3),
        marker=dict(size=8),
        hovertemplate="<b>Data:</b> %{x}<br><b>Sentiment:</b> %{y:.3f}<extra></extra>",
    )
    fig.add_hline(
        y=0.05,
        line_dash="dash",
        line_color="#22c55e",
        line_width=1.5,
        annotation_text="Pozitiv",
        annotation_font_color="#22c55e",
    )
    fig.add_hline(
        y=-0.05,
        line_dash="dash",
        line_color="#ef4444",
        line_width=1.5,
        annotation_text="Negativ",
        annotation_font_color="#ef4444",
    )
    fig.add_hline(
        y=0,
        line_dash="dot",
        line_color=DARK_TEXT_SECONDARY,
        line_width=1,
        opacity=0.5,
    )
    fig.update_yaxes(range=[-1, 1], title="Sentiment Score")
    fig.update_xaxes(title="Data")
    fig.update_layout(
        template=PLOTLY_DARK_TEMPLATE,
        title_font_size=16,
        title_font_color=DARK_TEXT,
        hovermode="x unified",
        height=350,
    )
    return fig


def create_sentiment_bar_chart(df_filtered):
    """
    Create sentiment count bar chart.
    
    Args:
        df_filtered (pd.DataFrame): Filtered dataframe
        
    Returns:
        plotly.graph_objects.Figure: Bar chart
    """
    if df_filtered.empty:
        return None

    sentiment_data = df_filtered.groupby("SentimentLabel").size().reset_index(
        name="Vlera"
    )

    fig = px.bar(
        sentiment_data,
        x="SentimentLabel",
        y="Vlera",
        text="Vlera",
        color="SentimentLabel",
        color_discrete_map={
            "Pozitiv": "#22c55e",
            "Negativ": "#ef4444",
            "Neutral": "#3b82f6",
        },
        title="Numri i Deklaratave sipas Sentimentit",
    )
    fig.update_traces(
        textposition="outside",
        textfont_size=12,
        textfont_color=DARK_TEXT,
        hovertemplate="<b>%{x}</b><br>Numri: %{y}<extra></extra>",
    )
    fig.update_layout(
        template=PLOTLY_DARK_TEMPLATE,
        title_font_size=16,
        title_font_color=DARK_TEXT,
        xaxis_title="Sentiment",
        yaxis_title="Numri i deklaratave",
        height=350,
    )
    return fig


def create_topics_bar_chart(df_filtered):
    """
    Create topics distribution bar chart.
    X-axis shows topic keyword labels (truncated) instead of topic ID.
    """
    if df_filtered.empty:
        return None

    topic_data = (
        df_filtered.groupby("Topic")
        .agg({"TopKeywords": "first", "Speech": "count"})
        .reset_index()
        .rename(columns={"Speech": "Vlera"})
    )
    # Label for display: first ~40 chars of keywords, or "Tema N: kw1, kw2..."
    topic_data["TopicLabel"] = topic_data.apply(
        lambda r: (str(r["TopKeywords"])[:42] + "…") if len(str(r["TopKeywords"])) > 42 else str(r["TopKeywords"]) or f"Tema {int(r['Topic'])}",
        axis=1,
    )

    fig = px.bar(
        topic_data,
        x="TopicLabel",
        y="Vlera",
        text="Vlera",
        color="Vlera",
        color_continuous_scale="viridis",
        hover_data={"TopicLabel": False, "TopKeywords": True, "Vlera": True, "Topic": True},
    )
    fig.update_traces(
        textposition="outside",
        textfont_size=11,
        textfont_color=DARK_TEXT,
        hovertemplate="<b>Tema %{customdata[2]}</b><br>Fjalëkyçe: %{customdata[0]}<br>Numri: %{y}<extra></extra>",
    )
    fig.update_xaxes(
        tickangle=-45,
        title="Tema (fjalëkyçe)",
        tickfont_size=10,
    )
    fig.update_layout(
        template=PLOTLY_DARK_TEMPLATE,
        title_font_size=16,
        title_font_color=DARK_TEXT,
        xaxis_title="Tema (fjalëkyçe)",
        yaxis_title="Numri i deklaratave",
        coloraxis_colorbar=dict(
            title="Numri",
            title_font_color=DARK_TEXT,
            tickfont_color=DARK_TEXT,
        ),
        height=400,
    )
    return fig


def create_wordcount_histogram(df_filtered):
    """
    Create word count distribution histogram.
    
    Args:
        df_filtered (pd.DataFrame): Filtered dataframe
        
    Returns:
        altair.Chart: Bar chart
    """
    if df_filtered.empty:
        return None

    bins = [0, 50, 100, 200, 500, 1000, 5000]
    labels = ["0-50", "51-100", "101-200", "201-500", "501-1000", "1000+"]
    
    df_copy = df_filtered.copy()
    df_copy["WordCountBin"] = pd.cut(
        df_copy["WordCount"],
        bins=bins,
        labels=labels,
        right=False,
    )
    
    style_data = df_copy.groupby("WordCountBin").size().reset_index(name="Numri")

    fig = (
        alt.Chart(style_data)
        .mark_bar(cornerRadius=4)
        .encode(
            x=alt.X("WordCountBin:O", title="Gjatësia e deklaratës (fjalë)", axis=alt.Axis(labelColor=DARK_TEXT, titleColor=DARK_TEXT, gridColor=DARK_GRID)),
            y=alt.Y("Numri:Q", title="Numri i deklaratave", axis=alt.Axis(labelColor=DARK_TEXT, titleColor=DARK_TEXT, gridColor=DARK_GRID)),
            tooltip=[alt.Tooltip("WordCountBin", title="Gjatësia"), alt.Tooltip("Numri", title="Numri i deklaratave", format=".0f")],
            color=alt.Color("Numri:Q", scale=alt.Scale(scheme="blues"), legend=None),
        )
        .properties(width=350, height=300)
        .configure_view(strokeWidth=0, fill=DARK_PAPER)
        .configure_axis(domainColor=DARK_GRID, tickColor=DARK_GRID)
        .configure_text(color=DARK_TEXT)
    )
    return fig


def create_speaker_comparison_chart(df, speakers_to_compare):
    """
    Create speaker TTR comparison chart.
    
    Args:
        df (pd.DataFrame): Full dataframe
        speakers_to_compare (list): List of speakers to compare
        
    Returns:
        altair.Chart: Bar chart
    """
    if not speakers_to_compare or df.empty:
        return None

    speaker_stats = df[df["Speaker"].isin(speakers_to_compare)].groupby("Speaker").agg(
        Avg_TTR=("TTR", "mean"),
        Count=("Speech_SQ", "size"),
    ).reset_index()

    fig = (
        alt.Chart(speaker_stats)
        .mark_bar(cornerRadius=4)
        .encode(
            x=alt.X("Avg_TTR", title="Pasuria Leksikore (TTR)", axis=alt.Axis(labelColor=DARK_TEXT, titleColor=DARK_TEXT, gridColor=DARK_GRID)),
            y=alt.Y("Speaker", title="Folësi", sort="-x", axis=alt.Axis(labelColor=DARK_TEXT, titleColor=DARK_TEXT)),
            color=alt.Color("Speaker", legend=None, scale=alt.Scale(scheme="category10")),
            tooltip=[
                alt.Tooltip("Speaker", title="Folësi"),
                alt.Tooltip("Avg_TTR", title="TTR mesatar", format=".3f"),
                alt.Tooltip("Count", title="Numri deklaratave", format=".0f"),
            ],
        )
        .properties(height=300)
        .configure_view(strokeWidth=0, fill=DARK_PAPER)
        .configure_axis(domainColor=DARK_GRID, tickColor=DARK_GRID)
        .configure_text(color=DARK_TEXT)
    )
    return fig


def create_speaker_sentiment_boxplot(df, speakers_to_compare):
    """
    Create speaker sentiment distribution boxplot.
    
    Args:
        df (pd.DataFrame): Full dataframe
        speakers_to_compare (list): List of speakers to compare
        
    Returns:
        plotly.graph_objects.Figure: Boxplot
    """
    if not speakers_to_compare or df.empty:
        return None

    comp_df = df[df["Speaker"].isin(speakers_to_compare)]

    fig = px.box(
        comp_df,
        x="SentimentScore",
        y="Speaker",
        color="Speaker",
        orientation="h",
        title="Shpërndarja e SentimentiScore (-1 Negativ, +1 Pozitiv)",
        points="outliers",
        color_discrete_sequence=px.colors.qualitative.Set3,
    )
    fig.update_traces(
        hovertemplate="<b>%{y}</b><br>Sentiment: %{x:.3f}<extra></extra>",
        marker=dict(size=4, opacity=0.6),
    )
    fig.update_xaxes(
        range=[-1.0, 1.0],
        title="Sentiment Score",
        gridcolor=DARK_GRID,
    )
    fig.update_yaxes(title="Folësi", gridcolor=DARK_GRID)
    fig.update_layout(
        template=PLOTLY_DARK_TEMPLATE,
        title_font_size=16,
        title_font_color=DARK_TEXT,
        height=350,
        showlegend=False,
        margin=dict(l=20, r=20, t=50, b=20),
        hovermode="closest",
    )
    return fig