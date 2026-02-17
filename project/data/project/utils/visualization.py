# ==========================================
# VISUALIZATION MODULE - DIELLA AI
# ==========================================

import pandas as pd
import plotly.express as px
import altair as alt
from config import (
    COLOR_POSITIVE,
    COLOR_NEGATIVE,
    COLOR_NEUTRAL,
)


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
            "Pozitiv": COLOR_POSITIVE,
            "Negativ": COLOR_NEGATIVE,
            "Neutral": COLOR_NEUTRAL,
        },
        title="Përqindja e Sentimenteve",
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
    )
    fig.add_hline(
        y=0.05,
        line_dash="dash",
        line_color=COLOR_POSITIVE,
        annotation_text="Pozitiv",
    )
    fig.add_hline(
        y=-0.05,
        line_dash="dash",
        line_color=COLOR_NEGATIVE,
        annotation_text="Negativ",
    )
    fig.update_yaxes(range=[-1, 1])
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
            "Pozitiv": COLOR_POSITIVE,
            "Negativ": COLOR_NEGATIVE,
            "Neutral": COLOR_NEUTRAL,
        },
        title="Numri i Deklaratave sipas Sentimentit",
    )
    fig.update_traces(textposition="outside")
    return fig


def create_topics_bar_chart(df_filtered):
    """
    Create topics distribution bar chart.
    
    Args:
        df_filtered (pd.DataFrame): Filtered dataframe
        
    Returns:
        plotly.graph_objects.Figure: Bar chart
    """
    if df_filtered.empty:
        return None

    topic_data = (
        df_filtered.groupby("Topic")
        .agg({"TopKeywords": "first", "Speech": "count"})
        .reset_index()
        .rename(columns={"Speech": "Vlera"})
    )

    fig = px.bar(
        topic_data,
        x="Topic",
        y="Vlera",
        text="Vlera",
        color="Vlera",
        color_continuous_scale="plasma",
        hover_data=["TopKeywords"],
    )
    fig.update_traces(textposition="outside")
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
        .mark_bar()
        .encode(
            x=alt.X("WordCountBin:O", title="Gjatësia e deklaratës (fjalë)"),
            y=alt.Y("Numri:Q", title="Numri i deklaratave"),
            tooltip=["WordCountBin", "Numri"],
            color=alt.Color("Numri:Q", scale=alt.Scale(scheme="blues")),
        )
        .properties(width=350, height=300)
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
        .mark_bar()
        .encode(
            x=alt.X("Avg_TTR", title="Pasuria Leksikore (TTR)"),
            y=alt.Y("Speaker", title="Folësi", sort="-x"),
            color=alt.Color("Speaker", legend=None),
            tooltip=["Speaker", "Avg_TTR", "Count"],
        )
        .properties(height=300)
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
        points="all",
    )
    fig.update_xaxes(range=[-1.0, 1.0])
    fig.update_layout(
        height=350,
        showlegend=False,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig