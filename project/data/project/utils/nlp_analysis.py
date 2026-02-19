# ==========================================
# NLP ANALYSIS MODULE - DIELLA AI
# ==========================================

import re
import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from config import (
    SENTIMENT_POSITIVE_THRESHOLD,
    SENTIMENT_NEGATIVE_THRESHOLD,
    NUM_TOPICS,
    NUM_TOP_WORDS,
    TFIDF_MAX_FEATURES,
    TFIDF_MIN_DF,
)


def calculate_ttr(text):
    """
    Calculate Type-Token Ratio (TTR) for a given text.
    TTR measures lexical diversity (unique words / total words).
    
    Args:
        text (str): Input text
        
    Returns:
        float: TTR value between 0 and 1
    """
    if pd.isna(text) or not text:
        return 0.0

    # Tokenize supporting Unicode characters (exclude numbers and underscores)
    words = re.findall(r"[^\W\d_]+", str(text).lower(), flags=re.UNICODE)

    if len(words) == 0:
        return 0.0

    unique_words = set(words)
    ttr = len(unique_words) / len(words)
    return ttr


def add_sentiment(df):
    """
    Add sentiment analysis columns to dataframe.
    Uses VADER sentiment analyzer on the 'Speech' column (English text).
    Declarations are stored in both English (Speech) and Albanian (Speech_SQ);
    we use English here so VADER, which is built for English, gives reliable scores.
    
    Args:
        df (pd.DataFrame): Input dataframe with 'Speech' column (English)
        
    Returns:
        pd.DataFrame: Dataframe with 'SentimentScore' and 'SentimentLabel' columns
    """
    try:
        analyzer = SentimentIntensityAnalyzer()
        df["SentimentScore"] = df["Speech"].apply(
            lambda x: analyzer.polarity_scores(str(x))["compound"]
        )
        df["SentimentLabel"] = df["SentimentScore"].apply(
            lambda x: "Pozitiv"
            if x > SENTIMENT_POSITIVE_THRESHOLD
            else (
                "Negativ"
                if x < SENTIMENT_NEGATIVE_THRESHOLD
                else "Neutral"
            )
        )
    except Exception as e:
        print(f"Sentiment analysis error: {e}")
        df["SentimentScore"] = 0.0
        df["SentimentLabel"] = "Neutral"

    return df


def add_topics(df):
    """
    Add topic modeling to dataframe using NMF.
    
    Args:
        df (pd.DataFrame): Input dataframe with 'Speech' column
        
    Returns:
        pd.DataFrame: Dataframe with 'Topic' and 'TopKeywords' columns
    """
    non_empty_speeches = (
        df["Speech"].str.strip().replace("", np.nan).dropna()
    )

    if len(non_empty_speeches) < 2:
        df["Topic"] = -1
        df["TopKeywords"] = ""
        return df

    try:
        # TF-IDF Vectorization
        tfidf_vectorizer = TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,
            min_df=TFIDF_MIN_DF,
            stop_words="english",
        )
        tfidf = tfidf_vectorizer.fit_transform(non_empty_speeches)

        # NMF Topic Modeling
        n_components = min(NUM_TOPICS, tfidf.shape[0] - 1)
        nmf_model = NMF(
            n_components=n_components,
            random_state=42,
            max_iter=1000,
        )
        nmf_model.fit(tfidf)

        # Extract top words per topic
        tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()
        top_words_per_topic = []

        for topic_idx, topic in enumerate(nmf_model.components_):
            top_indices = topic.argsort()[:-NUM_TOP_WORDS - 1:-1]
            top_words = [tfidf_feature_names[i] for i in top_indices]
            top_words_per_topic.append(", ".join(top_words))

        # Map topics to original dataframe
        topic_values = nmf_model.transform(
            tfidf_vectorizer.transform(
                df["Speech"].fillna("").astype(str)
            )
        )
        df["Topic"] = topic_values.argmax(axis=1)
        df["TopKeywords"] = df["Topic"].apply(
            lambda x: (
                top_words_per_topic[x]
                if x < len(top_words_per_topic)
                else ""
            )
        )
    except Exception as e:
        print(f"Topic modeling error: {e}")
        df["Topic"] = -1
        df["TopKeywords"] = ""

    return df


def get_nmf_artifacts_and_top_words(df, speech_col="Speech"):
    """
    Fit NMF on the corpus and return vectorizer, model, and top words per topic.
    Used for evaluation (e.g. topic coherence).
    """
    import numpy as np
    non_empty = (
        df[speech_col].astype(str).str.strip().replace("", np.nan).dropna()
    )
    if len(non_empty) < 2:
        return None, None, []

    tfidf_vectorizer = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        min_df=TFIDF_MIN_DF,
        stop_words="english",
    )
    tfidf = tfidf_vectorizer.fit_transform(non_empty)
    n_components = min(NUM_TOPICS, tfidf.shape[0] - 1)
    nmf_model = NMF(
        n_components=n_components,
        random_state=42,
        max_iter=1000,
    )
    nmf_model.fit(tfidf)
    feature_names = tfidf_vectorizer.get_feature_names_out()
    top_words_per_topic = []
    for topic in nmf_model.components_:
        top_idx = topic.argsort()[:-NUM_TOP_WORDS - 1:-1]
        top_words_per_topic.append([feature_names[i] for i in top_idx])
    return tfidf_vectorizer, nmf_model, top_words_per_topic