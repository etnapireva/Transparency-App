# ==========================================
# DATA LOADER MODULE - DIELLA AI
# ==========================================

import pandas as pd
from .nlp_analysis import calculate_ttr, add_sentiment, add_topics


def load_data(path):
    """
    Load and preprocess CSV data.
    
    Args:
        path (str): Path to CSV file
        
    Returns:
        tuple: (pd.DataFrame, error_message) or (None, error_message) if failed
    """
    try:
        df = pd.read_csv(path, parse_dates=["Date"], engine="python")
    except FileNotFoundError:
        return None, f"CSV file not found: {path}"
    except Exception as e:
        return None, f"Error loading CSV: {str(e)}"

    # Ensure required columns exist
    required_columns = ["Speech", "Speech_SQ", "Speaker", "Date"]
    for col in required_columns:
        if col not in df.columns:
            if col in ["Speech", "Speech_SQ", "Speaker"]:
                df[col] = ""
            elif col == "Date":
                df[col] = pd.NaT

    # Clean and standardize columns
    df["Speech"] = df["Speech"].fillna("").astype(str)
    df["Speech_SQ"] = df["Speech_SQ"].fillna("").astype(str)
    df["Speaker"] = df["Speaker"].fillna("Unknown").astype(str)

    # Calculate basic metrics
    df["WordCount"] = df["Speech"].apply(lambda x: len(str(x).split()))
    df["TTR"] = df["Speech_SQ"].apply(calculate_ttr)

    # Add NLP features
    df = add_sentiment(df)
    df = add_topics(df)

    return df, None