# ==========================================
# DIELLA AI - EVALUATION SCRIPT (THESIS)
# ==========================================
# Run: python run_evaluation.py
# - Sentiment: accuracy, F1, confusion matrix (requires evaluation_sentiment_gold.csv)
# - Topic coherence: NPMI on NMF topics (uses main data CSV)

import json
import re
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from config import DATA_PATH
from utils.nlp_analysis import add_sentiment, get_nmf_artifacts_and_top_words


# ---------- Paths ----------
BASE_DIR = Path(__file__).resolve().parent
GOLD_CSV = BASE_DIR / "evaluation_sentiment_gold.csv"
RESULTS_FILE = BASE_DIR / "evaluation_results.json"


def _normalize_label(s: str) -> str:
    """Map gold label to one of: Pozitiv, Neutral, Negativ."""
    if pd.isna(s):
        return "Neutral"
    s = str(s).strip().lower()
    if s in ("pozitiv", "positive", "pos"):
        return "Pozitiv"
    if s in ("negativ", "negative", "neg"):
        return "Negativ"
    return "Neutral"


def _tokenize_for_coherence(text: str) -> set:
    """Tokenize like TF-IDF (word chars only, lower)."""
    if pd.isna(text) or not text:
        return set()
    return set(re.findall(r"\b[a-z]+\b", str(text).lower()))


def evaluate_sentiment(gold_path: Path) -> dict | None:
    """
    Load gold CSV (columns: Speech, GoldLabel), run VADER, compute metrics.
    GoldLabel must be Pozitiv / Neutral / Negativ (or English equivalents).
    """
    if not gold_path.exists():
        print(f"Gold file not found: {gold_path}")
        print("Create it with columns: Speech, GoldLabel (Pozitiv | Neutral | Negativ)")
        return None

    gold = pd.read_csv(gold_path, encoding="utf-8")
    if "Speech" not in gold.columns or "GoldLabel" not in gold.columns:
        print("Gold CSV must have columns: Speech, GoldLabel")
        return None

    gold = gold.dropna(subset=["Speech"])
    gold["GoldLabel"] = gold["GoldLabel"].apply(_normalize_label)
    if len(gold) == 0:
        print("No rows in gold file after dropping empty Speech.")
        return None

    # Use same sentiment logic as the app
    gold = add_sentiment(gold.copy())
    y_true = gold["GoldLabel"].tolist()
    y_pred = gold["SentimentLabel"].tolist()

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    report = classification_report(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    results = {
        "sentiment": {
            "accuracy": round(float(acc), 4),
            "f1_macro": round(float(f1_macro), 4),
            "f1_weighted": round(float(f1_weighted), 4),
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
            "labels": sorted(set(y_true) | set(y_pred)),
            "n_samples": len(gold),
        }
    }
    return results


def compute_topic_coherence_npmi(df: pd.DataFrame, speech_col: str = "Speech") -> float | None:
    """
    Document-level NPMI coherence for NMF topics.
    Returns mean coherence over topics (higher = more coherent).
    """
    vec, nmf, top_words_per_topic = get_nmf_artifacts_and_top_words(df, speech_col)
    if vec is None or not top_words_per_topic:
        return None

    # Tokenized corpus: one set of words per document
    corpus = [
        _tokenize_for_coherence(text)
        for text in df[speech_col].fillna("").astype(str)
    ]
    D = len(corpus)
    if D == 0:
        return None

    def doc_count(word: str) -> int:
        return sum(1 for doc in corpus if word in doc)

    def doc_count_pair(w1: str, w2: str) -> int:
        return sum(1 for doc in corpus if w1 in doc and w2 in doc)

    eps = 1e-10
    npmi_scores = []
    for words in top_words_per_topic:
        words = [w for w in words if w]
        if len(words) < 2:
            continue
        pair_npmi = []
        for i in range(len(words)):
            for j in range(i + 1, len(words)):
                wi, wj = words[i], words[j]
                c_i = doc_count(wi) / D
                c_j = doc_count(wj) / D
                c_ij = doc_count_pair(wi, wj) / D
                c_i = max(c_i, eps)
                c_j = max(c_j, eps)
                c_ij = max(c_ij, eps)
                pmi = np.log(c_ij) - np.log(c_i) - np.log(c_j)
                npmi = pmi / (-np.log(c_ij)) if c_ij < 1 else 0.0
                pair_npmi.append(npmi)
        if pair_npmi:
            npmi_scores.append(np.mean(pair_npmi))

    if not npmi_scores:
        return None
    return float(np.mean(npmi_scores))


def run_topic_coherence(data_path: Path) -> dict | None:
    """Load main data (no NLP preprocess), fit NMF, return coherence."""
    if not data_path.exists():
        print(f"Data file not found: {data_path}")
        return None
    df = pd.read_csv(data_path, nrows=500)
    if "Speech" not in df.columns:
        print("Data CSV must have a 'Speech' column.")
        return None
    coherence = compute_topic_coherence_npmi(df)
    if coherence is None:
        return None
    return {"topic_coherence": {"npmi_mean": round(coherence, 4), "n_docs_used": len(df)}}


def main():
    results = {}

    # ----- Sentiment -----
    print("Running sentiment evaluation (gold labels)...")
    sent = evaluate_sentiment(GOLD_CSV)
    if sent:
        results.update(sent)
        print("\n--- Sentiment results ---")
        print(f"Accuracy: {sent['sentiment']['accuracy']}")
        print(f"F1 (macro): {sent['sentiment']['f1_macro']}")
        print(f"F1 (weighted): {sent['sentiment']['f1_weighted']}")
        print(f"N samples: {sent['sentiment']['n_samples']}")
        print("\nClassification report:\n", sent["sentiment"]["classification_report"])
        print("Confusion matrix:\n", np.array(sent["sentiment"]["confusion_matrix"]))

    # ----- Topic coherence -----
    print("\nRunning topic coherence (NPMI)...")
    data_path = BASE_DIR / DATA_PATH
    coh = run_topic_coherence(data_path)
    if coh:
        results.setdefault("topic_coherence", {}).update(coh["topic_coherence"])
        print(f"NPMI coherence: {coh['topic_coherence']['npmi_mean']}")

    if results:
        with open(RESULTS_FILE, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {RESULTS_FILE}")

    return results


if __name__ == "__main__":
    main()
