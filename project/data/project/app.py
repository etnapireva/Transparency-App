# app.py
import streamlit as st
import pandas as pd
import numpy as np
from transformers import pipeline
from sentence_transformers import SentenceTransformer

# --------------------------
# Load data and models
# --------------------------
df = pd.read_csv("data/diella_speeches_clean.csv")  # sigurohu që e ke pastruar
sentiment_pipe = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment", tokenizer="nlptown/bert-base-multilingual-uncased-sentiment")
embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
doc_embeddings = embedder.encode(df["Speech"].tolist(), normalize_embeddings=True, convert_to_numpy=True)

# --------------------------
# Helper functions (thjeshtuar)
# --------------------------
def split_sentences(text):
    import re
    return [s.strip() for s in re.split(r"(?<=[\.\!\?])\s+", text) if s.strip()]

def get_sentiment_label(text):
    res = sentiment_pipe(text[:512])[0]
    digits = "".join(ch for ch in res["label"] if ch.isdigit())
    n = int(digits) if digits else 3
    return "negative" if n <= 2 else "neutral" if n==3 else "positive"

def most_similar_speech_index(user_text):
    q_emb = embedder.encode([user_text], normalize_embeddings=True, convert_to_numpy=True)[0]
    sims = np.dot(doc_embeddings, q_emb)
    best_idx = int(np.argmax(sims))
    return best_idx, float(sims[best_idx])

def best_snippet_for_query(speech_text, user_text, max_sentences=2):
    sentences = split_sentences(speech_text)
    if len(sentences)==1: return sentences[0]
    q_emb = embedder.encode([user_text], normalize_embeddings=True, convert_to_numpy=True)[0]
    sent_embs = embedder.encode(sentences, normalize_embeddings=True, convert_to_numpy=True)
    sims = np.dot(sent_embs, q_emb)
    top_indices = sorted(np.argsort(-sims)[:max_sentences])
    return " ".join(sentences[i] for i in top_indices)

# --------------------------
# Streamlit UI
# --------------------------
st.title("🗨️ Pyet Ministreshën Diella")
user_input = st.text_input("Shkruaj pyetjen tënde këtu:")

if st.button("Dërgo") and user_input.strip():
    sentiment = get_sentiment_label(user_input)
    best_idx, best_sim = most_similar_speech_index(user_input)
    snippet = best_snippet_for_query(df.loc[best_idx, "Speech"], user_input)
    
    prefix = ""
    if sentiment=="positive": prefix="Ju falënderoj për mbështetjen. "
    if sentiment=="negative": prefix="E kuptoj shqetësimin tuaj. "
    
    response = prefix + snippet
    st.markdown(f"**Diella:** {response}")
    st.markdown(f"_Sentimenti yt i pyetjes: {sentiment}_")
