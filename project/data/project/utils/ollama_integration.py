# ==========================================
# VECTOR STORE & CONTEXT BUILDING MODULE
# ==========================================

import pandas as pd
from .vector_store import search_similar_documents
from config import MAX_QA_DOCS, MAX_CHARS_CONTEXT


def build_qa_context(query, model, index, df, max_docs=MAX_QA_DOCS, max_chars=MAX_CHARS_CONTEXT):
    """
    Build context from vector search results for Q&A.
    
    Args:
        query (str): Query in Albanian
        model: SentenceTransformer model
        index: FAISS index
        df (pd.DataFrame): Original dataframe
        max_docs (int): Maximum number of documents to include
        max_chars (int): Maximum characters for context
        
    Returns:
        tuple: (context_text, sources_list)
    """
    # Search similar documents
    relevant_docs = search_similar_documents(
        query,
        model,
        index,
        df,
        k=max_docs,
    )

    if relevant_docs.empty:
        return "", []

    # Remove duplicates
    relevant_docs = relevant_docs.drop_duplicates(subset=["Speech_SQ"])

    # Build numbered sources and context
    sources = []
    ctx_parts = []
    char_count = 0

    for i, (_, row) in enumerate(relevant_docs.iterrows(), start=1):
        date_str = (
            row["Date"].date()
            if pd.notna(row.get("Date"))
            else "-"
        )
        speech_sq = str(row.get("Speech_SQ", "")).strip()

        if not speech_sq:
            continue

        sources.append({
            "id": i,
            "speaker": row.get("Speaker", "-"),
            "date": date_str,
            "text": speech_sq,
        })

        part = f"[{i}] Deklaratë nga {row.get('Speaker', '-')} ({date_str}): {speech_sq}"
        ctx_parts.append(part)

        char_count += len(part)
        if char_count > max_chars:
            break

    context_text = "\n\n".join(ctx_parts)
    return context_text, sources