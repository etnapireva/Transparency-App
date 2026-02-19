# ==========================================
# VECTOR STORE MODULE - DIELLA AI
# ==========================================

import numpy as np
from config import VECTOR_MODEL


def build_vector_store(df):
    """
    Build FAISS vector store from Albanian speeches.
    
    Args:
        df (pd.DataFrame): Dataframe with 'Speech_SQ' column
        
    Returns:
        tuple: (SentenceTransformer model, FAISS index) or (None, None) if failed
    """
    try:
        from sentence_transformers import SentenceTransformer
        import faiss
    except ImportError:
        print("Error: sentence-transformers or faiss not installed")
        return None, None

    # Get texts
    texts = df["Speech_SQ"].fillna("").astype(str).tolist()
    if len(texts) == 0:
        return None, None

    try:
        # Load model
        model = SentenceTransformer(VECTOR_MODEL)

        # Encode texts to embeddings (show_progress_bar=False për Streamlit që të mos ngatërrohet me spinner)
        embeddings = model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        # Ensure float32 and proper shape
        embeddings = np.asarray(embeddings, dtype="float32")
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        # Build FAISS index
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)

        return model, index

    except Exception as e:
        print(f"Error building vector store: {e}")
        return None, None


def search_similar_documents(query_text, model, index, df, k=8):
    """
    Search for similar documents using vector similarity.
    
    Args:
        query_text (str): Query text in Albanian
        model: SentenceTransformer model
        index: FAISS index
        df (pd.DataFrame): Original dataframe
        k (int): Number of results to return
        
    Returns:
        pd.DataFrame: Dataframe with k most similar documents
    """
    if model is None or index is None:
        return pd.DataFrame()

    try:
        # Encode query
        q_embed = model.encode(
            [query_text],
            convert_to_numpy=True,
        )
        q_embed = np.asarray(q_embed, dtype="float32")

        # Search
        k = min(k, int(index.ntotal))
        if k <= 0:
            return pd.DataFrame()

        distances, indices = index.search(q_embed, k)

        # Get valid indices
        valid_idx = [int(i) for i in indices[0] if i != -1]
        if len(valid_idx) == 0:
            return pd.DataFrame()

        # Return matching rows
        return df.iloc[valid_idx].copy()

    except Exception as e:
        print(f"Error searching vector store: {e}")
        return pd.DataFrame()