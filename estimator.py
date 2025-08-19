# estimator.py (memory-optimized for Render free tier <= 512MB)
# --------------------------------------------------------------
# Drop-in module to compute embeddings and do ANN search with FAISS
# using very low RAM. Designed for CPU only and single-threaded runs.
#
# Environment variables you can set in Render:
#   EMB_MODEL=sentence-transformers/paraphrase-MiniLM-L3-v2
#   OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, NUMEXPR_NUM_THREADS=1
#   TOKENIZERS_PARALLELISM=false, FAISS_NUM_THREADS=1
#
# Gunicorn (render.yaml):
#   gunicorn -k gthread -w 1 --threads 1 -t 300 -b 0.0.0.0:$PORT app:app
#
# If you still get OOM, reduce MAX_SEQ_LEN to 128 and BATCH_SIZE to 4 or 2.

from __future__ import annotations
import os
import logging
from typing import List, Tuple, Optional

# --------- Threading & memory knobs (set before heavy imports) ----------
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("FAISS_NUM_THREADS", "1")

# -------------------------- Imports -------------------------------------
import numpy as np

try:
    import torch  # type: ignore
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
except Exception:
    pass

from sentence_transformers import SentenceTransformer  # type: ignore
import faiss  # type: ignore

# -------------------------- Config --------------------------------------
MODEL_NAME = os.environ.get("EMB_MODEL", "sentence-transformers/paraphrase-MiniLM-L3-v2")
MAX_SEQ_LEN = int(os.environ.get("MAX_SEQ_LEN", "256"))   # lower to 128 if needed
BATCH_SIZE  = int(os.environ.get("EMB_BATCH", "8"))       # lower to 4 or 2 if needed

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# ----------------------- Singleton Model --------------------------------
_model: Optional[SentenceTransformer] = None

def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        logger.info(f"Loading embeddings model: {MODEL_NAME}")
        _model = SentenceTransformer(MODEL_NAME, device="cpu")
        # Cut sequence length to keep tokenizer and attention small
        try:
            _model.max_seq_length = MAX_SEQ_LEN
        except Exception:
            pass
        logger.info("Model loaded.")
    return _model

# ----------------------- Embedding Utils --------------------------------
def embed_texts(texts: List[str]) -> np.ndarray:
    """Return L2-normalized embeddings as float32 (N, D).
    Keep batch size small to avoid spikes in RAM.
    """
    if not texts:
        return np.zeros((0, 0), dtype="float32")
    model = _get_model()
    embs = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,  # enables cosine via inner product
        show_progress_bar=False,
        convert_to_numpy=True,
    ).astype("float32")  # (N, D)
    return embs

# -------------------------- FAISS Index ---------------------------------
class SearchIndex:
    """Flat IP index (cosine if embeddings are normalized). Minimal RAM."""
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)

    def add(self, embs: np.ndarray) -> None:
        if embs.dtype != np.float32:
            embs = embs.astype(np.float32, copy=False)
        if embs.ndim != 2 or embs.shape[1] != self.dim:
            raise ValueError(f"Expected (N,{self.dim}) float32, got {embs.shape} {embs.dtype}")
        self.index.add(embs)

    def search(self, query_embs: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        if query_embs.dtype != np.float32:
            query_embs = query_embs.astype(np.float32, copy=False)
        D, I = self.index.search(query_embs, k)
        return D, I

def build_index(corpus_texts: List[str]) -> Tuple[SearchIndex, np.ndarray]:
    """Embeds corpus and builds a flat IP index. Returns (index, embeddings)."""
    embs = embed_texts(corpus_texts)  # (N, D)
    if embs.size == 0:
        raise ValueError("No embeddings built (empty corpus)." )
    idx = SearchIndex(dim=embs.shape[1])
    idx.add(embs)
    return idx, embs

def query(corpus_embs: np.ndarray, queries: List[str], k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """Quick helper when you don't want to persist the FAISS object separately."""
    if corpus_embs.ndim != 2:
        raise ValueError("corpus_embs must be 2D (N, D)")
    idx = SearchIndex(dim=corpus_embs.shape[1])
    idx.add(corpus_embs)
    q_embs = embed_texts(queries)
    return idx.search(q_embs, k=k)

# -------------------------- Fallback TF-IDF ------------------------------
# Optional: ultra-low-memory fallback if transformer fails to load.
# Enable by setting EMB_FALLBACK=1
USE_FALLBACK = os.environ.get("EMB_FALLBACK", "0") == "1"

_vectorizer = None
_fallback_matrix = None

def _ensure_fallback(corpus_texts: List[str]):
    global _vectorizer, _fallback_matrix
    if _vectorizer is None:
        from sklearn.feature_extraction.text import TfidfVectorizer  # lazy import
        _vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1,2))
        _fallback_matrix = _vectorizer.fit_transform(corpus_texts)

def embed_or_fallback(texts: List[str]) -> np.ndarray:
    if USE_FALLBACK:
        # Return L2-normalized TF-IDF vectors as float32 dense
        from sklearn.preprocessing import normalize
        _ensure_fallback(texts)
        X = _vectorizer.transform(texts)
        X = normalize(X, norm="l2").astype(np.float32)
        return X.toarray().astype(np.float32)
    # Default path
    try:
        return embed_texts(texts)
    except Exception as e:
        logger.warning(f"Falling back to TF-IDF due to: {e}")
        os.environ["EMB_FALLBACK"] = "1"
        return embed_or_fallback(texts)

# -------------------------- Small self-test ------------------------------
if __name__ == "__main__":
    data = [
        "Crear endpoint Flask para estimación de horas de ticket CESQ",
        "Implementar login automático en SII usando certificado digital",
        "Generar embeddings con MiniLM y buscar similares con FAISS",
        "Automatizar carga de clientes en Sovos desde CSV",
        "Dashboard HTML con filtros por trimestre y manager",
    ]
    idx, embs = build_index(data)
    D, I = query(embs, ["estimación de horas con embeddings"], k=3)
    print("Top-3:", I[0], D[0])
