# estimator.py (Render-friendly, with EmbeddingsFaissEstimator API)
# --------------------------------------------------------------
# Compatible shim exposing `EmbeddingsFaissEstimator` expected by app.py.
# Minimizes RAM for Render free tier (<=512 MB), CPU-only.
#
# Suggested env vars in Render:
#   HF_HOME=/var/data/hf_cache
#   EMB_MODEL=sentence-transformers/paraphrase-MiniLM-L3-v2
#   OMP_NUM_THREADS=1
#   MKL_NUM_THREADS=1
#   NUMEXPR_NUM_THREADS=1
#   TOKENIZERS_PARALLELISM=false
#   FAISS_NUM_THREADS=1
#   MAX_SEQ_LEN=256   (use 128 if still tight)
#   EMB_BATCH=8       (use 4 or 2 if still tight)
#
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
os.environ.setdefault("HF_HOME", "/var/data/hf_cache")

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

# -------------------------- Logging -------------------------------------
logger = logging.getLogger(__name__)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)

# -------------------------- Config --------------------------------------
DEFAULT_MODEL = os.environ.get("EMB_MODEL", "sentence-transformers/paraphrase-MiniLM-L3-v2")
MAX_SEQ_LEN = int(os.environ.get("MAX_SEQ_LEN", "256"))
BATCH_SIZE  = int(os.environ.get("EMB_BATCH", "8"))
USE_FALLBACK = os.environ.get("EMB_FALLBACK", "0") == "1"

# ----------------------- Internal helpers -------------------------------
_model: Optional[SentenceTransformer] = None

def _get_model(model_name: str) -> SentenceTransformer:
    global _model
    if _model is None:
        logger.info(f"Loading embeddings model: {model_name}")
        _model = SentenceTransformer(model_name, device="cpu")
        try:
            _model.max_seq_length = MAX_SEQ_LEN
        except Exception:
            pass
        logger.info("Model loaded.")
    return _model

def _embed(texts: List[str], model: SentenceTransformer) -> np.ndarray:
    if not texts:
        return np.zeros((0, 0), dtype="float32")
    embs = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,  # cosine via inner product
        show_progress_bar=False,
        convert_to_numpy=True,
    ).astype("float32")
    return embs

# ----------------------- Public functions (previous API) ----------------
def embed_texts(texts: List[str]) -> np.ndarray:
    model = _get_model(DEFAULT_MODEL)
    return _embed(texts, model)

class SearchIndex:
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
    embs = embed_texts(corpus_texts)
    if embs.size == 0:
        raise ValueError("No embeddings built (empty corpus).")
    idx = SearchIndex(dim=embs.shape[1])
    idx.add(embs)
    return idx, embs

def query(corpus_embs: np.ndarray, queries: List[str], k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    if corpus_embs.ndim != 2:
        raise ValueError("corpus_embs must be 2D (N, D)")
    idx = SearchIndex(dim=corpus_embs.shape[1])
    idx.add(corpus_embs)
    q_embs = embed_texts(queries)
    return idx.search(q_embs, k=k)

# ----------------------- Compatibility class ----------------------------
class EmbeddingsFaissEstimator:
    """
    Drop-in estimator expected by app.py:
      - __init__(model_name: str | None = None, max_seq_len: int = 256, batch_size: int = 8)
      - fit(corpus_texts: List[str]) -> None
      - search(query_texts: List[str], k: int = 5) -> Tuple[np.ndarray, np.ndarray]
      - embed(texts: List[str]) -> np.ndarray
      - add(texts: List[str]) -> None
    Attributes:
      - index: FAISS flat IP index
      - corpus_texts: original texts
      - corpus_embs: embeddings matrix (N, D)
      - dim: embedding dimension
    """
    def __init__(
        self,
        model_name: Optional[str] = None,
        max_seq_len: Optional[int] = None,
        batch_size: Optional[int] = None,
        use_fallback: Optional[bool] = None,
    ) -> None:
        self.model_name = model_name or DEFAULT_MODEL
        if max_seq_len is not None:
            os.environ["MAX_SEQ_LEN"] = str(max_seq_len)
        if batch_size is not None:
            os.environ["EMB_BATCH"] = str(batch_size)
        if use_fallback is not None:
            os.environ["EMB_FALLBACK"] = "1" if use_fallback else "0"

        self.model: Optional[SentenceTransformer] = None
        self.index: Optional[SearchIndex] = None
        self.corpus_texts: List[str] = []
        self.corpus_embs: Optional[np.ndarray] = None
        self.dim: Optional[int] = None

    def _ensure_model(self) -> SentenceTransformer:
        if self.model is None:
            self.model = _get_model(self.model_name)
        return self.model

    # Public API
    def embed(self, texts: List[str]) -> np.ndarray:
        model = self._ensure_model()
        return _embed(texts, model)

    def fit(self, corpus_texts: List[str]) -> None:
        self.corpus_texts = list(corpus_texts)
        embs = self.embed(self.corpus_texts)
        if embs.size == 0:
            raise ValueError("Empty corpus after embedding.")
        self.dim = int(embs.shape[1])
        self.index = SearchIndex(dim=self.dim)
        self.index.add(embs)
        self.corpus_embs = embs

    def add(self, texts: List[str]) -> None:
        if not texts:
            return
        if self.index is None or self.dim is None:
            # Treat as first fit
            self.fit(texts)
            return
        embs = self.embed(texts)
        if embs.shape[1] != self.dim:
            raise ValueError(f"Embedding dim changed. Expected {self.dim}, got {embs.shape[1]}")
        self.index.add(embs)
        # append to corpus
        if self.corpus_embs is None:
            self.corpus_embs = embs
        else:
            self.corpus_embs = np.vstack([self.corpus_embs, embs])
        self.corpus_texts.extend(texts)

    def search(self, query_texts: List[str], k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        if self.index is None:
            raise RuntimeError("Index not built. Call fit(corpus_texts) first.")
        q_embs = self.embed(query_texts)
        return self.index.search(q_embs, k=k)

# -------------------------- Self-test -----------------------------------
if __name__ == "__main__":
    data = [
        "Crear endpoint Flask para estimación de horas de ticket CESQ",
        "Implementar login automático en SII usando certificado digital",
        "Generar embeddings con MiniLM y buscar similares con FAISS",
        "Automatizar carga de clientes en Sovos desde CSV",
        "Dashboard HTML con filtros por trimestre y manager",
    ]
    est = EmbeddingsFaissEstimator()
    est.fit(data)
    D, I = est.search(["estimación de horas con embeddings"], k=3)
    print("Top-3:", I[0], D[0])
