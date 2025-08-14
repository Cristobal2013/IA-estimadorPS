import os
from pathlib import Path
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle

# =====================================
# Configuración de rutas y caché Render
# =====================================

DATA_DIR = Path(os.environ.get("DATA_DIR", str(Path(__file__).parent / "data")))
CACHE_DIR = Path(os.environ.get("HF_HOME", str(Path(__file__).parent / ".hf_cache")))

# Crear carpetas necesarias
DATA_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Variables de entorno para cache de HuggingFace/SBERT
os.environ["HF_HOME"] = str(CACHE_DIR)
os.environ["TRANSFORMERS_CACHE"] = str(CACHE_DIR)
os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(CACHE_DIR)

# Nombre del modelo, configurable por variable de entorno
MODEL_NAME = os.environ.get("EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


def _load_model():
    """Carga el modelo de embeddings usando carpeta de cache específica."""
    return SentenceTransformer(MODEL_NAME, cache_folder=str(CACHE_DIR))


# =====================================
# Clases y funciones
# =====================================

class EmbeddingsFaissEstimator:
    def __init__(self, tipo):
        self.tipo = tipo
        self.index_path = DATA_DIR / f"faiss_{tipo}.idx"
        self.meta_path = DATA_DIR / f"faiss_{tipo}.meta.pkl"
        self.model = None
        self.index = None
        self.meta = None

    def load(self):
        if not self.index_path.exists() or not self.meta_path.exists():
            raise FileNotFoundError(f"No existe índice para {self.tipo}")
        self.index = faiss.read_index(str(self.index_path))
        with open(self.meta_path, "rb") as f:
            self.meta = pickle.load(f)
        self.model = _load_model()

    def save(self):
        faiss.write_index(self.index, str(self.index_path))
        with open(self.meta_path, "wb") as f:
            pickle.dump(self.meta, f)

    def train(self, df: pd.DataFrame):
        self.model = _load_model()
        texts = df["text"].tolist()
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self.meta = df.to_dict(orient="records")
        self.save()

    def predict(self, texto, k=5):
        if self.model is None or self.index is None:
            raise RuntimeError("Debes cargar el índice primero")
        emb = self.model.encode([texto], convert_to_numpy=True)
        faiss.normalize_L2(emb)
        D, I = self.index.search(emb, k)
        results = []
        for idx, sim in zip(I[0], D[0]):
            if idx < 0:
                continue
            h = self.meta[idx].get("hours", None)
            results.append((idx, float(sim), h))
        return None, results


def load_labeled_dataframe(tipo):
    path = DATA_DIR / f"labeled_{tipo}.csv"
    if not path.exists():
        raise FileNotFoundError(f"No existe dataset etiquetado para {tipo}")
    return pd.read_csv(path)


def train_index_per_type():
    counts = {}
    for tipo in ["desarrollo", "implementacion"]:
        df = load_labeled_dataframe(tipo)
        est = EmbeddingsFaissEstimator(tipo)
        est.train(df)
        counts[tipo] = len(df)
    return counts
