import os
from pathlib import Path
import glob
import re
import pandas as pd
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer

# =========================
# Rutas y caché de modelos
# =========================
DATA_DIR = Path(os.environ.get("DATA_DIR", str(Path(__file__).parent / "data")))
CACHE_DIR = Path(os.environ.get("HF_HOME", str(Path(__file__).parent / ".hf_cache")))
DATA_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

os.environ["HF_HOME"] = str(CACHE_DIR)
os.environ["TRANSFORMERS_CACHE"] = str(CACHE_DIR)
os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(CACHE_DIR)

MODEL_NAME = os.environ.get("EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

def _load_model():
    return SentenceTransformer(MODEL_NAME, cache_folder=str(CACHE_DIR))

# =========================
# Utilidades de datos
# =========================
def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        # intenta con ; y con latin-1
        try:
            return pd.read_csv(path, sep=";", encoding="latin-1")
        except Exception:
            return pd.read_csv(path, encoding_errors="ignore")

def _standardize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df

HOUR_COL_CANDIDATES = [
    "hours", "hora", "horas", "hh", "time spent", "Σ time spent",
    "logged hours", "estimate", "original estimate"
]
TEXT_COL_CANDIDATES = [
    "text", "descripcion", "descripción", "summary", "titulo", "título",
    "tarea", "detalle", "observacion", "observación", "comments"
]

def _parse_hours(val) -> float | None:
    if pd.isna(val):
        return None
    s = str(val).strip().lower()
    if not s:
        return None
    # 1) hh:mm
    m = re.match(r"^\s*(\d+)\s*:\s*([0-5]?\d)\s*$", s)
    if m:
        return int(m.group(1)) + int(m.group(2))/60.0
    # 2) "90 min" / "90m"
    m = re.match(r"^\s*(\d+(?:[\.,]\d+)?)\s*(min|mins|m)\s*$", s)
    if m:
        v = float(m.group(1).replace(",", "."))
        return round(v/60.0, 4)
    # 3) "2 h" / "2 horas"
    m = re.match(r"^\s*(\d+(?:[\.,]\d+)?)\s*(h|hs|hora|horas)\s*$", s)
    if m:
        return float(m.group(1).replace(",", "."))
    # 4) número simple (2, 2.5, 2,5)
    m = re.match(r"^\s*\d+(?:[\.,]\d+)?\s*$", s)
    if m:
        return float(s.replace(",", "."))
    return None

def _pick_first_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def _load_catalog(tipo: str) -> pd.DataFrame:
    # Mapear por tipo a catálogo
    cat_name = "catalogo_cesq.csv" if tipo == "desarrollo" else "catalogo_pstc.csv"
    cat_path = DATA_DIR / cat_name
    if not cat_path.exists():
        return pd.DataFrame(columns=["text", "hours", "source"])

    df = _standardize_cols(_read_csv(cat_path))
    # Intentar localizar columnas de texto y horas
    text_col = _pick_first_col(df, TEXT_COL_CANDIDATES)
    hour_col = _pick_first_col(df, HOUR_COL_CANDIDATES)

    # Si ya vienen como text/hours, perfecto
    if text_col is None and "text" in df.columns:
        text_col = "text"
    if hour_col is None and "hours" in df.columns:
        hour_col = "hours"

    if text_col is None or hour_col is None:
        # Catálogo inválido
        return pd.DataFrame(columns=["text", "hours", "source"])

    df["text"] = df[text_col].astype(str).fillna("").str.strip()
    df["hours"] = df[hour_col].apply(_parse_hours)
    df = df[["text", "hours"]].dropna(subset=["text"])
    df["source"] = cat_name
    df = df.dropna(subset=["hours"])
    df = df[df["hours"] > 0]
    return df[["text", "hours", "source"]]

def _load_historico(tipo: str) -> pd.DataFrame:
    pats = ["*CESQ*.csv"] if tipo == "desarrollo" else ["*PSTC*.csv"]
    files = []
    for p in pats:
        files += glob.glob(str(DATA_DIR / p))
    rows = []
    for f in files:
        try:
            df = _standardize_cols(_read_csv(Path(f)))
            text_col = _pick_first_col(df, TEXT_COL_CANDIDATES)
            hour_col = _pick_first_col(df, HOUR_COL_CANDIDATES)
            if text_col and hour_col:
                part = pd.DataFrame({
                    "text": df[text_col].astype(str).fillna("").str.strip(),
                    "hours": df[hour_col].apply(_parse_hours)
                })
                part["source"] = Path(f).name
                part = part.dropna(subset=["hours"])
                part = part[part["hours"] > 0]
                rows.append(part)
        except Exception:
            continue
    if rows:
        out = pd.concat(rows, ignore_index=True)
        out = out[out["text"].str.len() > 0]
        return out[["text", "hours", "source"]]
    return pd.DataFrame(columns=["text", "hours", "source"])

def load_labeled_dataframe(tipo: str) -> pd.DataFrame:
    """
    1) Si existe labeled_{tipo}.csv, úsalo.
    2) Si no, construye dataset desde catálogo + históricos en DATA_DIR.
    """
    path = DATA_DIR / f"labeled_{tipo}.csv"
    if path.exists():
        df = _standardize_cols(_read_csv(path))
        if "text" in df.columns and "hours" in df.columns:
            df["hours"] = df["hours"].apply(_parse_hours)
            df = df.dropna(subset=["hours"])
            df = df[df["hours"] > 0]
            if not df.empty:
                df["source"] = path.name
                return df[["text", "hours", "source"]]

    # Fallback: catálogo + históricos
    cat = _load_catalog(tipo)
    hist = _load_historico(tipo)
    df = pd.concat([cat, hist], ignore_index=True) if not cat.empty or not hist.empty \
         else pd.DataFrame(columns=["text", "hours", "source"])

    if df.empty:
        raise FileNotFoundError(f"No hay datos con horas para {tipo}. "
                                f"Verifica {('catalogo_cesq.csv' if tipo=='desarrollo' else 'catalogo_pstc.csv')} "
                                f"o sube labeled_{tipo}.csv con columnas 'text' y 'hours'.")
    return df

# =========================
# Estimador FAISS
# =========================
class EmbeddingsFaissEstimator:
    def __init__(self, tipo: str):
        self.tipo = tipo
        self.index_path = DATA_DIR / f"faiss_{tipo}.idx"
        self.meta_path  = DATA_DIR / f"faiss_{tipo}.meta.pkl"
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
        texts = df["text"].astype(str).tolist()
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        dim = embeddings.shape[1]
        faiss.normalize_L2(embeddings)
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        self.meta = df.to_dict(orient="records")
        self.save()

    def predict(self, texto: str, k: int = 5):
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
            results.append((int(idx), float(sim), float(h) if h is not None else None))
        return None, results

# =========================
# Entrenamiento por tipo
# =========================
def train_index_per_type():
    counts = {}
    for tipo in ["desarrollo", "implementacion"]:
        df = load_labeled_dataframe(tipo)
        est = EmbeddingsFaissEstimator(tipo)
        est.train(df)
        counts[tipo] = len(df)
    return counts
