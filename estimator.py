
import os, re, json, glob
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

# ---------- Env & memory knobs ----------
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("FAISS_NUM_THREADS", "1")

DATA_DIR = Path(os.environ.get("DATA_DIR", str(Path(__file__).parent / "data")))
DATA_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR = DATA_DIR / "faiss"; INDEX_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR = DATA_DIR / "models"; MODEL_DIR.mkdir(parents=True, exist_ok=True)

# cache HF persistente
os.environ.setdefault("HF_HOME", str(MODEL_DIR / "hf_cache"))

CATALOGO_CESQ  = DATA_DIR / "catalogo_cesq.csv"
CATALOGO_PSTC  = DATA_DIR / "catalogo_pstc.csv"
NEW_EST_PATH   = DATA_DIR / "estimaciones_nuevas.csv"

MODEL_NAME  = os.environ.get("EMB_MODEL", "sentence-transformers/paraphrase-MiniLM-L3-v2")
MAX_SEQ_LEN = int(os.environ.get("MAX_SEQ_LEN", "256"))
EMB_BATCH   = int(os.environ.get("EMB_BATCH", "8"))
LAZY_BOOT   = os.environ.get("LAZY_BOOT", "1") == "1"  # evita entrenamientos pesados al arranque

# ---------- Embeddings (singleton) ----------
_MODEL: Optional[SentenceTransformer] = None
def _get_model() -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer(MODEL_NAME, device="cpu")
        try: _MODEL.max_seq_length = MAX_SEQ_LEN
        except Exception: pass
    return _MODEL

def _embed_texts(texts: List[str]) -> np.ndarray:
    if not texts: return np.zeros((0,0), dtype="float32")
    m = _get_model()
    embs = m.encode(
        texts, batch_size=EMB_BATCH, show_progress_bar=False,
        normalize_embeddings=True, convert_to_numpy=True
    )
    return np.asarray(embs, dtype="float32")

# ---------- FAISS helpers ----------
def _index_path(tag: str) -> Path: return INDEX_DIR / f"faiss_{tag}.index"
def _hours_path(tag: str) -> Path: return INDEX_DIR / f"faiss_{tag}_hours.json"
def _dataset_path(tag: str) -> Path: return INDEX_DIR / f"faiss_{tag}_dataset.csv"

def _save_index(index, tag: str, hours: List[float]):
    faiss.write_index(index, str(_index_path(tag)))
    Path(_hours_path(tag)).write_text(json.dumps(hours, ensure_ascii=False))

def _load_index(tag: str):
    ip = _index_path(tag); hp = _hours_path(tag)
    if not (ip.exists() and hp.exists()): return None, None
    idx = faiss.read_index(str(ip))
    hrs = json.loads(Path(hp).read_text())
    return idx, hrs

# ---------- CSV utils (livianos) ----------
def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    def clean(s: str) -> str: return str(s).replace("\ufeff","").strip()
    out = df.copy(); out.columns = [clean(c) for c in out.columns]; return out

def _load_catalog(path: Path) -> pd.DataFrame:
    if not path.exists(): return pd.DataFrame(columns=["text","hours","ticket","source"])
    df = pd.read_csv(path, encoding="utf-8"); df = _normalize_cols(df)
    name_col, hours_col = None, None
    for c in df.columns:
        lc = str(c).lower()
        if name_col is None and any(k in lc for k in ["tarea","nombre","descripción","descripcion","actividad","text","nombre tarea"]): name_col = c
        if hours_col is None and any(k in lc for k in ["hora","horas","hh","estimado","estimacion","estimación","hours"]): hours_col = c
    if name_col is None or hours_col is None:
        if len(df.columns) >= 2: name_col, hours_col = df.columns[:2]
        else: return pd.DataFrame(columns=["text","hours","ticket","source"])
    out = df[[name_col, hours_col]].dropna()
    out = out.rename(columns={name_col:"text", hours_col:"hours"})
    out["hours"] = pd.to_numeric(out["hours"], errors="coerce")
    out = out.dropna(subset=["hours"])
    out["ticket"] = [f"CAT-{i}" for i in range(len(out))]
    out["source"] = "catalog"
    return out[["text","hours","ticket","source"]].reset_index(drop=True)

def load_labeled_dataframe(tag: str) -> pd.DataFrame:
    ds = _dataset_path(tag)
    if ds.exists():
        try:
            df = pd.read_csv(ds, encoding="utf-8")
            req = [c for c in ["text","hours","ticket","source"] if c in df.columns]
            if set(["text","hours"]).issubset(req):
                if "ticket" not in df: df["ticket"] = [str(i) for i in range(len(df))]
                if "source" not in df: df["source"] = "unknown"
                return df[["text","hours","ticket","source"]].reset_index(drop=True)
        except Exception:
            pass
    # fallback a catálogos (muy ligeros)
    if tag == "desarrollo": df = _load_catalog(CATALOGO_CESQ)
    elif tag == "implementacion": df = _load_catalog(CATALOGO_PSTC)
    else: raise ValueError("tag inválido")
    if "ticket" not in df: df["ticket"] = [str(i) for i in range(len(df))]
    if "source" not in df: df["source"] = "unknown"
    return df[["text","hours","ticket","source"]].reset_index(drop=True)

# ---------- Estimator (API esperada por app.py) ----------
class EmbeddingsFaissEstimator:
    def __init__(self, tag: str):
        assert tag in ("desarrollo","implementacion")
        self.tag = tag
        self.index = None
        self.hours = None

    def fit(self, texts: List[str], hours: List[float]):
        embs = _embed_texts(texts)
        if embs.size == 0: raise ValueError("Corpus vacío")
        dim = embs.shape[1]
        idx = faiss.IndexFlatIP(dim)  # cosine gracias a normalize_embeddings=True
        idx.add(embs)
        self.index = idx
        self.hours = list(map(float, hours))

    def save(self):
        if self.index is None or self.hours is None: return
        _save_index(self.index, self.tag, self.hours)

    def load(self) -> bool:
        idx, hrs = _load_index(self.tag)
        if idx is None: return False
        self.index, self.hours = idx, hrs
        return True

    def predict(self, text: str, k: int = 5) -> Tuple[float, List[Tuple[int,float,float]]]:
        if self.index is None or self.hours is None:
            raise RuntimeError("Índice no cargado")
        q = _embed_texts([text])
        D, I = self.index.search(q, k)
        sims = D[0].tolist(); idxs = I[0].tolist()
        neigh = []
        for i, s in zip(idxs, sims):
            if i == -1: continue
            neigh.append((int(i), float(s), float(self.hours[i])))
        total = sum(s for _,s,_ in neigh) or 1.0
        est = sum(h*s for _,s,h in neigh)/total if neigh else 0.0
        return est, neigh

# ---------- Orquestación de entrenamiento ----------
def _build_from_frame(tag: str, df: pd.DataFrame) -> int:
    if df.empty: return 0
    est = EmbeddingsFaissEstimator(tag)
    est.fit(df["text"].astype(str).tolist(), df["hours"].astype(float).tolist())
    est.save()
    # Persistimos dataset para la UI
    df[["text","hours","ticket","source"]].to_csv(_dataset_path(tag), index=False, encoding="utf-8")
    return len(df)

def train_index_per_type(full: bool=False) -> Dict[str,int]:
    """
    Render-safe:
      - Si LAZY_BOOT=1 (default) y full=False: NO hace training pesado en el boot.
        * Intenta cargar índices existentes.
        * Si no existen, construye índices mínimos desde catálogos (ligeros).
      - Con full=True: fuerza construir desde datasets (si existen).
    """
    counts = {"desarrollo":0,"implementacion":0}
    lazy = LAZY_BOOT and not full

    if lazy:
        idx, hrs = _load_index("desarrollo")
        counts["desarrollo"] = len(hrs) if hrs else _build_from_frame("desarrollo", load_labeled_dataframe("desarrollo"))
    else:
        counts["desarrollo"] = _build_from_frame("desarrollo", load_labeled_dataframe("desarrollo"))

    if lazy:
        idx, hrs = _load_index("implementacion")
        counts["implementacion"] = len(hrs) if hrs else _build_from_frame("implementacion", load_labeled_dataframe("implementacion"))
    else:
        counts["implementacion"] = _build_from_frame("implementacion", load_labeled_dataframe("implementacion"))

    return counts
