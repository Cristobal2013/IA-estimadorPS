import os, re, json
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd

# --- Robust CSV reader patch (auto-added) ---
import pandas as _pd
_pd_read_csv_orig = _pd.read_csv
def _pd_read_csv_robust(*args, **kwargs):
    # Prefer safe defaults for messy CSVs
    if "on_bad_lines" not in kwargs:
        kwargs["on_bad_lines"] = "skip"
    # If python engine is used, low_memory is unsupported -> drop it
    if kwargs.get("engine") == "python":
        kwargs.pop("low_memory", None)
    # Encoding fallbacks
    given_enc = kwargs.pop("encoding", None)
    encodings = [given_enc, "utf-8", "latin-1"]
    last_err = None
    for enc in encodings:
        try:
            return _pd_read_csv_orig(*args, encoding=enc, **kwargs)
        except Exception as e:
            last_err = e
            # Retry forcing python engine (without low_memory)
            try_kwargs = dict(kwargs)
            try_kwargs["engine"] = "python"
            try_kwargs.pop("low_memory", None)
            try:
                return _pd_read_csv_orig(*args, encoding=enc, **try_kwargs)
            except Exception as e2:
                last_err = e2
                continue
    # Final attempt without explicit encoding
    try_kwargs = dict(kwargs)
    if try_kwargs.get("engine") == "python":
        try_kwargs.pop("low_memory", None)
    return _pd_read_csv_orig(*args, **try_kwargs)
_pd.read_csv = _pd_read_csv_robust
# --- End robust patch ---

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

# HF cache persistente
os.environ.setdefault("HF_HOME", str(MODEL_DIR / "hf_cache"))

CATALOGO_CESQ  = DATA_DIR / "catalogo_cesq.csv"
CATALOGO_PSTC  = DATA_DIR / "catalogo_pstc.csv"
NEW_EST_CSV    = DATA_DIR / "estimaciones_nuevas.csv"


MODEL_NAME  = os.environ.get("EMB_MODEL", "sentence-transformers/paraphrase-MiniLM-L3-v2")
MAX_SEQ_LEN = int(os.environ.get("MAX_SEQ_LEN", "256"))
EMB_BATCH   = int(os.environ.get("EMB_BATCH", "8"))
LAZY_BOOT   = os.environ.get("LAZY_BOOT", "1") == "1"  # evita reentrenos pesados en boot

# ---------- Embeddings (singleton) ----------
_MODEL: Optional[SentenceTransformer] = None
def _get_model() -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer(MODEL_NAME, device="cpu")
        try:
            _MODEL.max_seq_length = MAX_SEQ_LEN
        except Exception:
            pass
    return _MODEL

def _embed_texts(texts: List[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 0), dtype="float32")
    m = _get_model()
    embs = m.encode(
        texts,
        batch_size=EMB_BATCH,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
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

# ---------- CSV utils (robustos) ----------
def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    def clean(s: str) -> str:
        return str(s).replace("\ufeff", "").strip()
    out = df.copy()
    out.columns = [clean(c) for c in out.columns]
    return out

def _read_csv_smart(path: Path) -> pd.DataFrame:
    """
    Lee CSV probando encoding y separador para evitar one-column por ; o \t.
    Devuelve DF con columnas normalizadas.
    """
    encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252", "utf-16"]
    seps = [",", ";", "\t", "|"]
    best = None
    for enc in encodings:
        for sep in seps:
            try:
                df = pd.read_csv(path, encoding=enc, sep=sep, engine="python", low_memory=False)
                if df.shape[1] >= 2 and len(df) >= 1:
                    df = _normalize_cols(df)
                    if best is None or df.shape[1] > best.shape[1]:
                        best = df
            except Exception:
                continue
    if best is not None:
        return best
    return _normalize_cols(pd.read_csv(path, encoding="utf-8", low_memory=False))

_TEXT_HINTS   = ["summary","resumen","descripcion","descripción","title","título","nombre","text","nombre tarea"]
_HOURS_HINTS  = ["original estimate","horas","hours","estimado","estimacion","estimación","hh","hhs"]
_TICKET_HINTS = ["issue key","clave","ticket","id","key"]

def _guess_col(df: pd.DataFrame, hints: List[str], default: Optional[str]=None) -> Optional[str]:
    for c in df.columns:
        lc = str(c).lower()
        if any(h in lc for h in hints):
            return c
    return default

def _coerce_float(x):
    try:
        v = float(x)
        if pd.isna(v):
            return None
        return v
    except Exception:
        return None

def _make_rows(df: pd.DataFrame, text_col: str, hours_col: Optional[str]=None,
               ticket_col: Optional[str]=None, source: str="unknown") -> pd.DataFrame:
    out = pd.DataFrame()
    out["text"] = df[text_col].astype(str)
    out["hours"] = df[hours_col].apply(_coerce_float) if (hours_col and hours_col in df.columns) else None
    if ticket_col and ticket_col in df.columns:
        out["ticket"] = df[ticket_col].astype(str)
    else:
        out["ticket"] = [f"{source[:3].upper()}-{i}" for i in range(len(df))]
    out["source"] = source
    out = out[out["text"].str.strip() != ""].reset_index(drop=True)
    return out[["text","hours","ticket","source"]]


# ---------- Robust hour extraction (shared with app) ----------
import unicodedata as _unicodedata

_CAND_HOUR_KEYS = [
    "hours","hour","horas","hh","hhs","hrs","he","hr",
    "total_horas","hh_total","hh_totales",
    "horas_estimadas","estimacion_horas","hrs_estimadas","hrestimadas",
    "hh_estimadas","hhs_estimadas","estimacion_hhs","hhe","hh_est","hhestimadas"
]

def _strip_accents_est(s: str) -> str:
    try:
        return "".join(c for c in _unicodedata.normalize("NFD", str(s)) if _unicodedata.category(c) != "Mn")
    except Exception:
        return str(s)

def _norm_key_est(k: str) -> str:
    s = _strip_accents_est(str(k)).lower()
    return (s.replace(" ", "")
             .replace("_", "")
             .replace(".", "")
             .replace("-", "")
             .replace("'", "")
             .replace("’", "")
             .replace("´", ""))

_HOUR_NAME_RX_EST = re.compile(r"(hh|hhs|hrs?|hora|horas|hours|hhe|hr)", re.I)
_HOUR_TEXT_RX_EST = re.compile(r"(\d+(?:[.,]\d+)?)\s*(?:hh'?s?|hh|hrs?|horas?|h\b)", re.I)

def _extract_hours_from_row_est(row) -> float:
    # 0) mapa normalizado
    try:
        keys = list(row.index)
    except Exception:
        keys = list(getattr(row, "keys", lambda: [])())
    norm_map = { _norm_key_est(k): k for k in keys }

    # 1) por lista blanca exacta
    for cand in _CAND_HOUR_KEYS:
        kn = _norm_key_est(cand)
        orig = norm_map.get(kn)
        if orig is not None:
            try:
                v = float(str(row.get(orig, 0)).replace(",", "."))
            except Exception:
                v = 0.0
            if v > 0:
                return v

    # 2) por patrón flexible de nombre
    for nk, orig in norm_map.items():
        if _HOUR_NAME_RX_EST.search(nk):
            try:
                v = float(str(row.get(orig, 0)).replace(",", "."))
            except Exception:
                v = 0.0
            if v > 0:
                return v

    # 3) desde texto con unidad (incluye 'h' sola)
    for k in keys:
        val = row.get(k)
        if isinstance(val, str) and val:
            m = _HOUR_TEXT_RX_EST.findall(val)
            if m:
                try:
                    v = float(m[-1].replace(",", "."))
                except Exception:
                    v = 0.0
                if v > 0:
                    return v

    # 4) si nada, 0
    return 0.0

# ---------- Carga de catálogos para estimate_from_catalog ----------
def load_catalog(tipo: str):
    """
    Devuelve lista de (texto, horas) para 'Desarrollo' o 'Implementación'.
    """
    fname = CATALOGO_CESQ if tipo.lower().startswith("des") else CATALOGO_PSTC
    if not fname.exists():
        return []
    df = _read_csv_smart(fname)
    tcol = _guess_col(df, _TEXT_HINTS) or df.columns[0]
    hcol = _guess_col(df, _HOURS_HINTS) or (df.columns[1] if len(df.columns)>1 else None)
    rows = []
    for _, r in df.iterrows():
        txt = str(r.get(tcol, "")).strip()
        hrs = _coerce_float(r.get(hcol, None)) if hcol else None
        if txt and (hrs is not None):
            rows.append((txt, float(hrs)))
    return rows

# ---------- Loader principal: histórico + catálogo + nuevas ----------
def load_labeled_dataframe(tag: str) -> pd.DataFrame:
    """
    tag: 'desarrollo' | 'implementacion'
    Combina: histórico (Jira export) + catálogo + nuevas confirmaciones.
    Columnas: text, hours, ticket, source ('historic'|'catalog'|'new')
    """
    assert tag in ("desarrollo","implementacion")
    frames = []

    # 1) Histórico (Jira export)
    hist_path = DATA_DIR / ("EstimacionCESQ.csv" if tag == "desarrollo" else "EstimacionesPSTC.csv")
    if hist_path.exists():
        dfh = _read_csv_smart(hist_path)
        tcol = _guess_col(dfh, _TEXT_HINTS) or dfh.columns[0]
        hcol = _guess_col(dfh, _HOURS_HINTS)
        kcol = _guess_col(dfh, _TICKET_HINTS)
        frames.append(_make_rows(dfh, tcol, hcol, kcol, source="historic"))

    # 2) Catálogo base
    cat_path = CATALOGO_CESQ if tag == "desarrollo" else CATALOGO_PSTC
    if cat_path.exists():
        dfc = _read_csv_smart(cat_path)
        tcol = _guess_col(dfc, _TEXT_HINTS) or dfc.columns[0]
        hcol = _guess_col(dfc, _HOURS_HINTS) or (dfc.columns[1] if len(dfc.columns)>1 else None)
        frames.append(_make_rows(dfc, tcol, hcol, None, source="catalog"))

    # 3) Nuevas confirmaciones del usuario (acepta .csv o .cs)
    new_path = NEW_EST_CSV if NEW_EST_CSV.exists() else None
    if new_path:
        dfn = _read_csv_smart(new_path)
        if "tipo" in dfn.columns:
            mask = dfn["tipo"].astype(str).str.lower().str.contains("desarrollo" if tag == "desarrollo" else "pstc")
            dfn = dfn[mask]
        if not dfn.empty:
            tcol = "texto" if "texto" in dfn.columns else (_guess_col(dfn, _TEXT_HINTS) or dfn.columns[0])
            if "estimacion_final" in dfn.columns:
                hcol = "estimacion_final"
            elif "horas_reales" in dfn.columns:
                hcol = "horas_reales"
            else:
                hcol = _guess_col(dfn, _HOURS_HINTS)
            frames.append(_make_rows(dfn, tcol, hcol, None, source="new"))

    # Filtra frames vacíos para evitar FutureWarning de concat
    frames = [f for f in frames if isinstance(f, pd.DataFrame) and not f.empty]
    if not frames:
        return pd.DataFrame(columns=["text","hours","ticket","source"])

    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["text"]).reset_index(drop=True)
    return df[["text","hours","ticket","source"]]

# ---------- Estimator (API esperada por app.py) ----------
class EmbeddingsFaissEstimator:
    def __init__(self, tag: str):
        assert tag in ("desarrollo","implementacion")
        self.tag = tag
        self.index = None
        self.hours = None

    def fit(self, texts: List[str], hours: List[float]):
        embs = _embed_texts(texts)
        if embs.size == 0:
            raise ValueError("Corpus vacío")
        dim = embs.shape[1]
        idx = faiss.IndexFlatIP(dim)  # cosine gracias a normalize_embeddings=True
        idx.add(embs)
        self.index = idx
        self.hours = list(map(float, hours))

    def save(self):
        if self.index is None or self.hours is None:
            return
        _save_index(self.index, self.tag, self.hours)

    def load(self) -> bool:
        idx, hrs = _load_index(self.tag)
        if idx is None:
            return False
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
            if i == -1:
                continue
            neigh.append((int(i), float(s), float(self.hours[i])))
        # --- Promedio ponderado robusto ---
        if not neigh:
            return 0.0, []
        # ordenar por similitud
        neigh.sort(key=lambda t: t[1], reverse=True)
        # recorte por masa de similitud (80%)
        total_sim = sum(s for _, s, _ in neigh) or 1.0
        acc = 0.0
        used = []
        for (i, s, h) in neigh:
            used.append((i, s, h))
            acc += s
            if acc >= 0.80 * total_sim:
                break
        # clip percentil para horas (p10-p90) si hay suficientes
        hs = [h for (_, _, h) in used]
        if len(hs) >= 3:
            p10, p90 = np.percentile(hs, [10, 90])
            used = [(i, s, float(min(max(h, float(p10)), float(p90)))) for (i, s, h) in used]
        num = sum(h * s for _, s, h in used)
        den = sum(s for _, s, _ in used) or 1e-9
        est = num / den
        return est, neigh

# ---------- Orquestación de entrenamiento ----------
def _build_from_frame(tag: str, df: pd.DataFrame) -> int:
    if df.empty:
        # guarda dataset vacío para trazabilidad
        pd.DataFrame(columns=["text","hours","ticket","source"]).to_csv(_dataset_path(tag), index=False, encoding="utf-8")
        return 0
    est = EmbeddingsFaissEstimator(tag)
    hours = [float(h) if h is not None and not pd.isna(h) else 0.0 for h in df["hours"].tolist()]
    est.fit(df["text"].astype(str).tolist(), hours)
    est.save()
    df[["text","hours","ticket","source"]].to_csv(_dataset_path(tag), index=False, encoding="utf-8")
    return len(df)

def train_index_per_type(full: bool=False) -> Dict[str, int]:
    """
    Si full=True: reconstruye SIEMPRE desde (historic+catalog+new).
    Si full=False y LAZY_BOOT=1: intenta cargar índices; si no existen, crea mínimos.
    """
    counts = {"desarrollo": 0, "implementacion": 0}
    lazy = LAZY_BOOT and not full

    for tag in ["desarrollo", "implementacion"]:
        if lazy:
            idx, hrs = _load_index(tag)
            if idx is not None and hrs is not None:
                counts[tag] = len(hrs)
                continue
        df = load_labeled_dataframe(tag)
        counts[tag] = _build_from_frame(tag, df)

    return counts

# ---------- Estimación por catálogo (token overlap) ----------
def _tokens(s: str):
    return set([t for t in re.findall(r"[a-záéíóúñ]+", (s or "").lower()) if len(t) >= 3])

def estimate_from_catalog(texto: str, tipo: str, top_n: int = 3, min_cover: float = 0.35) -> float:
    """
    Calcula horas desde catálogo con token overlap.
    """
    cat = load_catalog(tipo)
    if not cat:
        return 0.0
    qtoks = _tokens(texto)
    if not qtoks:
        return 0.0

    scored = []
    for key, h in cat:
        ktoks = _tokens(key)
        if not ktoks:
            continue
        inter = len(qtoks & ktoks)
        cover = inter / max(1, len(ktoks))
        if cover >= min_cover:
            hnum = float(h or 0.0)
            if hnum > 0:
                scored.append((cover, hnum, key))

    scored.sort(key=lambda x: x[0], reverse=True)
    total = sum(h for _, h, _ in scored[:max(1, int(top_n))])
    return round(float(total), 2)
