
import os, re, json
from pathlib import Path
from typing import List, Dict, Optional, Tuple

try:
    import numpy as np  # noqa: F401
except Exception:  # numpy should exist per requirements, but degrade if not
    np = None

try:
    import pandas as pd
except Exception:
    pd = None

# Optional heavy deps
try:
    from sentence_transformers import SentenceTransformer  # noqa: F401
    import faiss  # noqa: F401
    _EMB_OK = True
except Exception:
    _EMB_OK = False

DATA_DIR = Path(__file__).resolve().parent / "data"

def _normalize_cols(df):
    df.columns = [str(c).strip() for c in df.columns]
    return df

def _find_text_and_hours_cols(df):
    name_col, hours_col = None, None
    for c in df.columns:
        lc = str(c).lower()
        if name_col is None and any(k in lc for k in ["tarea","name","summary","descripcion","descripción","actividad","text","titulo","title"]):
            name_col = c
        if hours_col is None and any(k in lc for k in ["hora","hhs","estimado","estimacion","estimación","hours","original estimate"]):
            hours_col = c
    if name_col is None and len(df.columns) >= 1:
        name_col = df.columns[0]
    if hours_col is None and len(df.columns) >= 2:
        hours_col = df.columns[1]
    return name_col, hours_col

def load_labeled_dataframe(path: Path):
    """Load a CSV and return DataFrame with columns: text, hours, ticket, source"""
    if pd is None or not Path(path).exists():
        import pandas as _pd
        return _pd.DataFrame(columns=["text","hours","ticket","source"])

    df = pd.read_csv(path, encoding="utf-8")
    df = _normalize_cols(df)
    name_col, hours_col = _find_text_and_hours_cols(df)

    if name_col not in df.columns:
        return pd.DataFrame(columns=["text","hours","ticket","source"])

    out = pd.DataFrame()
    out["text"] = df[name_col].astype(str).fillna("")
    if hours_col in df.columns:
        out["hours"] = pd.to_numeric(df[hours_col], errors="coerce")
    else:
        out["hours"] = None
    # Ticket/id if exists
    tcol = None
    for c in df.columns:
        if str(c).lower() in ["issue key","issue id","ticket","id","key"]:
            tcol = c; break
    out["ticket"] = df[tcol].astype(str) if tcol else [f"ROW-{i}" for i in range(len(out))]
    out["source"] = Path(path).name
    # keep only rows with text
    out = out[out["text"].str.strip()!=""].reset_index(drop=True)
    return out

def _tokenize(s: str) -> List[str]:
    return re.findall(r"[a-záéíóúñ0-9]+", (s or "").lower())

def _jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb: return 0.0
    return len(sa & sb) / len(sa | sb)

class EmbeddingsFaissEstimator:
    """Graceful estimator. If sentence-transformers/faiss not available, use Jaccard similarity."""
    def __init__(self):
        self.enabled = _EMB_OK
        self.rows = []  # list of dict(text, hours, ticket, source)

    def fit(self, df):
        self.rows = df.to_dict(orient="records")

    def query(self, text: str, top_k: int = 5) -> List[Dict]:
        toks_q = _tokenize(text)
        scored = []
        for r in self.rows:
            score = _jaccard(toks_q, _tokenize(r.get("text","")))
            scored.append((score, r))
        scored.sort(key=lambda x: x[0], reverse=True)
        out = []
        for score, r in scored[:top_k]:
            out.append({
                "ticket": r.get("ticket"),
                "descripcion": r.get("text"),
                "horas": r.get("hours"),
                "similitud": round(float(score), 3),
                "source": r.get("source"),
            })
        return out

def train_index_per_type() -> Dict[str, EmbeddingsFaissEstimator]:
    """Train two separate indexes for 'Desarrollo' and 'Implementación' from provided CSVs if present."""
    idx = {}
    # Desarrollo (CESQ)
    cesq_path = DATA_DIR / "EstimacionCESQ.csv"
    cesq_df = load_labeled_dataframe(cesq_path)
    est_cesq = EmbeddingsFaissEstimator()
    est_cesq.fit(cesq_df)
    idx["Desarrollo"] = est_cesq

    # Implementación (PSTC)
    pstc_path = DATA_DIR / "EstimacionesPSTC.csv"
    pstc_df = load_labeled_dataframe(pstc_path)
    est_pstc = EmbeddingsFaissEstimator()
    est_pstc.fit(pstc_df)
    idx["Implementación"] = est_pstc
    return idx

def load_catalog(tipo: str) -> Optional[List[Tuple[str, float]]]:
    fname = "catalogo_cesq.csv" if tipo == "Desarrollo" else "catalogo_pstc.csv"
    path = DATA_DIR / fname
    if not path.exists() or pd is None:
        return []
    df = pd.read_csv(path, encoding="utf-8")
    # Expect columns text, hours; if not, try best-effort
    txt_col = None; hrs_col = None
    for c in df.columns:
        lc = str(c).lower()
        if txt_col is None and any(k in lc for k in ["text","descripcion","descripción","tarea","actividad","nombre"]):
            txt_col = c
        if hrs_col is None and any(k in lc for k in ["hour","hora","hh","estim"]):
            hrs_col = c
    if txt_col is None:
        txt_col = df.columns[0]
    if hrs_col is None and len(df.columns) > 1:
        hrs_col = df.columns[1]
    rows = []
    for _, r in df.iterrows():
        t = str(r.get(txt_col, "")).strip()
        try:
            h = float(r.get(hrs_col, 0))
        except Exception:
            h = 0.0
        if t:
            rows.append((t, h))
    return rows

def estimate_from_catalog(texto: str, tipo: str) -> float:
    """Very simple rule: sum hours of catalog rows whose keyword appears in text; otherwise 0."""
    cat = load_catalog(tipo)
    if not cat:
        return 0.0
    t = (texto or "").lower()
    hours = 0.0
    for key, h in cat:
        k = key.lower()
        if len(k) > 2 and k in t:
            hours += float(h or 0.0)
    return hours

