import os
import re
import json
import glob
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

# =========================
# Configuración de rutas
# =========================
# Usa DATA_DIR si está definida (Azure: /home/site/data). Si no, ./data
DATA_DIR = Path(os.environ.get("DATA_DIR", str(Path(__file__).parent / "data")))
DATA_DIR.mkdir(parents=True, exist_ok=True)

MODEL_DIR = DATA_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

INDEX_DIR = DATA_DIR / "faiss"
INDEX_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = os.environ.get("EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
HF_HOME = os.environ.get("HF_HOME", str(MODEL_DIR / "hf_cache"))
os.environ["HF_HOME"] = HF_HOME

NEW_EST_PATH = DATA_DIR / "estimaciones_nuevas.csv"   # persistimos lo que el usuario guarda
CATALOGO_CESQ = DATA_DIR / "catalogo_cesq.csv"
CATALOGO_PSTC = DATA_DIR / "catalogo_pstc.csv"


def _dataset_csv_path(tag: str) -> Path:
    assert tag in ("desarrollo", "implementacion")
    return INDEX_DIR / f"faiss_{tag}_dataset.csv"


# =========================
# Ubicar CSV históricos
# =========================
def _find_csv(pattern: str) -> Path:
    """
    Busca un CSV por patrón primero en DATA_DIR y luego en el directorio del proyecto.
    Ejemplos de patrones: '*CESQ*.csv', '*PSTC*.csv'
    """
    candidates = []
    candidates.extend(sorted(glob.glob(str(DATA_DIR / pattern))))
    candidates.extend(sorted(glob.glob(str(Path(__file__).parent / pattern))))
    if not candidates:
        raise FileNotFoundError(f"No se encontró CSV con patrón: {pattern}")
    return Path(candidates[0])


def _cesq_path() -> Path:
    return _find_csv("*CESQ*.csv")


def _pstc_path() -> Path:
    return _find_csv("*PSTC*.csv")


# =========================
# Normalización / Ticket
# =========================
def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    def clean(s: str) -> str:
        return str(s).replace("\ufeff", "").strip()
    out = df.copy()
    out.columns = [clean(c) for c in out.columns]
    return out


def _pick_ticket_column(df: pd.DataFrame) -> Optional[str]:
    norm_map = {c.lower().replace(" ", ""): c for c in df.columns}
    for k in ["key", "issuekey", "issue key", "id", "ticket", "número", "numero", "nro", "n°"]:
        if k in norm_map:
            return norm_map[k]
    for c in df.columns:
        series = df[c].astype(str)
        if series.str.contains(r"\b(?:CESQ|PSTC)-\d+\b", regex=True).any():
            return c
    return None


# =========================
# Horas desde Comments
# =========================
def _extract_hours_from_comments_row(row) -> Optional[float]:
    # Junta todas las columnas que empiezan por "Comments"
    comment_cols = [c for c in row.index if str(c).startswith("Comments")]
    texts = [str(row[c]) for c in comment_cols if isinstance(row.get(c), str)]
    blob = "\n".join(texts) if texts else ""

    # 1) Tabla con Total (| Total | 6 |)
    m = re.search(r"(?mi)^\s*\|?\s*\*?total\*?\s*\|\s*\*?(\d+(?:[.,]\d+)?)\*?\s*\|?", blob)
    if m:
        return float(m.group(1).replace(",", "."))

    # 2) Tabla con HH y suma de filas
    if re.search(r"(?mi)^\s*\|.*\bHH\b.*\|", blob):
        total, any_row = 0.0, False
        for line in blob.splitlines():
            mrow = re.match(r"^\s*\|.*?\|\s*\*?(\d+(?:[.,]\d+)?)\*?\s*\|\s*$", line)
            if mrow and not re.search(r"(?i)\b(total|actividad|hh)\b", line):
                any_row = True
                total += float(mrow.group(1).replace(",", "."))
        if any_row and total > 0:
            return total

    low = blob.lower()

    # 3) "Total 6 h"
    m = re.search(r"total\W*(\d+(?:[.,]\d+)?)\s*h\b", low)
    if m:
        return float(m.group(1).replace(",", "."))

    # 4) Suma de "X h", "Y horas", "2hh"
    matches_h = re.findall(r"(?<!\d)(\d+(?:[.,]\d+)?)\s*h\b", low)
    if matches_h:
        return sum(float(x.replace(",", ".")) for x in matches_h)

    matches_u = re.findall(r"(\d+(?:[.,]\d+)?)\s*(?:horas|hora|hrs|hr|hh)\b", low)
    if matches_u:
        return sum(float(x.replace(",", ".")) for x in matches_u)

    # 5) Formatos 1:30 y "30 min"
    m = re.findall(r"(\d+):(\d{1,2})", low)
    if m:
        return sum(int(h) + int(mm) / 60.0 for h, mm in m)

    m = re.findall(r"(\d+(?:[.,]\d+)?)\s*min\b", low)
    if m:
        return sum(float(x.replace(",", ".")) / 60.0 for x in m)

    return None


def _build_text(df: pd.DataFrame) -> pd.Series:
    # Preferimos columnas típicas
    pref = [c for c in ["Summary", "Descripción", "Descripcion", "Description", "Issue Type", "Tipo", "Tipo de incidencia"] if c in df.columns]
    if pref:
        return df[pref].astype(str).agg(" . ".join, axis=1)

    # Fallback: las 3 columnas object más largas en promedio
    obj_cols = [c for c in df.columns if df[c].dtype == object]
    if not obj_cols:
        cand = df.columns[:3].tolist()
        return df[cand].astype(str).agg(" . ".join, axis=1)
    lengths = {c: df[c].astype(str).str.len().mean() for c in obj_cols}
    top = sorted(lengths, key=lengths.get, reverse=True)[:3]
    return df[top].astype(str).agg(" . ".join, axis=1)


# =========================
# Carga de datasets
# =========================
def _load_and_label_historic(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8", sep=None, engine="python")
    df = _normalize_cols(df)

    hours_from_comments = df.apply(_extract_hours_from_comments_row, axis=1)

    # Candidatas explícitas de horas
    candidates = [c for c in df.columns if str(c).lower() in [
        "horas", "hh", "tiempo", "time spent", "Σ time spent", "sum of time spent", "logged hours",
        "tiemposumado", "tiempoint", "tiempoinvertido", "esfuerzo", "original estimate",
        "remaining estimate", "estimate", "estimado", "estimacion", "estimación"
    ] or ("hora" in str(c).lower()) or ("time" in str(c).lower() and "spent" in str(c).lower())]

    hours_explicit = None

    def _parse(v):
        s = str(v).strip().lower().replace(",", ".")
        if not s or s in ("nan", "none"):
            return None
        m = re.match(r"^(\d+):(\d{1,2})$", s)
        if m:
            return int(m.group(1)) + int(m.group(2)) / 60.0
        m = re.match(r"^(\d+(?:\.\d+)?)\s*min", s)
        if m:
            return float(m.group(1)) / 60.0
        m = re.match(r"^(\d+(?:\.\d+)?)(?:\s*h|\s*hh)?$", s)
        if m:
            return float(m.group(1))
        return None

    for c in candidates:
        parsed = df[c].map(_parse)
        if parsed.notna().sum() > 0:
            hours_explicit = parsed
            break

    df["hours"] = hours_from_comments.fillna(hours_explicit) if hours_explicit is not None else hours_from_comments
    df["text"] = _build_text(df)

    tcol = _pick_ticket_column(df)
    df["ticket"] = df[tcol].astype(str) if tcol and tcol in df.columns else df.index.astype(str)

    df = df.dropna(subset=["hours"])[["text", "hours", "ticket"]].reset_index(drop=True)
    df["source"] = "historic"
    return df


def _load_new_estimations(tag: str) -> pd.DataFrame:
    """
    Lee data/estimaciones_nuevas.csv y devuelve filas del tipo solicitado.
    hours = horas_reales > estimacion_final > estimacion_ia
    """
    if not NEW_EST_PATH.exists():
        return pd.DataFrame(columns=["text", "hours", "ticket", "source"])

    df = pd.read_csv(NEW_EST_PATH, encoding="utf-8")
    df = _normalize_cols(df)
    if "tipo" not in df.columns or "texto" not in df.columns:
        return pd.DataFrame(columns=["text", "hours", "ticket", "source"])

    df["tipo"] = df["tipo"].astype(str).str.lower().map({
        "desarrollo": "desarrollo",
        "implementacion": "implementacion",
        "implementación": "implementacion",
        "impl": "implementacion",
        "dev": "desarrollo",
        "cesq": "desarrollo",
        "pstc": "implementacion",
    })

    df = df[df["tipo"] == tag].copy()
    if df.empty:
        return pd.DataFrame(columns=["text", "hours", "ticket", "source"])

    for col in ["horas_reales", "estimacion_final", "estimacion_ia"]:
        if col not in df.columns:
            df[col] = np.nan

    df["hours"] = df["horas_reales"].fillna(df["estimacion_final"]).fillna(df["estimacion_ia"])
    df = df.dropna(subset=["hours", "texto"]).copy()
    df = df[df["hours"].astype(float) > 0]

    df["ticket"] = [f"NEW-{i}" for i in range(len(df))]
    df = df.rename(columns={"texto": "text"})
    df = df[["text", "hours", "ticket"]].reset_index(drop=True)
    df["source"] = "new"
    return df


def _load_catalog(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["text", "hours", "ticket", "source"])

    df = pd.read_csv(path, encoding="utf-8")
    df = _normalize_cols(df)

    name_col, hours_col = None, None
    for c in df.columns:
        lc = str(c).lower()
        if name_col is None and any(k in lc for k in ["tarea", "nombre", "descripción", "descripcion", "actividad"]):
            name_col = c
        if hours_col is None and any(k in lc for k in ["hora", "horas", "hh", "estimado", "estimacion", "estimación"]):
            hours_col = c

    if name_col is None or hours_col is None:
        if len(df.columns) >= 2:
            name_col, hours_col = df.columns[:2]
        else:
            return pd.DataFrame(columns=["text", "hours", "ticket", "source"])

    out = df[[name_col, hours_col]].dropna()
    out = out.rename(columns={name_col: "text", hours_col: "hours"})
    out["hours"] = pd.to_numeric(out["hours"], errors="coerce")
    out = out.dropna(subset=["hours"])
    out["ticket"] = [f"CAT-{i}" for i in range(len(out))]
    out["text"] = out["text"].astype(str)
    out = out[["text", "hours", "ticket"]].reset_index(drop=True)
    out["source"] = "catalog"
    return out


# =========================
# Embeddings + FAISS
# =========================
def _embed_texts(texts: List[str]) -> np.ndarray:
    model = SentenceTransformer(MODEL_NAME)
    emb = model.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
    return np.asarray(emb, dtype="float32")


def _save_faiss(index, path: Path):
    faiss.write_index(index, str(path))


def _load_faiss(path: Path):
    return faiss.read_index(str(path))


class EmbeddingsFaissEstimator:
    def __init__(self, tag: str):
        assert tag in ("desarrollo", "implementacion")
        self.tag = tag
        self.index = None
        self.hours = None

    def _paths(self) -> Dict[str, Path]:
        stem = f"faiss_{self.tag}"
        return {
            "index": INDEX_DIR / f"{stem}.index",
            "hours": INDEX_DIR / f"{stem}_hours.json",
            "meta": INDEX_DIR / f"{stem}_meta.json",
        }

    def fit(self, texts: List[str], hours: List[float]):
        emb = _embed_texts(texts)
        dim = emb.shape[1]
        index = faiss.IndexFlatIP(dim)  # usamos producto interno con embeddings normalizados (≈ coseno)
        index.add(emb)
        self.index = index
        self.hours = list(map(float, hours))

    def save(self):
        p = self._paths()
        _save_faiss(self.index, p["index"])
        Path(p["hours"]).write_text(json.dumps(self.hours, ensure_ascii=False))
        Path(p["meta"]).write_text(json.dumps({"n_docs": len(self.hours), "model": MODEL_NAME}, ensure_ascii=False))

    def load(self):
        p = self._paths()
        self.index = _load_faiss(p["index"])
        self.hours = json.loads(Path(p["hours"]).read_text())
        meta = json.loads(Path(p["meta"]).read_text())
        return meta

    def predict(self, text: str, k: int = 5) -> Tuple[float, List[Tuple[int, float, float]]]:
        if self.index is None:
            raise RuntimeError("Índice no cargado")
        q = _embed_texts([text])
        D, I = self.index.search(q, k)
        sims = D[0].tolist()
        idxs = I[0].tolist()
        neighbors: List[Tuple[int, float, float]] = []
        for i, s in zip(idxs, sims):
            if i == -1:
                continue
            neighbors.append((int(i), float(s), float(self.hours[i])))
        total_sim = sum(s for _, s, _ in neighbors) or 1.0
        est = sum(h * s for _, s, h in neighbors) / total_sim if neighbors else 0.0
        return est, neighbors


# =========================
# Entrenamiento por tipo
# =========================
def train_index_per_type() -> Dict[str, int]:
    counts = {"desarrollo": 0, "implementacion": 0}

    # --- Desarrollo (CESQ) ---
    try:
        df_hist = _load_and_label_historic(_cesq_path())
    except Exception as e:
        print("WARN CESQ:", e)
        df_hist = pd.DataFrame(columns=["text", "hours", "ticket", "source"])
    df_new = _load_new_estimations("desarrollo")
    df_cat = _load_catalog(CATALOGO_CESQ)

    df_dev = pd.concat([df_hist, df_cat, df_new], ignore_index=True)
    if not df_dev.empty:
        model_dev = EmbeddingsFaissEstimator("desarrollo")
        model_dev.fit(df_dev["text"].tolist(), df_dev["hours"].tolist())
        model_dev.save()
        counts["desarrollo"] = len(df_dev)
        # Persistimos el dataset exacto (alineado con el índice)
        ds_path = _dataset_csv_path("desarrollo")
        cols = [c for c in ["text", "hours", "ticket", "source"] if c in df_dev.columns]
        df_dev[cols].reset_index(drop=True).to_csv(ds_path, index=False, encoding="utf-8")

    # --- Implementación (PSTC) ---
    try:
        df_hist = _load_and_label_historic(_pstc_path())
    except Exception as e:
        print("WARN PSTC:", e)
        df_hist = pd.DataFrame(columns=["text", "hours", "ticket", "source"])
    df_new = _load_new_estimations("implementacion")
    df_cat = _load_catalog(CATALOGO_PSTC)

    df_imp = pd.concat([df_hist, df_cat, df_new], ignore_index=True)
    if not df_imp.empty:
        model_imp = EmbeddingsFaissEstimator("implementacion")
        model_imp.fit(df_imp["text"].tolist(), df_imp["hours"].tolist())
        model_imp.save()
        counts["implementacion"] = len(df_imp)
        ds_path = _dataset_csv_path("implementacion")
        cols = [c for c in ["text", "hours", "ticket", "source"] if c in df_imp.columns]
        df_imp[cols].reset_index(drop=True).to_csv(ds_path, index=False, encoding="utf-8")

    if counts["desarrollo"] == 0 and counts["implementacion"] == 0:
        raise RuntimeError("No hay datos con horas en CESQ/PSTC ni en catálogos.")
    return counts


# =========================
# Dataset para la UI
# =========================
def load_labeled_dataframe(tag: str) -> pd.DataFrame:
    """
    Devuelve el dataset EXACTO usado para FAISS si existe (persistido),
    para que los índices de vecinos calcen 1:1 con lo mostrado en pantalla.
    Si no existe, reconstruye desde histórico + catálogo + nuevas.
    """
    ds_path = _dataset_csv_path(tag)
    if ds_path.exists():
        try:
            df = pd.read_csv(ds_path, encoding="utf-8")
            if "text" in df.columns and "hours" in df.columns:
                if "ticket" not in df.columns:
                    df["ticket"] = [str(i) for i in range(len(df))]
                if "source" not in df.columns:
                    df["source"] = "unknown"
                return df[["text", "hours", "ticket", "source"]].reset_index(drop=True)
        except Exception as e:
            print("WARN load_labeled_dataframe persisted read:", e)

    # Fallback (reconstruye)
    if tag == "desarrollo":
        try:
            df_hist = _load_and_label_historic(_cesq_path())
        except Exception:
            df_hist = pd.DataFrame(columns=["text", "hours", "ticket", "source"])
        df_new = _load_new_estimations("desarrollo")
        df_cat = _load_catalog(CATALOGO_CESQ)
        df = pd.concat([df_hist, df_cat, df_new], ignore_index=True)
    elif tag == "implementacion":
        try:
            df_hist = _load_and_label_historic(_pstc_path())
        except Exception:
            df_hist = pd.DataFrame(columns=["text", "hours", "ticket", "source"])
        df_new = _load_new_estimations("implementacion")
        df_cat = _load_catalog(CATALOGO_PSTC)
        df = pd.concat([df_hist, df_cat, df_new], ignore_index=True)
    else:
        raise ValueError("tag inválido: usa 'desarrollo' o 'implementacion'")

    if "ticket" not in df.columns:
        df["ticket"] = [str(i) for i in range(len(df))]
    if "source" not in df.columns:
        df["source"] = "unknown"
    return df.reset_index(drop=True)[["text", "hours", "ticket", "source"]]
