
import os, csv, json, math
from typing import List, Dict, Tuple, Optional
import numpy as np

from flask import Flask, render_template, request, redirect, url_for, flash

# ==== Config ====
ENABLE_EMBEDDINGS = os.environ.get("ENABLE_EMBEDDINGS", "0") in {"1","true","True","yes","YES"}
MODEL_NAME = os.environ.get("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
DATA_CSV = os.environ.get("DATA_CSV", "data/historicos.csv")
EMB_PATH = os.environ.get("EMB_PATH", "data/embeddings.npz")
TOP_K = int(os.environ.get("TOP_K", "5"))

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret")

# ==== Jinja filter ====
@app.template_filter("hfmt")
def hours_format(value):
    if value is None:
        return "-"
    try:
        return f"{float(value):.2f}"
    except Exception:
        return "-"

# ==== Catálogo heurístico (placeholder) ====
KEYWORD_WEIGHTS = {
    "api": 2.0, "selenium": 2.0, "excel": 1.5, "pdf": 1.5, "grafico": 1.5, "gráfico": 1.5,
    "dashboard": 2.0, "integracion": 2.0, "integración": 2.0, "reporte": 1.5, "sii": 2.5,
    "netlify": 1.0, "flask": 1.0, "faiss": 2.0, "embeddings": 2.0, "sendgrid": 1.0,
    "correo": 1.0, "email": 1.0, "pfx": 2.0, "certificado": 2.0, "firefox": 1.0
}
def round_quarter(x: float) -> float:
    return round(x * 4) / 4.0
def catalog_estimate(text: str) -> float:
    t = (text or "").lower()
    base = 2.0 if t else 0.0
    for k, w in KEYWORD_WEIGHTS.items():
        if k in t:
            base += w
    words = len(t.split())
    base += min(words / 200.0, 3.0)
    return max(0.5, round_quarter(base))

# ==== Embeddings (opcional) ====
_model = None
_kb_records: List[Dict] = []
_kb_matrix: Optional[np.ndarray] = None

def _try_import_model():
    global _model
    if not ENABLE_EMBEDDINGS:
        return False
    if _model is not None:
        return True
    try:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME)
        return True
    except Exception as e:
        app.logger.warning("No se pudo cargar el modelo de embeddings: %s", e)
        return False

def _load_kb_from_disk() -> bool:
    """Carga embeddings y registros desde NPZ si existe."""
    global _kb_records, _kb_matrix
    if not os.path.exists(EMB_PATH):
        return False
    try:
        npz = np.load(EMB_PATH, allow_pickle=True)
        _kb_matrix = npz["X"]
        _kb_records = list(npz["records"])
        return True
    except Exception as e:
        app.logger.warning("No se pudo leer %s: %s", EMB_PATH, e)
        return False

def _save_kb_to_disk():
    global _kb_records, _kb_matrix
    np.savez_compressed(EMB_PATH, X=_kb_matrix, records=np.array(_kb_records, dtype=object))

def _read_hist_csv(path: str) -> List[Dict]:
    records = []
    if not os.path.exists(path):
        return records
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, 1):
            rec = {
                "id": row.get("id") or row.get("ticket_id") or str(i),
                "tipo": row.get("tipo") or row.get("category") or "",
                "descripcion": (row.get("descripcion") or row.get("descripcion_limpia") or row.get("texto") or "").strip(),
                "horas": None
            }
            try:
                rec["horas"] = float(row.get("horas")) if row.get("horas") not in (None, "", "-") else None
            except Exception:
                rec["horas"] = None
            if rec["descripcion"]:
                records.append(rec)
    return records

def retrain_from_csv() -> Tuple[int, str]:
    """Lee CSV, calcula embeddings (si hay modelo) y guarda NPZ."""
    global _kb_records, _kb_matrix
    _kb_records = _read_hist_csv(DATA_CSV)
    if not _kb_records:
        return 0, "No hay registros en el CSV"
    if not _try_import_model():
        return len(_kb_records), "Modelo no disponible (instala sentence-transformers o habilita ENABLE_EMBEDDINGS=1)"
    try:
        from sentence_transformers import SentenceTransformer
        X = _model.encode([r["descripcion"] for r in _kb_records], normalize_embeddings=True)
        _kb_matrix = np.asarray(X, dtype="float32")
        _save_kb_to_disk()
        return len(_kb_records), "Embeddings recalculados"
    except Exception as e:
        return len(_kb_records), f"Fallo calculando embeddings: {e}"

def load_or_init_kb():
    # Primero intenta cargar embeddings ya guardados
    if _load_kb_from_disk():
        return
    # Si no existe, intenta entrenar si hay CSV
    if ENABLE_EMBEDDINGS and os.path.exists(DATA_CSV):
        retrain_from_csv()

def semantic_estimate(text: str) -> Tuple[Optional[float], List[Dict]]:
    if not text or not _try_import_model():
        return None, []
    if _kb_matrix is None or not len(_kb_records):
        return None, []
    q = _model.encode([text], normalize_embeddings=True)[0].astype("float32")
    sims = _kb_matrix @ q  # cosenos si están normalizados
    idx = np.argsort(-sims)[:TOP_K]
    hits = []
    weights = []
    horas_vals = []
    for i in idx:
        rec = dict(_kb_records[i])
        rec["sim"] = float(sims[i])
        hits.append(rec)
        if rec.get("horas") is not None:
            weights.append(max(0.0, float(sims[i])))
            horas_vals.append(float(rec["horas"]))
    if weights and sum(weights) > 0:
        est = sum(h * w for h, w in zip(horas_vals, weights)) / sum(weights)
    elif horas_vals:
        est = float(sum(horas_vals) / len(horas_vals))
    else:
        est = None
    return (round_quarter(est) if isinstance(est, (int, float)) else None), hits

# ==== App routes ====
@app.route("/", methods=["GET", "POST"])
def index():
    status = {
        "ok": True,
        "emb": bool(_kb_matrix is not None),
        "kb_count": len(_kb_records),
        "emb_enabled": ENABLE_EMBEDDINGS,
    }

    default_form = {"tipo": "CESQ", "descripcion": ""}

    estimate = None
    estimate_catalog = None
    estimate_semantic = None
    kb_hits: List[Dict] = []

    if request.method == "POST" and request.form.get("action") == "retrain":
        cnt, msg = retrain_from_csv()
        flash(f"Reentrenado: {cnt} registros. {msg}", "info")
    elif request.method == "POST":
        tipo = request.form.get("tipo") or "CESQ"
        descripcion = (request.form.get("descripcion") or "").strip()
        if not descripcion:
            flash("Por favor ingresa una descripción.", "warning")
        else:
            estimate_catalog = catalog_estimate(descripcion)
            estimate_semantic, kb_hits = semantic_estimate(descripcion)
            if estimate_semantic is None:
                estimate = estimate_catalog
            else:
                estimate = (estimate_catalog + estimate_semantic) / 2.0
                estimate = round_quarter(estimate)
        default_form.update({"tipo": tipo, "descripcion": descripcion})

    return render_template(
        "index.html",
        status=status,
        form=default_form,
        estimate=estimate,
        estimate_catalog=estimate_catalog,
        estimate_semantic=estimate_semantic,
        kb_hits=kb_hits,
        top_k=TOP_K,
    )

@app.route("/retrain", methods=["POST"])
def retrain_route():
    cnt, msg = retrain_from_csv()
    flash(f"Reentrenado: {cnt} registros. {msg}", "info")
    return redirect(url_for("index"))

if __name__ == "__main__":
    load_or_init_kb()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
