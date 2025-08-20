from flask import Flask, render_template, request, jsonify
from pathlib import Path
import csv, time, os, json

from estimator import EmbeddingsFaissEstimator, load_labeled_dataframe, train_index_per_type, estimate_from_catalog

app = Flask(__name__)

DATA_DIR = Path(os.environ.get("DATA_DIR", str(Path(__file__).parent / "data")))
DATA_DIR.mkdir(parents=True, exist_ok=True)
NEW_EST_CSV = DATA_DIR / "estimaciones_nuevas.csv"
CSV_FIELDS = ["timestamp","tipo","texto","horas","top_ticket","top_sim","metodo","autor","comentarios"]

def append_row_safe(row: dict):
    new_file = not NEW_EST_CSV.exists() or NEW_EST_CSV.stat().st_size == 0
    with open(NEW_EST_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new_file:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in CSV_FIELDS})

def _resolve_tag(tipo: str) -> str:
    t = (tipo or "").strip().lower()
    if t.startswith("imp") or "implement" in t or "pstc" in t:
        return "implementacion"
    return "desarrollo"

@app.route("/")
def index():
    return render_template("index.html")

@app.get("/api/health")
def api_health():
    return {"ok": True}

@app.get("/api/estimate")
def api_estimate_get():
    return {"ok": False, "usage": "POST /api/estimate con JSON {tipo, texto}."}, 200

@app.post("/api/estimate")
def api_estimate():
    data = request.get_json(force=True, silent=True) or {}
    tipo = data.get("tipo") or "desarrollo"
    texto = (data.get("texto") or "").strip()
    if not texto:
        return jsonify({"ok": False, "error": "texto vacío"}), 400

    tag = _resolve_tag(tipo)
    est = EmbeddingsFaissEstimator(tag)
    loaded = False
    try:
        loaded = est.load()
    except Exception:
        loaded = False
    if not loaded:
        train_index_per_type(full=True)
        est.load()

    # Predicción y top vecinos
    horas_est, neighbors = est.predict(texto, k=int(os.environ.get("TOPK", "15")))
    labeled = load_labeled_dataframe(tag).reset_index(drop=True)

    top = []
    for (idx, sim, h) in sorted(neighbors, key=lambda t: t[1], reverse=True)[:3]:
        if idx is None or idx < 0 or idx >= len(labeled):
            continue
        row = labeled.loc[idx]
        top.append({
            "ticket": str(row.get("ticket","")),
            "hours": float(h),
            "sim": round(float(sim), 3),
            "source": str(row.get("source","")),
            "text": str(row.get("text",""))[:480]
        })

    # Mezcla FAISS + catálogo (si aplica en tu estimator)
    try:
        cat_horas = estimate_from_catalog(texto, tag, top_n=3, min_cover=0.35)
    except Exception:
        cat_horas = 0.0
    alpha = float(os.environ.get("HYBRID_ALPHA", "0.8"))
    hybrid = round(alpha * float(horas_est or 0.0) + (1.0 - alpha) * float(cat_horas or 0.0), 2)

    return jsonify({"ok": True, "horas": float(hybrid), "metodo": "faiss+catalog", "top": top})

@app.get("/api/accept")
def api_accept_get():
    return {"ok": False, "usage": "POST /api/accept para guardar una estimación."}, 200

@app.post("/api/accept")
def api_accept():
    data = request.get_json(force=True, silent=True) or {}
    row = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tipo": _resolve_tag(data.get("tipo") or ""),
        "texto": (data.get("texto") or "").strip(),
        "horas": float(data.get("horas") or 0) or 0.0,
        "top_ticket": (data.get("top_ticket") or "").strip(),
        "top_sim": float(data.get("top_sim") or 0) or 0.0,
        "metodo": (data.get("metodo") or "faiss+catalog").strip(),
        "autor": (data.get("autor") or "web").strip(),
        "comentarios": (data.get("comentarios") or "").strip(),
    }
    append_row_safe(row)
    return jsonify({"ok": True})
