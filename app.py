
from flask import Flask, render_template, request, jsonify
from pathlib import Path
import csv, time, os, json, math, traceback

app = Flask(__name__, static_folder="static", template_folder="templates")

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

def _lazy_backend():
    """Importa estimator en tiempo de petición. Evita que errores/tiempos de arranque
    tumben el proceso (causando 502 y fallos en /static)."""
    try:
        from estimator import (
            EmbeddingsFaissEstimator,
            load_labeled_dataframe,
            train_index_per_type,
            estimate_from_catalog,
        )
        return {
            "EmbeddingsFaissEstimator": EmbeddingsFaissEstimator,
            "load_labeled_dataframe": load_labeled_dataframe,
            "train_index_per_type": train_index_per_type,
            "estimate_from_catalog": estimate_from_catalog,
            "ok": True, "err": None
        }
    except Exception as e:
        return {"ok": False, "err": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()}

@app.route("/")
def index():
    return render_template("index.html")

@app.get("/api/health")
def api_health():
    backend = _lazy_backend()
    return {
        "ok": True,
        "estimator_import": backend.get("ok", False),
        "estimator_error": backend.get("err")
    }

@app.post("/api/estimate")
def api_estimate():
    data = request.get_json(force=True, silent=True) or {}
    tipo = data.get("tipo") or "desarrollo"
    metodo = (data.get("metodo") or "faiss+catalog").strip().lower()
    texto = (data.get("texto") or "").strip()
    if not texto:
        return jsonify({"ok": False, "error": "texto vacío"}), 400

    tag = _resolve_tag(tipo)

    backend = _lazy_backend()
    if not backend.get("ok"):
        return jsonify({"ok": False, "error": "estimator import failed", "detail": backend.get("err")}), 500

    Emb = backend["EmbeddingsFaissEstimator"]
    load_df = backend["load_labeled_dataframe"]
    train_ix = backend["train_index_per_type"]
    est_cat = backend["estimate_from_catalog"]

    est = Emb(tag)
    loaded = False
    try:
        loaded = est.load()
    except Exception:
        loaded = False
    if not loaded:
        try:
            train_ix(full=True)
            est.load()
        except Exception as e:
            return jsonify({"ok": False, "error": f"no se pudo entrenar/cargar índice: {e}"}), 500

    # Predicción FAISS
    try:
        horas_faiss, neighbors = est.predict(texto, k=int(os.environ.get("TOPK", "15")))
    except Exception as e:
        return jsonify({"ok": False, "error": f"falló predict(): {e}"}), 500

    # Dataset etiquetado
    try:
        labeled = load_df(tag).reset_index(drop=True)
    except Exception as e:
        return jsonify({"ok": False, "error": f"falló load_labeled_dataframe(): {e}"}), 500

    def _to_float(x):
        try:
            return float(x)
        except Exception:
            return 0.0

    top = []
    try:
        for (idx, sim, h) in sorted(neighbors, key=lambda t: t[1], reverse=True)[:3]:
            if idx is None or idx < 0 or idx >= len(labeled):
                continue
            row = labeled.loc[idx]
            hours_row = row.get("hours", 0)
            hours_val = _to_float(h if h not in (None, "", 0) else hours_row)
            tk = str(row.get("ticket",""))
            tipo_badge = "CESQ" if tk.upper().startswith("CESQ") else ("PSTC" if tk.upper().startswith("PSTC") else ("CESQ" if tag=="desarrollo" else "PSTC"))
            top.append({
                "ticket": tk,
                "tipo": tipo_badge,
                "hours": hours_val,
                "sim": round(float(sim), 3),
                "source": str(row.get("source","")),
                "text": str(row.get("text",""))[:480]
            })
    except Exception as e:
        return jsonify({"ok": False, "error": f"falló armado de top: {e}"}), 500

    # Catálogo
    try:
        horas_catalog = est_cat(texto, tag, top_n=3, min_cover=0.35)
    except Exception:
        horas_catalog = 0.0

    metodo_norm = metodo if metodo in {"faiss","catalog","faiss+catalog"} else "faiss+catalog"
    alpha = float(os.environ.get("HYBRID_ALPHA", "0.8"))
    if metodo_norm == "faiss":
        hybrid = float(horas_faiss or 0.0)
    elif metodo_norm == "catalog":
        hybrid = float(horas_catalog or 0.0)
    else:
        hybrid = alpha * float(horas_faiss or 0.0) + (1.0 - alpha) * float(horas_catalog or 0.0)

    import math
    return jsonify({
        "ok": True,
        "horas": float(math.ceil(hybrid)),
        "metodo": metodo_norm,
        "detalle": {
            "faiss": float(horas_faiss or 0.0),
            "catalogo": float(horas_catalog or 0.0),
            "final_sin_redondeo": float(round(hybrid, 2)),
            "alpha": alpha if metodo_norm == "faiss+catalog" else None
        },
        "top": top
    })

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
