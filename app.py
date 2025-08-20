
from flask import Flask, render_template, request, jsonify
from pathlib import Path
import csv, time, os, json, math, traceback

app = Flask(__name__, static_folder="static", template_folder="templates")

DATA_DIR = Path(os.environ.get("DATA_DIR", str(Path(__file__).parent / "data")))
DATA_DIR.mkdir(parents=True, exist_ok=True)
NEW_EST_CSV = DATA_DIR / "estimaciones_nuevas.csv"
FULL_FIELDS = ["timestamp","tipo","texto","horas","top_ticket","top_sim","metodo","autor","comentarios"]

def _existing_fields(path: Path):
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            header = f.readline().strip()
        parts = [h.strip() for h in header.split(",")]
        return parts if all(parts) else None
    except Exception:
        return None

def _to_float(x):
    try:
        v = float(x)
    except Exception:
        return 0.0
    if math.isnan(v) or math.isinf(v):
        return 0.0
    return v

def append_row_safe(row: dict):
    fields = _existing_fields(NEW_EST_CSV) or FULL_FIELDS
    new_file = not NEW_EST_CSV.exists() or NEW_EST_CSV.stat().st_size == 0
    with open(NEW_EST_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new_file:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fields})

def _resolve_tag(tipo: str) -> str:
    t = (tipo or "").strip().lower()
    if t.startswith("imp") or "implement" in t or "pstc" in t:
        return "implementacion"
    return "desarrollo"

def _lazy_backend():
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

def _clean_json(o):
    if isinstance(o, float):
        return 0.0 if (math.isnan(o) or math.isinf(o)) else float(o)
    if isinstance(o, dict):
        return {k: _clean_json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean_json(v) for v in o]
    return o

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

    try:
        horas_faiss, neighbors = est.predict(texto, k=int(os.environ.get("TOPK", "15")))
        horas_faiss = _to_float(horas_faiss)
    except Exception as e:
        return jsonify({"ok": False, "error": f"falló predict(): {e}"}), 500

    try:
        labeled = load_df(tag).reset_index(drop=True)
    except Exception as e:
        return jsonify({"ok": False, "error": f"falló load_labeled_dataframe(): {e}"}), 500

    top = []
    try:
        for (idx, sim, h) in sorted(neighbors, key=lambda t: t[1], reverse=True)[:3]:
            if idx is None or idx < 0 or idx >= len(labeled):
                continue
            row = labeled.loc[idx]
            hours_row = _to_float(row.get("hours", 0))
            hours_val = _to_float(h) or hours_row
            sim_val = _to_float(sim)
            tk = str(row.get("ticket",""))
            tipo_badge = "CESQ" if tk.upper().startswith("CESQ") else ("PSTC" if tk.upper().startswith("PSTC") else ("CESQ" if tag=="desarrollo" else "PSTC"))
            top.append({
                "ticket": tk,
                "tipo": tipo_badge,
                "hours": hours_val,
                "sim": sim_val,
                "source": str(row.get("source","")),
                "text": str(row.get("text",""))[:480]
            })
    except Exception as e:
        return jsonify({"ok": False, "error": f"falló armado de top: {e}"}), 500

    try:
        horas_catalog = _to_float(est_cat(texto, tag, top_n=3, min_cover=0.35))
    except Exception:
        horas_catalog = 0.0

    metodo_norm = metodo if metodo in {"faiss","catalog","faiss+catalog"} else "faiss+catalog"
    alpha = _to_float(os.environ.get("HYBRID_ALPHA", "0.8"))
    if metodo_norm == "faiss":
        hybrid = horas_faiss
    elif metodo_norm == "catalog":
        hybrid = horas_catalog
    else:
        hybrid = alpha * horas_faiss + (1.0 - alpha) * horas_catalog

    resp = {
        "ok": True,
        "horas": float(math.ceil(_to_float(hybrid))),
        "metodo": metodo_norm,
        "detalle": {
            "faiss": horas_faiss,
            "catalogo": horas_catalog,
            "final_sin_redondeo": float(round(_to_float(hybrid), 2)),
            "alpha": alpha if metodo_norm == "faiss+catalog" else None
        },
        "top": top
    }
    return jsonify(_clean_json(resp))

@app.post("/api/accept")
def api_accept():
    data = request.get_json(force=True, silent=True) or {}
    row = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tipo": _resolve_tag(data.get("tipo") or ""),
        "texto": (data.get("texto") or "").strip(),
        "horas": _to_float(data.get("horas") or 0),
        "top_ticket": (data.get("top_ticket") or "").strip(),
        "top_sim": _to_float(data.get("top_sim") or 0),
        "metodo": (data.get("metodo") or "faiss+catalog").strip(),
        "autor": (data.get("autor") or "web").strip(),
        "comentarios": (data.get("comentarios") or "").strip(),
    }
    append_row_safe(row)
    return jsonify({"ok": True})
