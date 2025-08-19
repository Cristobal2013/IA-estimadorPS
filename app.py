
from flask import Flask, render_template, request, redirect, url_for, flash
from pathlib import Path
import os

from estimator import (
    EmbeddingsFaissEstimator,
    train_index_per_type,
    load_labeled_dataframe,
    estimate_from_catalog,
)

from hfmt_filter import register_jinja_filters

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", "dev-secret")

# Register Jinja filters (fixes "No filter named 'hfmt'" error)
register_jinja_filters(app)

# Global state (simple demo)
INDEXES = train_index_per_type()  # {'Desarrollo': est, 'Implementación': est}

def _kb_count():
    return sum(len(est.rows) for est in INDEXES.values())

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        return estimate()

    # initial page
    status = {
        "emb_enabled": any(getattr(est, "enabled", False) for est in INDEXES.values()),
        "kb_count": _kb_count()
    }
    ctx = dict(
        status=type("Obj",(),status),
        tipo="Implementación",
        texto="",
        top_k=5,
        estimate_catalog=0.0,
        estimate_semantic=None,
        estimate=0.0,
        kb_hits=[],
        w_catalog=0.5,
        w_semantic=0.5,
    )
    return render_template("index.html", **ctx)

@app.post("/estimate")
def estimate():
    tipo = request.form.get("tipo","Implementación")
    texto = request.form.get("texto","").strip()
    top_k = int(request.form.get("top_k","5") or 5)
    w_catalog = float(request.form.get("w_catalog","0.5") or 0.5)
    w_semantic = float(request.form.get("w_semantic","0.5") or 0.5)

    est = INDEXES.get(tipo)
    kb_hits = est.query(texto, top_k=top_k) if est else []
    # semantic estimate: mean of available hours among hits
    horas_hits = [h.get("horas") for h in kb_hits if h.get("horas") is not None]
    estimate_semantic = round(sum(horas_hits)/len(horas_hits),2) if horas_hits else None

    estimate_catalog = round(estimate_from_catalog(texto, tipo), 2)
    parts = []
    if estimate_semantic is not None: parts.append(w_semantic * estimate_semantic)
    if estimate_catalog is not None: parts.append(w_catalog * estimate_catalog)
    estimate = round(sum(parts), 2) if parts else 0.0

    status = {
        "emb_enabled": any(getattr(e, "enabled", False) for e in INDEXES.values()),
        "kb_count": _kb_count()
    }

    ctx = dict(
        status=type("Obj",(),status),
        tipo=tipo,
        texto=texto,
        top_k=top_k,
        estimate_catalog=estimate_catalog,
        estimate_semantic=estimate_semantic,
        estimate=estimate,
        kb_hits=kb_hits,
        w_catalog=w_catalog,
        w_semantic=w_semantic,
    )
    return render_template("index.html", **ctx)

@app.post("/retrain")
def retrain():
    global INDEXES
    INDEXES = train_index_per_type()
    flash("Índices reentrenados", "success")
    return redirect(url_for("index"))

# Backwards-compatible alias
@app.route("/retrain", methods=["POST"], endpoint="retrain_route")
def retrain_route():
    return retrain()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=True)
