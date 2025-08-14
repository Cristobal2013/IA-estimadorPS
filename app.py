import os
from pathlib import Path
import shutil
from flask import Flask, request, jsonify, render_template
from jinja2 import TemplateNotFound

# --- SEED persistente: copia CSV del repo a /var/data si faltan ---
def seed_persistent_data_dir():
    data_dir = Path(os.environ.get("DATA_DIR", "/var/data"))
    repo_data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    seed_files = [
        "EstimacionCESQ.csv",
        "EstimacionesPSTC.csv",
        "catalogo_cesq.csv",
        "catalogo_pstc.csv",
        "estimaciones_nuevas.csv",
    ]
    for name in seed_files:
        src = repo_data_dir / name
        dst = data_dir / name
        if src.exists() and not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    # archivo de nuevas estimaciones siempre presente
    (data_dir / "estimaciones_nuevas.csv").touch(exist_ok=True)

seed_persistent_data_dir()
# ---------------------------------------------------------------

from estimator import (
    EmbeddingsFaissEstimator,
    train_index_per_type,
    DATA_DIR,
)

app = Flask(__name__)

@app.route("/", methods=["GET", "HEAD"])
def index():
    # Health check de Render: no renderizamos plantilla en HEAD
    if request.method == "HEAD":
        return "", 200

    # No entrenamos al arrancar; solo comprobamos si existe el índice
    try:
        est = EmbeddingsFaissEstimator("desarrollo")
        est.load()
        status = "ok"
    except Exception:
        status = "no_index"

    # Pasamos un 'form' default para que el template no falle si lo usa
    default_form = {"tipo": "desarrollo", "texto": ""}

    try:
        return render_template("index.html", status=status, form=default_form)
    except TemplateNotFound:
        return jsonify({"status": status, "message": "UI simple: usa /retrain o /api/estimar"}), 200

@app.route("/retrain", methods=["POST", "GET"])
def retrain():
    try:
        rows = train_index_per_type()
        return jsonify({"ok": True, "rows": rows})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/api/estimar", methods=["POST"])
def api_estimar():
    data = request.get_json(force=True) if request.is_json else request.form
    tipo = data.get("tipo", "desarrollo")
    texto = data.get("texto", "")
    if not texto:
        return jsonify({"error": "Falta 'texto'"}), 400

    est = EmbeddingsFaissEstimator(tipo)
    try:
        est.load()
    except Exception:
        # Entrenamiento on-demand si falta el índice
        train_index_per_type()
        est.load()

    _, results = est.predict(texto, k=5)
    horas_vecinas = [r[2] for r in results if r[2] is not None]
    horas = round(sum(horas_vecinas) / len(horas_vecinas), 2) if horas_vecinas else None
    return jsonify({"horas": horas, "vecinos": results})

# Debug opcional
@app.route("/debug-rutas")
def debug_rutas():
    import glob
    return {
        "cwd": os.getcwd(),
        "DATA_DIR": str(DATA_DIR),
        "ls_DATA_DIR": sorted([os.path.basename(p) for p in glob.glob(str(DATA_DIR / '*'))]),
    }, 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
