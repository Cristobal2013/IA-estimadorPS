import os
from flask import Flask, request, jsonify, render_template
from estimator import EmbeddingsFaissEstimator
from pathlib import Path
import shutil

# --- SEED PERSISTENTE PARA RENDER ---
def seed_persistent_data_dir():
    # DATA_DIR: en Render será /var/data (persistente)
    data_dir = Path(os.environ.get("DATA_DIR", "/var/data"))
    # Carpeta data del repo (no persistente)
    repo_data_dir = Path(__file__).parent / "data"

    data_dir.mkdir(parents=True, exist_ok=True)

    seed_files = [
        "EstimacionCESQ.csv",
        "EstimacionesPSTC.csv",
        "catalogo_cesq.csv",
        "catalogo_pstc.csv",
        "estimaciones_nuevas.csv",  # si no existe, se creará
    ]

    for fname in seed_files:
        src = repo_data_dir / fname
        dst = data_dir / fname
        if src.exists() and not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    # asegura que exista el archivo para nuevas estimaciones
    (data_dir / "estimaciones_nuevas.csv").touch(exist_ok=True)
# --- FIN SEED ---

# Sembrar antes de crear la app
seed_persistent_data_dir()

app = Flask(__name__)

@app.route("/")
def index():
    try:
        estimator = EmbeddingsFaissEstimator("desarrollo")
        estimator.load()
    except Exception:
        try:
            estimator.train_index_per_type()
        except Exception as e:
            return f"WARN: No se pudo entrenar índices al iniciar: {e}", 500
    return render_template("index.html")

@app.route("/retrain", methods=["POST", "GET"])
def retrain():
    try:
        EmbeddingsFaissEstimator("desarrollo").train_index_per_type()
        EmbeddingsFaissEstimator("implementacion").train_index_per_type()
        return "Índices reentrenados", 200
    except Exception as e:
        return f"ERROR al reentrenar: {e}", 500

@app.route("/api/estimar", methods=["POST"])
def api_estimar():
    data = request.get_json()
    tipo = data.get("tipo", "desarrollo")
    texto = data.get("texto", "")
    try:
        estimator = EmbeddingsFaissEstimator(tipo)
        estimator.load()
        horas = estimator.estimate_hours(texto)
        return jsonify({"horas": horas})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Debug opcional para Render
@app.route("/debug-rutas")
def debug_rutas():
    from estimator import DATA_DIR
    import glob
    return {
        "cwd": os.getcwd(),
        "DATA_DIR": str(DATA_DIR),
        "ls_DATA_DIR": sorted([os.path.basename(p) for p in glob.glob(str(Path(os.environ.get("DATA_DIR", "/var/data")) / "*"))]),
    }, 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
