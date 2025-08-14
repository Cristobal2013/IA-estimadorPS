import os
from pathlib import Path
import shutil
from flask import Flask, request, jsonify, render_template
from estimator import EmbeddingsFaissEstimator, train_index_per_type, DATA_DIR

# -------- Seed de /var/data con los CSV del repo --------
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

    # Asegura archivo de nuevas estimaciones
    (data_dir / "estimaciones_nuevas.csv").touch(exist_ok=True)

seed_persistent_data_dir()
# ---------------------------------------------------------

app = Flask(__name__)

@app.route("/")
def index():
    # Intenta cargar; si no existe índice, entrena con catálogos/históricos
    try:
        EmbeddingsFaissEstimator("desarrollo").load()
    except Exception:
        try:
            train_index_per_type()
        except Exception as e:
            return f"WARN: No se pudo entrenar índices al iniciar: {e}", 500
    return render_template("index.html")

@app.route("/retrain", methods=["POST", "GET"])
def retrain():
    try:
        counts = train_index_per_type()
        return jsonify({"ok": True, "rows": counts})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/api/estimar", methods=["POST"])
def api_estimar():
    data = request.get_json(force=True)
    tipo = data.get("tipo", "desarrollo")
    texto = data.get("texto", "")
    try:
        est = EmbeddingsFaissEstimator(tipo)
        est.load()
    except Exception:
        train_index_per_type()
        est = EmbeddingsFaissEstimator(tipo)
        est.load()

    _, results = est.predict(texto, k=5)
    # Promedio simple de horas de los K vecinos con horas válidas
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
