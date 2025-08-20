from flask import Flask, render_template, request, jsonify
import csv, time, json
import pandas as pd
from pathlib import Path

app = Flask(__name__)
DATA_DIR = Path(__file__).parent / "data"
NEW_EST_CSV = DATA_DIR / "estimaciones_nuevas.csv"
NEW_EST_CSV.parent.mkdir(parents=True, exist_ok=True)

CSV_FIELDS = ["timestamp","tipo","texto","horas","top_ticket","top_sim","metodo","autor","comentarios"]

def append_row_safe(row: dict):
    new_file = not NEW_EST_CSV.exists() or NEW_EST_CSV.stat().st_size == 0
    with open(NEW_EST_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new_file:
            w.writeheader()
        out = {k: row.get(k, "") for k in CSV_FIELDS}
        w.writerow(out)

def load_new_estimations() -> pd.DataFrame:
    if NEW_EST_CSV.exists():
        df = pd.read_csv(NEW_EST_CSV)
        df_norm = pd.DataFrame({
            "ticket": df.get("top_ticket", pd.Series([""]*len(df))),
            "hours": df.get("horas", 0.0),
            "text": df.get("texto", ""),
            "source": "new",
            "tipo": df.get("tipo", "")
        })
        return df_norm
    return pd.DataFrame(columns=["ticket","hours","text","source","tipo"])

@app.route("/")
def index():
    return render_template("index.html")

@app.post("/api/estimate")
def api_estimate():
    data = request.get_json(force=True, silent=True) or {}
    result = {
        "horas": 12.5,
        "metodo": "faiss",
        "top": [
            {"ticket": "CESQ-3956", "hours": 5, "sim": 0.658, "text": "[CL][PPL] Ejemplo"},
            {"ticket": "CESQ-3957", "hours": 3, "sim": 0.642, "text": "[CHILE][ALSEA] Otro ejemplo"},
            {"ticket": "CESQ-3960", "hours": 8, "sim": 0.630, "text": "[PE][ABC] Más ejemplo"}
        ]
    }
    return app.response_class(
        response=json.dumps(result, ensure_ascii=False),
        mimetype="application/json"
    )

@app.post("/api/accept")
def api_accept():
    data = request.get_json(force=True, silent=True) or {}
    row = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tipo": (data.get("tipo") or "").strip().lower(),
        "texto": (data.get("texto") or "").strip(),
        "horas": float(data.get("horas") or 0) or 0.0,
        "top_ticket": (data.get("top_ticket") or "").strip(),
        "top_sim": float(data.get("top_sim") or 0) or 0.0,
        "metodo": (data.get("metodo") or "faiss").strip(),
        "autor": (data.get("autor") or "web").strip(),
        "comentarios": (data.get("comentarios") or "").strip(),
    }
    append_row_safe(row)
    return jsonify({"ok": True})

if __name__ == "__main__":
    app.run(debug=True)
