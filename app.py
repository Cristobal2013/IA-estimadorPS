
from flask import Flask, render_template, request, redirect, url_for, flash
import math
import os

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret")

@app.template_filter("hfmt")
def hours_format(value):
    """Return '-' for None, otherwise format to 2 decimals."""
    if value is None:
        return "-"
    try:
        return f"{float(value):.2f}"
    except Exception:
        return "-"

def round_quarter(x):
    return round(x * 4) / 4.0

KEYWORD_WEIGHTS = {
    # muy básico; ajusta a tu catálogo
    "api": 2.0,
    "selenium": 2.0,
    "excel": 1.5,
    "pdf": 1.5,
    "grafico": 1.5,
    "gráfico": 1.5,
    "dashboard": 2.0,
    "integracion": 2.0,
    "integración": 2.0,
    "reporte": 1.5,
    "sii": 2.5,
    "netlify": 1.0,
    "flask": 1.0,
    "faiss": 2.0,
    "embeddings": 2.0,
    "sendgrid": 1.0,
    "correo": 1.0,
    "email": 1.0,
    "pfx": 2.0,
    "certificado": 2.0,
    "firefox": 1.0
}

def catalog_estimate(text: str) -> float:
    """Heurística simple basada en palabras clave como stand-in del catálogo base."""
    t = (text or "").lower()
    base = 2.0 if t else 0.0
    for k, w in KEYWORD_WEIGHTS.items():
        if k in t:
            base += w
    # penaliza longitud
    words = len(t.split())
    base += min(words / 200.0, 3.0)  # hasta +3h por textos largos
    return max(0.5, round_quarter(base))

def combined_estimate(text: str):
    # placeholder: por ahora usamos solo catálogo
    ce = catalog_estimate(text)
    se = None   # aquí podrías integrar tu similitud semántica cuando esté lista
    final = ce if se is None else (ce + se) / 2.0
    return ce, se, final

@app.route("/", methods=["GET", "POST"])
def index():
    status = {"ok": True}

    default_form = {
        "tipo": "CESQ",
        "descripcion": ""
    }

    estimate = None
    estimate_catalog = None
    estimate_semantic = None

    if request.method == "POST":
        tipo = request.form.get("tipo") or "CESQ"
        descripcion = (request.form.get("descripcion") or "").strip()

        if not descripcion:
            flash("Por favor ingresa una descripción.", "warning")
        else:
            estimate_catalog, estimate_semantic, estimate = combined_estimate(descripcion)

        default_form.update({"tipo": tipo, "descripcion": descripcion})

    return render_template(
        "index.html",
        status=status,
        form=default_form,
        estimate=estimate,
        estimate_catalog=estimate_catalog,
        estimate_semantic=estimate_semantic,
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
