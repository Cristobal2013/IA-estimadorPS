from flask import Flask, render_template, request, jsonify
from pathlib import Path
import csv, time, os, json, math, re, traceback, unicodedata

app = Flask(__name__, static_folder="static", template_folder="templates")

# TOPK configurables (cálculo vs UI)
TOPK = int(os.environ.get("TOPK", "3"))
TOPK_UI = int(os.environ.get("TOPK_UI", str(TOPK)))
DEBUG_FLAG = os.environ.get("DEBUG", "0") == "1"

DATA_DIR = Path(os.environ.get("DATA_DIR", str(Path(__file__).parent / "data")))
DATA_DIR.mkdir(parents=True, exist_ok=True)
NEW_EST_CSV = DATA_DIR / "estimaciones_nuevas.csv"
FULL_FIELDS = ["timestamp","tipo","texto","horas_estimadas","horas_reales","diferencia","top_ticket","top_sim","metodo","autor","comentarios"]

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

# -------- Números robustos --------
_num_rx = re.compile(r"[-+]?(?:\d+[\.,]\d+|\d+)")
def _to_float(x):
    try:
        v = float(str(x).replace(",", ".").strip())
    except Exception:
        m = _num_rx.search(str(x))
        if not m:
            return 0.0
        try:
            v = float(m.group(0).replace(",", "."))
        except Exception:
            v = 0.0
    if math.isnan(v) or math.isinf(v):
        return 0.0
    return v

# ---------------- HORAS: detección robusta ----------------
CAND_HOUR_KEYS = [
    "hours","hour","horas","hh","hhs","hrs","he",
    "total_horas","hh_total","hh_totales",
    "horas_estimadas","estimacion_horas","hrs_estimadas","hrestimadas",
    "hh_estimadas","hhs_estimadas","estimacion_hhs","hhe","hh_est","hhestimadas"
]

_IGNORE_KEYS = {
    "sim","similaridad","similarity","cosine","cosinesim",
    "ticket","codigo","id","source","fuente","tipo","texto",
    "description","descripcion","desc","detalle",
    "fecha","fechacreacion","fechacierre","anio","ano","year","mes"
}

def _strip_accents(s: str) -> str:
    try:
        return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")
    except Exception:
        return s

def _norm_key(k: str) -> str:
    s = _strip_accents(str(k)).lower()
    return (s.replace(" ", "")
             .replace("_", "")
             .replace(".", "")
             .replace("-", "")
             .replace("'", "")
             .replace("’", "")
             .replace("´", ""))

_HOUR_NAME_RX = re.compile(r"(hh|hhs|hrs?|hora|horas|hours|hhe)", re.I)
_HOUR_TEXT_RX = re.compile(r"(\d+(?:[.,]\d+)?)\s*(?:hh'?s?|hh|hrs?|horas?)", re.I)

def _row_hours(row):
    try:
        keys = list(row.index)
    except Exception:
        keys = list(getattr(row, "keys", lambda: [])())
    norm_map = { _norm_key(k): k for k in keys }

    for cand in CAND_HOUR_KEYS:
        key_norm = _norm_key(cand)
        for nk, orig in norm_map.items():
            if nk == key_norm:
                v = _to_float(row.get(orig, 0))
                if v > 0:
                    return v

    for nk, orig in norm_map.items():
        if _HOUR_NAME_RX.search(nk):
            v = _to_float(row.get(orig, 0))
            if v > 0:
                return v

    for k in keys:
        val = row.get(k)
        if isinstance(val, str):
            m = _HOUR_TEXT_RX.findall(val)
            if m:
                try:
                    v = float(m[-1].replace(",", "."))
                except Exception:
                    v = 0.0
                if v > 0:
                    return v

    for k in keys:
        nk = _norm_key(k)
        if nk in _IGNORE_KEYS:
            continue
        v = _to_float(row.get(k, 0))
        if 0 < v < 1000:
            return v
    return 0.0

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
            predict_xgb,
        )
        return {
            "EmbeddingsFaissEstimator": EmbeddingsFaissEstimator,
            "load_labeled_dataframe": load_labeled_dataframe,
            "train_index_per_type": train_index_per_type,
            "estimate_from_catalog": estimate_from_catalog,
            "predict_xgb": predict_xgb,
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
    pred_xgb = backend["predict_xgb"]

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
        horas_faiss, neighbors = est.predict(texto, k=TOPK)
        horas_faiss = _to_float(horas_faiss)
    except Exception as e:
        return jsonify({"ok": False, "error": f"falló predict(): {e}"}), 500

    try:
        labeled = load_df(tag).reset_index(drop=True)
    except Exception as e:
        return jsonify({"ok": False, "error": f"falló load_labeled_dataframe(): {e}"}), 500

    top = []
    try:
        for (idx, sim, h) in sorted(neighbors, key=lambda t: t[1], reverse=True)[:TOPK_UI]:
            if idx is None or idx < 0 or idx >= len(labeled):
                continue
            row = labeled.loc[idx]
            h_neighbor = _to_float(h)
            h_row = _row_hours(row)
            hours_val = h_neighbor if h_neighbor > 0 else h_row
            sim_val = _to_float(sim)
            tk = str(row.get("ticket",""))
            tipo_badge = "CESQ" if tk.upper().startswith("CESQ") else ("PSTC" if tk.upper().startswith("PSTC") else ("CESQ" if tag=="desarrollo" else "PSTC"))
            top.append({
                "ticket": tk,
                "tipo": tipo_badge,
                "hours": float(hours_val),
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

    try:
        horas_xgb = _to_float(pred_xgb(texto, tag))
    except Exception:
        horas_xgb = 0.0

    metodo_norm = metodo if metodo in {"faiss","catalog","faiss+catalog","faiss+xgb+catalog"} else "faiss+xgb+catalog"
    complexity = (data.get("complexity") or "media").lower()
    bias_map = {"baja": -0.1, "media": 0.0, "alta": +0.1}
    bias = bias_map.get(complexity, 0.0)
    alpha = _to_float(os.environ.get("HYBRID_ALPHA", "0.8"))
    alpha_eff = max(0.3, min(0.98, alpha + bias))

    if metodo_norm == "faiss":
        hybrid = horas_faiss
    elif metodo_norm == "catalog":
        hybrid = horas_catalog
    elif metodo_norm == "faiss+catalog":
        hybrid = alpha_eff * horas_faiss + (1.0 - alpha_eff) * horas_catalog
    else:  # faiss+xgb+catalog
        if horas_xgb > 0:
            hybrid = 0.45 * horas_faiss + 0.40 * horas_xgb + 0.15 * horas_catalog
        else:
            hybrid = alpha_eff * horas_faiss + (1.0 - alpha_eff) * horas_catalog

    resp = {
        "ok": True,
        "horas": float(math.ceil(max(0.0, _to_float(hybrid)))),
        "metodo": metodo_norm,
        "detalle": {
            "faiss": float(horas_faiss),
            "xgb": float(horas_xgb),
            "catalogo": float(horas_catalog),
            "final_sin_redondeo": float(round(_to_float(hybrid), 2)),
            "complexity": complexity
        },
        "top": top
    }

    # 🔹 Log informativo
    print(f"[INFO] tipo={tipo}, complejidad={complexity}, α={alpha_eff:.2f}, horas={resp['horas']}")
    return jsonify(_clean_json(resp))

@app.post("/api/razonar")
def api_razonar():
    data = request.get_json(force=True, silent=True) or {}
    texto = (data.get("texto") or "").strip()
    tipo  = (data.get("tipo") or "desarrollo").strip().lower()
    horas = _to_float(data.get("horas") or 0)
    top   = data.get("top") or []

    if not texto:
        return jsonify({"ok": False, "error": "texto vacío"}), 400

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        return jsonify({"ok": False, "error": "GEMINI_API_KEY no configurada en variables de entorno"}), 503

    tipo_nombre = "CESQ (Desarrollo)" if "des" in tipo else "PSTC (Implementación)"

    similares_txt = ""
    for i, t in enumerate(top[:3], 1):
        similares_txt += (
            f"\n{i}. Ticket {t.get('ticket','?')} ({t.get('tipo','?')}) — "
            f"{t.get('hours', 0):.0f}h — Similitud: {t.get('sim', 0):.3f}\n"
            f"   Descripción: {str(t.get('text',''))[:250]}"
        )

    prompt = f"""Eres un experto en estimación de esfuerzo para tickets Jira en proyectos de software fiscal y tributario (Sovos).

TICKET A ESTIMAR ({tipo_nombre}):
{texto}

ESTIMACIÓN INICIAL DEL SISTEMA: {horas:.0f}h

TICKETS HISTÓRICOS MÁS SIMILARES ENCONTRADOS:{similares_txt}

Analiza y responde en formato estructurado con estas 4 secciones:

**1. Comparabilidad con históricos**
¿Los tickets similares son realmente comparables? ¿Qué tienen en común y qué difiere?

**2. Riesgos y complejidades adicionales**
¿Qué factores podría haber pasado por alto el sistema? (integraciones, país, normativa, dependencias)

**3. Rango de estimación recomendado**
Mínimo — Máximo horas, con justificación breve.

**4. Confianza**
Alta / Media / Baja — una oración explicando por qué.

Responde en español. Sé directo y práctico, sin introducción."""

    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)
        return jsonify({"ok": True, "razonamiento": response.text})
    except Exception as e:
        return jsonify({"ok": False, "error": f"Error Gemini: {e}"}), 500


@app.post("/api/accept")
def api_accept():
    data = request.get_json(force=True, silent=True) or {}
    horas_estimadas = _to_float(data.get("horas_estimadas") or data.get("horas") or 0)
    horas_reales    = _to_float(data.get("horas_reales") or 0)
    row = {
        "timestamp":      time.strftime("%Y-%m-%d %H:%M:%S"),
        "tipo":           _resolve_tag(data.get("tipo") or ""),
        "texto":          (data.get("texto") or "").strip(),
        "horas_estimadas": horas_estimadas,
        "horas_reales":    horas_reales,
        "diferencia":      round(horas_reales - horas_estimadas, 2),
        "top_ticket":     (data.get("top_ticket") or "").strip(),
        "top_sim":        _to_float(data.get("top_sim") or 0),
        "metodo":         (data.get("metodo") or "faiss+catalog").strip(),
        "autor":          (data.get("autor") or "web").strip(),
        "comentarios":    (data.get("comentarios") or "").strip(),
    }
    append_row_safe(row)
    return jsonify({"ok": True})
