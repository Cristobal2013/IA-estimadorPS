import streamlit as st
import pandas as pd
import os
import time
import math

from app import _lazy_backend, _resolve_tag, _to_float, append_row_safe
from db import get_table, save_table, init_db

init_db()  # migra CSVs a SQLite si es la primera vez

st.set_page_config(
    page_title="Estimador PS · Sovos",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════
# CSS GLOBAL — DM Sans, color palette, components
# ══════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&family=DM+Mono:wght@400;500&display=swap');

/* ── Variables Sovos ─────────────────────────────────── */
:root {
    --sovos-navy:   #0a0e33;
    --sovos-cyan:   #00bcff;
    --sovos-cyan2:  #33b5f3;
    --sovos-bg:     #f0f5fa;
    --sovos-white:  #ffffff;
    --border:       #dde3ea;
    --text-main:    #0a0e33;
    --text-muted:   #5c6680;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
}

/* ── App background ──────────────────────────────────── */
.stApp { background-color: var(--sovos-bg); }
.stApp > header { background-color: transparent !important; }
section[data-testid="stSidebar"] { background: var(--sovos-navy); }

/* ── Main title — barra Sovos ────────────────────────── */
h1 {
    background: linear-gradient(90deg, var(--sovos-navy) 0%, #12186a 100%);
    color: white !important;
    font-weight: 700 !important;
    font-size: 1.35rem !important;
    letter-spacing: 0.01em !important;
    padding: 16px 24px !important;
    border-radius: 10px !important;
    margin-bottom: 0 !important;
    border-bottom: 3px solid var(--sovos-cyan) !important;
    position: relative;
}
/* Punto de acento cyan junto al texto */
h1::before {
    content: '';
    display: inline-block;
    width: 8px;
    height: 8px;
    background: var(--sovos-cyan);
    border-radius: 50%;
    margin-right: 10px;
    vertical-align: middle;
    box-shadow: 0 0 8px var(--sovos-cyan);
}

/* ── Tabs ────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background-color: var(--sovos-navy);
    border-radius: 8px;
    padding: 5px 8px;
    gap: 3px;
    margin-top: 4px;
}
.stTabs [data-baseweb="tab"] {
    color: rgba(255,255,255,0.5) !important;
    font-weight: 500;
    font-size: 0.87rem;
    border-radius: 6px;
    padding: 8px 18px;
    transition: all 0.15s;
}
.stTabs [data-baseweb="tab"]:hover {
    color: rgba(255,255,255,0.85) !important;
    background: rgba(0,188,255,0.12) !important;
}
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background-color: var(--sovos-cyan) !important;
    color: var(--sovos-navy) !important;
    font-weight: 700 !important;
}
.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] { display: none !important; }

/* ── Primary buttons ─────────────────────────────────── */
div.stButton > button[kind="primary"] {
    background: var(--sovos-cyan) !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    font-size: 0.88rem !important;
    color: var(--sovos-navy) !important;
    padding: 0.5rem 1.4rem !important;
    box-shadow: 0 2px 10px rgba(0,188,255,0.35) !important;
    transition: all 0.15s !important;
    letter-spacing: 0.01em;
}
div.stButton > button[kind="primary"]:hover {
    background: var(--sovos-cyan2) !important;
    box-shadow: 0 4px 16px rgba(0,188,255,0.5) !important;
    transform: translateY(-1px);
}
div.stButton > button[kind="secondary"] {
    border-radius: 8px !important;
    font-weight: 500 !important;
    border-color: var(--border) !important;
}

/* ── Expanders ───────────────────────────────────────── */
details[data-testid="stExpander"] > summary {
    background: var(--sovos-white) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    padding: 12px 16px !important;
    color: var(--text-main) !important;
}
details[data-testid="stExpander"][open] > summary {
    border-radius: 8px 8px 0 0 !important;
    border-bottom-color: transparent !important;
    border-left: 3px solid var(--sovos-cyan) !important;
}
details[data-testid="stExpander"] > div[data-testid="stExpanderDetails"] {
    background: var(--sovos-white) !important;
    border: 1px solid var(--border) !important;
    border-top: none !important;
    border-radius: 0 0 8px 8px !important;
}

/* ── DataFrames / data_editor ────────────────────────── */
[data-testid="stDataFrame"], iframe[title="st_aggrid"] {
    border-radius: 8px !important;
    border: 1px solid var(--border) !important;
    overflow: hidden !important;
}

/* ── Metric cards ────────────────────────────────────── */
[data-testid="metric-container"] {
    background: var(--sovos-white);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px 16px;
}

/* ── HR dividers ─────────────────────────────────────── */
hr { border-color: var(--border) !important; margin: 18px 0 !important; }

/* ── Role total cards ────────────────────────────────── */
.rol-cards {
    display: flex;
    gap: 14px;
    margin: 18px 0 10px 0;
}
.rol-card {
    flex: 1;
    border-radius: 12px;
    padding: 18px 16px 14px;
    text-align: center;
    box-shadow: 0 1px 6px rgba(0,0,0,0.07);
}
.rol-card .lbl {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-bottom: 6px;
}
.rol-card .val {
    font-size: 2.1rem;
    font-weight: 700;
    line-height: 1;
    font-family: 'DM Mono', monospace;
}
.rol-card .unit {
    font-size: 0.82rem;
    font-weight: 500;
    margin-top: 4px;
    opacity: 0.6;
}
/* Colores con paleta Sovos */
.card-tc  { background:#e6f9ff; color:#007aad; border:1px solid #99e6ff; }
.card-sc  { background:#F0FDF4; color:#15803D; border:1px solid #BBF7D0; }
.card-pm  { background:#FFFBEB; color:#B45309; border:1px solid #FDE68A; }
.card-tot { background:var(--sovos-navy); color:white; border:none;
            box-shadow:0 4px 16px rgba(10,14,51,0.35);
            border-bottom: 3px solid var(--sovos-cyan); }
.card-tot .lbl  { opacity:0.55; }
.card-tot .unit { opacity:0.45; }

/* ── Project banner ──────────────────────────────────── */
.proy-banner {
    background: var(--sovos-white);
    border-left: 4px solid var(--sovos-cyan);
    border-radius: 0 10px 10px 0;
    padding: 14px 20px;
    margin: 14px 0 6px;
    box-shadow: 0 1px 6px rgba(0,0,0,0.06);
}
.proy-banner .b-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--sovos-navy);
    margin: 0 0 3px;
}
.proy-banner .b-sub {
    font-size: 0.83rem;
    color: var(--text-muted);
    margin: 0;
}

/* ── Estado badges ───────────────────────────────────── */
.badge {
    display:inline-flex; align-items:center;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.badge-estimado  { background:#e6f9ff; color:#007aad; }
.badge-corregido { background:#DCFCE7; color:#15803D; }
.badge-descartado{ background:#FEE2E2; color:#B91C1C; }

/* ── Info/warning/success boxes ──────────────────────── */
[data-testid="stAlert"] { border-radius: 8px !important; }

/* ── Subheaders ──────────────────────────────────────── */
h2, h3 { color: var(--sovos-navy) !important; font-weight: 600 !important; }

/* ── Inputs / Selectbox / Multiselect ───────────────── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: var(--sovos-white) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-main) !important;
    font-family: 'DM Sans', sans-serif !important;
    padding: 8px 12px !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: var(--sovos-cyan) !important;
    box-shadow: 0 0 0 3px rgba(0,188,255,0.18) !important;
}

/* Selectbox */
.stSelectbox > div > div,
.stSelectbox > div > div > div {
    background: var(--sovos-white) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-main) !important;
}

/* Multiselect */
.stMultiSelect > div > div {
    background: var(--sovos-white) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 8px !important;
    min-height: 44px !important;
    color: var(--text-main) !important;
}
.stMultiSelect > div > div:focus-within {
    border-color: var(--sovos-cyan) !important;
    box-shadow: 0 0 0 3px rgba(0,188,255,0.18) !important;
}
.stMultiSelect [data-baseweb="tag"] {
    background: #e6f9ff !important;
    color: #007aad !important;
    border-radius: 6px !important;
    font-weight: 500 !important;
    font-size: 0.82rem !important;
}

/* Labels de formulario */
.stTextInput label,
.stTextArea label,
.stSelectbox label,
.stMultiSelect label,
.stNumberInput label,
[data-testid="stWidgetLabel"] {
    color: var(--text-muted) !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.04em !important;
    margin-bottom: 4px !important;
}

/* Contenedor de columnas — fondo blanco como card */
.stColumn > div {
    background: var(--sovos-white);
    border-radius: 10px;
    padding: 18px 20px !important;
    border: 1px solid var(--border);
}

/* Select slider — acento cyan */
.stSlider > div { padding: 4px 0; }
[data-testid="stSlider"] > div > div > div {
    background: var(--sovos-cyan) !important;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
# DATOS
# ══════════════════════════════════════════════════════
@st.cache_data(ttl=60)
def _load_roles():
    df = get_table("catalogo_roles")
    return df if not df.empty else pd.DataFrame()


# ══════════════════════════════════════════════════════
# ESTIMACIÓN IA (FAISS / XGB / CATÁLOGO)
# ══════════════════════════════════════════════════════
def _estimar_texto(texto, tag, metodo, complexity, backend):
    Emb      = backend["EmbeddingsFaissEstimator"]
    est_cat  = backend["estimate_from_catalog"]
    train_ix = backend["train_index_per_type"]
    load_df  = backend["load_labeled_dataframe"]

    est = Emb(tag)
    try:
        loaded = est.load()
    except Exception:
        loaded = False
    if not loaded:
        train_ix(full=True)
        est.load()

    horas_faiss, neighbors = est.predict(texto, k=5)
    horas_faiss = _to_float(horas_faiss)

    labeled = load_df(tag).reset_index(drop=True)
    top_tickets = []
    for (idx, sim, h) in neighbors[:3]:
        if idx is not None and 0 <= idx < len(labeled):
            row = labeled.loc[idx]
            top_tickets.append({
                "ticket": str(row.get("ticket", "N/A")),
                "hours": float(h),
                "sim": float(sim),
                "text": str(row.get("text", ""))[:200],
            })

    horas_catalog = 0.0
    catalog_match = ""
    try:
        from estimator import estimate_from_catalog_with_match
        horas_catalog, catalog_match = estimate_from_catalog_with_match(texto, tag, top_n=3, min_cover=0.35)
        horas_catalog = _to_float(horas_catalog)
    except Exception:
        try:
            horas_catalog = _to_float(est_cat(texto, tag, top_n=3, min_cover=0.35))
        except Exception:
            horas_catalog = 0.0

    horas_xgb = 0.0
    try:
        from estimator import predict_xgb
        horas_xgb = _to_float(predict_xgb(texto, tag))
    except Exception:
        horas_xgb = 0.0

    # ── Confianza FAISS (similitud del vecino más cercano) ────────────
    sim_top = top_tickets[0]["sim"] if top_tickets else 0.0

    if metodo == "faiss":
        final = horas_faiss
    elif metodo == "catalog":
        final = horas_catalog
    elif metodo == "faiss+catalog":
        # Pesos adaptativos: si FAISS no es confiable, catálogo gana peso
        if sim_top >= 0.75:
            final = 0.85 * horas_faiss + 0.15 * horas_catalog
        elif sim_top >= 0.55:
            final = 0.75 * horas_faiss + 0.25 * horas_catalog
        else:
            final = 0.55 * horas_faiss + 0.45 * horas_catalog
    else:  # faiss+xgb+catalog — pesos adaptativos por confianza FAISS
        if horas_xgb > 0:
            if sim_top >= 0.75:        # alta confianza FAISS
                w_f, w_x, w_c = 0.55, 0.35, 0.10
            elif sim_top >= 0.55:      # confianza media
                w_f, w_x, w_c = 0.45, 0.35, 0.20
            else:                      # baja confianza: catálogo toma más peso
                w_f, w_x, w_c = 0.30, 0.30, 0.40
            # Si no hay match en catálogo, redistribuir su peso a FAISS/XGB
            if horas_catalog == 0.0:
                w_f += w_c * 0.6
                w_x += w_c * 0.4
                w_c = 0.0
            final = w_f * horas_faiss + w_x * horas_xgb + w_c * horas_catalog
        else:
            if sim_top >= 0.75:
                final = 0.85 * horas_faiss + 0.15 * horas_catalog
            elif sim_top >= 0.55:
                final = 0.75 * horas_faiss + 0.25 * horas_catalog
            else:
                final = 0.55 * horas_faiss + 0.45 * horas_catalog

    # Multiplicador de complejidad aplicado siempre al resultado final
    cx_mult = {"baja": 0.85, "media": 1.00, "alta": 1.20}
    final = final * cx_mult.get(complexity, 1.00)

    # ── Rango desde horas reales de vecinos (no multiplicadores fijos) ──
    # Si tenemos ≥3 vecinos → rango = percentil 25-75 de sus horas
    # Si tenemos 2          → rango = min-max de sus horas
    # Si tenemos 0-1        → ±20% de la estimación final
    hs = [item["hours"] for item in top_tickets] if top_tickets else []
    if len(hs) >= 3:
        rango_min = max(1, int(math.ceil(float(sorted(hs)[0]))))
        rango_max = int(math.ceil(float(sorted(hs)[-1])))
        # Usar p25-p75 si el spread es muy amplio (>3x)
        if rango_max > rango_min * 3:
            rango_min = max(1, int(math.ceil(sorted(hs)[len(hs)//4])))
            rango_max = int(math.ceil(sorted(hs)[3*len(hs)//4]))
    elif len(hs) == 2:
        rango_min = max(1, int(math.ceil(min(hs))))
        rango_max = int(math.ceil(max(hs)))
    else:
        rango_min = max(1, int(math.ceil(final * 0.80)))
        rango_max = int(math.ceil(final * 1.25))

    return {
        "horas": final,
        "faiss": horas_faiss,
        "xgb": horas_xgb,
        "catalogo": horas_catalog,
        "catalog_match": catalog_match,
        "top": top_tickets,
        "rango_min": rango_min,
        "rango_max": rango_max,
    }


# ══════════════════════════════════════════════════════
# DISTRIBUCIÓN TC/SC/PM PARA TAREAS IA
# ══════════════════════════════════════════════════════
def _get_role_ratios(texto: str, df_roles: pd.DataFrame, integ_pref: str):
    """Busca el paquete más similar y retorna proporciones (tc_r, sc_r, pm_r, nombre_match)."""
    import unicodedata, re

    if df_roles.empty or not texto.strip():
        return 1.0, 0.0, 0.0, ""

    def _norm(s):
        s = unicodedata.normalize("NFD", str(s))
        s = "".join(c for c in s if unicodedata.category(c) != "Mn")
        return re.sub(r"[^a-z0-9\s]", " ", s.lower())

    STOP = {"de", "la", "el", "en", "y", "a", "con", "por", "para", "del",
            "los", "las", "un", "una", "se", "su", "que", "es", "al", "lo"}

    def tokenize(s):
        return set(_norm(s).split()) - STOP

    query = tokenize(texto)
    if not query:
        return 1.0, 0.0, 0.0, ""

    best_score = 0.0
    best_row = None
    best_name = ""
    seen: set = set()

    for _, row in df_roles.iterrows():
        key = (row["paquete"], row["escenario"])
        if key in seen:
            continue
        seen.add(key)

        pkg_tokens = tokenize(f"{row['paquete']} {row['escenario']}")
        if not pkg_tokens:
            continue

        inter = query & pkg_tokens
        score = len(inter) / max(len(query), len(pkg_tokens))

        if score > best_score:
            best_score = score
            # Obtener fila con la integración preferida
            sub = df_roles[(df_roles["paquete"] == row["paquete"]) & (df_roles["escenario"] == row["escenario"])]
            m = sub[sub["integracion"] == integ_pref]
            if m.empty:
                m = sub[sub["integracion"] == "Estandar"]
            if m.empty:
                m = sub.iloc[:1]
            if not m.empty:
                best_row = m.iloc[0]
                best_name = f"{row['paquete']} · {row['escenario']}"

    if best_row is None or float(best_row["total"]) == 0:
        return 1.0, 0.0, 0.0, ""

    total = float(best_row["total"])
    return (
        float(best_row["tc"]) / total,
        float(best_row["sc"]) / total,
        float(best_row["pm"]) / total,
        best_name,
    )


# ══════════════════════════════════════════════════════
# PARSER DE AJUSTE MANUAL POR ROL
# ══════════════════════════════════════════════════════
def _parse_ajuste_rol(texto: str) -> dict | None:
    """
    Detecta si el texto describe un ajuste directo de horas a un rol específico.
    Ejemplos reconocidos:
      "5 horas SC", "SC +3h reuniones", "añadir 4h a TC", "PM 2 horas gestión"
    Retorna dict con tc/sc/pm/total o None si no detecta patrón.
    """
    import re

    t = texto.lower().strip()

    # Extraer primer número (incluye decimales)
    num = re.search(r"(\d+(?:[.,]\d+)?)", t)
    if not num:
        return None
    horas = float(num.group(1).replace(",", "."))
    if horas <= 0:
        return None

    # Detectar rol explícito
    if re.search(r"\bsc\b", t):
        return {"tc": 0.0, "sc": horas, "pm": 0.0, "total": horas, "rol": "SC"}
    if re.search(r"\btc\b", t):
        return {"tc": horas, "sc": 0.0, "pm": 0.0, "total": horas, "rol": "TC"}
    if re.search(r"\bpm\b", t):
        return {"tc": 0.0, "sc": 0.0, "pm": horas, "total": horas, "rol": "PM"}

    return None  # sin rol explícito → usar FAISS


# ══════════════════════════════════════════════════════
# AJUSTE INTELIGENTE DE HORAS (overhead compartido)
# ══════════════════════════════════════════════════════
def _ajuste_proyecto(filas: list) -> dict:
    """
    Para proyectos multi-paquete, el overhead de PM (kickoff, SOW, cierre)
    no se multiplica por cada paquete — se comparte. El paquete más pesado
    'dueña' el overhead completo; los adicionales solo aportan su trabajo
    técnico específico (~30% del PM incremental).
    TC y SC sí son aditivos porque el trabajo técnico/solución es único por paquete.
    """
    filas_cat = [f for f in filas if f["Origen"] == "📦 Catálogo"]
    filas_ia  = [f for f in filas if f["Origen"] != "📦 Catálogo"]

    total_tc_raw = sum(f["TC (h)"] for f in filas)
    total_sc_raw = sum(f["SC (h)"] for f in filas)
    total_pm_raw = sum(f["PM (h)"] for f in filas)

    n_cat = len(filas_cat)
    if n_cat <= 1:
        return {
            "tc": total_tc_raw, "sc": total_sc_raw, "pm": total_pm_raw,
            "total_raw": total_tc_raw + total_sc_raw + total_pm_raw,
            "total_ajustado": total_tc_raw + total_sc_raw + total_pm_raw,
            "ahorro_pm": 0.0, "nota": "",
        }

    # PM ajustado: el mayor paquete paga el 100%, los demás el 30%
    pms_cat = sorted([f["PM (h)"] for f in filas_cat], reverse=True)
    pm_cat_ajustado = pms_cat[0] + sum(p * 0.30 for p in pms_cat[1:])
    pm_ia = sum(f["PM (h)"] for f in filas_ia)
    pm_total_ajustado = round(pm_cat_ajustado + pm_ia, 1)
    ahorro = round(total_pm_raw - pm_total_ajustado, 1)

    nota = (
        f"PM reducido en **{ahorro:.1f}h** por overhead compartido entre "
        f"{n_cat} paquetes (kickoff, SOW, cierre son una sola vez)."
    )
    return {
        "tc": total_tc_raw,
        "sc": total_sc_raw,
        "pm": pm_total_ajustado,
        "total_raw": total_tc_raw + total_sc_raw + total_pm_raw,
        "total_ajustado": total_tc_raw + total_sc_raw + pm_total_ajustado,
        "ahorro_pm": ahorro,
        "nota": nota,
    }


# ══════════════════════════════════════════════════════
# GUARDAR / CARGAR PROYECTOS
# ══════════════════════════════════════════════════════
import json

_PROYECTOS_FILE = os.path.join(os.path.dirname(__file__), "data", "proyectos_guardados.json")

def _load_proyectos() -> list:
    if not os.path.exists(_PROYECTOS_FILE):
        return []
    try:
        with open(_PROYECTOS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def _save_proyecto(entry: dict):
    proyectos = _load_proyectos()
    proyectos.append(entry)
    with open(_PROYECTOS_FILE, "w", encoding="utf-8") as f:
        json.dump(proyectos, f, ensure_ascii=False, indent=2)

def _update_proyecto(idx: int, data: dict):
    """Actualiza campos de un proyecto existente por índice."""
    proyectos = _load_proyectos()
    if 0 <= idx < len(proyectos):
        proyectos[idx].update(data)
        with open(_PROYECTOS_FILE, "w", encoding="utf-8") as f:
            json.dump(proyectos, f, ensure_ascii=False, indent=2)

def _delete_proyecto(idx: int):
    """Elimina un proyecto por índice."""
    proyectos = _load_proyectos()
    if 0 <= idx < len(proyectos):
        proyectos.pop(idx)
        with open(_PROYECTOS_FILE, "w", encoding="utf-8") as f:
            json.dump(proyectos, f, ensure_ascii=False, indent=2)


# ══════════════════════════════════════════════════════
# ANÁLISIS GEMINI DEL PROYECTO COMPLETO
# ══════════════════════════════════════════════════════
def _gemini_proyecto(nombre: str, integ: str, filas: list, ajuste: dict) -> str | None:
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        return None

    lineas = []
    for f in filas:
        if f["Origen"] == "📦 Catálogo":
            lineas.append(
                f"  • {f['Paquete / Tarea']} ({f['Integración']}): "
                f"TC={f['TC (h)']}h | SC={f['SC (h)']}h | PM={f['PM (h)']}h"
            )
        else:
            lineas.append(f"  • [IA] {f['Paquete / Tarea']}: ~{f['Total (h)']}h")

    prompt = f"""Eres experto en estimación de proyectos PS para Sovos (Latinoamérica).

PROYECTO: {nombre or 'Sin nombre'}
INTEGRACIÓN: {integ}

COMPONENTES SELECCIONADOS:
{chr(10).join(lineas)}

SUMA CRUDA: TC={ajuste['tc']:.1f}h | SC={ajuste['sc']:.1f}h | PM={ajuste['total_raw'] - ajuste['tc'] - ajuste['sc']:.1f}h | Total={ajuste['total_raw']:.1f}h
ESTIMACIÓN AJUSTADA (overhead PM compartido): Total={ajuste['total_ajustado']:.1f}h

Analiza este proyecto y responde con estas 3 secciones:

**1. Solapamientos detectados**
¿Hay configuraciones o actividades que aplican a múltiples paquetes y podrían ejecutarse en paralelo o una sola vez?

**2. Riesgos que podrían aumentar el esfuerzo**
Máximo 3 puntos. Sé específico al tipo de integración y paquetes seleccionados.

**3. Rango recomendado**
TC: Xh–Xh | SC: Xh–Xh | PM: Xh–Xh | Total: Xh–Xh (mínimo–máximo)
Una oración explicando el rango.

Responde en español, máximo 220 palabras. Sin introducción."""

    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash")
        return model.generate_content(prompt).text
    except Exception as e:
        return f"_(Error Gemini: {e})_"


# ══════════════════════════════════════════════════════
# DEDUPLICACIÓN INTELIGENTE DE TAREAS
# ══════════════════════════════════════════════════════
def _overlap_score(a: str, b: str) -> float:
    """Jaccard sobre tokens normalizados ≥ 3 letras, sin stopwords."""
    import re, unicodedata
    def _toks(s):
        s = unicodedata.normalize("NFD", str(s))
        s = "".join(c for c in s if unicodedata.category(c) != "Mn")
        toks = set(re.findall(r"[a-z]{3,}", s.lower()))
        return toks - {"del","los","las","una","que","por","con","para","desde",
                       "hasta","todo","todos","cada","nuevo","nueva"}
    ta, tb = _toks(a), _toks(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _deduplicar_filas(filas: list) -> tuple:
    """
    Detecta duplicados y trabajo compartido entre todos los ítems.

    Reglas:
    • Cat vs Cat mismo paquete: TC del 2º ítem en adelante −25% (setup compartido)
    • Cat vs Cat texto >= 0.65: alerta de posible duplicado, sin modificar horas
    • IA vs Catálogo >= 0.65 → reduce horas IA en (solapamiento × horas), marca motivo
    • IA vs Catálogo 0.40–0.65 → aviso en Referencia IA, sin tocar horas
    • IA vs IA        >= 0.70 → fusiona: la de más horas absorbe a la otra (horas=0)

    Devuelve (filas_ajustadas, lista_de_alertas_str)
    """
    _CAT = {"📦 Catálogo", "🔍 Búsqueda"}
    filas = [f.copy() for f in filas]
    alertas: list[str] = []

    cat_idx  = [i for i, f in enumerate(filas) if f["Origen"] in _CAT]
    cat_rows = [filas[i] for i in cat_idx]
    ia_idx   = [i for i, f in enumerate(filas) if f["Origen"] not in _CAT
                and "✏️ Manual" not in f["Origen"]]

    # ── Paso 0: Catálogo vs Catálogo ──────────────────
    # a) Duplicado textual (≥ 0.65): alerta sin modificar
    # b) Mismo paquete base: TC del ítem posterior −25% por setup compartido
    paquetes_vistos: dict[str, int] = {}   # paquete_base → índice del primero

    for i in cat_idx:
        f = filas[i]
        nombre  = f["Paquete / Tarea"]
        paquete = nombre.split(" · ")[0].strip() if " · " in nombre else nombre

        # a) Solapamiento textual alto con cualquier ítem anterior del catálogo
        for j in cat_idx:
            if j >= i:
                break
            s = _overlap_score(nombre, filas[j]["Paquete / Tarea"])
            if s >= 0.65:
                alertas.append(
                    f"⚠️ **{nombre}** es {int(s*100)}% similar a "
                    f"**{filas[j]['Paquete / Tarea']}** — verifica si no son el mismo componente"
                )

        # b) Mismo paquete base → descuento de setup en TC
        if paquete in paquetes_vistos:
            orig_tc = f["TC (h)"]
            f["TC (h)"] = max(0, int(round(orig_tc * 0.75)))
            ahorro_tc = orig_tc - f["TC (h)"]
            if ahorro_tc > 0:
                f["Total (h)"] = f["TC (h)"] + f["SC (h)"] + f["PM (h)"]
                alertas.append(
                    f"**{nombre}** comparte setup de plataforma con "
                    f"**{filas[paquetes_vistos[paquete]]['Paquete / Tarea']}** "
                    f"— TC reducido {orig_tc}h → {f['TC (h)']}h (−{ahorro_tc}h setup duplicado)"
                )
        else:
            paquetes_vistos[paquete] = i

    # ── Paso 1: IA vs Catálogo ────────────────────────
    for i in ia_idx:
        f = filas[i]
        best_s, best_cat = 0.0, None
        for cat in cat_rows:
            s = _overlap_score(f["Paquete / Tarea"], cat["Paquete / Tarea"])
            if s > best_s:
                best_s, best_cat = s, cat

        if best_cat and best_s >= 0.65:
            # Reducción proporcional: lo que el catálogo ya cubre, la IA no debe cobrar
            factor   = max(0.0, 1.0 - best_s)
            orig_tot = f["Total (h)"]
            f["TC (h)"] = max(0, int(round(f["TC (h)"] * factor)))
            f["SC (h)"] = max(0, int(round(f["SC (h)"] * factor)))
            f["PM (h)"] = max(0, int(round(f["PM (h)"] * factor)))
            f["Total (h)"] = f["TC (h)"] + f["SC (h)"] + f["PM (h)"]
            nota = (f"⚠️ {int(best_s*100)}% cubierto por catálogo "
                    f"'{best_cat['Paquete / Tarea']}' · {orig_tot}h→{f['Total (h)']}h")
            f["Referencia IA"] = nota
            alertas.append(
                f"**{f['Paquete / Tarea']}** solapa {int(best_s*100)}% con catálogo "
                f"**{best_cat['Paquete / Tarea']}** — horas reducidas "
                f"{orig_tot}h → {f['Total (h)']}h"
            )

        elif best_cat and best_s >= 0.40:
            old_ref = f["Referencia IA"]
            f["Referencia IA"] = (
                (old_ref.rstrip("—").strip() + " · " if old_ref not in ("—", "") else "")
                + f"⚠️ Similar a: {best_cat['Paquete / Tarea']}"
            )

    # ── Paso 2: IA vs IA (fusión de tareas similares) ─
    fusionados: set = set()
    for pi, idx_a in enumerate(ia_idx):
        if idx_a in fusionados:
            continue
        for idx_b in ia_idx[pi + 1:]:
            if idx_b in fusionados:
                continue
            s = _overlap_score(
                filas[idx_a]["Paquete / Tarea"],
                filas[idx_b]["Paquete / Tarea"]
            )
            if s >= 0.70:
                winner = idx_a if filas[idx_a]["Total (h)"] >= filas[idx_b]["Total (h)"] else idx_b
                loser  = idx_b if winner == idx_a else idx_a
                loser_name   = filas[loser]["Paquete / Tarea"]
                winner_name  = filas[winner]["Paquete / Tarea"]
                alertas.append(
                    f"**{loser_name}** es {int(s*100)}% similar a **{winner_name}** "
                    f"— fusionadas para evitar duplicado"
                )
                filas[loser]["TC (h)"]        = 0
                filas[loser]["SC (h)"]        = 0
                filas[loser]["PM (h)"]        = 0
                filas[loser]["Total (h)"]     = 0
                filas[loser]["Referencia IA"] = f"🔗 Fusionado con: {winner_name}"
                fusionados.add(loser)

    return filas, alertas


# ══════════════════════════════════════════════════════
# BÚSQUEDA INTELIGENTE EN CATÁLOGO
# Usa sinónimos del dominio PS + recall sobre catálogo
# ══════════════════════════════════════════════════════

# Diccionario de sinónimos del dominio PS/fiscal
# Clave = término que puede escribir el usuario → expansión = términos del catálogo
_PS_SYNONYMS: dict[str, list[str]] = {
    # Templates / Plantillas
    "xsl":          ["plantilla", "template", "modificacion"],
    "xslt":         ["plantilla", "template"],
    "template":     ["plantilla"],
    "membrete":     ["logo", "imagen", "plantilla", "campo"],
    "visualizacion":["plantilla", "template", "diseño"],
    "mail":         ["plantilla", "documento", "correo"],
    "email":        ["plantilla", "correo", "campo"],
    "pdf":          ["plantilla", "documento"],
    "impresion":    ["plantilla"],
    "layout":       ["plantilla", "diseño"],
    "diseño":       ["plantilla"],
    "formato":      ["plantilla", "campo"],
    # Modificaciones
    "actualizar":   ["modificacion", "modificar", "cambio", "correccion"],
    "actualización":["modificacion", "modificar"],
    "corregir":     ["modificacion", "correccion", "cambio"],
    "corrección":   ["modificacion"],
    "cambiar":      ["modificacion", "cambio"],
    "ajustar":      ["modificacion"],
    "reemplazar":   ["modificacion", "campo"],
    "eliminar":     ["modificacion", "campo"],
    "agregar":      ["modificacion", "campo", "nuevo"],
    "añadir":       ["modificacion", "campo", "nuevo"],
    "incluir":      ["modificacion", "campo"],
    # Campos
    "campo":        ["campo"],
    "texto":        ["campo"],
    "dato":         ["campo"],
    "firma":        ["campo", "plantilla"],
    "dirección":    ["campo"],
    "direccion":    ["campo"],
    "telefono":     ["campo"],
    "teléfono":     ["campo"],
    "rut":          ["campo", "dato"],
    "logo":         ["logo", "imagen"],
    "imagen":       ["logo", "imagen"],
    "nombre":       ["campo", "dato"],
    "razon":        ["campo", "dato"],
    "giro":         ["campo", "dato"],
    # Procesos PS
    "habilitacion": ["habilitacion", "activacion", "nuevo"],
    "habilitar":    ["habilitacion", "activacion"],
    "activar":      ["habilitacion", "activacion"],
    "configurar":   ["configuracion"],
    "configuracion":["configuracion"],
    "implementar":  ["implementacion"],
    "implementacion":["implementacion"],
    "integracion":  ["integracion"],
    "migracion":    ["migracion"],
    "migrar":       ["migracion"],
    "certificado":  ["certificacion", "certificado"],
    "certificacion":["certificacion"],
    "go-live":      ["golive", "acompañamiento", "soporte"],
    "golive":       ["golive", "acompañamiento"],
    "soporte":      ["soporte", "acompañamiento"],
    "prueba":       ["testing", "qa"],
    "testing":      ["testing", "qa"],
    # Documentos SII / fiscales
    "dte":          ["dte", "documento", "factura"],
    "factura":      ["factura", "dte"],
    "boleta":       ["boleta", "dte"],
    "nota":         ["nota", "dte"],
    "guia":         ["guia", "dte"],
    "liquidacion":  ["liquidacion", "dte"],
    "cesion":       ["cesion"],
    "recepcion":    ["recepcion"],
    "reclamacion":  ["reclamacion"],
    "ppl":          ["ppl"],
    "coapi":        ["coapi"],
    "res154":       ["res154"],
    "res55":        ["res55"],
    # Módulos / Sistemas
    "gateway":      ["gateway", "gw", "configuracion"],
    "epos":         ["epos", "pos", "gateway"],
    "pos":          ["pos", "gateway"],
    "api":          ["api", "integracion"],
    "sftp":         ["sftp", "integracion"],
    "xml":          ["xml", "plantilla", "documento"],
    "json":         ["json", "api"],
    # Generales
    "nuevo":        ["nuevo", "habilitacion", "creacion"],
    "nueva":        ["nuevo", "habilitacion", "creacion"],
    "creacion":     ["creacion", "nuevo"],
    "crear":        ["creacion", "nuevo"],
    "desde":        ["creacion", "nuevo"],
    "sociedad":     ["sociedad", "rut", "empresa"],
    "empresa":      ["empresa", "cliente"],
    "cliente":      ["cliente"],
    "mandato":      ["mandato", "requerimiento"],
    "requerimiento":["requerimiento"],
}


def _expandir_tokens(tokens: set) -> set:
    """Expande el conjunto de tokens con sinónimos del dominio PS."""
    expandidos = set(tokens)
    for t in tokens:
        for sinónimo in _PS_SYNONYMS.get(t, []):
            expandidos.add(sinónimo)
    return expandidos


def _buscar_catalogo_libre(texto: str, df_roles: pd.DataFrame, top_n: int = 6) -> list:
    """
    Busca componentes en df_roles.
    Usa expansión de sinónimos del dominio PS + métrica combinada:
      - Recall sobre catálogo (qué % del catálogo está cubierto por la query)
      - Jaccard penaliza si la query no tiene nada que ver
    Esto permite que descripciones largas con vocabulario natural encuentren
    ítems del catálogo aunque usen términos distintos (XSL → plantilla, etc.)
    """
    import unicodedata, re as _re

    STOP = {"de","la","el","en","y","a","con","por","para","del","los","las",
            "un","una","se","su","que","es","al","lo","desde","hasta","debe",
            "decir","ser","por","sin","mas","pero","como","este","esta","esos"}

    def _norm(s):
        s = unicodedata.normalize("NFD", str(s))
        s = "".join(c for c in s if unicodedata.category(c) != "Mn")
        return _re.sub(r"[^a-z0-9\s]", " ", s.lower())

    def tokenize(s):
        return set(_norm(s).split()) - STOP

    raw_q = tokenize(texto)
    if not raw_q:
        return []
    qtoks = _expandir_tokens(raw_q)   # query expandida con sinónimos

    seen: set = set()
    resultados = []
    for _, row in df_roles.iterrows():
        key = (row["paquete"], row["escenario"])
        if key in seen:
            continue
        seen.add(key)
        etiqueta = f"{row['paquete']} · {row['escenario']}"
        # También expandir tokens del catálogo (para mayor cobertura)
        cat_raw  = tokenize(etiqueta)
        ktoks    = _expandir_tokens(cat_raw)
        if not ktoks:
            continue

        inter = qtoks & ktoks
        if not inter:
            continue

        # Recall: qué % de los tokens del catálogo están en la query expandida
        recall   = len(inter) / len(cat_raw) if cat_raw else 0.0
        # Jaccard: penaliza si la query tiene vocabulario muy distinto
        jaccard  = len(inter) / len(qtoks | ktoks)
        # Score final: recall pesa más (permite queries largas)
        score    = 0.70 * recall + 0.30 * jaccard

        if score > 0.08:
            # Mostrar qué tokens originales (sin expandir) generaron el match
            inter_display = sorted((raw_q | cat_raw) & inter)
            resultados.append({
                "opt":   etiqueta,
                "score": score,
                "inter": inter_display or sorted(inter)[:4],
            })

    resultados.sort(key=lambda x: x["score"], reverse=True)
    return resultados[:top_n]


# ══════════════════════════════════════════════════════
# ESTADO DE SESIÓN
# ══════════════════════════════════════════════════════
for _k, _v in {
    "resultado_proyecto": None,
    "_proy_propuestos":   None,
    "_proy_ia_impl":      None,
    "_proy_ia_desq":      None,
}.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

df_roles = _load_roles()

# ══════════════════════════════════════════════════════
# TÍTULO
# ══════════════════════════════════════════════════════
st.title("Estimador de Esfuerzo PS · Sovos")

tab_proy, tab_historial, tab_catalogo = st.tabs([
    "📋 Estimador de Proyecto",
    "📊 Historial y Precisión",
    "📚 Catálogo",
])


# ══════════════════════════════════════════════════════
# TAB 1 — ESTIMADOR DE PROYECTO
# ══════════════════════════════════════════════════════
with tab_proy:

    # ── Fila de encabezado: ticket + integración ──────
    hdr1, hdr2 = st.columns([3, 1])
    with hdr1:
        nombre_proy = st.text_input(
            "Proyecto / Ticket", placeholder="Ej: PSTC-1234 · Cliente ABC", key="nombre_proy"
        )
    with hdr2:
        integ_proy = st.selectbox(
            "Integración del cliente",
            ["Full ASP", "Mixto", "On Premise", "Estandar"],
            key="integ_proy",
            help="Full ASP = menor TC, On Premise = mayor TC",
        )

    # ── Descripción del alcance ────────────────────────
    descr_proy = st.text_area(
        "Describe el alcance del proyecto",
        height=140,
        placeholder=(
            "Ej: Implementar PPL con plantillas personalizadas, certificado digital, "
            "parametría fiscal y soporte post go-live..."
        ),
        key="descr_proy",
    )

    # ── Opciones avanzadas (colapsadas por defecto) ────
    with st.expander("⚙️ Opciones avanzadas"):
        cx_extra = st.select_slider(
            "Complejidad", options=["baja", "media", "alta"], value="media", key="cx_proy",
            help="Baja = −15% horas · Alta = +20% horas",
        )

    metodo_proy = "faiss+xgb+catalog"  # siempre el método más preciso

    # Botón de búsqueda (primer paso)
    _, col_btn = st.columns([4, 1])
    with col_btn:
        btn_buscar = st.button(
            "🔍 Buscar componentes", type="primary",
            use_container_width=True, key="btn_buscar",
            disabled=not descr_proy.strip(),
        )

    # ── Paso 1: búsqueda catálogo + tickets históricos (ambos índices) ─
    if btn_buscar and descr_proy.strip():
        with st.spinner("🔍 Buscando en catálogo y tickets históricos (Implementación + Desarrollo)..."):
            cat_hits   = _buscar_catalogo_libre(descr_proy.strip(), df_roles, top_n=5)
            ia_impl    = None
            ia_desq    = None
            try:
                backend = _lazy_backend()
                if backend.get("ok"):
                    ia_impl = _estimar_texto(
                        descr_proy.strip(), "implementacion",
                        "faiss+xgb+catalog", cx_extra, backend
                    )
                    ia_desq = _estimar_texto(
                        descr_proy.strip(), "desarrollo",
                        "faiss+xgb+catalog", cx_extra, backend
                    )
            except Exception:
                pass
        propuestos = [h for h in cat_hits if h["score"] >= 0.50]
        st.session_state["_proy_propuestos"] = propuestos
        st.session_state["_proy_ia_impl"]    = ia_impl
        st.session_state["_proy_ia_desq"]    = ia_desq
        st.session_state["_proy_descr"]      = descr_proy.strip()
        st.session_state["_proy_nombre"]     = nombre_proy
        st.session_state["_proy_integ"]      = integ_proy
        st.session_state["_proy_cx"]         = cx_extra
        st.session_state.resultado_proyecto  = None

    # ── Paso 2: confirmación de componentes ──────────
    propuestos = st.session_state.get("_proy_propuestos")
    if propuestos is not None:
        texto_est  = st.session_state.get("_proy_descr", descr_proy.strip())
        integ_conf = st.session_state.get("_proy_integ", integ_proy)
        cx_conf    = st.session_state.get("_proy_cx", cx_extra)
        ia_impl    = st.session_state.get("_proy_ia_impl")
        ia_desq    = st.session_state.get("_proy_ia_desq")
        # compat con sesiones antiguas que usaban _proy_ia_result
        ia_result  = ia_impl or st.session_state.get("_proy_ia_result")

        st.divider()
        col_cat, col_hist = st.columns([3, 2])

        # ── Columna izquierda: catálogo ──────────────
        with col_cat:
            if propuestos:
                st.markdown(f"**📦 {len(propuestos)} componente(s) del catálogo** — desmarca los que no aplican:")
            else:
                st.info("No encontré componentes del catálogo. La IA estimará el proyecto completo.")
            seleccionados = []
            for hi, h in enumerate(propuestos):
                hrs = None
                if " · " in h["opt"]:
                    paq_h, esc_h = h["opt"].split(" · ", 1)
                    sub_h = df_roles[(df_roles["paquete"] == paq_h) & (df_roles["escenario"] == esc_h)]
                    m_h = sub_h[sub_h["integracion"] == integ_conf]
                    if m_h.empty: m_h = sub_h[sub_h["integracion"] == "Estandar"]
                    if m_h.empty and not sub_h.empty: m_h = sub_h.iloc[:1]
                    if not m_h.empty:
                        r_h = m_h.iloc[0]
                        hrs = {"TC": int(round(float(r_h["tc"]))),
                               "SC": int(round(float(r_h["sc"]))),
                               "PM": int(round(float(r_h["pm"])))}
                conf_icon = "🟢" if h["score"] >= 0.65 else "🟡"
                hrs_txt = (f"TC={hrs['TC']} SC={hrs['SC']} PM={hrs['PM']} "
                           f"**Total={hrs['TC']+hrs['SC']+hrs['PM']}h**") if hrs else "horas N/D"
                checked = st.checkbox(
                    f"{conf_icon} **{h['opt']}** — {hrs_txt}",
                    value=True, key=f"chk_prop_{hi}",
                )
                if checked:
                    seleccionados.append(h["opt"])

        # ── Columna derecha: tickets históricos (impl + desq) ──
        with col_hist:
            st.markdown("**🗃️ Tickets históricos similares**")

            def _mostrar_tickets(res, etiqueta, color):
                if not res or not res.get("top"):
                    return
                sim_top = res["top"][0]["sim"] if res["top"] else 0
                if sim_top < 0.55:
                    return
                st.markdown(
                    f"<span style='background:{color};color:white;font-size:0.72rem;"
                    f"padding:2px 8px;border-radius:10px'>{etiqueta}</span>",
                    unsafe_allow_html=True,
                )
                for tick in res["top"][:2]:
                    sim_pct  = int(tick["sim"] * 100)
                    sim_icon = "🟢" if tick["sim"] >= 0.75 else ("🟡" if tick["sim"] >= 0.55 else "🔴")
                    with st.expander(
                        f"{sim_icon} **{tick['ticket']}** — {int(tick['hours'])}h · {sim_pct}%"
                    ):
                        st.caption(tick["text"][:280] + ("..." if len(tick["text"]) > 280 else ""))
                h_est = math.ceil(res["horas"])
                st.markdown(
                    f"<div style='background:#f0f5fa;border-left:3px solid {color};"
                    f"padding:6px 10px;border-radius:6px;margin:4px 0 10px 0'>"
                    f"<span style='font-size:0.75rem;color:#5c6680'>IA {etiqueta}</span> "
                    f"<b>{h_est}h</b> "
                    f"<span style='color:#64748b;font-size:0.8rem'>"
                    f"({res.get('rango_min',0)}–{res.get('rango_max',0)}h)</span></div>",
                    unsafe_allow_html=True,
                )

            hay_tickets = False
            if ia_impl and ia_impl.get("top") and ia_impl["top"][0]["sim"] >= 0.55:
                _mostrar_tickets(ia_impl, "Implementación (PSTC)", "#0a0e33")
                hay_tickets = True
            if ia_desq and ia_desq.get("top") and ia_desq["top"][0]["sim"] >= 0.55:
                _mostrar_tickets(ia_desq, "Desarrollo (CESQ)", "#00767a")
                hay_tickets = True
            if not hay_tickets:
                st.caption("Sin tickets similares encontrados en el historial.")

        _, col_calc = st.columns([4, 1])
        with col_calc:
            btn_calcular = st.button(
                "🚀 Calcular estimación", type="primary",
                use_container_width=True, key="btn_calcular",
            )

        # ── Paso 3: cálculo final ─────────────────────
        if btn_calcular:
            filas_res: list[dict] = []

            # Componentes confirmados del catálogo
            for opt in seleccionados:
                paq, esc = opt.split(" · ", 1)
                sub = df_roles[(df_roles["paquete"] == paq) & (df_roles["escenario"] == esc)]
                m = sub[sub["integracion"] == integ_conf]
                if m.empty: m = sub[sub["integracion"] == "Estandar"]
                if m.empty and not sub.empty: m = sub.iloc[:1]
                if not m.empty:
                    row = m.iloc[0]
                    _tc = int(round(float(row["tc"])))
                    _sc = int(round(float(row["sc"])))
                    _pm = int(round(float(row["pm"])))
                    filas_res.append({
                        "Origen":          "📦 Catálogo",
                        "Paquete / Tarea": f"{paq} · {esc}",
                        "Integración":     row["integracion"],
                        "TC (h)":          _tc,
                        "SC (h)":          _sc,
                        "PM (h)":          _pm,
                        "Total (h)":       _tc + _sc + _pm,
                        "Referencia IA":   "—",
                    })

            # IA — agrega fila por cada índice con match relevante
            def _fila_ia(r, tipo_label, integ_label):
                if not r or not r.get("top"):
                    return None
                top1 = r["top"][0]
                if top1["sim"] < 0.55:
                    return None
                horas = max(1, math.ceil(r["horas"]))
                sim_v = top1["sim"]
                conf  = "🟢 Alta" if sim_v >= 0.80 else ("🟡 Media" if sim_v >= 0.60 else "🔴 Baja")
                ref   = f"{conf} · {r['rango_min']}-{r['rango_max']}h · Ref:{top1['ticket']}"
                return {
                    "Origen":          f"🤖 IA ({tipo_label})",
                    "Paquete / Tarea": texto_est[:75] + ("..." if len(texto_est) > 75 else ""),
                    "Integración":     integ_label,
                    "TC (h)":          int(round(horas * 0.75)),
                    "SC (h)":          int(round(horas * 0.20)),
                    "PM (h)":          int(round(horas * 0.05)),
                    "Total (h)":       horas,
                    "Referencia IA":   ref,
                }

            for _r, _tl, _il in [
                (ia_impl, "Impl", "Implementación"),
                (ia_desq, "Desq", "Desarrollo"),
            ]:
                fila = _fila_ia(_r, _tl, _il)
                if fila:
                    filas_res.append(fila)

            if not filas_res:
                st.warning("⚠️ No se encontraron resultados.")
            else:
                filas_res, alertas_dup = _deduplicar_filas(filas_res)
                if alertas_dup:
                    with st.expander(
                        f"🔄 {len(alertas_dup)} ajuste(s) para evitar duplicados", expanded=True
                    ):
                        st.caption("La IA ajustó horas solapadas entre catálogo e IA.")
                        for a in alertas_dup:
                            st.markdown(f"• {a}")
                st.session_state.resultado_proyecto = {
                    "filas":       filas_res,
                    "nombre":      st.session_state.get("_proy_nombre", nombre_proy),
                    "integracion": integ_conf,
                }
                st.session_state["_proy_propuestos"] = None

    # ── Resultados del proyecto ──────────────────────
    if st.session_state.resultado_proyecto:
        rd = st.session_state.resultado_proyecto
        filas = rd["filas"]

        st.divider()
        titulo_res = f"Estimación: {rd['nombre']}" if rd["nombre"] else "Estimación del Proyecto"
        st.markdown(f"""
        <div class="proy-banner">
          <p class="b-title">📊 {titulo_res}</p>
          <p class="b-sub">Integración base: <strong>{rd['integracion']}</strong></p>
        </div>
        """, unsafe_allow_html=True)

        df_res = pd.DataFrame(filas)

        edited = st.data_editor(
            df_res,
            column_config={
                "TC (h)": st.column_config.NumberColumn(
                    "👷 TC (h) ✏️", min_value=0, max_value=2000, step=1,
                    help="Editable: ajusta horas de TC directamente",
                ),
                "SC (h)": st.column_config.NumberColumn(
                    "💼 SC (h) ✏️", min_value=0, max_value=2000, step=1,
                    help="Editable: ajusta horas de SC directamente",
                ),
                "PM (h)": st.column_config.NumberColumn(
                    "📋 PM (h) ✏️", min_value=0, max_value=2000, step=1,
                    help="Editable: ajusta horas de PM directamente",
                ),
                "Total (h)":     st.column_config.NumberColumn("🔢 Total", format="%d"),
                "Origen":        st.column_config.TextColumn("Origen",      width="small"),
                "Referencia IA": st.column_config.TextColumn("Ref. IA",     width="medium"),
                "Integración":   st.column_config.TextColumn("Integración", width="small"),
            },
            disabled=["Origen", "Paquete / Tarea", "Integración", "Total (h)", "Referencia IA"],
            hide_index=True,
            use_container_width=True,
        )

        # Columna Total calculada desde TC+SC+PM editados
        edited["Total (h)"] = edited["TC (h)"] + edited["SC (h)"] + edited["PM (h)"]

        # ── Totales desde tabla editada
        total_tc    = edited["TC (h)"].sum()
        total_sc    = edited["SC (h)"].sum()
        total_pm    = edited["PM (h)"].sum()
        total_bruto = total_tc + total_sc + total_pm

        # ── Ajuste inteligente (overhead PM compartido) sobre valores editados
        ajuste = _ajuste_proyecto(edited.to_dict("records"))

        st.markdown("---")
        st.subheader("⏱️ Totales por Rol")

        _r = lambda v: int(round(v))
        # Valores a mostrar en las cards (ajustados si hay ahorro PM)
        _tc_d  = _r(ajuste["tc"])
        _sc_d  = _r(ajuste["sc"])
        _pm_d  = _r(ajuste["pm"])
        _tot_d = _r(ajuste["total_ajustado"])

        st.markdown(f"""
        <div class="rol-cards">
          <div class="rol-card card-tc">
            <div class="lbl">👷 TC</div>
            <div class="val">{_tc_d}</div>
            <div class="unit">horas</div>
          </div>
          <div class="rol-card card-sc">
            <div class="lbl">💼 SC</div>
            <div class="val">{_sc_d}</div>
            <div class="unit">horas</div>
          </div>
          <div class="rol-card card-pm">
            <div class="lbl">📋 PM</div>
            <div class="val">{_pm_d}</div>
            <div class="unit">horas</div>
          </div>
          <div class="rol-card card-tot">
            <div class="lbl">🔢 Total</div>
            <div class="val">{_tot_d}</div>
            <div class="unit">horas</div>
          </div>
        </div>
        """, unsafe_allow_html=True)
        if ajuste["ahorro_pm"] > 0:
            st.caption(f"💡 {ajuste['nota']} — ahorro PM: **{_r(ajuste['ahorro_pm'])}h**")

        # ── Análisis Gemini
        st.divider()
        st.subheader("🧠 Análisis IA del Proyecto")
        if os.environ.get("GEMINI_API_KEY"):
            with st.spinner("Analizando el proyecto con IA..."):
                analisis = _gemini_proyecto(rd["nombre"], rd["integracion"], filas, ajuste)
            if analisis:
                st.markdown(analisis)
        else:
            st.caption("_(Configura GEMINI_API_KEY para obtener análisis de solapamientos y riesgos)_")

        # Resumen para copiar
        with st.expander("📋 Copiar resumen"):
            titulo_copia = rd["nombre"] or "Proyecto"
            lines = [
                f"Estimación: {titulo_copia}",
                f"Integración: {rd['integracion']}",
                "",
            ]
            for row_ed in edited.to_dict("records"):
                tc_e = int(round(row_ed["TC (h)"]))
                sc_e = int(round(row_ed["SC (h)"]))
                pm_e = int(round(row_ed["PM (h)"]))
                tot_e = tc_e + sc_e + pm_e
                lines.append(
                    f"• {row_ed['Paquete / Tarea']}: "
                    f"TC={tc_e}h | SC={sc_e}h | PM={pm_e}h | Total={tot_e}h"
                )
            lines += [
                "",
                f"SUMA DIRECTA   → TC:{_r(total_tc)}h | SC:{_r(total_sc)}h | PM:{_r(total_pm)}h | Total:{_r(total_bruto)}h",
                f"AJUSTADO       → TC:{_r(ajuste['tc'])}h | SC:{_r(ajuste['sc'])}h | PM:{_r(ajuste['pm'])}h | Total:{_r(ajuste['total_ajustado'])}h",
            ]
            st.code("\n".join(lines), language=None)

        # Guardar estimación
        st.divider()
        nota_save = st.text_input("Nota / detalle (opcional)", placeholder="Ej: cliente nuevo, solo emisión, go-live en abril...", key="nota_save_proy")
        if st.button("💾 Guardar Estimación", type="primary", key="btn_save_proy"):
            _save_proyecto({
                "timestamp":   time.strftime("%Y-%m-%d %H:%M:%S"),
                "nombre":      rd["nombre"],
                "nota":        nota_save,
                "integracion": rd["integracion"],
                "componentes": [
                    {
                        "origen": row["Origen"],
                        "tarea":  row["Paquete / Tarea"],
                        "TC":     int(round(row["TC (h)"])),
                        "SC":     int(round(row["SC (h)"])),
                        "PM":     int(round(row["PM (h)"])),
                        "total":  int(round(row["TC (h)"])) + int(round(row["SC (h)"])) + int(round(row["PM (h)"])),
                    }
                    for row in edited.to_dict("records")
                ],
                "totales": {
                    "TC":             _r(total_tc),
                    "SC":             _r(total_sc),
                    "PM":             _r(total_pm),
                    "total_bruto":    _r(total_bruto),
                    "total_ajustado": _r(ajuste["total_ajustado"]),
                },
            })
            st.success("✅ Proyecto guardado con desglose TC/SC/PM.")


# ══════════════════════════════════════════════════════
# TAB 3 — HISTORIAL Y PRECISIÓN
# ══════════════════════════════════════════════════════
with tab_historial:
    csv_path = os.path.join(os.path.dirname(__file__), "data", "estimaciones_nuevas.csv")

    if os.path.exists(csv_path):
        try:
            df_fb = pd.read_csv(csv_path)
            if "horas_reales" in df_fb.columns and "horas_estimadas" in df_fb.columns:
                df_v = df_fb[(df_fb["horas_reales"] > 0) & (df_fb["horas_estimadas"] > 0)].copy()
                if len(df_v) >= 3:
                    st.subheader("📊 Precisión del Modelo")
                    df_v["error_abs"] = (df_v["horas_reales"] - df_v["horas_estimadas"]).abs()
                    df_v["error_pct"] = df_v["error_abs"] / df_v["horas_reales"] * 100
                    mae  = df_v["error_abs"].mean()
                    mape = df_v["error_pct"].mean()
                    bias = (df_v["horas_estimadas"] - df_v["horas_reales"]).mean()
                    n    = len(df_v)

                    h1, h2, h3, h4 = st.columns(4)
                    h1.metric("Estimaciones con feedback", n)
                    h2.metric("Error promedio (MAE)",       f"{mae:.1f}h")
                    h3.metric("Error % promedio",            f"{mape:.0f}%")
                    tendencia = (
                        f"+{bias:.1f}h (sobreestima)" if bias > 0.5
                        else (f"{bias:.1f}h (subestima)" if bias < -0.5 else "Calibrado ✓")
                    )
                    h4.metric("Tendencia", tendencia)

                    st.subheader("Últimas 20 estimaciones")
                    st.dataframe(df_fb.tail(20), use_container_width=True, hide_index=True)
                else:
                    st.info("Se necesitan al menos 3 estimaciones con horas reales para mostrar métricas.")
        except Exception:
            pass
    else:
        st.info("Aún no hay estimaciones guardadas.")

    st.divider()

    # ── Proyectos guardados
    st.subheader("📁 Proyectos Guardados")
    proyectos = _load_proyectos()
    if not proyectos:
        st.info("Aún no hay proyectos guardados. Estima un proyecto y guárdalo desde el Tab 1.")
    else:
        for real_idx in range(len(proyectos) - 1, -1, -1):
            p = proyectos[real_idx]
            estado = p.get("estado", "estimado")
            estado_icon = {"estimado": "📋", "corregido": "✅", "descartado": "❌"}.get(estado, "📋")
            label = (
                f"{estado_icon} {p['nombre'] or 'Sin nombre'} · "
                f"{p['timestamp'][:10]} · {p['integracion']} · "
                f"Total: {p['totales']['total_ajustado']}h"
            )
            if p.get("horas_reales"):
                label += f" → Real: {p['horas_reales']}h"

            with st.expander(label):
                # Estado badge
                badge_class = f"badge badge-{estado}"
                badge_label = {"estimado": "📋 Estimado", "corregido": "✅ Corregido", "descartado": "❌ Descartado"}.get(estado, estado)
                st.markdown(f'<span class="{badge_class}">{badge_label}</span>', unsafe_allow_html=True)
                if p.get("nota"):
                    st.caption(f"📝 {p['nota']}")

                df_comp = pd.DataFrame(p["componentes"])

                # Tabla editable por rol
                edited_comp = st.data_editor(
                    df_comp,
                    column_config={
                        "TC":    st.column_config.NumberColumn("👷 TC (h)", min_value=0, step=1),
                        "SC":    st.column_config.NumberColumn("💼 SC (h)", min_value=0, step=1),
                        "PM":    st.column_config.NumberColumn("📋 PM (h)", min_value=0, step=1),
                        "total": st.column_config.NumberColumn("🔢 Total",  format="%d"),
                    },
                    disabled=["origen", "tarea", "total"],
                    hide_index=True,
                    use_container_width=True,
                    key=f"comp_editor_{real_idx}",
                )

                # Totales recalculados desde la tabla editada
                tc_sum = int(edited_comp["TC"].sum())
                sc_sum = int(edited_comp["SC"].sum())
                pm_sum = int(edited_comp["PM"].sum())
                tot_sum = tc_sum + sc_sum + pm_sum
                st.markdown(f"""
                <div class="rol-cards" style="margin:10px 0 6px;">
                  <div class="rol-card card-tc" style="padding:10px 12px;">
                    <div class="lbl">👷 TC</div>
                    <div class="val" style="font-size:1.5rem;">{tc_sum}</div>
                    <div class="unit">h</div>
                  </div>
                  <div class="rol-card card-sc" style="padding:10px 12px;">
                    <div class="lbl">💼 SC</div>
                    <div class="val" style="font-size:1.5rem;">{sc_sum}</div>
                    <div class="unit">h</div>
                  </div>
                  <div class="rol-card card-pm" style="padding:10px 12px;">
                    <div class="lbl">📋 PM</div>
                    <div class="val" style="font-size:1.5rem;">{pm_sum}</div>
                    <div class="unit">h</div>
                  </div>
                  <div class="rol-card card-tot" style="padding:10px 12px;">
                    <div class="lbl">🔢 Total</div>
                    <div class="val" style="font-size:1.5rem;">{tot_sum}</div>
                    <div class="unit">h</div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

                # ── Corregir estimación
                st.markdown("---")
                col_hr, col_nota, col_acciones = st.columns([2, 3, 2])
                with col_hr:
                    hr_real = st.number_input(
                        "Horas reales totales", min_value=0, step=1,
                        value=int(p.get("horas_reales", 0)),
                        key=f"hr_real_{real_idx}",
                    )
                with col_nota:
                    corr_nota = st.text_input(
                        "Nota de corrección",
                        value=p.get("correccion_nota", ""),
                        placeholder="Ej: faltaron horas de testing...",
                        key=f"corr_nota_{real_idx}",
                    )
                with col_acciones:
                    st.markdown("<br>", unsafe_allow_html=True)
                    bc1, bc2 = st.columns(2)
                    with bc1:
                        if st.button("✅ Corregir", key=f"corr_{real_idx}"):
                            comp_corregidos = edited_comp.to_dict("records")
                            for row in comp_corregidos:
                                row["total"] = int(row["TC"]) + int(row["SC"]) + int(row["PM"])
                            _update_proyecto(real_idx, {
                                "estado":          "corregido",
                                "horas_reales":    hr_real,
                                "correccion_nota": corr_nota,
                                "corregido_en":    time.strftime("%Y-%m-%d %H:%M:%S"),
                                "componentes":     comp_corregidos,
                                "totales": {
                                    "TC":             tc_sum,
                                    "SC":             sc_sum,
                                    "PM":             pm_sum,
                                    "total_bruto":    tot_sum,
                                    "total_ajustado": tot_sum,
                                },
                            })
                            st.rerun()
                    with bc2:
                        if st.button("🗑️ Eliminar", key=f"del_{real_idx}"):
                            _delete_proyecto(real_idx)
                            st.rerun()

        # Descarga JSON
        st.divider()
        st.subheader("📥 Descargar Datos")
        st.caption("Descarga regularmente — Streamlit puede borrar archivos al reiniciarse.")
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            if os.path.exists(_PROYECTOS_FILE):
                with open(_PROYECTOS_FILE, "rb") as f:
                    st.download_button(
                        "⬇️ Descargar proyectos_guardados.json",
                        data=f,
                        file_name="proyectos_guardados.json",
                        mime="application/json",
                    )
        with col_d2:
            if os.path.exists(csv_path):
                with open(csv_path, "rb") as f:
                    st.download_button(
                        "⬇️ Descargar estimaciones_nuevas.csv",
                        data=f,
                        file_name="estimaciones_nuevas.csv",
                        mime="text/csv",
                    )


# ══════════════════════════════════════════════════════
# TAB 4 — CATÁLOGO
# ══════════════════════════════════════════════════════
with tab_catalogo:
    st.subheader("📚 Editor de Catálogos")
    st.caption("Edita directamente los datos que usa el estimador. Los cambios se guardan en la base de datos.")

    _TABLAS_CATALOGO = {
        "Roles por Paquete":       ("catalogo_roles",  "Horas TC/SC/PM por paquete, escenario e integración"),
        "Tareas PSTC":             ("catalogo_pstc",   "Tareas unitarias de implementación con horas y categoría"),
        "Tareas CESQ":             ("catalogo_cesq",   "Tareas unitarias de desarrollo con horas y categoría"),
        "Histórico PSTC (FAISS)":  ("historico_pstc",  "Tickets históricos usados para entrenar el modelo de implementación"),
        "Histórico CESQ (FAISS)":  ("historico_cesq",  "Tickets históricos usados para entrenar el modelo de desarrollo"),
    }

    tabla_sel = st.selectbox(
        "Selecciona un catálogo",
        options=list(_TABLAS_CATALOGO.keys()),
        key="cat_sel",
    )

    tabla_db, tabla_desc = _TABLAS_CATALOGO[tabla_sel]
    st.caption(tabla_desc)

    df_cat = get_table(tabla_db)

    if df_cat.empty:
        st.warning(f"La tabla '{tabla_db}' está vacía o no existe.")
    else:
        st.caption(f"**{len(df_cat)} registros** · Edita las celdas y presiona Guardar.")

        edited_cat = st.data_editor(
            df_cat,
            num_rows="dynamic",
            hide_index=True,
            use_container_width=True,
            key=f"editor_{tabla_db}",
        )

        col_s, col_r = st.columns([1, 5])
        with col_s:
            if st.button("💾 Guardar cambios", type="primary", key=f"save_{tabla_db}"):
                save_table(tabla_db, edited_cat)
                st.cache_data.clear()
                st.success(f"✅ '{tabla_sel}' guardado — {len(edited_cat)} registros.")
        with col_r:
            st.caption("⚠️ Los cambios en Histórico PSTC/CESQ requieren re-entrenar el índice FAISS para tener efecto en las estimaciones IA.")
