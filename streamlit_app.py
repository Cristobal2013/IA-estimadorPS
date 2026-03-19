import streamlit as st
import pandas as pd
import os
import time
import math

from app import _lazy_backend, _resolve_tag, _to_float, append_row_safe

st.set_page_config(
    page_title="Estimador PS · Sovos",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════
# DATOS
# ══════════════════════════════════════════════════════
@st.cache_data
def _load_roles():
    p = os.path.join(os.path.dirname(__file__), "data", "catalogo_roles.csv")
    if not os.path.exists(p):
        return pd.DataFrame()
    return pd.read_csv(p)


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

    alpha = 0.8
    bias_map = {"baja": -0.1, "media": 0.0, "alta": +0.1}
    alpha_eff = max(0.3, min(0.98, alpha + bias_map.get(complexity, 0.0)))

    if metodo == "faiss":
        final = horas_faiss
    elif metodo == "catalog":
        final = horas_catalog
    elif metodo == "faiss+catalog":
        final = alpha_eff * horas_faiss + (1.0 - alpha_eff) * horas_catalog
    else:  # faiss+xgb+catalog
        if horas_xgb > 0:
            final = 0.45 * horas_faiss + 0.40 * horas_xgb + 0.15 * horas_catalog
        else:
            final = alpha_eff * horas_faiss + (1.0 - alpha_eff) * horas_catalog

    hs = [item["hours"] for item in top_tickets] if top_tickets else []
    rango_min = max(0, math.ceil(final * 0.75)) if len(hs) >= 2 else math.ceil(final)
    rango_max = math.ceil(final * 1.35) if len(hs) >= 2 else math.ceil(final)

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
# ESTADO DE SESIÓN
# ══════════════════════════════════════════════════════
for _k, _v in {
    "resultado_libre": None,
    "resultado_proyecto": None,
    "tareas_extra": [{"texto": "", "tipo": "Implementación"}],
}.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

df_roles = _load_roles()

# ══════════════════════════════════════════════════════
# TÍTULO
# ══════════════════════════════════════════════════════
st.title("🤖 Estimador de Esfuerzo PS · Sovos")

tab_proy, tab_libre, tab_historial = st.tabs([
    "📋 Estimador de Proyecto",
    "📝 Consulta Rápida",
    "📊 Historial y Precisión",
])


# ══════════════════════════════════════════════════════
# TAB 1 — ESTIMADOR DE PROYECTO (formulario estructurado)
# ══════════════════════════════════════════════════════
with tab_proy:

    # ── Encabezado ───────────────────────────────────
    nombre_proy = st.text_input(
        "Proyecto / Ticket", placeholder="Ej: PSTC-1234 · Cliente ABC", key="nombre_proy"
    )

    st.divider()

    col_sel, col_extra = st.columns([3, 2])

    # ── SELECCIÓN FLAT — un solo multiselect con todos los componentes
    with col_sel:
        st.subheader("📦 Componentes del Proyecto")
        st.caption("Busca por nombre, tecnología o normativa. Puedes combinar libremente.")

        # Controles de configuración compactos
        c1, c2, c3 = st.columns(3)
        with c1:
            integ_proy = st.selectbox("Integración", ["Full ASP", "Mixto", "On Premise", "Estandar"], key="integ_proy")
        with c2:
            cx_extra = st.select_slider("Complejidad", options=["baja", "media", "alta"], value="media", key="cx_proy")
        with c3:
            metodo_proy = st.selectbox("Método IA", ["faiss+xgb+catalog", "faiss+catalog", "faiss", "catalog"], key="met_proy")

        if df_roles.empty:
            st.warning("No se encontró catalogo_roles.csv en data/")
            componentes_sel = []
        else:
            # Construir opciones: "PAQUETE · ESCENARIO" únicas
            opciones: list[str] = []
            seen_opts: set = set()
            for _, row in df_roles.iterrows():
                opt = f"{row['paquete']} · {row['escenario']}"
                if opt not in seen_opts:
                    seen_opts.add(opt)
                    opciones.append(opt)

            componentes_sel = st.multiselect(
                "Selecciona los componentes",
                options=opciones,
                placeholder="Busca: Plantillas, PPL, Coapi, RES154 PPL...",
                key="componentes_sel",
            )

            # Preview en tiempo real de lo seleccionado
            if componentes_sel:
                prev = []
                for opt in componentes_sel:
                    paq, esc = opt.split(" · ", 1)
                    sub = df_roles[(df_roles["paquete"] == paq) & (df_roles["escenario"] == esc)]
                    m = sub[sub["integracion"] == integ_proy]
                    if m.empty:
                        m = sub[sub["integracion"] == "Estandar"]
                    if m.empty and not sub.empty:
                        m = sub.iloc[:1]
                    if not m.empty:
                        r = m.iloc[0]
                        prev.append({
                            "Componente": opt,
                            "Integración": r["integracion"],
                            "TC": r["tc"], "SC": r["sc"], "PM": r["pm"], "Total": r["total"],
                        })
                if prev:
                    st.dataframe(
                        pd.DataFrame(prev),
                        hide_index=True,
                        use_container_width=True,
                        column_config={
                            "TC": st.column_config.NumberColumn("TC (h)", format="%.1f"),
                            "SC": st.column_config.NumberColumn("SC (h)", format="%.1f"),
                            "PM": st.column_config.NumberColumn("PM (h)", format="%.1f"),
                            "Total": st.column_config.NumberColumn("Total (h)", format="%.1f"),
                        },
                    )

    # ── Tareas adicionales (IA)
    with col_extra:
        st.subheader("✍️ Tareas Adicionales (IA)")
        st.caption("Para actividades no en el catálogo. La IA estima con tickets históricos.")

        tareas_ed = []
        for i, t in enumerate(st.session_state.tareas_extra):
            if isinstance(t, dict):
                t_texto = t.get("texto", "")
                t_tipo  = t.get("tipo", "Implementación")
            else:
                t_texto, t_tipo = t, "Implementación"

            ct, ctipo, cd = st.columns([4, 2, 1])
            with ct:
                val = st.text_input(
                    f"t{i}", value=t_texto, key=f"textra_{i}",
                    placeholder="Ej: Soporte post go-live...",
                    label_visibility="collapsed",
                )
            with ctipo:
                tipo_t = st.selectbox(
                    "ti", ["Implementación", "Desarrollo"],
                    index=0 if t_tipo == "Implementación" else 1,
                    key=f"ttipo_{i}", label_visibility="collapsed",
                )
            with cd:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("✕", key=f"del_extra_{i}") and len(st.session_state.tareas_extra) > 1:
                    st.session_state.tareas_extra.pop(i)
                    st.rerun()
            tareas_ed.append({"texto": val, "tipo": tipo_t})
        st.session_state.tareas_extra = tareas_ed

        if st.button("➕ Agregar tarea", key="add_extra"):
            st.session_state.tareas_extra.append({"texto": "", "tipo": "Implementación"})
            st.rerun()

    # ── Botón estimar ────────────────────────────────
    st.divider()
    _, col_btn = st.columns([4, 1])
    with col_btn:
        btn_proy = st.button("🚀 Estimar Proyecto", type="primary", use_container_width=True, key="btn_proy")

    # ── Cálculo ──────────────────────────────────────
    if btn_proy:
        filas_res: list[dict] = []

        # 1) Componentes seleccionados del catálogo
        for opt in (componentes_sel if not df_roles.empty else []):
            paq, esc = opt.split(" · ", 1)
            sub = df_roles[(df_roles["paquete"] == paq) & (df_roles["escenario"] == esc)]
            m = sub[sub["integracion"] == integ_proy]
            if m.empty:
                m = sub[sub["integracion"] == "Estandar"]
            if m.empty and not sub.empty:
                m = sub.iloc[:1]
            if not m.empty:
                row = m.iloc[0]
                filas_res.append({
                    "Origen":          "📦 Catálogo",
                    "Paquete / Tarea": f"{paq} · {esc}",
                    "Integración":     row["integracion"],
                    "TC (h)":          float(row["tc"]),
                    "SC (h)":          float(row["sc"]),
                    "PM (h)":          float(row["pm"]),
                    "Referencia IA":   "—",
                })

        # 2) Tareas adicionales → IA
        tareas_validas = [
            t if isinstance(t, dict) else {"texto": t, "tipo": "Implementación"}
            for t in st.session_state.tareas_extra
            if (t.get("texto", "") if isinstance(t, dict) else t).strip()
        ]
        if tareas_validas:
            with st.spinner(f"🧠 Estimando {len(tareas_validas)} tarea(s) adicional(es)..."):
                try:
                    backend = _lazy_backend()
                    if not backend.get("ok"):
                        st.error(f"Error cargando motor IA: {backend.get('err')}")
                    else:
                        for tarea_item in tareas_validas:
                            tarea  = tarea_item["texto"].strip()
                            tipo_t = tarea_item.get("tipo", "Implementación")

                            # ── Detectar ajuste manual por rol (ej: "5h SC reuniones")
                            ajuste_manual = _parse_ajuste_rol(tarea)
                            if ajuste_manual:
                                filas_res.append({
                                    "Origen":          f"✏️ Manual ({ajuste_manual['rol']})",
                                    "Paquete / Tarea": tarea,
                                    "Integración":     tipo_t,
                                    "TC (h)":          ajuste_manual["tc"],
                                    "SC (h)":          ajuste_manual["sc"],
                                    "PM (h)":          ajuste_manual["pm"],
                                    "Referencia IA":   f"Ajuste directo {ajuste_manual['rol']}",
                                })
                                continue

                            # ── Sin patrón de rol → estimar con FAISS/XGB
                            tag_proy = _resolve_tag(tipo_t)
                            r = _estimar_texto(tarea, tag_proy, metodo_proy, cx_extra, backend)
                            horas_ia = max(1, math.ceil(r["horas"]))
                            top1 = r["top"][0] if r["top"] else None
                            ref = f"FAISS:{r['faiss']:.0f}h"
                            if r["xgb"] > 0:
                                ref += f" · XGB:{r['xgb']:.0f}h"
                            if top1:
                                ref += f" · Ref:{top1['ticket']}({top1['sim']:.2f})"

                            # Distribución fija: la dificultad va a TC
                            tc_h = round(horas_ia * 0.75, 1)
                            sc_h = round(horas_ia * 0.20, 1)
                            pm_h = round(horas_ia * 0.05, 1)

                            filas_res.append({
                                "Origen":          f"🤖 IA ({tipo_t[:4]})",
                                "Paquete / Tarea": tarea,
                                "Integración":     tipo_t,
                                "TC (h)":          tc_h,
                                "SC (h)":          sc_h,
                                "PM (h)":          pm_h,
                                "Referencia IA":   ref,
                            })
                except Exception as e:
                    st.error(f"Error IA: {e}")

        if not filas_res:
            st.warning("⚠️ Selecciona al menos un escenario del catálogo o agrega una tarea.")
        else:
            st.session_state.resultado_proyecto = {
                "filas":       filas_res,
                "nombre":      nombre_proy,
                "integracion": integ_proy,
            }

    # ── Resultados del proyecto ──────────────────────
    if st.session_state.resultado_proyecto:
        rd = st.session_state.resultado_proyecto
        filas = rd["filas"]

        st.divider()
        titulo_res = f"📊 Estimación: {rd['nombre']}" if rd["nombre"] else "📊 Estimación del Proyecto"
        st.subheader(titulo_res)
        st.caption(f"Integración base: **{rd['integracion']}**")

        df_res = pd.DataFrame(filas)

        edited = st.data_editor(
            df_res,
            column_config={
                "TC (h)": st.column_config.NumberColumn(
                    "👷 TC (h) ✏️", min_value=0, max_value=2000, step=0.5,
                    help="Editable: ajusta horas de TC directamente",
                ),
                "SC (h)": st.column_config.NumberColumn(
                    "💼 SC (h) ✏️", min_value=0, max_value=2000, step=0.5,
                    help="Editable: ajusta horas de SC directamente",
                ),
                "PM (h)": st.column_config.NumberColumn(
                    "📋 PM (h) ✏️", min_value=0, max_value=2000, step=0.5,
                    help="Editable: ajusta horas de PM directamente",
                ),
                "Origen":        st.column_config.TextColumn("Origen",      width="small"),
                "Referencia IA": st.column_config.TextColumn("Ref. IA",     width="medium"),
                "Integración":   st.column_config.TextColumn("Integración", width="small"),
            },
            disabled=["Origen", "Paquete / Tarea", "Integración", "Referencia IA"],
            hide_index=True,
            use_container_width=True,
        )

        # ── Totales desde tabla editada
        total_tc    = edited["TC (h)"].sum()
        total_sc    = edited["SC (h)"].sum()
        total_pm    = edited["PM (h)"].sum()
        total_bruto = total_tc + total_sc + total_pm

        # ── Ajuste inteligente (overhead PM compartido) sobre valores editados
        ajuste = _ajuste_proyecto(edited.to_dict("records"))

        st.markdown("---")
        st.subheader("⏱️ Totales por Rol")

        df_totales = pd.DataFrame({
            "":          ["Suma directa", "Ajustado (PM compartido)"],
            "👷 TC":     [round(total_tc, 1),        round(ajuste["tc"], 1)],
            "💼 SC":     [round(total_sc, 1),        round(ajuste["sc"], 1)],
            "📋 PM":     [round(total_pm, 1),        round(ajuste["pm"], 1)],
            "🔢 Total":  [round(total_bruto, 1),     round(ajuste["total_ajustado"], 1)],
        })
        st.dataframe(df_totales, hide_index=True, use_container_width=True)

        if ajuste["ahorro_pm"] > 0:
            st.caption(f"💡 {ajuste['nota']} — ahorro: **{ajuste['ahorro_pm']:.1f}h PM**")

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
                tc_e = row_ed["TC (h)"]
                sc_e = row_ed["SC (h)"]
                pm_e = row_ed["PM (h)"]
                tot_e = tc_e + sc_e + pm_e
                lines.append(
                    f"• {row_ed['Paquete / Tarea']}: "
                    f"TC={tc_e:.1f}h | SC={sc_e:.1f}h | PM={pm_e:.1f}h | Total={tot_e:.1f}h"
                )
            lines += [
                "",
                f"SUMA DIRECTA   → TC:{total_tc:.1f}h | SC:{total_sc:.1f}h | PM:{total_pm:.1f}h | Total:{total_bruto:.1f}h",
                f"AJUSTADO       → TC:{ajuste['tc']:.1f}h | SC:{ajuste['sc']:.1f}h | PM:{ajuste['pm']:.1f}h | Total:{ajuste['total_ajustado']:.1f}h",
            ]
            st.code("\n".join(lines), language=None)

        # Guardar estimación
        st.divider()
        with st.form("form_proy_save"):
            st.subheader("💾 Guardar Estimación")
            cp1, cp2 = st.columns(2)
            with cp1:
                hr_real_p = st.number_input("Horas reales (post-proyecto)", min_value=0.0, step=1.0)
            with cp2:
                com_p = st.text_input("Comentarios")
            if st.form_submit_button("Guardar", type="primary"):
                append_row_safe({
                    "timestamp":       time.strftime("%Y-%m-%d %H:%M:%S"),
                    "tipo":            "implementacion",
                    "texto":           rd["nombre"],
                    "horas_estimadas": float(ajuste["total_ajustado"]),
                    "horas_reales":    float(hr_real_p),
                    "diferencia":      round(float(hr_real_p) - ajuste["total_ajustado"], 2),
                    "top_ticket":      "",
                    "top_sim":         0,
                    "metodo":          "catalogo_proyecto",
                    "autor":           "streamlit_proyecto",
                    "comentarios":     com_p,
                })
                st.success("✅ Guardado correctamente.")


# ══════════════════════════════════════════════════════
# TAB 2 — CONSULTA RÁPIDA (texto libre)
# ══════════════════════════════════════════════════════
with tab_libre:
    col_izq, col_der = st.columns([2, 1])
    with col_izq:
        texto_libre = st.text_area(
            "Descripción del Requerimiento",
            height=200,
            placeholder="Pega aquí el correo o la descripción del ticket...",
        )
    with col_der:
        st.subheader("Configuración")
        tipo_l    = st.selectbox("Tipo de Tarea", ["Desarrollo", "Implementación"], key="tipo_libre")
        metodo_l  = st.selectbox("Método", ["faiss+xgb+catalog", "faiss+catalog", "faiss", "catalog"], key="met_libre")
        complex_l = st.select_slider("Complejidad", options=["baja", "media", "alta"], value="media", key="cx_libre")
        btn_libre = st.button("🚀 Calcular Estimación", use_container_width=True, type="primary", key="btn_libre")

    if btn_libre:
        if not texto_libre.strip():
            st.error("⚠️ Ingresa una descripción.")
        else:
            with st.spinner("🧠 Analizando..."):
                try:
                    backend = _lazy_backend()
                    if not backend.get("ok"):
                        st.error(f"Error cargando motor: {backend.get('err')}")
                        st.stop()
                    tag_l = _resolve_tag(tipo_l)
                    res_l = _estimar_texto(texto_libre, tag_l, metodo_l, complex_l, backend)
                    res_l.update({"texto": texto_libre, "tipo": tag_l, "metodo": metodo_l})
                    st.session_state.resultado_libre = res_l
                except Exception as e:
                    st.error(f"Error: {e}")

    if st.session_state.resultado_libre:
        res = st.session_state.resultado_libre
        st.divider()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("⏱️ Estimación Final", f"{math.ceil(res['horas'])} hrs")
        m2.metric("📚 FAISS",            f"{res['faiss']:.1f} hrs")
        m3.metric("🤖 XGBoost",          f"{res.get('xgb', 0):.1f} hrs")
        m4.metric("📋 Catálogo",         f"{res['catalogo']:.1f} hrs")

        rmin = res.get("rango_min", math.ceil(res["horas"]))
        rmax = res.get("rango_max", math.ceil(res["horas"]))
        info_parts = [f"**Rango probable:** {rmin}h — {rmax}h"]
        if res.get("catalog_match"):
            info_parts.append(f"**Mejor match catálogo:** _{res['catalog_match']}_")
        st.info("  ·  ".join(info_parts))

        if res["top"]:
            st.subheader("Tickets Similares Encontrados")
            for item in res["top"]:
                with st.expander(f"{item['ticket']} ({item['hours']}h) — Similitud: {item['sim']:.2f}"):
                    st.write(item["text"])

        st.divider()
        with st.form("form_libre_save"):
            st.subheader("💾 Guardar Feedback")
            fl1, fl2, fl3 = st.columns(3)
            with fl1:
                st.metric("Estimación IA", f"{math.ceil(res['horas'])}h")
            with fl2:
                hr_real_l = st.number_input("Horas reales que tomó", min_value=0.0, step=0.5)
            with fl3:
                com_l = st.text_input("Comentario / ID ticket")
            if st.form_submit_button("Confirmar y Guardar", type="primary"):
                horas_est = float(math.ceil(res["horas"]))
                append_row_safe({
                    "timestamp":       time.strftime("%Y-%m-%d %H:%M:%S"),
                    "tipo":            res["tipo"],
                    "texto":           res["texto"],
                    "horas_estimadas": horas_est,
                    "horas_reales":    float(hr_real_l),
                    "diferencia":      round(float(hr_real_l) - horas_est, 2),
                    "top_ticket":      res["top"][0]["ticket"] if res["top"] else "",
                    "top_sim":         res["top"][0]["sim"]    if res["top"] else 0,
                    "metodo":          res["metodo"],
                    "autor":           "streamlit_ui",
                    "comentarios":     com_l,
                })
                if hr_real_l > 0:
                    try:
                        bk = _lazy_backend()
                        if bk.get("ok"):
                            est_obj = bk["EmbeddingsFaissEstimator"](res["tipo"])
                            if est_obj.load():
                                est_obj.add_one(res["texto"], float(hr_real_l))
                                st.success("✅ Guardado e índice actualizado.")
                                st.stop()
                    except Exception:
                        pass
                st.success("✅ Guardado.")


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
    st.subheader("📥 Descargar Datos")
    st.caption("Streamlit borra los archivos al reiniciarse. Descarga tu CSV regularmente.")
    if os.path.exists(csv_path):
        with open(csv_path, "rb") as f:
            st.download_button(
                label="⬇️ Descargar estimaciones_nuevas.csv",
                data=f,
                file_name="estimaciones_nuevas.csv",
                mime="text/csv",
            )
    else:
        st.info("Aún no hay datos para descargar.")
