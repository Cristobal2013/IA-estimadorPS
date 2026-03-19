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

    # ── Encabezado del proyecto ──────────────────────
    st.subheader("1️⃣ Datos del Proyecto")
    c1, c2, c3 = st.columns([4, 1, 1])
    with c1:
        nombre_proy = st.text_input(
            "Nombre del proyecto / Ticket",
            placeholder="Ej: PSTC-1234 · Cliente ABC",
            key="nombre_proy",
        )
    with c2:
        integ_proy = st.selectbox(
            "Integración base",
            ["Full ASP", "Mixto", "On Premise", "Estandar"],
            key="integ_proy",
        )
    with c3:
        cx_extra = st.select_slider(
            "Complejidad extra",
            options=["baja", "media", "alta"],
            value="media",
            key="cx_proy",
        )
    st.caption("💡 Los paquetes del catálogo son siempre **Implementación**. Cada tarea adicional puede ser Desarrollo o Implementación.")

    st.divider()

    # ── Sección paquetes + tareas ────────────────────
    col_paq, col_extra = st.columns([3, 2])

    # ── Paquetes del catálogo
    with col_paq:
        st.subheader("2️⃣ Paquetes del Catálogo")

        if df_roles.empty:
            st.warning("No se encontró catalogo_roles.csv en data/")
        else:
            paquetes_disp = sorted(df_roles["paquete"].unique())
            paquetes_sel = st.multiselect(
                "Selecciona los paquetes del proyecto",
                options=paquetes_disp,
                placeholder="Busca o selecciona: Plantillas, PPL, Coapi...",
                key="paquetes_sel",
            )

            # Guarda qué escenarios están checked
            escenarios_check: dict = {}

            for paq in paquetes_sel:
                subset = df_roles[df_roles["paquete"] == paq]
                integ_disp = list(subset["integracion"].unique())

                # Elegir integración más apropiada para este paquete
                if integ_proy in integ_disp:
                    integ_use = integ_proy
                elif "Estandar" in integ_disp:
                    integ_use = "Estandar"
                else:
                    integ_use = integ_disp[0]

                subset_integ = subset[subset["integracion"] == integ_use]

                with st.expander(f"**{paq}** · {integ_use}", expanded=True):
                    for _, row in subset_integ.iterrows():
                        esc = row["escenario"]
                        label = (
                            f"{esc}  —  "
                            f"TC: **{row['tc']}h** · SC: **{row['sc']}h** · "
                            f"PM: **{row['pm']}h**  →  Total: **{row['total']}h**"
                        )
                        checked = st.checkbox(label, key=f"chk_{paq}_{esc}")
                        escenarios_check[(paq, esc, integ_use)] = (checked, row)

    # ── Tareas adicionales (IA)
    with col_extra:
        st.subheader("3️⃣ Tareas Adicionales (IA)")
        st.caption(
            "Para actividades no contempladas en el catálogo. "
            "La IA estima las horas basándose en tickets históricos."
        )

        tareas_ed = []
        for i, t in enumerate(st.session_state.tareas_extra):
            # Soporte tanto dict nuevo como string legacy
            if isinstance(t, dict):
                t_texto = t.get("texto", "")
                t_tipo  = t.get("tipo", "Implementación")
            else:
                t_texto = t
                t_tipo  = "Implementación"

            ct, ctipo, cd = st.columns([4, 2, 1])
            with ct:
                val = st.text_input(
                    f"Tarea {i + 1}",
                    value=t_texto,
                    key=f"textra_{i}",
                    placeholder="Ej: Soporte post go-live, dev especial...",
                    label_visibility="collapsed",
                )
            with ctipo:
                tipo_t = st.selectbox(
                    "Tipo",
                    ["Implementación", "Desarrollo"],
                    index=0 if t_tipo == "Implementación" else 1,
                    key=f"ttipo_{i}",
                    label_visibility="collapsed",
                )
            with cd:
                st.markdown("<br>", unsafe_allow_html=True)
                if (
                    st.button("✕", key=f"del_extra_{i}", help="Eliminar")
                    and len(st.session_state.tareas_extra) > 1
                ):
                    st.session_state.tareas_extra.pop(i)
                    st.rerun()
            tareas_ed.append({"texto": val, "tipo": tipo_t})
        st.session_state.tareas_extra = tareas_ed

        if st.button("➕ Agregar tarea", key="add_extra"):
            st.session_state.tareas_extra.append({"texto": "", "tipo": "Implementación"})
            st.rerun()

        st.markdown("---")
        metodo_proy = st.selectbox(
            "Método IA",
            ["faiss+xgb+catalog", "faiss+catalog", "faiss", "catalog"],
            key="met_proy",
        )

    # ── Botón estimar ────────────────────────────────
    st.divider()
    _, col_btn = st.columns([4, 1])
    with col_btn:
        btn_proy = st.button(
            "🚀 Estimar Proyecto", type="primary", use_container_width=True, key="btn_proy"
        )

    # ── Cálculo ──────────────────────────────────────
    if btn_proy:
        filas_res: list[dict] = []

        # 1) Paquetes seleccionados del catálogo
        for (paq, esc, integ_use), (checked, row) in escenarios_check.items():
            if not checked:
                continue
            filas_res.append({
                "Origen":          "📦 Catálogo",
                "Paquete / Tarea": f"{paq} · {esc}",
                "Integración":     integ_use,
                "TC (h)":          float(row["tc"]),
                "SC (h)":          float(row["sc"]),
                "PM (h)":          float(row["pm"]),
                "Total (h)":       float(row["total"]),
                "Horas ajustadas": float(row["total"]),
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
                            tarea    = tarea_item["texto"].strip()
                            tipo_t   = tarea_item.get("tipo", "Implementación")
                            tag_proy = _resolve_tag(tipo_t)
                            r = _estimar_texto(tarea, tag_proy, metodo_proy, cx_extra, backend)
                            horas_ia = max(1, math.ceil(r["horas"]))
                            top1 = r["top"][0] if r["top"] else None
                            ref = f"FAISS:{r['faiss']:.0f}h"
                            if r["xgb"] > 0:
                                ref += f" · XGB:{r['xgb']:.0f}h"
                            if top1:
                                ref += f" · Ref:{top1['ticket']}({top1['sim']:.2f})"

                            # Distribuir horas por rol según paquete más similar
                            tc_r, sc_r, pm_r, matched = _get_role_ratios(tarea, df_roles, integ_proy)
                            tc_h = round(horas_ia * tc_r, 1)
                            sc_h = round(horas_ia * sc_r, 1)
                            pm_h = round(horas_ia * pm_r, 1)
                            if matched:
                                ref += f" · Roles≈{matched}"

                            filas_res.append({
                                "Origen":          f"🤖 IA ({tipo_t[:4]})",
                                "Paquete / Tarea": tarea,
                                "Integración":     tipo_t,
                                "TC (h)":          tc_h,
                                "SC (h)":          sc_h,
                                "PM (h)":          pm_h,
                                "Total (h)":       float(horas_ia),
                                "Horas ajustadas": float(horas_ia),
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
                "Horas ajustadas": st.column_config.NumberColumn(
                    "Horas ajustadas ✏️",
                    min_value=0,
                    max_value=5000,
                    step=0.5,
                    help="Edita para ajustar la estimación de cada ítem",
                ),
                "Origen":          st.column_config.TextColumn("Origen",     width="small"),
                "Referencia IA":   st.column_config.TextColumn("Ref. IA",    width="medium"),
                "Integración":     st.column_config.TextColumn("Integración",width="small"),
            },
            disabled=[
                "Origen", "Paquete / Tarea", "Integración",
                "TC (h)", "SC (h)", "PM (h)", "Total (h)", "Referencia IA",
            ],
            hide_index=True,
            use_container_width=True,
        )

        # Totales
        total_tc        = df_res["TC (h)"].sum()
        total_sc        = df_res["SC (h)"].sum()
        total_pm        = df_res["PM (h)"].sum()
        total_catalogo  = df_res["Total (h)"].sum()
        total_ajustado  = edited["Horas ajustadas"].sum()
        diff = total_ajustado - total_catalogo

        st.markdown("---")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("👷 TC",            f"{total_tc:.1f}h")
        c2.metric("💼 SC",            f"{total_sc:.1f}h")
        c3.metric("📋 PM",            f"{total_pm:.1f}h")
        c4.metric("🔢 Total estimado", f"{total_catalogo:.1f}h")
        c5.metric(
            "✏️ Total ajustado",
            f"{total_ajustado:.1f}h",
            delta=f"{diff:+.1f}h" if diff != 0 else "Sin cambios",
        )

        # Gráfico por rol
        st.bar_chart(
            pd.DataFrame({"Rol": ["TC", "SC", "PM"], "Horas": [total_tc, total_sc, total_pm]}).set_index("Rol")
        )

        # Resumen para copiar
        with st.expander("📋 Copiar resumen"):
            titulo_copia = rd["nombre"] or "Proyecto"
            lines = [
                f"Estimación: {titulo_copia}",
                f"Integración: {rd['integracion']}",
                "",
            ]
            for f in filas:
                if f["Origen"] == "📦 Catálogo":
                    lines.append(
                        f"• {f['Paquete / Tarea']}: "
                        f"TC={f['TC (h)']}h | SC={f['SC (h)']}h | PM={f['PM (h)']}h | Total={f['Total (h)']}h"
                    )
                else:
                    lines.append(f"• {f['Paquete / Tarea']}: ~{f['Total (h)']}h (estimado IA)")
            lines += [
                "",
                f"TOTALES POR ROL → TC:{total_tc:.1f}h | SC:{total_sc:.1f}h | PM:{total_pm:.1f}h",
                f"TOTAL PROYECTO : {total_ajustado:.1f}h",
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
                    "horas_estimadas": float(total_ajustado),
                    "horas_reales":    float(hr_real_p),
                    "diferencia":      round(float(hr_real_p) - total_ajustado, 2),
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
