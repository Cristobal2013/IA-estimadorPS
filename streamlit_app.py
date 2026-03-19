import streamlit as st
import pandas as pd
import os
import time
import math


# --- IMPORTAMOS TU LÓGICA EXISTENTE ---
from app import _lazy_backend, _resolve_tag, _to_float, append_row_safe

# Configuración de la página
st.set_page_config(page_title="Estimador IA", layout="wide")

# --- ESTADO DE LA SESIÓN ---
if 'resultado' not in st.session_state:
    st.session_state.resultado = None
if 'resultado_desglose' not in st.session_state:
    st.session_state.resultado_desglose = None
if 'tareas' not in st.session_state:
    st.session_state.tareas = [""]


# --- FUNCIÓN REUTILIZABLE DE ESTIMACIÓN ---
def _estimar_texto(texto, tag, metodo, complexity, backend):
    Emb        = backend["EmbeddingsFaissEstimator"]
    est_cat    = backend["estimate_from_catalog"]
    train_ix   = backend["train_index_per_type"]
    load_df    = backend["load_labeled_dataframe"]

    est = Emb(tag)
    try:
        loaded = est.load()
    except:
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
                "text": str(row.get("text", ""))[:200] + "..."
            })

    horas_catalog = 0.0
    catalog_match = ""
    try:
        from estimator import estimate_from_catalog_with_match
        horas_catalog, catalog_match = estimate_from_catalog_with_match(texto, tag, top_n=3, min_cover=0.35)
        horas_catalog = _to_float(horas_catalog)
    except:
        try:
            horas_catalog = _to_float(est_cat(texto, tag, top_n=3, min_cover=0.35))
        except:
            horas_catalog = 0.0

    horas_xgb = 0.0
    try:
        from estimator import predict_xgb
        horas_xgb = _to_float(predict_xgb(texto, tag))
    except:
        horas_xgb = 0.0

    alpha = 0.8
    bias_map = {"baja": -0.1, "media": 0.0, "alta": +0.1}
    alpha_eff = max(0.3, min(0.98, alpha + bias_map[complexity]))

    if metodo == "faiss":
        final = horas_faiss
    elif metodo == "catalog":
        final = horas_catalog
    elif metodo == "faiss+catalog":
        final = alpha_eff * horas_faiss + (1.0 - alpha_eff) * horas_catalog
    else:
        if horas_xgb > 0:
            final = 0.45 * horas_faiss + 0.40 * horas_xgb + 0.15 * horas_catalog
        else:
            final = alpha_eff * horas_faiss + (1.0 - alpha_eff) * horas_catalog

    # Rango de confianza basado en varianza de vecinos FAISS
    hs_neigh = [h for (_, _, h) in top_tickets] if top_tickets else []
    if len(hs_neigh) >= 2:
        rango_min = max(0, math.ceil(final * 0.75))
        rango_max = math.ceil(final * 1.35)
    else:
        rango_min = rango_max = math.ceil(final)

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


# --- TÍTULO ---
st.title("🤖 Estimador de Esfuerzo (IA)")

tab_libre, tab_desglose = st.tabs(["📝 Texto Libre", "📋 Desglose por Tareas"])


# ══════════════════════════════════════════════════
# TAB 1 — TEXTO LIBRE (comportamiento original)
# ══════════════════════════════════════════════════
with tab_libre:
    col_izq, col_der = st.columns([2, 1])

    with col_izq:
        texto = st.text_area("Descripción del Requerimiento", height=200,
                             placeholder="Pega aquí el correo o la descripción del ticket...")
    with col_der:
        st.subheader("Configuración")
        tipo       = st.selectbox("Tipo de Tarea", ["Desarrollo", "Implementación"], key="tipo_libre")
        metodo     = st.selectbox("Método", ["faiss+xgb+catalog", "faiss+catalog", "faiss", "catalog"], key="met_libre")
        complexity = st.select_slider("Complejidad", options=["baja", "media", "alta"], value="media", key="cx_libre")
        btn_estimar = st.button("🚀 Calcular Estimación", use_container_width=True, type="primary", key="btn_libre")

    if btn_estimar:
        if not texto.strip():
            st.error("⚠️ Por favor ingresa una descripción.")
        else:
            with st.spinner("🧠 Analizando..."):
                try:
                    backend = _lazy_backend()
                    if not backend["ok"]:
                        st.error(f"Error cargando motor IA: {backend.get('err')}")
                        st.stop()
                    tag = _resolve_tag(tipo)
                    res = _estimar_texto(texto, tag, metodo, complexity, backend)
                    res.update({"texto": texto, "tipo": tag, "metodo": metodo})
                    st.session_state.resultado = res
                except Exception as e:
                    st.error(f"Error durante el cálculo: {str(e)}")

# ══════════════════════════════════════════════════
# TAB 2 — DESGLOSE POR TAREAS
# ══════════════════════════════════════════════════
with tab_desglose:
    st.markdown("Ingresa cada tarea por separado. El sistema estima cada una y suma el total.")

    col_conf1, col_conf2, col_conf3 = st.columns(3)
    with col_conf1:
        tipo_d     = st.selectbox("Tipo de Tarea", ["Desarrollo", "Implementación"], key="tipo_des")
    with col_conf2:
        metodo_d   = st.selectbox("Método", ["faiss+xgb+catalog", "faiss+catalog", "faiss", "catalog"], key="met_des")
    with col_conf3:
        complex_d  = st.select_slider("Complejidad", options=["baja", "media", "alta"], value="media", key="cx_des")

    st.markdown("---")
    st.markdown("**Tareas a estimar:**")

    # Inputs dinámicos de tareas
    tareas_editadas = []
    for i, tarea in enumerate(st.session_state.tareas):
        col_txt, col_del = st.columns([10, 1])
        with col_txt:
            val = st.text_input(f"Tarea {i+1}", value=tarea, key=f"tarea_{i}",
                                placeholder="Ej: Modificar template 3 campos, instalacion UAT PRD...")
            tareas_editadas.append(val)
        with col_del:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("✕", key=f"del_{i}", help="Eliminar tarea") and len(st.session_state.tareas) > 1:
                st.session_state.tareas.pop(i)
                st.rerun()

    st.session_state.tareas = tareas_editadas

    col_add, col_est = st.columns([1, 3])
    with col_add:
        if st.button("➕ Agregar tarea"):
            st.session_state.tareas.append("")
            st.rerun()
    with col_est:
        btn_desglose = st.button("🚀 Estimar todas las tareas", type="primary", use_container_width=True)

    if btn_desglose:
        tareas_validas = [t.strip() for t in st.session_state.tareas if t.strip()]
        if not tareas_validas:
            st.error("⚠️ Ingresa al menos una tarea.")
        else:
            with st.spinner(f"🧠 Estimando {len(tareas_validas)} tareas..."):
                try:
                    backend = _lazy_backend()
                    if not backend["ok"]:
                        st.error(f"Error cargando motor IA: {backend.get('err')}")
                        st.stop()
                    tag = _resolve_tag(tipo_d)
                    resultados_desglose = []
                    for tarea in tareas_validas:
                        r = _estimar_texto(tarea, tag, metodo_d, complex_d, backend)
                        resultados_desglose.append({
                            "tarea": tarea,
                            "horas": r["horas"],
                            "faiss": r["faiss"],
                            "xgb": r["xgb"],
                            "catalogo": r["catalogo"],
                            "catalog_match": r.get("catalog_match", ""),
                        })
                    st.session_state.resultado_desglose = {
                        "filas": resultados_desglose,
                        "tipo": tag,
                        "metodo": metodo_d,
                    }
                except Exception as e:
                    st.error(f"Error: {str(e)}")

    # Mostrar resultado desglose
    if st.session_state.resultado_desglose:
        rd = st.session_state.resultado_desglose
        st.divider()
        st.subheader("📊 Desglose por Tarea")
        st.caption("Puedes ajustar las horas en la columna **'Horas ajustadas'** antes de guardar.")

        filas = rd["filas"]
        total_ia = sum(math.ceil(f["horas"]) for f in filas)

        # Tabla editable con st.data_editor
        df_edit = pd.DataFrame([{
            "Tarea":              f["tarea"],
            "IA estima":          round(f["horas"], 1),
            "Horas ajustadas":    math.ceil(f["horas"]),
            "FAISS":              round(f["faiss"], 1),
            "XGBoost":            round(f["xgb"], 1),
            "Catálogo":           round(f["catalogo"], 1),
            "Mejor match catálogo": f.get("catalog_match", "—"),
        } for f in filas])

        edited = st.data_editor(
            df_edit,
            column_config={
                "Horas ajustadas": st.column_config.NumberColumn(
                    "Horas ajustadas ✏️", min_value=0, max_value=2000, step=1,
                    help="Edita este valor para ajustar la estimación"
                ),
            },
            disabled=["Tarea", "IA estima", "FAISS", "XGBoost", "Catálogo", "Mejor match catálogo"],
            hide_index=True,
            use_container_width=True,
        )

        total_ajustado = int(edited["Horas ajustadas"].sum())
        diferencia = total_ajustado - total_ia

        c1, c2, c3 = st.columns(3)
        c1.metric("🤖 Total IA", f"{total_ia}h")
        c2.metric("✏️ Total Ajustado", f"{total_ajustado}h",
                  delta=f"{diferencia:+d}h" if diferencia != 0 else "Sin cambios")
        c3.metric("Tareas", len(filas))

        # Guardar feedback del desglose
        st.divider()
        st.subheader("💾 Guardar Feedback")
        with st.form("form_desglose"):
            col_a, col_b = st.columns(2)
            with col_a:
                horas_reales_d = st.number_input("Horas reales totales (una vez terminado)", min_value=0.0, step=0.5, key="hr_des")
            with col_b:
                comentarios_d = st.text_input("Comentario / ID ticket", key="com_des")
            submitted_d = st.form_submit_button("Confirmar y Guardar", type="primary")
            if submitted_d:
                texto_combinado = " | ".join(f["tarea"] for f in filas)
                nueva_fila = {
                    "timestamp":       time.strftime("%Y-%m-%d %H:%M:%S"),
                    "tipo":            rd["tipo"],
                    "texto":           texto_combinado,
                    "horas_estimadas": float(total_ajustado),
                    "horas_reales":    float(horas_reales_d),
                    "diferencia":      round(float(horas_reales_d) - total_ajustado, 2),
                    "top_ticket":      "",
                    "top_sim":         0,
                    "metodo":          rd["metodo"],
                    "autor":           "streamlit_desglose",
                    "comentarios":     comentarios_d,
                }
                try:
                    append_row_safe(nueva_fila)
                    st.success("✅ Guardado correctamente.")
                except Exception as e:
                    st.error(f"Error guardando: {e}")


# --- MOSTRAR RESULTADOS TAB LIBRE ---
if st.session_state.resultado:
    res = st.session_state.resultado
    
    st.divider()
    
    # Métricas grandes
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("⏱️ Estimación Final", f"{math.ceil(res['horas'])} hrs")
    m2.metric("📚 FAISS (similitud)", f"{res['faiss']:.1f} hrs")
    m3.metric("🤖 XGBoost (regresión)", f"{res.get('xgb', 0):.1f} hrs")
    m4.metric("📋 Catálogo", f"{res['catalogo']:.1f} hrs")

    # Rango de confianza y match catálogo
    rmin = res.get("rango_min", math.ceil(res["horas"]))
    rmax = res.get("rango_max", math.ceil(res["horas"]))
    cat_match = res.get("catalog_match", "")
    info_parts = [f"**Rango probable:** {rmin}h — {rmax}h"]
    if cat_match:
        info_parts.append(f"**Mejor match catálogo:** _{cat_match}_")
    st.info("  ·  ".join(info_parts))

    # Tabla de tickets similares
    st.subheader("Tickets Similares Encontrados")
    if res['top']:
        for item in res['top']:
            with st.expander(f"{item['ticket']} ({item['hours']} hrs) - Similitud: {item['sim']:.2f}"):
                st.write(f"**Texto:** {item['text']}")
    else:
        st.info("No se encontraron tickets similares relevantes.")

    # --- SECCIÓN DE GUARDADO (FEEDBACK) ---
    st.divider()
    st.subheader("💾 Guardar Feedback")
    st.caption("Ingresa las horas reales una vez terminado el ticket. Esto mejora el modelo con el tiempo.")

    with st.form("form_guardar"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Estimación IA", f"{math.ceil(res['horas'])} h")
        with c2:
            horas_reales = st.number_input("Horas reales que tomó", min_value=0.0, step=0.5,
                                           help="¿Cuántas horas tomó realmente el ticket?")
        with c3:
            comentarios = st.text_input("Comentario / ID ticket")

        submitted = st.form_submit_button("Confirmar y Guardar", type="primary")

        if submitted:
            horas_estimadas = float(math.ceil(res['horas']))
            nueva_fila = {
                "timestamp":       time.strftime("%Y-%m-%d %H:%M:%S"),
                "tipo":            res['tipo'],
                "texto":           res['texto'],
                "horas_estimadas": horas_estimadas,
                "horas_reales":    float(horas_reales),
                "diferencia":      round(float(horas_reales) - horas_estimadas, 2),
                "top_ticket":      res['top'][0]['ticket'] if res['top'] else "",
                "top_sim":         res['top'][0]['sim'] if res['top'] else 0,
                "metodo":          res['metodo'],
                "autor":           "streamlit_ui",
                "comentarios":     comentarios,
            }
            try:
                append_row_safe(nueva_fila)

                # Actualiza el índice FAISS de forma incremental (sin reconstruir todo)
                if horas_reales > 0:
                    try:
                        backend = _lazy_backend()
                        if backend["ok"]:
                            Emb = backend["EmbeddingsFaissEstimator"]
                            est = Emb(res['tipo'])
                            if est.load():
                                # Usa horas_reales si hay, si no usa estimadas
                                est.add_one(res['texto'], float(horas_reales))
                                st.success(f"✅ Guardado e índice actualizado con {horas_reales}h reales.")
                            else:
                                st.success("✅ Guardado. Índice se actualizará en el próximo reentrenamiento.")
                        else:
                            st.success("✅ Guardado.")
                    except Exception as e_idx:
                        st.success(f"✅ Guardado. (Índice no actualizado: {e_idx})")
                else:
                    st.success("✅ Guardado. Ingresa horas reales para mejorar el modelo.")
            except Exception as e:
                st.error(f"Error guardando: {e}")

# --- MÉTRICAS DE PRECISIÓN DEL MODELO ---
csv_path = "data/estimaciones_nuevas.csv"
if os.path.exists(csv_path):
    try:
        df_fb = pd.read_csv(csv_path)
        if "horas_reales" in df_fb.columns and "horas_estimadas" in df_fb.columns:
            df_valid = df_fb[(df_fb["horas_reales"] > 0) & (df_fb["horas_estimadas"] > 0)].copy()
            if len(df_valid) >= 3:
                st.divider()
                st.subheader("📊 Precisión del Modelo")
                df_valid["error_abs"] = (df_valid["horas_reales"] - df_valid["horas_estimadas"]).abs()
                df_valid["error_pct"] = df_valid["error_abs"] / df_valid["horas_reales"] * 100
                mae  = df_valid["error_abs"].mean()
                mape = df_valid["error_pct"].mean()
                bias = (df_valid["horas_estimadas"] - df_valid["horas_reales"]).mean()
                n    = len(df_valid)
                a1, a2, a3, a4 = st.columns(4)
                a1.metric("Registros con feedback", n)
                a2.metric("Error promedio (MAE)", f"{mae:.1f} h")
                a3.metric("Error % promedio", f"{mape:.0f}%")
                tendencia = f"+{bias:.1f}h (sobreestima)" if bias > 0.5 else (f"{bias:.1f}h (subestima)" if bias < -0.5 else "Calibrado ✓")
                a4.metric("Tendencia", tendencia)
                st.caption(f"Basado en {n} confirmaciones con horas reales ingresadas.")
    except Exception:
        pass

# --- DESCARGA DE DATOS (IMPORTANTE PARA PERSISTENCIA) ---
st.divider()
st.subheader("📥 Descargar Datos")
st.markdown("Streamlit borra los archivos al reiniciarse. **Descarga tu CSV regularmente.**")

csv_path = "data/estimaciones_nuevas.csv"
if os.path.exists(csv_path):
    with open(csv_path, "rb") as f:
        st.download_button(
            label="Descargar estimaciones_nuevas.csv",
            data=f,
            file_name="estimaciones_nuevas.csv",
            mime="text/csv"
        )
else:
    st.info("Aún no hay estimaciones guardadas para descargar.")
