import streamlit as st
import pandas as pd
import os
import time

# --- IMPORTAMOS TU LÓGICA EXISTENTE ---
# Esto reutiliza lo que ya programaste en app.py y estimator.py
from app import _lazy_backend, _resolve_tag, _to_float, append_row_safe

# Configuración de la página
st.set_page_config(page_title="Estimador IA", layout="wide")

# --- ESTADO DE LA SESIÓN ---
# Necesario para recordar la estimación cuando tocas otros botones
if 'resultado' not in st.session_state:
    st.session_state.resultado = None
if 'texto_input' not in st.session_state:
    st.session_state.texto_input = ""

# --- TÍTULO Y ENTRADAS ---
st.title("🤖 Estimador de Esfuerzo (IA)")
st.markdown("Ingresa el requerimiento para buscar tickets similares y calcular horas.")

col_izq, col_der = st.columns([2, 1])

with col_izq:
    texto = st.text_area("Descripción del Requerimiento", height=200, placeholder="Pega aquí el correo o la descripción del ticket...")

with col_der:
    st.subheader("Configuración")
    tipo = st.selectbox("Tipo de Tarea", ["Desarrollo", "Implementación"])
    metodo = st.selectbox("Método", ["faiss+catalog", "faiss", "catalog"])
    complexity = st.select_slider("Complejidad Percibida", options=["baja", "media", "alta"], value="media")
    
    btn_estimar = st.button("🚀 Calcular Estimación", use_container_width=True, type="primary")

# --- LÓGICA DE ESTIMACIÓN (AL CLICAR BOTÓN) ---
if btn_estimar:
    if not texto.strip():
        st.error("⚠️ Por favor ingresa una descripción.")
    else:
        with st.spinner("🧠 Analizando similitud semántica y catálogo..."):
            try:
                # 1. Cargar el backend (IA)
                backend = _lazy_backend()
                if not backend["ok"]:
                    st.error(f"Error cargando motor IA: {backend.get('err')}")
                    st.stop()

                # 2. Preparar herramientas
                Emb = backend["EmbeddingsFaissEstimator"]
                est_cat = backend["estimate_from_catalog"]
                train_ix = backend["train_index_per_type"]
                load_df = backend["load_labeled_dataframe"]
                
                tag = _resolve_tag(tipo)

                # 3. Cargar o Entrenar al vuelo si falta el índice
                est = Emb(tag)
                loaded = False
                try:
                    loaded = est.load()
                except:
                    loaded = False
                
                if not loaded:
                    st.warning("Entrenando índice por primera vez (esto puede tardar unos segundos)...")
                    train_ix(full=True)
                    est.load()

                # 4. Predicción FAISS (Histórico)
                horas_faiss, neighbors = est.predict(texto, k=5)
                horas_faiss = _to_float(horas_faiss)

                # 5. Recuperar detalles de los vecinos (Top Tickets)
                labeled = load_df(tag).reset_index(drop=True)
                top_tickets = []
                for (idx, sim, h) in neighbors[:3]: # Top 3 para mostrar
                    if idx is not None and 0 <= idx < len(labeled):
                        row = labeled.loc[idx]
                        top_tickets.append({
                            "ticket": str(row.get("ticket", "N/A")),
                            "hours": float(h),
                            "sim": float(sim),
                            "text": str(row.get("text", ""))[:200] + "..."
                        })

                # 6. Predicción Catálogo
                horas_catalog = 0.0
                try:
                    horas_catalog = _to_float(est_cat(texto, tag, top_n=3, min_cover=0.35))
                except:
                    horas_catalog = 0.0

                # 7. Cálculo Híbrido (Igual que en tu app.py)
                alpha = 0.8
                bias_map = {"baja": -0.1, "media": 0.0, "alta": +0.1}
                alpha_eff = max(0.3, min(0.98, alpha + bias_map[complexity]))

                if metodo == "faiss":
                    final = horas_faiss
                elif metodo == "catalog":
                    final = horas_catalog
                else:
                    final = alpha_eff * horas_faiss + (1.0 - alpha_eff) * horas_catalog

                # Guardar en sesión para que no se borre al interactuar
                st.session_state.resultado = {
                    "horas": final,
                    "faiss": horas_faiss,
                    "catalogo": horas_catalog,
                    "top": top_tickets,
                    "texto": texto,
                    "tipo": tag,
                    "metodo": metodo
                }

            except Exception as e:
                st.error(f"Error durante el cálculo: {str(e)}")

# --- MOSTRAR RESULTADOS ---
if st.session_state.resultado:
    res = st.session_state.resultado
    
    st.divider()
    
    # Métricas grandes
    m1, m2, m3 = st.columns(3)
    m1.metric("⏱️ Estimación Final", f"{res['horas']:.1f} hrs")
    m2.metric("📚 Base Histórica", f"{res['faiss']:.1f} hrs")
    m3.metric("📋 Catálogo", f"{res['catalogo']:.1f} hrs")

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
    st.caption("Ajusta las horas si es necesario y guarda. ¡Descarga el CSV antes de salir!")

    with st.form("form_guardar"):
        c1, c2 = st.columns(2)
        with c1:
            horas_reales = st.number_input("Horas Corregidas (Reales)", value=float(res['horas']))
        with c2:
            comentarios = st.text_input("Comentarios / ID Ticket Nuevo")
        
        submitted = st.form_submit_button("Confirmar y Guardar")
        
        if submitted:
            nueva_fila = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "tipo": res['tipo'],
                "texto": res['texto'],
                "horas": horas_reales,
                "top_ticket": res['top'][0]['ticket'] if res['top'] else "",
                "top_sim": res['top'][0]['sim'] if res['top'] else 0,
                "metodo": res['metodo'],
                "autor": "streamlit_ui",
                "comentarios": comentarios
            }
            try:
                append_row_safe(nueva_fila)
                st.success("✅ Guardado en memoria temporal.")
            except Exception as e:
                st.error(f"Error guardando: {e}")

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
