import streamlit as st
import os
# Importamos la lógica de tu backend actual
# Nota: Asegúrate de que las funciones _lazy_backend o similares sean accesibles
from app import _lazy_backend, _resolve_tag, _to_float

st.title("Estimador IA - CESQ/PSTC")

# Entrada del usuario
texto = st.text_area("Descripción del requerimiento:", height=150)
tipo = st.selectbox("Tipo:", ["Desarrollo", "Implementación"])
metodo = st.selectbox("Método:", ["faiss+catalog", "faiss", "catalog"])
complexity = st.select_slider("Complejidad:", options=["baja", "media", "alta"], value="media")

if st.button("Estimar"):
    if not texto:
        st.error("Por favor ingresa un texto.")
    else:
        with st.spinner("Calculando estimación..."):
            # --- Aquí replicamos la lógica de tu endpoint /api/estimate ---
            backend = _lazy_backend()
            if not backend["ok"]:
                st.error(f"Error cargando el modelo: {backend.get('err')}")
            else:
                # Extraer funciones
                Emb = backend["EmbeddingsFaissEstimator"]
                est_cat = backend["estimate_from_catalog"]
                train_ix = backend["train_index_per_type"]
                
                tag = _resolve_tag(tipo)
                
                # Carga / Entrenamiento
                est = Emb(tag)
                if not est.load():
                    train_ix(full=True)
                    est.load()
                
                # Predicción FAISS
                horas_faiss, _ = est.predict(texto, k=3)
                horas_faiss = _to_float(horas_faiss)
                
                # Predicción Catálogo
                horas_catalog = _to_float(est_cat(texto, tag, top_n=3, min_cover=0.35))
                
                # Híbrido
                alpha = 0.8 # Valor por defecto
                bias_map = {"baja": -0.1, "media": 0.0, "alta": +0.1}
                alpha_eff = max(0.3, min(0.98, alpha + bias_map[complexity]))
                
                if metodo == "faiss":
                    final = horas_faiss
                elif metodo == "catalog":
                    final = horas_catalog
                else:
                    final = alpha_eff * horas_faiss + (1.0 - alpha_eff) * horas_catalog
                
                # Resultado
                st.success(f"Estimación: {final:.2f} horas")
                st.json({
                    "faiss": horas_faiss,
                    "catalogo": horas_catalog,
                    "alpha_usado": alpha_eff
                })
