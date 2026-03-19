---
title: IA Estimador PS
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: 1.41.0
app_file: streamlit_app.py
pinned: false
---

# IA Estimador PS

Estimador de esfuerzo (horas) para tickets Jira CESQ y PSTC usando IA semántica (FAISS + sentence-transformers) y análisis con Gemini AI.

## Variables de entorno (configurar en HF Spaces → Settings → Variables)

| Variable | Descripción |
|---|---|
| `GEMINI_API_KEY` | API key de Google AI Studio (gratis en aistudio.google.com) |

## Datos

Los archivos CSV históricos van en la carpeta `data/`:
- `EstimacionCESQ.csv` — histórico tickets CESQ (Desarrollo)
- `EstimacionesPSTC.csv` — histórico tickets PSTC (Implementación)
- `catalogo_cesq.csv` — catálogo base CESQ
- `catalogo_pstc.csv` — catálogo base PSTC
