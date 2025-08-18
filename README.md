
# IA Estimador (Catálogo + Históricos + Reentrenar + Upload)

## Variables de entorno
- ENABLE_EMBEDDINGS=1
- MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2 (opcional)
- DATA_CSV=data/historicos.csv
- EMB_PATH=data/embeddings.npz
- TOP_K=5

## Pasos en Render
1) Build Command: `pip install -r requirements.txt`
2) Start Command: `gunicorn -k gthread -w 1 --threads 1 -t 300 -b 0.0.0.0:$PORT app:app`
3) Env Vars: `ENABLE_EMBEDDINGS=1`
4) Sube tu CSV de históricos desde la UI (botón "Subir CSV") y luego pulsa "Reentrenar histórico".

## Formato CSV
id,tipo,descripcion,horas
TCK-001,CESQ,Crear endpoint...,8
...
