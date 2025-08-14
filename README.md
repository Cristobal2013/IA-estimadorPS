
# Estimador de Horas (RHEL · Embeddings + FAISS · Dual índices)

## Requisitos
- Python 3.8 en RHEL 7.6
- Salida a internet (para descargar el modelo la primera vez) **o** copiar manualmente la caché de HuggingFace.

## Instalación
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
pip install -r requirements.txt
```

> Si tu red requiere proxy, exporta HTTP(S)_PROXY antes de `pip install`.
> Los pesos del modelo (all-MiniLM-L6-v2) se guardarán en `data/models/hf_cache`.

## Ejecutar
```bash
python app.py
```
- Entrena dos índices (CESQ=desarrollo, PSTC=implementación) con embeddings
- Corre en http://0.0.0.0:7860

## Modo offline
En una máquina con internet:
```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').save_pretrained('all-MiniLM-L6-v2')"
```
Copia la carpeta `all-MiniLM-L6-v2/` a `data/models/hf_cache/models--sentence-transformers--all-MiniLM-L6-v2/` (estructura estándar de HF o usa ENV `HF_HOME` apuntando a la carpeta).

## Variables útiles
- `EMB_MODEL`: nombre del modelo HF a usar (por defecto `sentence-transformers/all-MiniLM-L6-v2`)
- `HF_HOME`: ruta de caché HF (por defecto `data/models/hf_cache`)
