
# IA Estimador (mínimo viable para Render)

Este paquete soluciona el error `jinja2.exceptions.UndefinedError: 'estimate' is undefined`
pasando variables por defecto al template y blindando el render con un filtro `hfmt`.

## Deploy en Render
1. Crear servicio Web (Python).
2. Build Command: `pip install -r requirements.txt`
3. Start Command: `gunicorn -k gthread -w 2 -b 0.0.0.0:$PORT app:app`
4. (Opcional) `SECRET_KEY` en variables de entorno.

## Estructura
- app.py
- templates/index.html
- static/styles.css
- requirements.txt
- Procfile

## Extender
- Implementa tu lógica real de estimación en `combined_estimate()`.
- Si agregas estimador por similitud, pasa también `estimate_semantic` y ajusta el cálculo final.
