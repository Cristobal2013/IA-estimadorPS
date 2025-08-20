# Paquete de correcciones

Incluye:
- app.py  -> agrega /api/estimate, /api/accept y /api/health (no elimina nada de lo existente).
- estimator.py -> corrige FutureWarning en concat de DataFrames y fallback NEW_EST_CSV.
- templates/index.html -> tu HTML con manejo de errores y estado de carga.
- static/styles.css -> estilos mínimos.

Instalación:
1) Reemplaza estos archivos en tu repo manteniendo el resto igual.
2) Verifica que exista la carpeta 'data/'. Al guardar, se crea 'data/estimaciones_nuevas.csv'.
3) Despliega en Render.
