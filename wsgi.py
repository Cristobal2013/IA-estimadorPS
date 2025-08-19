# wsgi.py
import os, sys, traceback

# Asegurar que la raíz está en el PYTHONPATH
sys.path.insert(0, os.path.dirname(__file__))

try:
    # Caso usual: app.py define `app = Flask(__name__)`
    from app import app as application
except Exception:
    try:
        # Alternativa: app.py define `application = Flask(__name__)`
        from app import application  # noqa: F401
    except Exception:
        try:
            # Alternativa factory: app.py expone create_app()
            from app import create_app
            application = create_app()
        except Exception:
            # MOSTRAR el error real en logs
            print("==== APP IMPORT FAILED ====", flush=True)
            traceback.print_exc()
            # App mínima para no caer en bucle de import
            from flask import Flask
            application = Flask(__name__)

# Gunicorn por defecto busca `app`, exponemos alias
app = application
