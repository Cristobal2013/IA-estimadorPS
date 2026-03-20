"""
db.py — Capa SQLite para el Estimador PS
Migra automáticamente los CSV existentes en la primera ejecución.
"""
import os
import sqlite3
from pathlib import Path

import pandas as pd

DATA_DIR = Path(os.environ.get("DATA_DIR", str(Path(__file__).parent / "data")))
DB_PATH  = DATA_DIR / "estimador.db"

# Mapa CSV → nombre de tabla
_CSV_TABLES = {
    "catalogo_roles.csv":   "catalogo_roles",
    "catalogo_pstc.csv":    "catalogo_pstc",
    "catalogo_cesq.csv":    "catalogo_cesq",
    "EstimacionesPSTC.csv": "historico_pstc",
    "EstimacionCESQ.csv":   "historico_cesq",
}


def _conn() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH)


def init_db():
    """
    Inicializa la BD. Si una tabla no existe, la crea migrando el CSV correspondiente.
    Los CSV originales NO se eliminan (sirven de respaldo).
    """
    con = _conn()
    migrated = []
    for csv_name, table_name in _CSV_TABLES.items():
        csv_path = DATA_DIR / csv_name
        # Verificar si la tabla ya existe
        exists = con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,)
        ).fetchone()
        if exists:
            continue
        # Migrar desde CSV si existe
        if csv_path.exists():
            try:
                df = _read_csv_safe(csv_path)
                df.to_sql(table_name, con, if_exists="replace", index=False)
                migrated.append(f"{table_name} ({len(df)} filas)")
            except Exception as e:
                print(f"[db] Error migrando {csv_name}: {e}")
        else:
            # Crear tabla vacía con schema mínimo según el tipo
            _create_empty_table(con, table_name)
    con.commit()
    con.close()
    if migrated:
        print(f"[db] Migración completada: {', '.join(migrated)}")


def _create_empty_table(con: sqlite3.Connection, table_name: str):
    schemas = {
        "catalogo_roles":  "paquete TEXT, escenario TEXT, integracion TEXT, tc REAL, sc REAL, pm REAL, total REAL",
        "catalogo_pstc":   "text TEXT, hours REAL, categoria TEXT",
        "catalogo_cesq":   "text TEXT, hours REAL, categoria TEXT",
        "historico_pstc":  "text TEXT, hours REAL, ticket TEXT, source TEXT",
        "historico_cesq":  "text TEXT, hours REAL, ticket TEXT, source TEXT",
    }
    schema = schemas.get(table_name, "id INTEGER PRIMARY KEY")
    con.execute(f"CREATE TABLE IF NOT EXISTS {table_name} ({schema})")


def _read_csv_safe(path: Path) -> pd.DataFrame:
    """Lee un CSV con fallback de encodings."""
    for enc in ["utf-8", "latin-1", "utf-8-sig"]:
        try:
            return pd.read_csv(path, encoding=enc, on_bad_lines="skip", low_memory=False)
        except Exception:
            continue
    return pd.read_csv(path, on_bad_lines="skip", low_memory=False)


# ── API pública ────────────────────────────────────────

def get_table(table_name: str) -> pd.DataFrame:
    """Retorna una tabla completa como DataFrame."""
    init_db()
    try:
        con = _conn()
        df = pd.read_sql(f"SELECT * FROM {table_name}", con)
        con.close()
        return df
    except Exception:
        return pd.DataFrame()


def save_table(table_name: str, df: pd.DataFrame):
    """Reemplaza toda la tabla con el DataFrame dado."""
    init_db()
    con = _conn()
    df.to_sql(table_name, con, if_exists="replace", index=False)
    con.commit()
    con.close()


def append_row(table_name: str, row: dict):
    """Agrega una fila a la tabla."""
    init_db()
    con = _conn()
    df_row = pd.DataFrame([row])
    df_row.to_sql(table_name, con, if_exists="append", index=False)
    con.commit()
    con.close()


def delete_rows(table_name: str, indices: list):
    """Elimina filas por rowid (índice 0-based del DataFrame)."""
    init_db()
    df = get_table(table_name)
    df = df.drop(index=indices).reset_index(drop=True)
    save_table(table_name, df)


def table_exists(table_name: str) -> bool:
    init_db()
    con = _conn()
    exists = con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,)
    ).fetchone() is not None
    con.close()
    return exists
