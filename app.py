from flask import Flask, render_template, request, redirect, url_for, flash
from pathlib import Path
import pandas as pd
import datetime as dt
import re, math, os

from estimator import (
    EmbeddingsFaissEstimator,
    train_index_per_type,
    load_labeled_dataframe,
)

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", "dev-secret")


# -------------------------
# Helpers
# -------------------------
def _extract_field_count(text: str) -> int:
    if not text:
        return 0
    m = re.search(r"(\d+)\s*(campos?|columnas?|fields?)", text.lower())
    return int(m.group(1)) if m else 0


def _neighbor_based_cost_per_field(neighbors, labeled_df) -> float:
    numer, denom = 0.0, 0.0
    for idx, sim, h in neighbors:
        if idx is None or idx < 0 or idx >= len(labeled_df):
            continue
        t = labeled_df.loc[idx, "text"]
        n = _extract_field_count(t)
        if n > 0:
            numer += float(h)
            denom += float(n)
    if denom > 0:
        return numer / denom
    return 0.35  # fallback: 0.35 h/campo


def _estimate_with_softmax(neighbors, temperature=0.20):
    """Softmax(sim/temperature): más frío => más peso al top1."""
    if not neighbors:
        return 0.0, []
    sims = [max(0.0, s) for _, s, _ in neighbors]
    exps = [math.exp(s/temperature) for s in sims]
    Z = sum(exps) or 1.0
    weights = [e/Z for e in exps]
    hours = [h for _, _, h in neighbors]
    estimate = sum(w*h for w, h in zip(weights, hours))
    return estimate, weights


def _filter_neighbors_by_source(neighbors, labeled_df, origen):
    """Filtra vecinos por source: 'historic' | 'catalog' | 'new' | 'todos'."""
    if origen in (None, "", "todos"):
        return neighbors
    keep = []
    for idx, sim, h in neighbors:
        if idx is None or idx < 0 or idx >= len(labeled_df):
            continue
        src = labeled_df.loc[idx].get("source", "unknown")
        if src == origen:
            keep.append((idx, sim, h))
    return keep


# Entrenar índices al iniciar (histórico + catálogos + nuevas)
try:
    counts = train_index_per_type()
    print(
        f"Índices entrenados. "
        f"Desarrollo={counts.get('desarrollo',0)}, "
        f"Implementación={counts.get('implementacion',0)}"
    )
except Exception as e:
    print("WARN: No se pudo entrenar índices al iniciar:", e)


# -------------------------
# Rutas
# -------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    estimate = None
    estimate_top1 = None
    neighbors = []
    debug_info = {}
    form = {"tipo": "desarrollo", "texto": "", "metodo": "softmax", "origen": "todos"}

    if request.method == "POST":
        accion = request.form.get("accion", "estimar")
        form["tipo"] = request.form.get("tipo", "desarrollo")
        form["texto"] = request.form.get("texto", "")
        form["metodo"] = request.form.get("metodo", "softmax")
        form["origen"] = request.form.get("origen", "todos")

        tag = "desarrollo" if form["tipo"] == "desarrollo" else "implementacion"

        # Cargar índice (si falla, reentrena una vez)
        est = EmbeddingsFaissEstimator(tag)
        try:
            est.load()
        except Exception:
            try:
                train_index_per_type()
                est.load()
            except Exception as e2:
                return (f"No se pudo cargar/entrenar el índice [{tag}]: {e2}", 500)

        if accion == "guardar":
            # Persistir confirmación del usuario
            metodo_usado = request.form.get("metodo_usado", form["metodo"])
            estimacion_ia = request.form.get("estimacion_ia")
            estimacion_top1 = request.form.get("estimacion_top1")
            estimacion_final = request.form.get("estimacion_final")
            horas_reales = request.form.get("horas_reales")

            # Si no llegó estimacion_final, elegir según método
            if not estimacion_final:
                if metodo_usado == "top1" and estimacion_top1:
                    estimacion_final = estimacion_top1
                elif metodo_usado == "softmax" and estimacion_ia:
                    estimacion_final = estimacion_ia

            def to_float(x):
                try:
                    return float(x) if x not in (None, "", "-") else None
                except Exception:
                    return None

            out = Path("data/estimaciones_nuevas.csv")
            out.parent.mkdir(parents=True, exist_ok=True)
            row = {
                "fecha": dt.datetime.now().isoformat(timespec="seconds"),
                "tipo": form["tipo"],                # CESQ (desarrollo) o PSTC (implementación)
                "texto": form["texto"],
                "metodo_usado": metodo_usado,        # softmax o top1
                "estimacion_ia": to_float(estimacion_ia),
                "estimacion_top1": to_float(estimacion_top1),
                "estimacion_final": to_float(estimacion_final),
                "horas_reales": to_float(horas_reales),
            }
            df_out = pd.DataFrame([row])
            header = not out.exists()
            df_out.to_csv(out, mode="a", header=header, index=False, encoding="utf-8")

            flash("Estimación guardada. El modelo aprenderá con futuros reentrenos.", "ok")
            return redirect(url_for("index"))

        # === Estimar ===
        # Vecinos (diagnóstico pre y post filtro)
        est_soft_unused, neighbors_all = est.predict(form["texto"], k=30)
        labeled = load_labeled_dataframe(tag).reset_index(drop=True)

        # Conteo por origen (antes de filtrar)
        counts_all = {"historic":0, "catalog":0, "new":0, "unknown":0}
        rows_all = []
        for (i,s,h) in neighbors_all:
            if i is None or i < 0 or i >= len(labeled):
                continue
            src_i = labeled.loc[i].get("source","unknown")
            counts_all[src_i] = counts_all.get(src_i,0)+1
            rows_all.append({
                "idx": int(i),
                "sim": float(round(s,4)),
                "hours": float(round(h,4)),
                "ticket": str(labeled.loc[i].get("ticket","?")),
                "source": src_i,
                "text": str(labeled.loc[i].get("text",""))[:180]
            })

        # Aplicar filtro de origen a los vecinos usados
        neighbors = _filter_neighbors_by_source(neighbors_all, labeled, form.get("origen", "todos"))
        debug_info = {
            "tipo": tag,
            "texto": form.get("texto",""),
            "origen": form.get("origen","todos"),
            "k_all": len(neighbors_all),
            "k_used": len(neighbors),
            "counts_all": counts_all,
            "neighbors_all": rows_all[:12],
        }

        if neighbors:
            # Top1: si origen=todos, preferir histórico en casi-empate
            if form.get("metodo") == "top1" and form.get("origen", "todos") == "todos":
                best_idx, best_sim, best_h = max(neighbors, key=lambda t: t[1])
                hist_candidates = [(i,s,h) for (i,s,h) in neighbors if labeled.loc[i].get("source","?")=="historic"]
                if hist_candidates:
                    h_idx, h_sim, h_h = max(hist_candidates, key=lambda t: t[1])
                    estimate_top1 = h_h if (best_sim - h_sim) <= 0.02 else best_h
                else:
                    estimate_top1 = best_h
            else:
                estimate_top1 = sorted(neighbors, key=lambda t: t[1], reverse=True)[0][2]
        else:
            estimate_top1 = None

        # Estrategia elegible por el usuario
        if form["metodo"] == "top1" and neighbors:
            estimate = estimate_top1
        else:
            est_soft, _ = _estimate_with_softmax(neighbors, temperature=0.20)
            estimate = est_soft

        # Ajuste por cantidad de campos
        N = _extract_field_count(form["texto"])
        if N > 0:
            cpf = _neighbor_based_cost_per_field(neighbors, labeled)
            alpha = 0.3  # 30% método elegido + 70% costo_por_campo
            base = alpha * float(estimate or 0.0) + (1.0 - alpha) * (cpf * N)
            min_per_field = 0.25
            estimate = max(base, N * min_per_field)

    # Render vecinos (solo top 3)
    examples = []
    if neighbors:
        tag = "desarrollo" if form["tipo"] == "desarrollo" else "implementacion"
        labeled = load_labeled_dataframe(tag).reset_index(drop=True)
        neighbors = sorted(neighbors, key=lambda t: t[1], reverse=True)[:3]
        for idx, sim, h in neighbors:
            if idx is None or idx < 0 or idx >= len(labeled):
                continue
            row = labeled.loc[idx]
            txt = row["text"]
            tkt = row["ticket"]
            src = row.get("source", "?")
            examples.append({
                "ticket": tkt,
                "texto": (txt[:500] + ("..." if len(txt) > 500 else "")),
                "horas": h,
                "sim": round(sim, 3),
                "source": src,
            })

    return render_template(
        "index.html",
        form=form,
        estimate=estimate,
        estimate_top1=estimate_top1,
        neighbors=neighbors,
        examples=examples,
        debug_info=debug_info
    )


@app.route("/retrain", methods=["POST"])
def retrain():
    try:
        counts = train_index_per_type()
        msg = (f"Reentrenado. Desarrollo={counts.get('desarrollo',0)}, "
               f"Implementación={counts.get('implementacion',0)}")
        flash(msg, "ok")
    except Exception as e:
        flash(f"Error al reentrenar: {e}", "err")
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=True)


# --- Minimal upload endpoint to match template's url_for('upload_route') ---
try:
    if "upload_route" not in app.view_functions:
        from werkzeug.utils import secure_filename
        from flask import request, redirect, url_for, flash
        from pathlib import Path as _Path
        import os as _os

        @app.route("/upload", methods=["POST"], endpoint="upload_route")
        def upload_route():
            try:
                file = request.files.get("file")
                if file is None or file.filename == "":
                    flash("Selecciona un CSV para subir.", "err")
                    return redirect(url_for("index"))
                fname = secure_filename(file.filename)
                base_dir = _Path(_os.environ.get("DATA_DIR", str(_Path(__file__).parent / "data")))
                upload_dir = base_dir / "uploads"
                upload_dir.mkdir(parents=True, exist_ok=True)
                dest = upload_dir / fname
                file.save(dest)
                flash(f"CSV subido: {fname}", "ok")
            except Exception as e:
                flash(f"Error al subir CSV: {e}", "err")
            return redirect(url_for("index"))
except Exception:
    pass
