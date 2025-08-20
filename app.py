# app.py — orden de import seguro + app expuesta a nivel módulo

from flask import Flask, render_template, request, redirect, url_for, flash
from pathlib import Path
import pandas as pd
import datetime as dt
import re, math, os, json

# === 1) Crear y EXPONER la app ANTES de cualquier otra cosa ===
app = Flask(__name__)
application = app  # alias por si algún runner usa application

# === 2) Filtro hfmt con fallback (por si falta hfmt_filter.py) ===
try:
    from hfmt_filter import register_jinja_filters, _hfmt
except Exception:
    def _hfmt(v):
        try:
            x = float(v)
            if x != x:  # NaN
                return "—"
            return f"{round(x, 2):g}"
        except Exception:
            return "—"
    def register_jinja_filters(flask_app):
        flask_app.add_template_filter(_hfmt, name="hfmt")

# Registrar filtro (idempotente)
register_jinja_filters(app)
app.add_template_filter(_hfmt, name="hfmt")

@app.before_request
def _ensure_hfmt():
    if "hfmt" not in app.jinja_env.filters:
        app.add_template_filter(_hfmt, name="hfmt")

# === 3) Recién ahora importa lo pesado que usa la app ===
from estimator import (
    EmbeddingsFaissEstimator,
    train_index_per_type,
    load_labeled_dataframe,
)

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
        t = str(labeled_df.loc[idx, "text"])
        n = _extract_field_count(t)
        if n > 0:
            try:
                numer += float(h); denom += float(n)
            except Exception:
                pass
    if denom > 0:
        return numer / denom
    return 0.35  # fallback


def _estimate_with_softmax(neighbors, temperature=0.20):
    if not neighbors:
        return 0.0, []
    sims = [max(0.0, float(s)) for _, s, _ in neighbors]
    exps = [math.exp(s / temperature) for s in sims]
    Z = sum(exps) or 1.0
    weights = [e / Z for e in exps]
    hours = [float(h) for _, _, h in neighbors]
    estimate = sum(w * h for w, h in zip(weights, hours))
    return estimate, weights


def _filter_neighbors_by_source(neighbors, labeled_df, origen):
    if origen in (None, "", "todos"):
        return neighbors
    keep = []
    for idx, sim, h in neighbors:
        if idx is None or idx < 0 or idx >= len(labeled_df):
            continue
        src = str(labeled_df.loc[idx].get("source", "unknown"))
        if src == origen:
            keep.append((idx, sim, h))
    return keep


def _status_obj():
    try:
        import faiss  # noqa
        from sentence_transformers import SentenceTransformer  # noqa
        emb_enabled = True
    except Exception:
        emb_enabled = False
    kb_count = 0
    try:
        kb_count += len(load_labeled_dataframe("desarrollo"))
    except Exception:
        pass
    try:
        kb_count += len(load_labeled_dataframe("implementacion"))
    except Exception:
        pass
    return type("Obj", (), {"emb_enabled": emb_enabled, "kb_count": kb_count})


# -------------------------
# Entrenamiento al iniciar (opcional completo si TRAIN_FULL_ON_BOOT=1)
# -------------------------
try:
    boot_full = os.environ.get("TRAIN_FULL_ON_BOOT", "0") == "1"
    counts = train_index_per_type(full=boot_full)
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

    form = {
        "tipo": "desarrollo",
        "texto": "",
        "metodo": "softmax",
        "origen": "todos",
    }

    if request.method == "POST":
        accion = request.form.get("accion", "estimar")
        form["tipo"] = request.form.get("tipo", "desarrollo")
        form["texto"] = request.form.get("texto", "")
        form["metodo"] = request.form.get("metodo", "softmax")
        form["origen"] = request.form.get("origen", "todos")

        tag = "desarrollo" if form["tipo"] == "desarrollo" else "implementacion"

        est = EmbeddingsFaissEstimator(tag)
        loaded = False
        try:
            loaded = est.load()
        except Exception:
            loaded = False
        if not loaded:
            try:
                train_index_per_type(full=True)  # fuerza histórico+catálogo+nuevos
                est.load()
            except Exception as e2:
                return (f"No se pudo cargar/entrenar el índice [{tag}]: {e2}", 500)

        if accion == "guardar":
            metodo_usado = request.form.get("metodo_usado", form["metodo"])
            estimacion_ia = request.form.get("estimacion_ia")
            estimacion_top1 = request.form.get("estimacion_top1")
            estimacion_final = request.form.get("estimacion_final")
            horas_reales = request.form.get("horas_reales")

            if not estimacion_final:
                if metodo_usado == "top1" and estimacion_top1:
                    estimacion_final = estimacion_top1
                elif metodo_usado == "softmax" and estimacion_ia:
                    estimacion_final = estimacion_ia

            def to_float(x):
                try:
                    return float(x) if x not in (None, "", "-", "—") else None
                except Exception:
                    return None

            out = Path("data/estimaciones_nuevas.csv")
            out.parent.mkdir(parents=True, exist_ok=True)
            row = {
                "fecha": dt.datetime.now().isoformat(timespec="seconds"),
                "tipo": form["tipo"],
                "texto": form["texto"],
                "metodo_usado": metodo_usado,
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
        est_soft_unused, neighbors_all = est.predict(form["texto"], k=30)
        labeled = load_labeled_dataframe(tag).reset_index(drop=True)

        counts_all = {"historic": 0, "catalog": 0, "new": 0, "unknown": 0}
        rows_all = []
        for (i, s, h) in neighbors_all:
            if i is None or i < 0 or i >= len(labeled):
                continue
            src_i = str(labeled.loc[i].get("source", "unknown"))
            counts_all[src_i] = counts_all.get(src_i, 0) + 1
            rows_all.append({
                "idx": int(i),
                "sim": float(round(s, 4)),
                "hours": float(round(h, 4)),
                "ticket": str(labeled.loc[i].get("ticket", "?")),
                "source": src_i,
                "text": str(labeled.loc[i].get("text", ""))[:180]
            })

        neighbors = _filter_neighbors_by_source(neighbors_all, labeled, form.get("origen", "todos"))
        debug_info = {
            "tipo": tag,
            "texto": form.get("texto", ""),
            "origen": form.get("origen", "todos"),
            "k_all": len(neighbors_all),
            "k_used": len(neighbors),
            "counts_all": counts_all,
            "neighbors_all": rows_all[:12],
        }

        if neighbors:
            if form.get("metodo") == "top1" and form.get("origen", "todos") == "todos":
                best_idx, best_sim, best_h = max(neighbors, key=lambda t: t[1])
                hist_candidates = [(i, s, h) for (i, s, h) in neighbors
                                   if str(labeled.loc[i].get("source", "?")) == "historic"]
                if hist_candidates:
                    h_idx, h_sim, h_h = max(hist_candidates, key=lambda t: t[1])
                    estimate_top1 = h_h if (best_sim - h_sim) <= 0.02 else best_h
                else:
                    estimate_top1 = best_h
            else:
                estimate_top1 = sorted(neighbors, key=lambda t: t[1], reverse=True)[0][2]
        else:
            estimate_top1 = None

        if form["metodo"] == "top1" and neighbors:
            estimate = estimate_top1
        else:
            est_soft, _ = _estimate_with_softmax(neighbors, temperature=0.20)
            estimate = est_soft

        N = _extract_field_count(form["texto"])
        if N > 0:
            cpf = _neighbor_based_cost_per_field(neighbors, labeled)
            alpha = 0.3
            base = alpha * float(estimate or 0.0) + (1.0 - alpha) * (cpf * N)
            min_per_field = 0.25
            estimate = max(base, N * min_per_field)

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

    ctx = dict(
        form=form,
        estimate=estimate,
        estimate_top1=estimate_top1,
        neighbors=neighbors,
        examples=examples,
        debug_info=debug_info,
        status=_status_obj(),
        kb_hits=examples,
        top_k=len(examples) or 0,
        estimate_catalog=0.0,
        estimate_semantic=None,
        w_catalog=0.5,
        w_semantic=0.5,
    )
    return render_template("index.html", **ctx)


@app.route("/retrain", methods=["POST"])
def retrain():
    try:
        counts = train_index_per_type(full=True)
        msg = (f"Reentrenado. Desarrollo={counts.get('desarrollo',0)}, "
               f"Implementación={counts.get('implementacion',0)}")
        flash(msg, "ok")
    except Exception as e:
        flash(f"Error al reentrenar: {e}", "err")
    return redirect(url_for("index"))


@app.get("/debug")
def debug_route():
    out = []
    for tag in ["desarrollo", "implementacion"]:
        try:
            df = load_labeled_dataframe(tag)
            counts = df["source"].value_counts(dropna=False).to_dict() if not df.empty else {}
            samples = {}
            for src in ["historic", "catalog", "new"]:
                s = df[df["source"] == src][["ticket", "hours", "text"]].head(2)
                samples[src] = s.to_dict(orient="records")
            out.append({"tag": tag, "n": len(df), "counts": counts, "samples": samples})
        except Exception as e:
            out.append({"tag": tag, "error": str(e)})
    return app.response_class(
        response=json.dumps(out, ensure_ascii=False, indent=2),
        mimetype="application/json"
    )


if __name__ == "__main__":
    # Local: python app.py ; En Render usamos gunicorn wsgi:app
    app.run(host="0.0.0.0", port=7860, debug=True)
