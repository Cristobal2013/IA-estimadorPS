
# hfmt_filter.py — registers 'hfmt' Jinja filter
def _hfmt(value):
    try:
        v = float(value)
        if abs(v - round(v)) < 1e-9:
            return str(int(round(v)))
        return f"{v:.2f}".rstrip("0").rstrip(".")
    except Exception:
        return value

def register_jinja_filters(app):
    # Force-register the filter even if templates are compiled early
    app.add_template_filter(_hfmt, name="hfmt")
