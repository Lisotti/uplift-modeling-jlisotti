"""Dashboard de simulación de campañas, genérico por dataset (Objetivo 7).

Ejecutar:
    make dashboard      # equivale a: streamlit run dashboard/app.py

Muestra cualquier dataset de ``configs/`` que tenga un modelo entrenado
(``make train DATASET=<nombre>``). Todo el contenido —outcomes, tratamientos,
supuestos de negocio, formulario de cliente y segmentos— sale de la config.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uplift.data import load_split, treatment_index  # noqa: E402
from uplift.dataset import available_configs, load_config  # noqa: E402
from uplift.features import add_segments, quantile_bins  # noqa: E402
from uplift.learners import LEARNER_LABELS  # noqa: E402
from uplift.policy import assign, evaluate_action, value_matrix  # noqa: E402
from uplift.predict import load_bundle, score  # noqa: E402

st.set_page_config(page_title="Uplift — Simulador de campañas", page_icon="📧", layout="wide")

TEAL, GOLD, BLUE, CORAL, GRAY = "#00897B", "#C08A00", "#3D6CC9", "#D9472B", "#8A8A8A"
ARM_COLORS = [TEAL, GOLD, BLUE, CORAL]


def money(v, dec=0):
    # "$" se escapa: Streamlit interpreta $…$ como LaTeX en captions/markdown
    return f"US\\$ {v:,.{dec}f}"


@st.cache_resource
def get_bundle(name):
    return load_bundle(name)


@st.cache_data
def get_test(name):
    return load_split("test", load_config(name))


@st.cache_data
def get_test_uplifts(name):
    """Uplift en test del learner seleccionado por outcome (precalculado al entrenar)."""
    cfg = load_config(name)
    sel = get_bundle(name)["selected"]
    npz = np.load(cfg.models_dir / "test_uplift.npz")
    return {o: npz[f"{o}__{sel[o]}"].astype(float) for o in cfg.outcome_names}


@st.cache_data(show_spinner="Calculando curva de presupuesto…")
def budget_curve(name, objective, values_items, cost):
    cfg = load_config(name)
    test = get_test(name)
    t = treatment_index(test[cfg.treatment_col], cfg)
    V = value_matrix(get_test_uplifts(name), cfg, objective, dict(values_items), cost)
    return pd.DataFrame([{"budget": b, **evaluate_action(assign(V, budget=b), test, t, cfg, dict(values_items), cost)}
                         for b in np.linspace(0, 1, 21)])


trained = [d for d in available_configs() if load_config(d).bundle_path.exists()]
if not trained:
    st.error("No hay modelos entrenados. Ejecutá `make train DATASET=<nombre>` y recargá la página.")
    st.stop()

# ── Sidebar ────────────────────────────────────────────────────────
st.sidebar.title("Parámetros de la campaña")
name = st.sidebar.selectbox("Dataset / experimento", trained, format_func=lambda d: load_config(d).title,
                            index=trained.index("hillstrom") if "hillstrom" in trained else 0)
cfg = load_config(name)
bundle = get_bundle(name)
test = get_test(name)
t = treatment_index(test[cfg.treatment_col], cfg)
UP = get_test_uplifts(name)
B = cfg.business

obj_labels = {"profit": "Beneficio (valor × incremental − costo)"} if B["values"] else {}
obj_labels.update({o.name: f"{o.label} incremental" for o in cfg.outcomes})
objective = st.sidebar.selectbox("Objetivo de optimización", list(obj_labels), format_func=obj_labels.get)
base_size = st.sidebar.number_input("Tamaño de la base de clientes", 1_000, 100_000_000,
                                    int(B["default_base_size"]), 1_000)
budget = st.sidebar.slider("Presupuesto: % máximo de la base a contactar", 0, 100, 40, 5) / 100
cost = st.sidebar.number_input(B["cost_label"], 0.0, 100.0, float(B["cost_per_contact"]), 0.001, format="%.3f")
values = {}
for o, v in B["values"].items():
    if v.get("max", 1) <= 1:
        values[o] = st.sidebar.slider(v["label"], float(v.get("min", 0.0)), float(v["max"]),
                                      float(v["default"]), float(v.get("step", 0.05)))
    else:
        values[o] = st.sidebar.number_input(v["label"], float(v.get("min", 0.0)), float(v["max"]),
                                            float(v["default"]), float(v.get("step", 1.0)))
st.sidebar.caption("Modelos: " + ", ".join(f"{o}: {k}" for o, k in bundle["selected"].items())
                   + f" · entrenado {bundle['trained_at'][:10]}")

V = value_matrix(UP, cfg, objective, values, cost)
scale = base_size / 1000

st.title("📧 Simulador de campañas con Uplift Modeling")
st.caption(f"**{cfg.title}.** Estimaciones fuera de muestra sobre el set de test ({len(test):,} clientes) "
           "mediante Inverse Propensity Weighting. Valores incrementales frente a no contactar.")

tab1, tab2, tab3, tab4 = st.tabs(["Simulador", "Segmentos", "Cliente individual", "Modelo y datos"])

# ── Tab 1: simulador ──────────────────────────────────────────────
with tab1:
    rng = np.random.default_rng(0)
    n = len(t)
    rand = np.zeros(n, dtype=int)
    sel = rng.choice(n, int(budget * n), replace=False)
    rand[sel] = rng.integers(1, cfg.n_treatments + 1, len(sel))
    policies = {"Modelo de uplift": assign(V, budget=budget), "Envío aleatorio (mismo presupuesto)": rand}
    for k, arm in enumerate(cfg.treatments, start=1):
        policies[f"Contactar a todos — {arm.name}"] = np.full(n, k)
    res = {k: evaluate_action(a, test, t, cfg, values, cost) for k, a in policies.items()}
    m = res["Modelo de uplift"]

    cols = st.columns(1 + len(cfg.outcomes) + (1 if B["values"] else 0))
    cols[0].metric("Clientes contactados", f"{m['share_contacted'] * base_size:,.0f}",
                   f"{m['share_contacted']:.0%} de la base")
    for c, o in zip(cols[1:], cfg.outcomes):
        is_money = not o.binary and "US$" in o.label
        val = m[f"inc_{o.name}_per_1000"] * scale
        lo, hi = m[f"inc_{o.name}_lo"] * scale, m[f"inc_{o.name}_hi"] * scale
        c.metric(f"{o.label.split(' (')[0]} incremental", f"US$ {val:,.0f}" if is_money else f"{val:,.0f}")
        c.caption(f"IC 95 %: {money(lo)} a {money(hi)}" if is_money else f"IC 95 %: {lo:,.1f} a {hi:,.1f}")
    if B["values"]:
        cols[-1].metric("Beneficio neto", f"US$ {m['profit_per_1000'] * scale:,.0f}",
                        f"ROI {m['roi']:.1f}x" if m["cost_per_1000"] > 0 else None)
        cols[-1].caption(f"IC 95 %: {money(m['profit_lo'] * scale)} a {money(m['profit_hi'] * scale)}")

    a_model = policies["Modelo de uplift"]
    n_events = int(((t == a_model) & (a_model > 0) & (test[cfg.stratify_outcome] > 0)).sum())
    st.caption(
        f"ℹ️ Las cifras se estiman con IPW: sólo cuentan los clientes que en el experimento recibieron el "
        f"tratamiento que el modelo les asigna. Con este presupuesto la estimación de "
        f"{cfg.outcome(cfg.stratify_outcome).label.lower()} se apoya en **{n_events:,} eventos**; con pocos "
        "eventos el intervalo de confianza es amplio y las diferencias que caen dentro del intervalo "
        "(por ejemplo, entre dos presupuestos) no son estadísticamente significativas."
    )

    comp = pd.DataFrame(res).T
    disp = {"contactados": (comp.share_contacted * base_size).round(0)}
    for o in cfg.outcomes:
        d = 1 if comp[f"inc_{o.name}_per_1000"].abs().max() * scale < 1000 else 0
        disp[f"{o.label} inc."] = (comp[f"inc_{o.name}_per_1000"] * scale).round(d)
        disp[f"IC 95 % {o.label.lower()}"] = [f"{lo * scale:,.{d}f} a {hi * scale:,.{d}f}"
                                               for lo, hi in zip(comp[f"inc_{o.name}_lo"], comp[f"inc_{o.name}_hi"])]
    disp["costo (US$)"] = (comp.cost_per_1000 * scale).round(0)
    if B["values"]:
        disp["beneficio (US$)"] = (comp.profit_per_1000 * scale).round(0)
        disp["IC 95 % beneficio"] = [f"{lo * scale:,.0f} a {hi * scale:,.0f}"
                                     for lo, hi in zip(comp.profit_lo, comp.profit_hi)]
    st.subheader("Comparación de políticas")
    st.dataframe(pd.DataFrame(disp, index=comp.index), width="stretch")

    st.subheader("Curva de presupuesto")
    curve = budget_curve(name, objective, tuple(values.items()), cost)
    metric_opts = (["profit"] if B["values"] else []) + cfg.outcome_names
    metric_lbl = {"profit": "Beneficio", **{o.name: o.label for o in cfg.outcomes}}
    metric = st.radio("Métrica", metric_opts, horizontal=True, format_func=metric_lbl.get)
    key, lo_k, hi_k = (("profit_per_1000", "profit_lo", "profit_hi") if metric == "profit"
                       else (f"inc_{metric}_per_1000", f"inc_{metric}_lo", f"inc_{metric}_hi"))
    fig = go.Figure()
    x = curve.budget * 100
    fig.add_trace(go.Scatter(x=pd.concat([x, x[::-1]]), y=pd.concat([curve[hi_k], curve[lo_k][::-1]]) * scale,
                             fill="toself", fillcolor="rgba(0,137,123,0.18)", line=dict(width=0),
                             hoverinfo="skip", name="IC 95 % del modelo"))
    fig.add_trace(go.Scatter(x=x, y=curve[key] * scale, mode="lines+markers", name="Modelo",
                             line=dict(color=TEAL, width=3)))
    dashes = ["dash", "dot", "dashdot"]
    for k, arm in enumerate(cfg.treatments, start=1):
        full = res[f"Contactar a todos — {arm.name}"][key]
        fig.add_trace(go.Scatter(x=[0, 100], y=[0, full * scale], mode="lines", name=f"Aleatorio ({arm.name})",
                                 line=dict(color=GRAY if k == 1 else GOLD, dash=dashes[(k - 1) % 3])))
    fig.add_vline(x=budget * 100, line_color=CORAL)
    fig.update_layout(xaxis_title="% de la base contactada", yaxis_title="Incremental", height=420,
                      template="plotly_white", legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, width="stretch")
    st.caption("La banda sombreada es el intervalo de confianza al 95 %. Las subas y bajas de la curva "
               "dentro de la banda son ruido de estimación, no efectos reales del presupuesto.")
    best = curve.loc[curve[key].idxmax()]
    unit = money if metric == "profit" else (lambda v: f"{v:,.0f}")
    st.info(f"El máximo de {metric_lbl[metric].lower()} se alcanza contactando al **{best.budget:.0%}** de la "
            f"base ({unit(best[key] * scale)} para {base_size:,} clientes; IC 95 %: {unit(best[lo_k] * scale)} a "
            f"{unit(best[hi_k] * scale)}). Conviene leerlo como una región de presupuestos razonables, "
            "no como un punto exacto.")

# ── Tab 2: segmentos ──────────────────────────────────────────────
with tab2:
    df = add_segments(test[cfg.raw_features], cfg)
    seg_opts = [*cfg.segments.keys(), *cfg.raw_features]
    c1, c2 = st.columns(2)
    seg = c1.selectbox("Segmentar por", seg_opts)
    out_name = c2.selectbox("Outcome", cfg.outcome_names, format_func=lambda o: cfg.outcome(o).label)
    o = cfg.outcome(out_name)
    bins, order = quantile_bins(df[seg])
    if seg in cfg.segments:
        order = [x for x in cfg.segments[seg] if x in set(order)]
    mult = 100 if o.binary else 1
    figs = go.Figure()
    for k, arm in enumerate(cfg.treatments):
        means = pd.Series(mult * UP[out_name][:, k], index=df.index).groupby(bins).mean().reindex(order)
        figs.add_trace(go.Bar(x=order, y=means.values, name=arm.name, marker_color=ARM_COLORS[k % 4]))
    figs.update_layout(barmode="group", template="plotly_white", height=380,
                       title=f"CATE medio en {o.label.lower()} ({'pp' if o.binary else 'unidades'}) por {seg}",
                       legend=dict(orientation="h", y=1.12))
    st.plotly_chart(figs, width="stretch")
    names = {0: "No contactar", **{k + 1: a.name for k, a in enumerate(cfg.treatments)}}
    action = pd.Series(assign(V, budget=budget)).map(names)
    mix = pd.crosstab(bins.values, action.values, normalize="index").reindex(order) * 100
    figm = go.Figure()
    for col in mix.columns:
        color = GRAY if col == "No contactar" else ARM_COLORS[[a.name for a in cfg.treatments].index(col) % 4]
        figm.add_trace(go.Bar(x=mix.index, y=mix[col], name=col, marker_color=color))
    figm.update_layout(barmode="stack", template="plotly_white", height=360, yaxis_title="% del segmento",
                       title=f"Acción recomendada por {seg} (presupuesto {budget:.0%})")
    st.plotly_chart(figm, width="stretch")

# ── Tab 3: cliente individual ─────────────────────────────────────
with tab3:
    ranges = cfg.validation.get("ranges") or {}
    ref = test[cfg.raw_features]
    with st.form("cliente"):
        inputs = {}
        specs = [s for s in cfg.feature_specs if s["type"] != "interaction"]
        fcols = st.columns(3)
        for i, s in enumerate(specs):
            c = fcols[i % 3]
            nm, kind = s["name"], s["type"]
            if kind == "binary":
                inputs[nm] = int(c.checkbox(nm, bool(ref[nm].median())))
            elif kind == "categorical":
                inputs[nm] = c.selectbox(nm, s["categories"])
            else:
                lo, hi = ranges.get(nm, [None, None])
                lo = float(ref[nm].min()) if lo is None else float(lo)
                hi = float(ref[nm].max()) if hi is None else float(hi)
                inputs[nm] = c.number_input(nm, lo, hi, float(ref[nm].median()))
        ok = st.form_submit_button("Calcular uplift")
    if ok:
        r = score(pd.DataFrame([inputs]), cfg, bundle, objective, values, cost).iloc[0]
        rec = {a.label: f"📨 Contactar con {a.name}" for a in cfg.treatments}
        rec[cfg.none_action] = "🚫 No contactar"
        st.success(f"Recomendación: **{rec[r.recommended_treatment]}**")
        tbl = pd.DataFrame({a.name: [r[f"uplift_{oo.name}_{a.label}"] * (100 if oo.binary else 1)
                                     for oo in cfg.outcomes] for a in cfg.treatments},
                           index=[f"{oo.label} ({'pp' if oo.binary else 'unidades'})" for oo in cfg.outcomes])
        st.dataframe(tbl.round(3), width="stretch")

# ── Tab 4: modelo y datos ─────────────────────────────────────────
with tab4:
    st.write(f"**{cfg.title}** — {cfg.description}")
    nr = bundle.get("n_rows") or {}
    if nr:
        st.caption(f"Filas: {nr['total']:,} en el dataset · train {nr['train']:,} · test {nr['test']:,}")
    eda_p = cfg.reports_dir / "eda_summary.json"
    if eda_p.exists():
        eda = json.load(open(eda_p))
        st.subheader("Viabilidad del experimento (EDA)")
        st.dataframe(pd.DataFrame(eda["viability"]), width="stretch", hide_index=True)
    st.subheader("Learner seleccionado y métricas en test")
    rows = []
    for oname, mt in bundle["test_metrics"].items():
        for a in cfg.treatments:
            rows.append({"outcome": oname, "learner": LEARNER_LABELS[bundle["selected"][oname]], "tratamiento": a.name,
                         **{k: v for k, v in mt[a.label].items() if not isinstance(v, list)}})
    st.dataframe(pd.DataFrame(rows).round(4), width="stretch", hide_index=True)
    mon = cfg.monitoring_dir / "monitoring_summary.csv"
    if mon.exists():
        st.subheader("Monitoreo de drift (ventanas simuladas)")
        st.dataframe(pd.read_csv(mon).round(4), width="stretch", hide_index=True)
