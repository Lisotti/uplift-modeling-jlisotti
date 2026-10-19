"""Figuras de resultados de un dataset (``python -m uplift.figures --dataset X``).

Requiere haber corrido ``uplift.train`` (y ``uplift.monitoring`` para la figura de
monitoreo). Las figuras se guardan en ``artifacts/<dataset>/reports/figures/``;
las del EDA las genera ``uplift.eda``.
"""

from __future__ import annotations

import argparse
import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

from .data import load_split, treatment_index
from .dataset import DatasetConfig, load_config
from .features import FeatureBuilder, add_segments, quantile_bins
from .learners import LEARNER_LABELS
from .metrics import qini_curve, uplift_by_decile
from .plotting import BLUE, CORAL, GOLD, GRAY, INK, LIGHT, MUTED, TEAL, arm_colors, save

warnings.filterwarnings("ignore")


def _load(cfg: DatasetConfig):
    test = load_split("test", cfg)
    npz = np.load(cfg.models_dir / "test_uplift.npz")
    up = {}
    for key in npz.files:
        o, k = key.split("__")
        up.setdefault(o, {})[k] = npz[key].astype(float)
    metrics = json.load(open(cfg.reports_dir / "metrics.json"))
    return test, up, metrics


def _ranking_outcome(cfg: DatasetConfig) -> str:
    """Outcome cuyo modelo define la política de negocio (el valorizado, si existe)."""
    vals = list(cfg.business["values"])
    return vals[0] if vals else cfg.primary_outcome


def fig_cv(cfg, metrics):
    cv = pd.DataFrame(metrics["cv_summary"])
    outs = cfg.outcomes
    fig, axes = plt.subplots(1, len(outs), figsize=(3.7 * len(outs), 3.6))
    axes = np.atleast_1d(axes)
    reps = cfg.modeling["cv_folds"] * cfg.modeling["cv_repeats"]
    for ax, o in zip(axes, outs):
        d = cv[cv.outcome == o.name].sort_values("mean")
        sel = metrics["selected"][o.name]
        cols = [CORAL if k == "baseline" else (TEAL if k == sel else LIGHT) for k in d.learner]
        ax.barh([LEARNER_LABELS[k] for k in d.learner], d["mean"], xerr=1.96 * d["sem"], color=cols,
                height=0.6, error_kw=dict(ecolor=MUTED, lw=0.8, capsize=2))
        ax.axvline(0, color=INK, lw=0.8)
        ax.set_title(o.label)
        ax.set_xlabel(f"Qini relativo (media ± IC 95 %, {reps} folds)")
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", labelsize=8)
    fig.suptitle("Validación cruzada sobre train — teal: seleccionado · coral: baseline",
                 fontweight="bold", color=INK, y=1.02)
    fig.tight_layout()
    save(fig, cfg.figures_dir / "cv_model_selection.png")


def fig_qini(cfg, test, up, metrics):
    t = treatment_index(test[cfg.treatment_col], cfg)
    nr, nc = len(cfg.outcomes), cfg.n_treatments
    fig, axes = plt.subplots(nr, nc, figsize=(5 * nc, 3.5 * nr), squeeze=False)
    for i, o in enumerate(cfg.outcomes):
        y = test[o.name].to_numpy(float)
        sel = metrics["selected"][o.name]
        for j, arm in enumerate(cfg.treatments):
            k = j + 1
            ax = axes[i, j]
            m = (t == 0) | (t == k)
            w = (t[m] == k).astype(int)
            for key, u in up[o.name].items():
                if key in (sel, "baseline"):
                    continue
                x, q = qini_curve(y[m], w, u[m, k - 1])
                ax.plot(x / x[-1] * 100, q, color=LIGHT, lw=1)
            for key, col, lab in [("baseline", CORAL, "Baseline (response)"), (sel, TEAL, LEARNER_LABELS[sel] + " ★")]:
                x, q = qini_curve(y[m], w, up[o.name][key][m, k - 1])
                ax.plot(x / x[-1] * 100, q, color=col, lw=2.2, label=lab)
            ax.plot([0, 100], [0, q[-1]], color=GRAY, ls="--", lw=1.2, label="Aleatorio")
            ax.plot([], [], color=LIGHT, lw=1, label="Otros meta-learners")
            ax.set_title(f"{o.label} · {arm.name}")
            ax.set_xlabel("% de clientes contactados (ordenados por score)")
            ax.set_ylabel("Incremental acumulado (Qini)")
            if j == 0:
                ax.legend(loc="upper left", fontsize=7.5)
    fig.tight_layout()
    save(fig, cfg.figures_dir / "qini_curves.png")


def fig_deciles(cfg, test, up, metrics):
    t = treatment_index(test[cfg.treatment_col], cfg)
    colors = arm_colors(cfg)
    outs = [o for o in cfg.outcomes if o.binary]
    nr, nc = len(outs), cfg.n_treatments
    fig, axes = plt.subplots(nr, nc, figsize=(5 * nc, 3.2 * nr), squeeze=False)
    for i, o in enumerate(outs):
        sel = metrics["selected"][o.name]
        for j, arm in enumerate(cfg.treatments):
            k = j + 1
            ax = axes[i, j]
            m = (t == 0) | (t == k)
            d = uplift_by_decile(test[o.name].to_numpy()[m], (t[m] == k).astype(int), up[o.name][sel][m, k - 1])
            avg = 100 * (test[o.name][t == k].mean() - test[o.name][t == 0].mean())
            cols = [colors[arm.label] if v >= 0 else CORAL for v in d.uplift]
            ax.bar(d.decile, 100 * d.uplift, color=cols, width=0.7)
            ax.axhline(avg, color=INK, ls="--", lw=1)
            ax.text(10.4, avg, f" promedio\n {avg:.2f} pp", va="center", fontsize=8, color=INK)
            ax.set_xticks(range(1, 11))
            ax.set_title(f"{o.label} · {arm.name} ({LEARNER_LABELS[sel]})")
            ax.set_xlabel("Decil de uplift predicho (1 = mayor)")
            ax.set_ylabel("Uplift observado (pp)")
            ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, cfg.figures_dir / "uplift_by_decile.png")


def fig_budget(cfg):
    has_value = bool(cfg.business["values"])
    left = "profit_per_1000" if has_value else f"inc_{cfg.primary_outcome}_per_1000"
    right = f"inc_{cfg.stratify_outcome}_per_1000"
    palette = [BLUE, GOLD, TEAL, CORAL]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    rank = _ranking_outcome(cfg)
    bench = pd.read_csv(cfg.tables_dir / f"policy_benchmark_{rank}.csv").set_index("policy")
    for i, o in enumerate(cfg.outcomes):
        bc = pd.read_csv(cfg.tables_dir / f"budget_curve_{o.name}.csv")
        lab = f"Ranking por uplift en {o.label.split(' ')[0].lower()}"
        for ax, col in ((axes[0], left), (axes[1], right)):
            ax.plot(bc.budget * 100, bc[col], color=palette[i % 4], marker="o", ms=3.5, label=lab)
    styles = ["--", ":", "-."]
    for ax, col in ((axes[0], left), (axes[1], right)):
        for j, arm in enumerate(cfg.treatments):
            v = bench.loc[f"Enviar a todos — {arm.name}", col]
            ax.plot([0, 100], [0, v], color=GRAY, ls=styles[j % 3], lw=1.2, label=f"Aleatorio ({arm.name})")
        ax.set_xlabel("% de la base contactada (presupuesto)")
    axes[0].set_ylabel("US$ por cada 1.000 clientes" if has_value else "Por cada 1.000 clientes")
    axes[0].set_title("Beneficio neto incremental" if has_value else f"{cfg.outcome(cfg.primary_outcome).label} incremental")
    axes[1].set_ylabel("Por cada 1.000 clientes")
    axes[1].set_title(f"{cfg.outcome(cfg.stratify_outcome).label} incremental")
    axes[0].legend(fontsize=7.8, loc="upper left")
    fig.tight_layout()
    save(fig, cfg.figures_dir / "budget_curves.png")


def fig_policy_benchmark(cfg):
    rank = _ranking_outcome(cfg)
    has_value = bool(cfg.business["values"])
    col = "profit_per_1000" if has_value else f"inc_{rank}_per_1000"
    bp = pd.read_csv(cfg.tables_dir / f"policy_benchmark_{rank}.csv")
    bp = bp[bp.policy != "No enviar"]
    bc = pd.read_csv(cfg.tables_dir / f"budget_curve_{rank}.csv")
    best = bc.loc[bc[col].idxmax()]
    bp = pd.concat([bp, pd.DataFrame([{"policy": f"Modelo (óptimo, {best.budget:.0%})", col: best[col],
                                       "share_contacted": best.share_contacted}])], ignore_index=True)
    bp = bp.iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 0.45 * len(bp) + 0.8))
    cols = [TEAL if p.startswith("Modelo") else LIGHT for p in bp.policy]
    bars = ax.barh(bp.policy, bp[col], color=cols, height=0.6)
    pre = "US$ " if has_value else ""
    for b, v, s in zip(bars, bp[col], bp.share_contacted):
        ax.text(max(v, 0) + abs(bp[col]).max() * 0.02, b.get_y() + b.get_height() / 2,
                f"{pre}{v:,.1f}  ({s:.0%} contactado)", va="center", fontsize=8.5, color=INK)
    ax.set_xlabel(("Beneficio neto incremental" if has_value else f"{cfg.outcome(rank).label} incremental")
                  + " por 1.000 clientes" + (" (US$)" if has_value else ""))
    ax.set_title("Comparación de políticas (test, IPW)")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(min(0, bp[col].min() * 1.1), bp[col].max() * 1.5)
    save(fig, cfg.figures_dir / "policy_benchmark.png")


def fig_cate_distribution(cfg, up, metrics):
    colors = arm_colors(cfg)
    outs = cfg.outcomes
    fig, axes = plt.subplots(1, len(outs), figsize=(3.7 * len(outs), 3.1))
    axes = np.atleast_1d(axes)
    for ax, o in zip(axes, outs):
        sel = metrics["selected"][o.name]
        mult = 100 if o.binary else 1
        for k, arm in enumerate(cfg.treatments):
            v = mult * up[o.name][sel][:, k]
            lo, hi = np.percentile(v, [0.5, 99.5])
            ax.hist(np.clip(v, lo, hi), bins=50, color=colors[arm.label], alpha=0.65, label=arm.name)
        ax.axvline(0, color=INK, lw=0.8)
        ax.set_title(f"{o.label} — {LEARNER_LABELS[sel]}")
        ax.set_xlabel("CATE predicho" + (" (pp)" if o.binary else ""))
        ax.grid(axis="x", visible=False)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    save(fig, cfg.figures_dir / "cate_distribution.png")


def _surrogate(cfg, test, up, metrics, max_rows=200_000):
    from sklearn.tree import DecisionTreeRegressor

    o = cfg.primary_outcome
    sel = metrics["selected"][o]
    u = up[o][sel]
    k = int(np.argmax(u.std(axis=0)))  # tratamiento con efecto más heterogéneo
    X = FeatureBuilder.from_config(cfg).transform(test)
    X = X.drop(columns=[c for c in X.columns if c.startswith("log_")])
    y = (100 if cfg.outcome(o).binary else 1) * u[:, k]
    if len(X) > max_rows:
        idx = np.random.default_rng(0).choice(len(X), max_rows, replace=False)
        X, y = X.iloc[idx], y[idx]
    leaf = max(200, len(X) // 25)
    tree = DecisionTreeRegressor(max_depth=3, min_samples_leaf=leaf, random_state=0).fit(X, y)
    return tree, X, y, k


def fig_surrogate(cfg, test, up, metrics):
    from sklearn.tree import plot_tree

    tree, X, y, k = _surrogate(cfg, test, up, metrics)
    r2 = tree.score(X, y)
    arm = cfg.treatments[k]
    o = cfg.outcome(cfg.primary_outcome)
    fig, ax = plt.subplots(figsize=(12, 5))
    plot_tree(tree, feature_names=list(X.columns), filled=True, rounded=True, impurity=False,
              precision=2, fontsize=8, ax=ax, proportion=False)
    ax.set_title(f"Árbol sustituto del CATE en {o.label.lower()} — {arm.name}"
                 f" ({'pp' if o.binary else 'unidades'}). R² = {r2:.2f}")
    save(fig, cfg.figures_dir / "surrogate_tree.png")
    imp = pd.Series(tree.feature_importances_, index=X.columns).sort_values(ascending=False)
    return {"surrogate_r2": float(r2), "surrogate_arm": arm.label,
            "surrogate_importance": {c: float(v) for c, v in imp.items() if v > 0}}


def fig_cate_segments(cfg, test, up, metrics, importance: dict):
    """Heatmap de CATE medio en el outcome primario por dos dimensiones."""
    o = cfg.outcome(cfg.primary_outcome)
    sel = metrics["selected"][o.name]
    d = add_segments(test, cfg)
    dims = cfg.figures.get("heatmap")
    if not dims:
        raw = [f for f in importance if f in d.columns][:2]
        dims = raw if len(raw) == 2 else cfg.raw_features[:2]
    labels, orders = {}, {}
    for c in dims:
        labels[c], orders[c] = quantile_bins(d[c], q=4)
    mult = 100 if o.binary else 1
    cmaps = ["Greens", "YlOrBr", "Blues", "Reds"]
    fig, axes = plt.subplots(1, cfg.n_treatments, figsize=(5 * cfg.n_treatments, 3.3), squeeze=False)
    for k, arm in enumerate(cfg.treatments):
        ax = axes[0, k]
        frame = pd.DataFrame({"r": labels[dims[0]], "c": labels[dims[1]], "u": mult * up[o.name][sel][:, k]})
        piv = frame.pivot_table(index="r", columns="c", values="u", aggfunc="mean")
        rows = [x for x in cfg.segments[dims[0]]] if dims[0] in cfg.segments else orders[dims[0]]
        piv = piv.reindex(index=[x for x in rows if x in piv.index],
                          columns=[x for x in orders[dims[1]] if x in piv.columns])
        vmax = max(1e-9, np.nanmax(piv.to_numpy()))
        im = ax.imshow(piv.to_numpy(), cmap=cmaps[k % 4], vmin=min(0, np.nanmin(piv.to_numpy())), vmax=vmax)
        ax.set_xticks(range(piv.shape[1]), [str(c)[:16] for c in piv.columns], rotation=20, fontsize=8)
        ax.set_yticks(range(piv.shape[0]), piv.index)
        ax.set_xlabel(dims[1])
        ax.set_ylabel(dims[0])
        for (r, c), v in np.ndenumerate(piv.to_numpy()):
            ax.text(c, r, f"{v:.2f}" if abs(v) < 1 else f"{v:.1f}", ha="center", va="center", fontsize=9,
                    color="white" if v > 0.6 * vmax else INK)
        ax.grid(False)
        ax.set_title(f"CATE medio en {o.label.lower()} ({'pp' if o.binary else ''}) · {arm.name}")
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    save(fig, cfg.figures_dir / "cate_segments.png")
    return list(dims)


def fig_monitoring(cfg):
    p = cfg.monitoring_dir / "monitoring_summary.csv"
    if not p.exists():
        return
    s = pd.read_csv(p)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.3))
    cols = [CORAL if a else TEAL for a in s.alert]
    axes[0].bar(s.window, s.psi_max, color=cols, width=0.6)
    axes[0].axhline(0.2, color=INK, ls="--", lw=1)
    axes[0].text(len(s) - 0.5, 0.205, "umbral 0,20", ha="right", va="bottom", fontsize=8)
    axes[0].set_title("PSI máximo entre features por ventana")
    axes[0].grid(axis="x", visible=False)
    qcol = next((c for c in s.columns if c.startswith(f"qini_rel_{cfg.primary_outcome}_")), None)
    if qcol:
        axes[1].plot(s.window, s[qcol], marker="o", color=TEAL,
                     label=f"Qini rel. {cfg.outcome(cfg.primary_outcome).label.lower()} · {cfg.treatments[0].name}")
    axes[1].plot(s.window, s.share_recommended_contact, marker="s", color=BLUE, label="% recomendado contactar")
    axes[1].set_title("Performance y predicciones por ventana")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    save(fig, cfg.figures_dir / "monitoring.png")


# ── Diagramas (independientes del dataset) ─────────────────────────
def _box(ax, x, y, w, h, text, color, fc="white", fs=8.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=color, lw=1.6))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK, wrap=True)


def _arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.3))


def fig_architecture(path):
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4.6)
    ax.axis("off")
    steps = [("Config YAML\n+ datos crudos", 0), ("Validación del\ncontrato de datos", 1),
             ("EDA + split\nestratificado", 2), ("Feature\nengineering", 3),
             ("CV repetida\nS/T/X · LGBM/lineal", 4), ("Evaluación test\nQini · AUUC · IPW", 5)]
    bw, step = 1.62, 1.82
    for txt, i in steps:
        _box(ax, 0.1 + i * step, 3.25, bw, 0.9, txt, TEAL, "#E6F3F1", fs=8)
    for i in range(5):
        _arrow(ax, 0.1 + i * step + bw, 3.7, 0.1 + (i + 1) * step, 3.7)
    _box(ax, 4.2, 1.75, 2.6, 0.85, "Bundle del modelo por dataset\nartifacts/<dataset>/models", BLUE, "#EAF0FA")
    _box(ax, 7.6, 1.75, 2.9, 0.85, "MLflow tracking\nexperimento uplift-<dataset>", BLUE, "#EAF0FA")
    _arrow(ax, 9.45, 3.25, 6.3, 2.6)
    _arrow(ax, 9.45, 3.25, 9.0, 2.6)
    _box(ax, 0.3, 0.2, 2.6, 0.95, "API REST (FastAPI)\n/{dataset}/predict · /datasets", GOLD, "#FBF3DF")
    _box(ax, 3.3, 0.2, 2.6, 0.95, "Dashboard (Streamlit)\nselector de dataset", GOLD, "#FBF3DF")
    _box(ax, 6.3, 0.2, 2.6, 0.95, "Monitoreo (Evidently + PSI)\ndrift de datos y performance", CORAL, "#FBE9E5")
    _box(ax, 9.3, 0.2, 1.6, 0.95, "Tests\npytest", MUTED, "#F1F3F5")
    for x in (1.6, 4.6, 7.6):
        _arrow(ax, 5.5, 1.75, x, 1.15)
    ax.text(0.1, 4.45, "Pipeline reproducible y genérico (python -m uplift.train --dataset X)",
            fontsize=10, fontweight="bold", color=INK)
    ax.text(0.1, 1.35, "Serving y operación", fontsize=10, fontweight="bold", color=INK)
    save(fig, path / "architecture.png")


def fig_crisp(path):
    fig, ax = plt.subplots(figsize=(10, 2.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2.6)
    ax.axis("off")
    phases = [("1. Comprensión\ndel negocio", "Entrega 1"), ("2. Comprensión\nde los datos", "Entrega 2 (EDA)"),
              ("3. Preparación\nde los datos", "Validación + FE"), ("4. Modelado", "Meta-learners"),
              ("5. Evaluación", "Qini/AUUC + IPW"), ("6. Despliegue", "API · Dashboard · Drift")]
    for i, (p, d) in enumerate(phases):
        x = 0.1 + i * 1.65
        _box(ax, x, 1.0, 1.45, 1.0, p, TEAL if i < 3 else BLUE, "#E6F3F1" if i < 3 else "#EAF0FA", fs=8.5)
        ax.text(x + 0.725, 0.65, d, ha="center", fontsize=8, color=MUTED)
        if i < 5:
            _arrow(ax, x + 1.45, 1.5, x + 1.65, 1.5)
    ax.annotate("", xy=(0.8, 2.05), xytext=(9.15, 2.05),
                arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=1, ls="--", connectionstyle="arc3,rad=0.15"))
    ax.text(5, 2.45, "iteración", ha="center", fontsize=8, color=MUTED)
    save(fig, path / "crisp_dm.png")


def run(dataset: "str | DatasetConfig" = "hillstrom") -> dict:
    cfg = dataset if isinstance(dataset, DatasetConfig) else load_config(dataset)
    test, up, metrics = _load(cfg)
    fig_cv(cfg, metrics)
    fig_qini(cfg, test, up, metrics)
    fig_deciles(cfg, test, up, metrics)
    fig_budget(cfg)
    fig_policy_benchmark(cfg)
    fig_cate_distribution(cfg, up, metrics)
    meta = fig_surrogate(cfg, test, up, metrics)
    meta["heatmap"] = fig_cate_segments(cfg, test, up, metrics, meta["surrogate_importance"])
    fig_monitoring(cfg)
    fig_architecture(cfg.figures_dir)
    fig_crisp(cfg.figures_dir)
    with open(cfg.reports_dir / "figures_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("Figuras en", cfg.figures_dir)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="hillstrom")
    run(ap.parse_args().dataset)


if __name__ == "__main__":
    main()
