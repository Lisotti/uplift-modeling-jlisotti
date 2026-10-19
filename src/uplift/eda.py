"""Análisis exploratorio genérico para experimentos de uplift (CRISP-DM fase 2).

Uso:
    python -m uplift.eda --dataset criteo

Responde, para cualquier dataset declarado en ``configs/``, las preguntas que
condicionan el modelado de uplift:

1. **Calidad**: faltantes, duplicados, restricciones lógicas, dominios.
2. **Validez del experimento**: balance de covariables entre brazos
   (diferencia de medias estandarizada, SMD) — en un RCT debe ser |SMD| < 0,1.
3. **Efecto promedio** (ATE) de cada tratamiento sobre cada outcome, con IC 95 %
   y test de hipótesis (z para proporciones, Welch para continuos).
4. **Distribuciones** de features y outcomes (asimetría, ceros, outliers IQR).
5. **Correlaciones** (Spearman) entre features y outcomes.
6. **Heterogeneidad**: ATE por tramos de cada feature — si varía, el CATE es estimable.
7. **Veredicto de viabilidad** a partir de criterios explícitos.

Salidas en ``artifacts/<dataset>/reports/``: ``eda_summary.json``, tablas
``tables/eda_*.csv`` y figuras ``figures/eda_*.png``.
"""

from __future__ import annotations

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from .config import RANDOM_STATE
from .data import DataValidationError, load_raw, treatment_index, validate
from .dataset import DatasetConfig, load_config
from .features import add_segments, quantile_bins
from .plotting import BLUE, CORAL, GRAY, INK, MUTED, TEAL, arm_colors, save

PLOT_SAMPLE = 200_000


def _numeric_features(cfg):
    return [f["name"] for f in cfg.feature_specs if f["type"] == "numeric"]


def _discrete_features(cfg):
    return [f["name"] for f in cfg.feature_specs if f["type"] in ("binary", "categorical")]


# ── 1. Calidad ─────────────────────────────────────────────────────
def quality(df: pd.DataFrame, cfg: DatasetConfig) -> dict:
    cols = [*cfg.raw_features, cfg.treatment_col, *cfg.outcome_names]
    try:
        validate(df, cfg)
        contract = "ok"
    except DataValidationError as e:
        contract = str(e)
    dup = int(df.duplicated().sum()) if len(df) <= 2_000_000 else int(
        df.sample(2_000_000, random_state=RANDOM_STATE).duplicated().sum())
    return {
        "rows": int(len(df)), "columns": int(df.shape[1]),
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 1),
        "missing_total": int(df[cols].isna().sum().sum()),
        "missing_by_column": {c: int(v) for c, v in df[cols].isna().sum().items() if v},
        "duplicated_rows": dup, "duplicates_sampled": len(df) > 2_000_000,
        "data_contract": contract,
        "excluded_columns": cfg.excluded_columns,
    }


# ── 2. Balance del RCT ─────────────────────────────────────────────
def balance(df: pd.DataFrame, t: np.ndarray, cfg: DatasetConfig) -> pd.DataFrame:
    """SMD de cada feature (y categoría) entre cada tratamiento y el control."""
    rows = []
    cols = {}
    for s in cfg.feature_specs:
        if s["type"] == "numeric" or s["type"] == "binary":
            cols[s["name"]] = df[s["name"]].astype(float).to_numpy()
        elif s["type"] == "categorical":
            for c in s["categories"]:
                cols[f"{s['name']}={c}"] = (df[s["name"]] == c).astype(float).to_numpy()
    for k, arm in enumerate(cfg.treatments, start=1):
        for name, v in cols.items():
            a, b = v[t == k], v[t == 0]
            sd = np.sqrt((a.var() + b.var()) / 2)
            rows.append({"treatment": arm.label, "feature": name, "mean_treated": a.mean(),
                         "mean_control": b.mean(), "smd": (a.mean() - b.mean()) / sd if sd > 0 else 0.0})
    return pd.DataFrame(rows)


def arm_counts(t: np.ndarray, cfg: DatasetConfig) -> pd.DataFrame:
    n = np.bincount(t, minlength=len(cfg.arms))
    return pd.DataFrame({"arm": [a.name for a in cfg.arms], "label": [a.label for a in cfg.arms],
                         "n": n, "share": n / n.sum()})


# ── 3. Efecto promedio ────────────────────────────────────────────
def ate_table(df: pd.DataFrame, t: np.ndarray, cfg: DatasetConfig) -> pd.DataFrame:
    rows = []
    for o in cfg.outcomes:
        y = df[o.name].to_numpy(dtype=float)
        y0 = y[t == 0]
        rows.append({"outcome": o.name, "arm": cfg.control.label, "n": len(y0), "mean": y0.mean(),
                     "mean_lo": y0.mean() - 1.96 * y0.std(ddof=1) / np.sqrt(len(y0)),
                     "mean_hi": y0.mean() + 1.96 * y0.std(ddof=1) / np.sqrt(len(y0))})
        for k, arm in enumerate(cfg.treatments, start=1):
            y1 = y[t == k]
            diff = y1.mean() - y0.mean()
            se = np.sqrt(y1.var(ddof=1) / len(y1) + y0.var(ddof=1) / len(y0))
            if o.binary:
                p = (y1.sum() + y0.sum()) / (len(y1) + len(y0))
                se0 = np.sqrt(p * (1 - p) * (1 / len(y1) + 1 / len(y0)))
                pval = 2 * stats.norm.sf(abs(diff / se0)) if se0 > 0 else 1.0
            else:
                pval = float(stats.ttest_ind(y1, y0, equal_var=False).pvalue)
            rows.append({"outcome": o.name, "arm": arm.label, "n": len(y1), "mean": y1.mean(),
                         "mean_lo": y1.mean() - 1.96 * y1.std(ddof=1) / np.sqrt(len(y1)),
                         "mean_hi": y1.mean() + 1.96 * y1.std(ddof=1) / np.sqrt(len(y1)),
                         "ate": diff, "ate_lo": diff - 1.96 * se, "ate_hi": diff + 1.96 * se,
                         "relative_lift": diff / y0.mean() if y0.mean() else np.nan, "p_value": pval})
    return pd.DataFrame(rows)


# ── 4. Distribuciones ─────────────────────────────────────────────
def numeric_summary(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for c in cols:
        v = df[c].astype(float)
        q1, q3 = v.quantile([0.25, 0.75])
        iqr = q3 - q1
        rows.append({"variable": c, "mean": v.mean(), "median": v.median(), "std": v.std(),
                     "min": v.min(), "max": v.max(), "skewness": float(stats.skew(v)),
                     "n_unique": int(v.nunique()), "zero_share": float((v == 0).mean()),
                     "outliers_iqr": int(((v < q1 - 1.5 * iqr) | (v > q3 + 1.5 * iqr)).sum())})
    return pd.DataFrame(rows)


# ── 6. Heterogeneidad ─────────────────────────────────────────────
def heterogeneity(df: pd.DataFrame, t: np.ndarray, cfg: DatasetConfig, outcome: str) -> pd.DataFrame:
    """ATE del ``outcome`` por tramo de cada feature y segmento derivado."""
    y = df[outcome].to_numpy(dtype=float)
    rows = []
    cols = [*cfg.raw_features, *cfg.segments.keys()]
    for col in cols:
        bins, order = quantile_bins(df[col])
        if col in cfg.segments:
            order = [x for x in cfg.segments[col] if x in set(order)]
        for b in order:
            m = (bins == b).to_numpy()
            y0 = y[m & (t == 0)]
            for k, arm in enumerate(cfg.treatments, start=1):
                y1 = y[m & (t == k)]
                if len(y0) < 30 or len(y1) < 30:
                    continue
                d = y1.mean() - y0.mean()
                se = np.sqrt(y1.var(ddof=1) / len(y1) + y0.var(ddof=1) / len(y0))
                rows.append({"feature": col, "bin": str(b), "arm": arm.label, "n": int(m.sum()),
                             "rate_control": y0.mean(), "rate_treated": y1.mean(),
                             "ate": d, "ate_lo": d - 1.96 * se, "ate_hi": d + 1.96 * se})
    out = pd.DataFrame(rows)
    # Test de heterogeneidad por feature: χ² de Cochran (Q) sobre los ATE por tramo
    q = []
    for (f, a), g in out.groupby(["feature", "arm"]):
        w = 1 / ((g.ate_hi - g.ate_lo) / 3.92) ** 2
        mu = (w * g.ate).sum() / w.sum()
        Q = float((w * (g.ate - mu) ** 2).sum())
        q.append({"feature": f, "arm": a, "Q": Q, "df": len(g) - 1,
                  "p_value": float(stats.chi2.sf(Q, len(g) - 1)) if len(g) > 1 else 1.0,
                  "ate_range": float(g.ate.max() - g.ate.min())})
    out.attrs["cochran"] = pd.DataFrame(q).sort_values("Q", ascending=False)
    return out


# ── Figuras ───────────────────────────────────────────────────────
def fig_rates(ate: pd.DataFrame, cfg: DatasetConfig) -> None:
    colors = arm_colors(cfg)
    names = {a.label: a.name for a in cfg.arms}
    fig, axes = plt.subplots(1, len(cfg.outcomes), figsize=(3.4 * len(cfg.outcomes), 3.1))
    axes = np.atleast_1d(axes)
    for ax, o in zip(axes, cfg.outcomes):
        d = ate[ate.outcome == o.name]
        mult = 100 if o.binary else 1
        vals = d["mean"].to_numpy() * mult
        err = (d["mean_hi"] - d["mean"]).to_numpy() * mult
        bars = ax.bar([names[a] for a in d.arm], vals, yerr=err, width=0.6, capsize=3,
                      color=[colors[a] for a in d.arm], error_kw=dict(ecolor=MUTED, lw=1))
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v * 0.5, f"{v:.2f}" + ("%" if o.binary else ""),
                    ha="center", color="white", fontsize=9, fontweight="bold")
        ax.set_title(o.label + (" — tasa (%)" if o.binary else " — media por cliente"))
        ax.grid(axis="x", visible=False)
        ax.tick_params(axis="x", labelsize=8.5)
    fig.suptitle("Outcomes por grupo experimental (IC 95 %)", fontweight="bold", color=INK, y=1.03)
    save(fig, cfg.figures_dir / "eda_rates_by_group.png")


def fig_balance(bal: pd.DataFrame, cfg: DatasetConfig) -> None:
    colors = arm_colors(cfg)
    feats = bal.feature.unique()
    fig, ax = plt.subplots(figsize=(6.5, max(2.5, 0.28 * len(feats) + 1)))
    y = np.arange(len(feats))
    for i, arm in enumerate(cfg.treatments):
        d = bal[bal.treatment == arm.label].set_index("feature").reindex(feats)
        ax.scatter(d.smd, y + (i - (cfg.n_treatments - 1) / 2) * 0.2, color=colors[arm.label],
                   s=28, label=f"{arm.name} vs control", zorder=3)
    for x in (-0.1, 0.1):
        ax.axvline(x, color=CORAL, ls="--", lw=1)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(y, feats, fontsize=8)
    ax.set_xlabel("Diferencia de medias estandarizada (SMD)")
    ax.set_title("Balance de covariables entre brazos (umbral |SMD| = 0,1)")
    ax.legend(fontsize=8, loc="lower right")
    save(fig, cfg.figures_dir / "eda_balance.png")


def fig_distributions(df: pd.DataFrame, cfg: DatasetConfig) -> None:
    s = df.sample(min(len(df), PLOT_SAMPLE), random_state=RANDOM_STATE)
    cols = [*_numeric_features(cfg), *_discrete_features(cfg)]
    n = len(cols)
    ncol = 4 if n > 6 else 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 2.3 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, c in zip(axes, cols):
        v = s[c]
        if c in _numeric_features(cfg) and v.nunique() > 15:
            lo, hi = v.quantile([0.005, 0.995])
            ax.hist(v.clip(lo, hi), bins=40, color=BLUE, alpha=0.85)
        else:
            vc = v.value_counts().sort_index()
            ax.bar(vc.index.astype(str), vc.values, color=BLUE, width=0.6)
            ax.tick_params(axis="x", labelsize=7, rotation=30)
        ax.set_title(c, fontsize=9)
        ax.grid(axis="x", visible=False)
        ax.tick_params(axis="y", labelsize=7)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("Distribución de las features (percentiles 0,5–99,5 % en numéricas)",
                 fontweight="bold", color=INK, y=1.01)
    fig.tight_layout()
    save(fig, cfg.figures_dir / "eda_distributions.png")


def fig_correlation(df: pd.DataFrame, cfg: DatasetConfig) -> pd.DataFrame:
    s = df.sample(min(len(df), PLOT_SAMPLE), random_state=RANDOM_STATE)
    cols = [c for c in [*_numeric_features(cfg),
                        *[f["name"] for f in cfg.feature_specs if f["type"] == "binary"],
                        *cfg.outcome_names] if s[c].nunique() > 1]
    corr = s[cols].astype(float).corr(method="spearman")
    fig, ax = plt.subplots(figsize=(0.45 * len(cols) + 2.5, 0.45 * len(cols) + 1.8))
    im = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=90, fontsize=8)
    ax.set_yticks(range(len(cols)), cols, fontsize=8)
    ax.grid(False)
    if len(cols) <= 16:
        for (i, j), v in np.ndenumerate(corr.to_numpy()):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if abs(v) > 0.5 else INK)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Correlación de Spearman (features y outcomes)")
    save(fig, cfg.figures_dir / "eda_correlation.png")
    return corr


def fig_heterogeneity(het: pd.DataFrame, cfg: DatasetConfig, outcome: str) -> None:
    colors = arm_colors(cfg)
    names = {a.label: a.name for a in cfg.treatments}
    o = cfg.outcome(outcome)
    mult, unit = (100, "pp") if o.binary else (1, "")
    # (a) segmento derivado principal, si existe (figura del informe de Hillstrom)
    if cfg.segments:
        seg = next(iter(cfg.segments))
        d = het[het.feature == seg]
        order = list(cfg.segments[seg].keys())
        fig, ax = plt.subplots(figsize=(6.4, 3.2))
        x = np.arange(len(order))
        w = 0.8 / cfg.n_treatments
        for i, arm in enumerate(cfg.treatments):
            dd = d[d.arm == arm.label].set_index("bin").reindex(order)
            bars = ax.bar(x + (i - (cfg.n_treatments - 1) / 2) * w, dd.ate * mult, w * 0.95,
                          color=colors[arm.label], label=f"Email {arm.name}" if cfg.name == "hillstrom" else arm.name)
            for b, val in zip(bars, dd.ate * mult):
                ax.text(b.get_x() + b.get_width() / 2, val + 0.25, f"{val:.1f}", ha="center", fontsize=8.5, color=INK)
        ax.set_xticks(x, order)
        ax.set_ylabel(f"Uplift en {o.label.lower()} ({unit})")
        ax.set_title(f"Heterogeneidad del efecto según {seg}")
        ax.legend(loc="upper left")
        ax.grid(axis="x", visible=False)
        save(fig, cfg.figures_dir / "eda_heterogeneity.png")
    # (b) small multiples de las features con mayor heterogeneidad
    coch = het.attrs["cochran"]
    top = coch.groupby("feature").Q.max().sort_values(ascending=False).head(6).index.tolist()
    ncol = 3
    nrow = int(np.ceil(len(top) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 2.7 * nrow), squeeze=False)
    for ax, f in zip(axes.ravel(), top):
        d = het[het.feature == f]
        bins = list(dict.fromkeys(d.bin))  # orden ya definido en heterogeneity()
        x = np.arange(len(bins))
        w = 0.8 / cfg.n_treatments
        for i, arm in enumerate(cfg.treatments):
            dd = d[d.arm == arm.label].set_index("bin").reindex(bins)
            ax.bar(x + (i - (cfg.n_treatments - 1) / 2) * w, dd.ate * mult, w * 0.95,
                   yerr=[(dd.ate - dd.ate_lo) * mult, (dd.ate_hi - dd.ate) * mult],
                   color=colors[arm.label], error_kw=dict(ecolor=MUTED, lw=0.8, capsize=2),
                   label=names[arm.label])
        ax.axhline(0, color=INK, lw=0.8)
        ax.set_xticks(x, [b if len(b) <= 22 else b[:20] + "…" for b in bins], rotation=30, fontsize=6.5, ha="right")
        ax.set_title(f, fontsize=9.5)
        ax.grid(axis="x", visible=False)
    for ax in axes.ravel()[len(top):]:
        ax.axis("off")
    axes[0, 0].set_ylabel(f"ATE en {o.label.lower()} ({unit})")
    axes[0, 0].legend(fontsize=7.5)
    fig.suptitle(f"Efecto del tratamiento por tramo — features con mayor heterogeneidad ({o.label.lower()})",
                 fontweight="bold", color=INK, y=1.02)
    fig.tight_layout()
    save(fig, cfg.figures_dir / "eda_heterogeneity_features.png")


def fig_outcome_zero_inflation(df: pd.DataFrame, cfg: DatasetConfig) -> None:
    cont = [o for o in cfg.outcomes if not o.binary]
    if not cont:
        return
    o = cont[0]
    v = df[o.name]
    pos = v[v > 0]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), gridspec_kw={"width_ratios": [1, 2], "wspace": 0.3})
    shares = [(v == 0).mean() * 100, (v > 0).mean() * 100]
    bars = axes[0].bar(["= 0", "> 0"], shares, color=[GRAY, TEAL], width=0.6)
    for b, s in zip(bars, shares):
        axes[0].annotate(f"{s:.1f} %".replace(".", ","), (b.get_x() + b.get_width() / 2, s),
                         ha="center", va="bottom", fontsize=9, color=INK, xytext=(0, 2), textcoords="offset points")
    axes[0].set_ylabel("% de clientes")
    axes[0].set_ylim(0, 110)
    axes[0].set_title("Ceros vs positivos")
    axes[1].hist(pos, bins=40, color=TEAL)
    axes[1].axvline(pos.median(), color=CORAL, ls="--", lw=1.2)
    axes[1].set_title(f"Entre quienes compraron (n = {len(pos):,}; mediana {pos.median():.1f})"
                      .replace(",", "X").replace(".", ",").replace("X", "."))
    axes[1].set_xlabel(o.label)
    axes[1].set_ylabel("Clientes")
    fig.suptitle(f"Distribución de {o.label}", fontweight="bold", color=INK, y=1.04)
    for ax in axes:
        ax.grid(axis="x", visible=False)
    save(fig, cfg.figures_dir / "eda_outcome_distribution.png")


# ── Veredicto ─────────────────────────────────────────────────────
def viability(q: dict, bal: pd.DataFrame, ate: pd.DataFrame, het: pd.DataFrame, t, df, cfg) -> list[dict]:
    prim = cfg.primary_outcome
    a = ate[(ate.outcome == prim) & ate.ate.notna()]
    coch = het.attrs["cochran"]
    min_events = min(int(df.loc[t == k, o.name].astype(bool).sum())
                     for o in cfg.outcomes if o.binary for k in range(len(cfg.arms)))
    checks = [
        ("Sin valores faltantes", q["missing_total"] == 0, f"{q['missing_total']} faltantes"),
        ("Contrato de datos (dominios y restricciones lógicas)", q["data_contract"] == "ok", q["data_contract"]),
        ("Aleatorización válida (|SMD| < 0,1 en todas las covariables)", bool(bal.smd.abs().max() < 0.1),
         f"|SMD| máx. = {bal.smd.abs().max():.3f}"),
        (f"Efecto promedio significativo en {prim} (p < 0,05)", bool((a.p_value < 0.05).all()),
         "; ".join(f"{r.arm}: {r.ate * 100:+.2f} pp (p={r.p_value:.1e})" for r in a.itertuples())),
        ("Heterogeneidad del efecto detectable (Cochran Q, p < 0,05)", bool((coch.p_value < 0.05).any()),
         f"{int((coch.p_value < 0.05).sum())} de {len(coch)} pares feature × tratamiento"),
        ("Eventos suficientes por brazo (≥ 100 en cada outcome binario)", min_events >= 100,
         f"mínimo = {min_events:,}"),
    ]
    return [{"criterio": c, "aprobado": ok, "detalle": d} for c, ok, d in checks]


def run(dataset: "str | DatasetConfig" = "hillstrom") -> dict:
    cfg = dataset if isinstance(dataset, DatasetConfig) else load_config(dataset)
    cfg.make_dirs()
    print(f"=== EDA: {cfg.name} ===")
    df = load_raw(cfg, validate_data=False)
    df = add_segments(df, cfg)
    t = treatment_index(df[cfg.treatment_col], cfg)
    q = quality(df, cfg)
    print("  calidad ok")
    counts = arm_counts(t, cfg)
    bal = balance(df, t, cfg)
    ate = ate_table(df, t, cfg)
    num = numeric_summary(df, [*_numeric_features(cfg), *[o.name for o in cfg.outcomes if not o.binary]])
    print("  balance, ATE y distribuciones ok")
    het = {o.name: heterogeneity(df, t, cfg, o.name) for o in cfg.outcomes if o.binary}
    prim_het = het[cfg.primary_outcome]
    print("  heterogeneidad ok")

    for name, tab in [("eda_arm_counts", counts), ("eda_balance", bal), ("eda_ate", ate),
                      ("eda_numeric_summary", num)]:
        tab.to_csv(cfg.tables_dir / f"{name}.csv", index=False)
    for o, h in het.items():
        h.to_csv(cfg.tables_dir / f"eda_heterogeneity_{o}.csv", index=False)
        h.attrs["cochran"].to_csv(cfg.tables_dir / f"eda_heterogeneity_test_{o}.csv", index=False)

    fig_rates(ate, cfg)
    fig_balance(bal, cfg)
    fig_distributions(df, cfg)
    corr = fig_correlation(df, cfg)
    fig_heterogeneity(prim_het, cfg, cfg.primary_outcome)
    fig_outcome_zero_inflation(df, cfg)
    print("  figuras ok")

    verdict = viability(q, bal, ate, prim_het, t, df, cfg)
    summary = {
        "dataset": cfg.name, "title": cfg.title, "quality": q,
        "arms": counts.to_dict(orient="records"),
        "max_abs_smd": float(bal.smd.abs().max()),
        "ate": ate.to_dict(orient="records"),
        "outcome_rates": {o.name: float(df[o.name].mean()) for o in cfg.outcomes},
        "events": {o.name: int(df[o.name].astype(bool).sum()) for o in cfg.outcomes},
        "top_heterogeneity": prim_het.attrs["cochran"].head(8).to_dict(orient="records"),
        "strong_correlations": [
            {"a": a, "b": b, "rho": float(corr.loc[a, b])}
            for i, a in enumerate(corr.columns) for b in corr.columns[i + 1:]
            if abs(corr.loc[a, b]) >= 0.7],
        "viability": verdict,
        "viable": all(v["aprobado"] for v in verdict),
    }
    with open(cfg.reports_dir / "eda_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=float)
    print(f"  veredicto: {'VIABLE' if summary['viable'] else 'REVISAR'} "
          f"({sum(v['aprobado'] for v in verdict)}/{len(verdict)} criterios)")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="hillstrom")
    run(ap.parse_args().dataset)


if __name__ == "__main__":
    main()
