"""Monitoreo de drift de datos, de predicciones y de performance (Objetivo 8).

Los datasets son estáticos, por lo que la operación en producción se **simula**
partiendo el conjunto de test en ventanas sucesivas (W1…Wn, "ventanas") y
agregando una ventana con drift inducido: un re-muestreo ponderado según el
escenario declarado en ``monitoring.drift_scenario`` de la config. Al re-muestrear
filas reales se conservan los outcomes observados y se puede medir también la
degradación de performance.

Uso:
    python -m uplift.monitoring --dataset hillstrom
Salidas en ``artifacts/<dataset>/reports/monitoring/``: reportes HTML de Evidently
por ventana y ``monitoring_summary.csv`` con PSI, share de columnas con drift,
uplift medio predicho y Qini relativo observado por ventana.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .config import RANDOM_STATE
from .data import load_split, treatment_index
from .dataset import DatasetConfig, load_config
from .metrics import qini_relative
from .predict import load_bundle, score

PSI_ALERT = 0.20  # regla práctica: PSI > 0,2 = cambio poblacional significativo
DRIFT_SHARE_ALERT = 0.30
EVIDENTLY_MAX_ROWS = 50_000


def psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    """Population Stability Index entre una referencia y una ventana."""
    eps = 1e-4
    if expected.dtype == object or str(expected.dtype) == "category" or expected.nunique() <= 12:
        cats = sorted(set(expected.unique()) | set(actual.unique()), key=str)
        e = expected.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
        a = actual.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    else:
        # misma precisión en ambos lados: la referencia puede venir de CSV (float64) y la
        # ventana de parquet (float32); con masas puntuales, esa diferencia cambiaría de tramo
        # valores idénticos y produciría un PSI espurio
        expected = expected.astype(np.float32).astype(np.float64)
        actual = actual.astype(np.float32).astype(np.float64)
        edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        e = np.histogram(expected, edges)[0] / len(expected)
        a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, eps, None), np.clip(a, eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


def scenario_weights(df: pd.DataFrame, scenario: list[dict]) -> np.ndarray:
    """Pesos de re-muestreo del escenario de drift (producto de reglas)."""
    w = np.ones(len(df))
    for r in scenario:
        col, op, val = df[r["column"]], r["op"], r["value"]
        if op == "le":
            cond = col <= val
        elif op == "ge":
            cond = col >= val
        elif op == "eq":
            cond = col == val
        elif op == "quantile_ge":
            cond = col >= col.quantile(val)
        elif op == "quantile_le":
            cond = col <= col.quantile(val)
        else:
            raise ValueError(f"Operador de drift desconocido: {op}")
        w *= np.where(cond.to_numpy(), r["weight"], 1.0)
    return w / w.sum()


def make_windows(test: pd.DataFrame, cfg: DatasetConfig, seed: int = RANDOM_STATE) -> dict:
    """Ventanas simuladas: n particiones estables + una con drift inducido."""
    n_windows = cfg.monitoring["n_windows"]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(test))
    windows = {f"W{i + 1}": test.iloc[part].reset_index(drop=True)
               for i, part in enumerate(np.array_split(idx, n_windows))}
    if cfg.monitoring["drift_scenario"]:
        p = scenario_weights(test, cfg.monitoring["drift_scenario"])
        drift_idx = rng.choice(len(test), size=len(test) // n_windows, replace=True, p=p)
        windows[f"W{n_windows + 1} (drift)"] = test.iloc[drift_idx].reset_index(drop=True)
    return windows


def evidently_report(reference: pd.DataFrame, current: pd.DataFrame, path) -> float:
    """Genera el reporte HTML de Evidently y devuelve el share de columnas con drift."""
    from evidently.metric_preset import DataDriftPreset
    from evidently.report import Report

    rep = Report(metrics=[DataDriftPreset()])
    rep.run(reference_data=reference, current_data=current)
    rep.save_html(str(path))
    res = rep.as_dict()["metrics"][0]["result"]
    return float(res.get("share_of_drifted_columns", np.nan))


def run(dataset: "str | DatasetConfig" = "hillstrom", use_evidently: bool = True) -> pd.DataFrame:
    cfg = dataset if isinstance(dataset, DatasetConfig) else load_config(dataset)
    cfg.monitoring_dir.mkdir(parents=True, exist_ok=True)
    bundle = load_bundle(cfg)
    reference = load_split("reference", cfg)
    test = load_split("test", cfg)
    feats = cfg.raw_features
    first = cfg.treatments[0].label
    obj = "profit" if cfg.business["values"] else cfg.primary_outcome
    ref_scores = score(reference, cfg, bundle, objective=obj)
    ref_ev = reference[feats].sample(min(len(reference), EVIDENTLY_MAX_ROWS), random_state=RANDOM_STATE)

    rows = []
    for name, win in make_windows(test, cfg).items():
        sc = score(win[feats], cfg, bundle, objective=obj)
        t = treatment_index(win[cfg.treatment_col], cfg)
        row = {"window": name, "n": len(win)}
        for f in feats:
            row[f"psi_{f}"] = psi(reference[f], win[f])
        row["psi_max"] = max(row[f"psi_{f}"] for f in feats)
        col = f"uplift_{cfg.primary_outcome}_{first}"
        row["psi_prediction"] = psi(ref_scores[col], sc[col])
        row["mean_uplift_primary"] = float(sc[col].mean())
        row["share_recommended_contact"] = float((sc["recommended_treatment"] != cfg.none_action).mean())
        for o in cfg.outcomes:
            if not o.binary:
                continue
            for k, arm in enumerate(cfg.treatments, start=1):
                m = (t == 0) | (t == k)
                row[f"qini_rel_{o.name}_{arm.label}"] = qini_relative(
                    win[o.name].to_numpy()[m], (t[m] == k).astype(int),
                    sc[f"uplift_{o.name}_{arm.label}"].to_numpy()[m])
        if use_evidently:
            cur = win[feats].sample(min(len(win), EVIDENTLY_MAX_ROWS), random_state=RANDOM_STATE)
            row["evidently_share_drifted"] = evidently_report(
                ref_ev, cur, cfg.monitoring_dir / f"data_drift_{name.split()[0].lower()}.html")
        row["alert"] = bool(row["psi_max"] > PSI_ALERT
                            or row.get("evidently_share_drifted", 0) > DRIFT_SHARE_ALERT)
        rows.append(row)
        print(f"  {name}: PSI máx={row['psi_max']:.3f} alerta={row['alert']}")

    summary = pd.DataFrame(rows)
    summary.to_csv(cfg.monitoring_dir / "monitoring_summary.csv", index=False)
    return summary


def main():
    import warnings

    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="hillstrom")
    ap.add_argument("--no-evidently", action="store_true")
    a = ap.parse_args()
    run(a.dataset, use_evidently=not a.no_evidently)


if __name__ == "__main__":
    main()
