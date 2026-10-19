"""Pipeline de entrenamiento reproducible end-to-end (genérico por dataset).

Uso:
    python -m uplift.train --dataset hillstrom
    python -m uplift.train --dataset criteo --sample-size 1000000 --no-mlflow

Pasos:
 1. Ingesta + validación del contrato de datos (``configs/<dataset>.yaml``).
 2. Muestra estratificada opcional (datasets grandes) y split train/test
    estratificado por tratamiento × outcome raro.
 3. Feature engineering declarativo.
 4. Selección de modelo por validación cruzada (K-fold repetido) sobre train,
    con el Qini relativo promedio de los tratamientos como criterio.
 5. Re-entrenamiento del mejor learner por outcome sobre todo train y
    evaluación final en test (métricas + IC bootstrap).
 6. Simulación de políticas de targeting bajo presupuesto (IPW).
 7. Persistencia en ``artifacts/<dataset>/``: bundle joblib, métricas, tablas
    y runs de MLflow (experimento ``uplift-<dataset>``).
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold

from . import __version__
from .config import MLRUNS_DIR, RANDOM_STATE
from .data import load_raw, sample, save_splits, split, treatment_index
from .dataset import DatasetConfig, load_config
from .features import FeatureBuilder
from .learners import LGBM_PARAMS, build_learners, set_n_jobs
from .metrics import bootstrap_metric, evaluate, qini_relative
from .policy import benchmark_policies, budget_curve


def pair_metrics(uplift: np.ndarray, y: np.ndarray, t: np.ndarray, binary: bool, cfg: DatasetConfig) -> dict:
    """Métricas por tratamiento (k vs control) usando la columna k-1 del uplift."""
    out = {}
    for k, arm in enumerate(cfg.treatments, start=1):
        mask = (t == 0) | (t == k)
        out[arm.label] = evaluate(y[mask], (t[mask] == k).astype(int), uplift[mask, k - 1], binary)
    out["mean_qini_relative"] = float(np.mean([out[a.label]["qini_relative"] for a in cfg.treatments]))
    return out


def _qini_by_arm(y, t, u, cfg):
    q = []
    for k in range(1, cfg.n_treatments + 1):
        m = (t == 0) | (t == k)
        q.append(qini_relative(y[m], (t[m] == k).astype(int), u[m, k - 1]))
    return q


def cross_validate(X: pd.DataFrame, t: np.ndarray, df: pd.DataFrame, cfg: DatasetConfig) -> pd.DataFrame:
    """Qini relativo por learner y outcome en K-fold estratificado repetido sobre train.

    Se repite el K-fold con distintas semillas porque la métrica de uplift es muy
    ruidosa en outcomes raros (d.e. entre folds del orden de la media); promediar
    más particiones reduce la varianza del criterio de selección.
    """
    folds, repeats = cfg.modeling["cv_folds"], cfg.modeling["cv_repeats"]
    strata = t * 10 + df[cfg.stratify_outcome].to_numpy().astype(int)
    skf = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=RANDOM_STATE)
    rows = []
    for o in cfg.outcomes:
        y = df[o.name].to_numpy(dtype=float)
        for fold, (tr, va) in enumerate(skf.split(X, strata)):
            for key, learner in build_learners(o.binary, cfg.n_treatments, o.hurdle).items():
                learner.fit(X.iloc[tr], t[tr], y[tr])
                q = _qini_by_arm(y[va], t[va], learner.predict_uplift(X.iloc[va]), cfg)
                row = {"outcome": o.name, "learner": key, "fold": fold, "qini_rel_mean": float(np.mean(q))}
                row.update({f"qini_rel_{a.label}": v for a, v in zip(cfg.treatments, q)})
                rows.append(row)
            print(f"  CV {o.name} fold {fold + 1}/{folds * repeats} ok", flush=True)
    return pd.DataFrame(rows)


def run(dataset: str | DatasetConfig = "hillstrom", use_mlflow: bool = True,
        sample_size: int | None = None, folds: int | None = None, repeats: int | None = None,
        n_boot: int | None = None) -> dict:
    base = dataset if isinstance(dataset, DatasetConfig) else load_config(dataset)
    cfg = DatasetConfig(raw=copy.deepcopy(base.raw))  # copia: los overrides no tocan la config cacheada
    overrides = {"cv_folds": folds, "cv_repeats": repeats, "n_boot": n_boot, "sample_size": sample_size}
    cfg.raw.setdefault("modeling", {}).update({k: v for k, v in overrides.items() if v})
    m = cfg.modeling
    n_boot = m["n_boot"]
    set_n_jobs(m["n_jobs"])
    t0 = time.time()
    cfg.make_dirs()
    print(f"=== Dataset: {cfg.name} ({cfg.title}) ===")

    print("[1/6] Ingesta y validación")
    df = load_raw(cfg)
    n_total = len(df)
    df = sample(df, cfg)
    print(f"    filas: {n_total:,} → modelado sobre {len(df):,}")
    train, test = split(df, cfg)
    save_splits(train, test, cfg)
    del df

    print("[2/6] Feature engineering")
    fb = FeatureBuilder.from_config(cfg)
    X_tr, X_te = fb.fit_transform(train), fb.transform(test)
    t_tr = treatment_index(train[cfg.treatment_col], cfg)
    t_te = treatment_index(test[cfg.treatment_col], cfg)

    print(f"[3/6] Validación cruzada ({m['cv_folds']} folds × {m['cv_repeats']} repeticiones)")
    cv = cross_validate(X_tr, t_tr, train, cfg)
    cv.to_csv(cfg.tables_dir / "cv_results.csv", index=False)
    cv_summary = cv.groupby(["outcome", "learner"])["qini_rel_mean"].agg(["mean", "std", "median"]).reset_index()
    cv_summary["sem"] = cv_summary["std"] / np.sqrt(m["cv_folds"] * m["cv_repeats"])
    cv_summary.to_csv(cfg.tables_dir / "cv_summary.csv", index=False)
    causal = cv_summary[cv_summary.learner != "baseline"]
    selected = causal.loc[causal.groupby("outcome")["mean"].idxmax()].set_index("outcome")["learner"].to_dict()
    print("    Seleccionados:", selected)

    print("[4/6] Entrenamiento final y evaluación en test")
    results, fitted, test_uplift = {}, {}, {}
    for o in cfg.outcomes:
        y_tr, y_te = train[o.name].to_numpy(dtype=float), test[o.name].to_numpy(dtype=float)
        results[o.name], fitted[o.name], test_uplift[o.name] = {}, {}, {}
        for key, learner in build_learners(o.binary, cfg.n_treatments, o.hurdle).items():
            learner.fit(X_tr, t_tr, y_tr)
            u = learner.predict_uplift(X_te)
            fitted[o.name][key], test_uplift[o.name][key] = learner, u
            results[o.name][key] = pair_metrics(u, y_te, t_te, o.binary, cfg)
        for key in {selected[o.name], "baseline"}:
            for k, arm in enumerate(cfg.treatments, start=1):
                mask = (t_te == 0) | (t_te == k)
                lo, hi = bootstrap_metric(qini_relative, y_te[mask], (t_te[mask] == k).astype(int),
                                          test_uplift[o.name][key][mask, k - 1], n_boot=n_boot, seed=k)
                results[o.name][key][arm.label]["qini_relative_ci95"] = [lo, hi]

    print("[5/6] Simulación de políticas (IPW)")
    for o in cfg.outcome_names:
        u = test_uplift[o][selected[o]]
        budget_curve(u, test, t_te, cfg).to_csv(cfg.tables_dir / f"budget_curve_{o}.csv", index=False)
        benchmark_policies(u, test, t_te, cfg, budget=0.5).to_csv(
            cfg.tables_dir / f"policy_benchmark_{o}.csv", index=False)

    rows = []
    for o, by_learner in results.items():
        for key, mt in by_learner.items():
            for arm in cfg.treatments:
                rows.append({"outcome": o, "learner": key, "treatment": arm.label,
                             **{k: v for k, v in mt[arm.label].items() if not isinstance(v, list)}})
    summary = pd.DataFrame(rows)
    summary.to_csv(cfg.tables_dir / "test_metrics.csv", index=False)

    print("[6/6] Persistencia de artefactos")
    ref = train[cfg.raw_features]
    ref.sample(min(len(ref), 200_000), random_state=RANDOM_STATE).to_csv(cfg.data_dir / "reference.csv", index=False)
    bundle = {
        "version": __version__,
        "dataset": cfg.name,
        "config": cfg.raw,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_rows": {"total": n_total, "train": len(train), "test": len(test)},
        "feature_builder": fb,
        "feature_names": list(fb.get_feature_names_out()),
        "models": {o: fitted[o][selected[o]] for o in cfg.outcome_names},
        "selected": selected,
        "test_metrics": {o: results[o][selected[o]] for o in cfg.outcome_names},
    }
    joblib.dump(bundle, cfg.bundle_path, compress=3)
    np.savez_compressed(cfg.models_dir / "test_uplift.npz",
                        **{f"{o}__{k}": v.astype(np.float32) for o in cfg.outcome_names
                           for k, v in test_uplift[o].items()})
    with open(cfg.reports_dir / "metrics.json", "w") as f:
        json.dump({"dataset": cfg.name, "selected": selected, "results": results,
                   "n_rows": bundle["n_rows"], "cv_summary": cv_summary.to_dict(orient="records")}, f, indent=2)

    if use_mlflow:
        log_mlflow(cfg, results, cv_summary, selected)
    print(f"Listo en {time.time() - t0:.0f}s")
    return {"selected": selected, "results": results, "summary": summary, "cv_summary": cv_summary}


def log_mlflow(cfg: DatasetConfig, results: dict, cv_summary: pd.DataFrame, selected: dict) -> None:
    import mlflow

    mlflow.set_tracking_uri(MLRUNS_DIR.as_uri())
    mlflow.set_experiment(cfg.mlflow_experiment)
    with mlflow.start_run(run_name=f"train-{datetime.now():%Y%m%d-%H%M%S}"):
        mlflow.set_tag("dataset", cfg.name)
        mlflow.log_params({f"lgbm_{k}": v for k, v in LGBM_PARAMS.items()})
        mlflow.log_params({f"modeling_{k}": v for k, v in cfg.modeling.items()})
        mlflow.log_params({f"selected_{o}": s for o, s in selected.items()})
        for outcome, by_learner in results.items():
            for key, mt in by_learner.items():
                with mlflow.start_run(run_name=f"{outcome}-{key}", nested=True):
                    mlflow.set_tags({"outcome": outcome, "learner": key,
                                     "selected": str(selected[outcome] == key)})
                    cvr = cv_summary[(cv_summary.outcome == outcome) & (cv_summary.learner == key)]
                    if len(cvr):
                        mlflow.log_metric("cv_qini_rel_mean", float(cvr["mean"].iloc[0]))
                    for arm in cfg.treatments:
                        for name, v in mt[arm.label].items():
                            if not isinstance(v, list) and v == v:
                                mlflow.log_metric(f"test_{arm.label}_{name}", float(v))
        mlflow.log_artifact(str(cfg.bundle_path))
        mlflow.log_artifact(str(cfg.reports_dir / "metrics.json"))
        mlflow.log_artifacts(str(cfg.tables_dir), artifact_path="tables")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="hillstrom", help="nombre de configs/<dataset>.yaml")
    ap.add_argument("--sample-size", type=int, default=None, help="filas a usar (pisa la config)")
    ap.add_argument("--folds", type=int, default=None)
    ap.add_argument("--repeats", type=int, default=None)
    ap.add_argument("--n-boot", type=int, default=None)
    ap.add_argument("--no-mlflow", action="store_true")
    a = ap.parse_args()
    run(a.dataset, use_mlflow=not a.no_mlflow, sample_size=a.sample_size, folds=a.folds,
        repeats=a.repeats, n_boot=a.n_boot)


if __name__ == "__main__":
    main()
