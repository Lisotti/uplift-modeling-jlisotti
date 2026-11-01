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
    con el Qini relativo promedio de los tratamientos como criterio. La misma
    CV deja predicciones *out-of-fold* (cross-fitting) de cada learner en train.
 5. Re-entrenamiento del mejor learner por outcome sobre todo train y
    evaluación final en test (métricas + IC bootstrap), con pruebas de
    significancia: t-test corregido sobre los folds de la CV, test de
    permutación del Qini y bootstrap pareado frente al baseline en test, y
    conteo de eventos / efecto mínimo detectable por brazo.
 6. Selección del presupuesto óptimo sobre las predicciones out-of-fold de train
    (validación) y evaluación de esa política, ya fija, en test (IPW). El set de
    test no interviene en ninguna decisión: sólo se usa para reportar.
 6b. Índice sustituto (si la config declara ``surrogate``): la visita se valoriza
    con el embudo P(compra | visita) × ticket medio estimado en train, la política
    se rankea por ese valor y el presupuesto se elige en validación; en test se
    reporta el beneficio observado (IPW) y el sustituto, y se valida el supuesto.
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
from .metrics import (ate_mde, bootstrap_metric, corrected_resampled_ttest, evaluate, event_counts,
                      paired_bootstrap_diff, permutation_pvalue, qini_relative)
from .policy import benchmark_policies, budget_criterion, budget_curve, ranking_gain, select_budget
from .surrogate import SurrogateIndex


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


def cross_validate(X: pd.DataFrame, t: np.ndarray, df: pd.DataFrame,
                   cfg: DatasetConfig) -> tuple[pd.DataFrame, dict[str, dict[str, np.ndarray]]]:
    """Qini relativo por learner y outcome en K-fold estratificado repetido sobre train.

    Se repite el K-fold con distintas semillas porque la métrica de uplift es muy
    ruidosa en outcomes raros (d.e. entre folds del orden de la media); promediar
    más particiones reduce la varianza del criterio de selección.

    Devuelve además las predicciones *out-of-fold* de cada learner: cada fila de
    train recibe el uplift estimado por modelos que no la vieron al entrenar
    (promediado sobre las repeticiones). Son el conjunto de validación sobre el
    que se eligen los hiperparámetros de la política (presupuesto) sin tocar test.
    """
    folds, repeats = cfg.modeling["cv_folds"], cfg.modeling["cv_repeats"]
    strata = t * 10 + df[cfg.stratify_outcome].to_numpy().astype(int)
    skf = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=RANDOM_STATE)
    rows = []
    oof: dict[str, dict[str, np.ndarray]] = {}
    for o in cfg.outcomes:
        y = df[o.name].to_numpy(dtype=float)
        oof[o.name] = {}
        for fold, (tr, va) in enumerate(skf.split(X, strata)):
            for key, learner in build_learners(o.binary, cfg.n_treatments, o.hurdle, cfg.modeling["extra_learners"]).items():
                learner.fit(X.iloc[tr], t[tr], y[tr])
                u = learner.predict_uplift(X.iloc[va])
                acc = oof[o.name].setdefault(key, np.zeros((len(X), cfg.n_treatments)))
                acc[va] += u / repeats  # cada repetición cubre cada fila exactamente una vez
                q = _qini_by_arm(y[va], t[va], u, cfg)
                row = {"outcome": o.name, "learner": key, "fold": fold, "qini_rel_mean": float(np.mean(q))}
                row.update({f"qini_rel_{a.label}": v for a, v in zip(cfg.treatments, q)})
                rows.append(row)
            print(f"  CV {o.name} fold {fold + 1}/{folds * repeats} ok", flush=True)
    return pd.DataFrame(rows), oof


def cv_significance(cv: pd.DataFrame, selected: dict, cfg: DatasetConfig, n_train: int) -> pd.DataFrame:
    """Compara el learner seleccionado contra cada alternativa con los mismos folds de la CV.

    Usa el t-test corregido de Nadeau & Bengio: con folds solapados el t-test
    clásico declara diferencias significativas que no lo son.
    """
    n_te = n_train / cfg.modeling["cv_folds"]
    rows = []
    for o, d in cv.groupby("outcome", sort=False):
        best = d[d.learner == selected[o]].set_index("fold")["qini_rel_mean"]
        for rival in d.learner.unique():
            if rival == selected[o]:
                continue
            other = d[d.learner == rival].set_index("fold")["qini_rel_mean"]
            diffs = (best - other.reindex(best.index)).to_numpy()
            t_stat, p = corrected_resampled_ttest(diffs, n_train - n_te, n_te)
            rows.append({"outcome": o, "selected": selected[o], "rival": rival, "mean_diff": float(diffs.mean()),
                         "t_corrected": t_stat, "p_value": p, "n_folds": len(diffs)})
    return pd.DataFrame(rows)


def significance_on_test(results: dict, test_uplift: dict, test: pd.DataFrame, t_te: np.ndarray,
                      selected: dict, cfg: DatasetConfig, n_boot: int, n_perm: int) -> pd.DataFrame:
    """Significancia en test del learner seleccionado, por outcome y tratamiento.

    * ``qini_perm_pvalue``: ¿el ranking es mejor que el azar?
    * ``diff_vs_baseline``: Qini(seleccionado) − Qini(baseline) con bootstrap pareado.
    * eventos por grupo y efecto mínimo detectable (ATE) con 80 % de potencia.
    Agrega los mismos campos a ``results`` (métricas del learner seleccionado).
    """
    rows = []
    for o in cfg.outcomes:
        y_te, sel = test[o.name].to_numpy(dtype=float), selected[o.name]
        for k, arm in enumerate(cfg.treatments, start=1):
            mask = (t_te == 0) | (t_te == k)
            yk, wk = y_te[mask], (t_te[mask] == k).astype(int)
            s_sel = test_uplift[o.name][sel][mask, k - 1]
            s_base = test_uplift[o.name]["baseline"][mask, k - 1]
            p_perm = permutation_pvalue(qini_relative, yk, wk, s_sel, n_perm=n_perm, seed=k)
            d = paired_bootstrap_diff(qini_relative, yk, wk, s_sel, s_base, n_boot=n_boot, seed=k)
            info = {"qini_perm_pvalue": p_perm, "qini_diff_vs_baseline": d["diff"],
                    "qini_diff_vs_baseline_ci95": d["ci95"], "qini_diff_vs_baseline_pvalue": d["p_value"],
                    **event_counts(yk, wk), "ate": float(yk[wk == 1].mean() - yk[wk == 0].mean()),
                    "ate_mde_80": ate_mde(yk, wk)}
            results[o.name][sel][arm.label].update(info)
            row = {"outcome": o.name, "treatment": arm.label, "learner": sel,
                   "qini_relative": results[o.name][sel][arm.label]["qini_relative"]}
            row.update({k2: v for k2, v in info.items() if not isinstance(v, list)})
            row["diff_ci95_lo"], row["diff_ci95_hi"] = d["ci95"]
            rows.append(row)
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
    cv, oof = cross_validate(X_tr, t_tr, train, cfg)
    cv.to_csv(cfg.tables_dir / "cv_results.csv", index=False)
    cv_summary = cv.groupby(["outcome", "learner"])["qini_rel_mean"].agg(["mean", "std", "median"]).reset_index()
    cv_summary["sem"] = cv_summary["std"] / np.sqrt(m["cv_folds"] * m["cv_repeats"])
    cv_summary.to_csv(cfg.tables_dir / "cv_summary.csv", index=False)
    causal = cv_summary[cv_summary.learner != "baseline"]
    selected = causal.loc[causal.groupby("outcome")["mean"].idxmax()].set_index("outcome")["learner"].to_dict()
    print("    Seleccionados:", selected)
    cv_sig = cv_significance(cv, selected, cfg, len(train))
    cv_sig.to_csv(cfg.tables_dir / "cv_significance.csv", index=False)

    print("[4/6] Entrenamiento final y evaluación en test")
    results, fitted, test_uplift = {}, {}, {}
    for o in cfg.outcomes:
        y_tr, y_te = train[o.name].to_numpy(dtype=float), test[o.name].to_numpy(dtype=float)
        results[o.name], fitted[o.name], test_uplift[o.name] = {}, {}, {}
        for key, learner in build_learners(o.binary, cfg.n_treatments, o.hurdle, cfg.modeling["extra_learners"]).items():
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

    print(f"    Significancia en test ({m['n_perm']} permutaciones, {n_boot} bootstrap pareados)")
    sig = significance_on_test(results, test_uplift, test, t_te, selected, cfg, n_boot, m["n_perm"])
    sig.to_csv(cfg.tables_dir / "test_significance.csv", index=False)
    for r in sig.itertuples():
        print(f"    {r.outcome}/{r.treatment}: eventos T={r.events_treated} C={r.events_control} · "
              f"p(perm)={r.qini_perm_pvalue:.3f} · Δ vs baseline={r.qini_diff_vs_baseline:+.3f} "
              f"(p={r.qini_diff_vs_baseline_pvalue:.3f})")

    print("[5/6] Presupuesto elegido en validación (OOF de train) y evaluación de políticas en test (IPW)")
    budget_sel = {}
    for o in cfg.outcome_names:
        crit = budget_criterion(cfg, o)
        # Validación: curva sobre predicciones out-of-fold de train → se elige el presupuesto
        val_curve = budget_curve(oof[o][selected[o]], train, t_tr, cfg)
        val_curve.to_csv(cfg.tables_dir / f"budget_curve_{o}_validation.csv", index=False)
        b = select_budget(val_curve, crit)
        # Test: la política ya está fija; sólo se mide
        u = test_uplift[o][selected[o]]
        test_curve = budget_curve(u, test, t_te, cfg)
        test_curve.to_csv(cfg.tables_dir / f"budget_curve_{o}.csv", index=False)
        bench = benchmark_policies(u, test, t_te, cfg, budget=0.5, selected_budget=b)
        bench.to_csv(cfg.tables_dir / f"policy_benchmark_{o}.csv", index=False)
        lo_k, hi_k = ("profit_lo", "profit_hi") if crit == "profit_per_1000" else (f"inc_{o}_lo", f"inc_{o}_hi")
        at_b = bench.iloc[-1]
        budget_sel[o] = {"budget": b, "criterion": crit,
                         "validation_value": float(val_curve.set_index("budget").loc[b, crit]),
                         "test_value": float(at_b[crit]), "test_ci95": [float(at_b[lo_k]), float(at_b[hi_k])],
                         "share_contacted_test": float(at_b["share_contacted"])}
        print(f"    {o}: presupuesto {b:.0%} (elegido en validación) → test {crit} = {at_b[crit]:,.2f}")
    sur, sur_test, sur_oof = run_surrogate(cfg, train, test, X_tr, X_te, t_tr, t_te, oof, test_uplift, selected,
                                           budget_sel)
    with open(cfg.reports_dir / "budget_selection.json", "w") as f:
        json.dump(budget_sel, f, indent=2)

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
        "selected_budget": {o: v["budget"] for o, v in budget_sel.items()},
        "surrogate": sur,
        "test_metrics": {o: results[o][selected[o]] for o in cfg.outcome_names},
    }
    joblib.dump(bundle, cfg.bundle_path, compress=3)
    np.savez_compressed(cfg.models_dir / "test_uplift.npz",
                        **{f"{o}__{k}": v.astype(np.float32) for o in cfg.outcome_names
                           for k, v in test_uplift[o].items()},
                        **({"surrogate__index": sur_test.astype(np.float32)} if sur is not None else {}))
    # Uplift out-of-fold del learner seleccionado (para elegir presupuesto en el dashboard sin mirar test)
    np.savez_compressed(cfg.models_dir / "train_oof_uplift.npz",
                        **{o: oof[o][selected[o]].astype(np.float32) for o in cfg.outcome_names},
                        **({"surrogate": sur_oof.astype(np.float32)} if sur is not None else {}))
    with open(cfg.reports_dir / "metrics.json", "w") as f:
        json.dump({"dataset": cfg.name, "selected": selected, "budget_selection": budget_sel, "results": results,
                   "n_rows": bundle["n_rows"], "cv_summary": cv_summary.to_dict(orient="records"),
                   "cv_significance": cv_sig.to_dict(orient="records")}, f, indent=2)

    if use_mlflow:
        log_mlflow(cfg, results, cv_summary, selected)
    print(f"Listo en {time.time() - t0:.0f}s")
    return {"selected": selected, "results": results, "summary": summary, "cv_summary": cv_summary,
            "budget_selection": budget_sel}


def run_surrogate(cfg: DatasetConfig, train, test, X_tr, X_te, t_tr, t_te, oof, test_uplift, selected,
                  budget_sel: dict):
    """Política valorizada con el índice sustituto (ver ``uplift.surrogate``).

    Usa el CATE en visita del learner seleccionado: out-of-fold en train para elegir el
    presupuesto (con μ_0 también cross-fitted) y el modelo final en test para evaluar.
    """
    sur = SurrogateIndex.from_config(cfg)
    if sur is None:
        return None, None, None
    print("[6b] Índice sustituto: valorización de la visita con el embudo de conversión")
    sur.fit(train, t_tr, X_tr, cfg)
    p = sur.proxy
    w = cfg.value_defaults.get(sur.value, 0.0)
    cost = cfg.business["cost_per_contact"]
    s_tr, s_te = sur.stratum(train, cfg), sur.stratum(test, cfg)
    v_oof = sur.value_uplift(oof[p][selected[p]], sur.mu0_oof(X_tr, t_tr, train, cfg.modeling["cv_folds"]), s_tr)
    v_te = sur.value_uplift(test_uplift[p][selected[p]], sur.mu0(X_te), s_te)
    ys_tr, ys_te = sur.outcome(train, t_tr, cfg), sur.outcome(test, t_te, cfg)
    crit = "profit_sur_per_1000" if w else "inc_surrogate_per_1000"
    val_curve = budget_curve(w * v_oof - cost, train, t_tr, cfg, surrogate=ys_tr)
    val_curve.to_csv(cfg.tables_dir / "budget_curve_surrogate_validation.csv", index=False)
    b = select_budget(val_curve, crit)
    budget_curve(w * v_te - cost, test, t_te, cfg, surrogate=ys_te).to_csv(
        cfg.tables_dir / "budget_curve_surrogate.csv", index=False)
    bench = benchmark_policies(w * v_te - cost, test, t_te, cfg, budget=0.5, selected_budget=b, surrogate=ys_te)
    bench.to_csv(cfg.tables_dir / "policy_benchmark_surrogate.csv", index=False)
    sur.summary().to_csv(cfg.tables_dir / "surrogate_params.csv", index=False)
    # Valor del ranking: modelo vs. el mejor tratamiento enviado al azar con el mismo presupuesto
    ranking_gain(w * v_te - cost, test, t_te, cfg, ys_te, weight=w or 1.0, cost=cost if w else 0.0).to_csv(
        cfg.tables_dir / "surrogate_ranking_gain.csv", index=False)
    chk = sur.check(test, t_te, cfg)
    chk.to_csv(cfg.tables_dir / "surrogate_check.csv", index=False)
    at_b = bench.iloc[-1]
    obs_crit = "profit_per_1000" if w else f"inc_{sur.value}_per_1000"
    budget_sel["surrogate"] = {
        "budget": b, "criterion": crit,
        "validation_value": float(val_curve.set_index("budget").loc[b, crit]),
        "test_value_surrogate": float(at_b.get(crit, np.nan)),
        "test_value_observed": float(at_b.get(obs_crit, np.nan)),
        "share_contacted_test": float(at_b["share_contacted"])}
    for r in chk[chk.estimate == "diferencia"].itertuples():
        print(f"    {r.treatment}: ATE(observado) − ATE(sustituto) = {r.ate:+.3f} [{r.ci_lo:+.3f}; {r.ci_hi:+.3f}]")
    print(f"    presupuesto {b:.0%} (validación) → test: sustituto {at_b.get(crit, np.nan):,.2f} · "
          f"observado {at_b.get(obs_crit, np.nan):,.2f}")
    return sur, v_te, v_oof


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
