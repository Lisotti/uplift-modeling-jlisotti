"""Test end-to-end del pipeline genérico con un dataset sintético estilo Criteo.

Verifica que, dada solamente una config YAML, el pipeline completo (EDA,
entrenamiento, monitoreo y figuras) funcione con un tratamiento binario,
features numéricas anónimas y una variable post-tratamiento excluida.
"""

from __future__ import annotations

import shutil

import numpy as np
import pandas as pd
import pytest
import yaml

from uplift.dataset import load_config


@pytest.fixture(scope="module")
def synthetic_cfg(tmp_path_factory):
    d = tmp_path_factory.mktemp("synthetic")
    rng = np.random.default_rng(7)
    n = 30_000
    X = rng.normal(size=(n, 4)).astype("float32")
    t = (rng.random(n) < 0.8).astype(int)
    logit = -2.0 + 0.5 * X[:, 0] + t * (0.3 + 0.8 * (X[:, 1] > 0))
    visit = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    conv = (visit * (rng.random(n) < 0.3)).astype(int)
    df = pd.DataFrame(X, columns=["f0", "f1", "f2", "f3"])
    df["treatment"], df["visit"], df["conversion"] = t, visit, conv
    df["exposure"] = t
    df.to_csv(d / "data.csv", index=False)
    cfg = {
        "name": "pytest_synthetic", "title": "Sintético",
        "source": {"path": str(d / "data.csv")},
        "treatment": {"column": "treatment", "control": {"value": 0, "label": "control", "name": "Control"},
                      "arms": [{"value": 1, "label": "ads", "name": "Ads"}], "none_action": "no_ads"},
        "outcomes": [{"name": "visit", "type": "binary", "label": "Visita"},
                     {"name": "conversion", "type": "binary", "label": "Conversión"}],
        "primary_outcome": "visit", "stratify_outcome": "conversion",
        "features": [{"name": f"f{i}", "type": "numeric"} for i in range(4)],
        "excluded_columns": {"exposure": "post-tratamiento"},
        "validation": {"implies": [["conversion", "visit"]]},
        "modeling": {"cv_folds": 2, "cv_repeats": 1, "n_boot": 10, "n_jobs": 2},
        "business": {"cost_per_contact": 0.01, "values": {"conversion": {"label": "Valor", "default": 20.0}}},
        "surrogate": {"proxy": "visit", "conversion": "conversion", "value": "conversion", "strata": None},
        "monitoring": {"n_windows": 2,
                       "drift_scenario": [{"column": "f0", "op": "quantile_ge", "value": 0.75, "weight": 3.0}]},
    }
    path = d / "pytest_synthetic.yaml"
    path.write_text(yaml.safe_dump(cfg))
    c = load_config(str(path))
    yield c
    shutil.rmtree(c.artifacts_dir, ignore_errors=True)


def test_end_to_end(synthetic_cfg):
    from uplift import eda, figures, monitoring, train
    from uplift.predict import load_bundle, score

    summary = eda.run(synthetic_cfg)
    assert summary["viable"]
    out = train.run(synthetic_cfg, use_mlflow=False)
    assert set(out["selected"]) == {"visit", "conversion"}
    assert synthetic_cfg.bundle_path.exists()
    # El presupuesto se elige en validación (OOF de train) y se guarda junto al modelo
    sel = out["budget_selection"]
    assert {"visit", "conversion"} <= set(sel) and all(0 <= v["budget"] <= 1 for v in sel.values())
    assert (synthetic_cfg.reports_dir / "budget_selection.json").exists()
    assert (synthetic_cfg.models_dir / "train_oof_uplift.npz").exists()
    # Pruebas de significancia (CV con t-test corregido y test con permutación / bootstrap pareado)
    sig = pd.read_csv(synthetic_cfg.tables_dir / "test_significance.csv")
    assert len(sig) == 2 and sig.qini_perm_pvalue.between(0, 1).all() and (sig.events_treated > 0).all()
    assert (synthetic_cfg.tables_dir / "cv_significance.csv").exists()
    # Índice sustituto: política valorizada, validación del supuesto y objetivo en el scoring
    assert "surrogate" in sel and 0 <= sel["surrogate"]["budget"] <= 1
    for f in ("budget_curve_surrogate.csv", "policy_benchmark_surrogate.csv", "surrogate_check.csv",
              "surrogate_ranking_gain.csv"):
        assert (synthetic_cfg.tables_dir / f).exists()

    customers = pd.DataFrame([{f"f{i}": 0.5 for i in range(4)}, {f"f{i}": -1.0 for i in range(4)}])
    sc = score(customers, synthetic_cfg, load_bundle(synthetic_cfg), objective="profit")
    assert set(sc.recommended_treatment) <= {"ads", "no_ads"}
    assert {"uplift_visit_ads", "uplift_conversion_ads", "expected_profit_usd"} <= set(sc.columns)
    sc2 = score(customers, synthetic_cfg, load_bundle(synthetic_cfg), objective="profit_surrogate")
    assert "expected_value_surrogate_ads" in sc2.columns and set(sc2.recommended_treatment) <= {"ads", "no_ads"}

    mon = monitoring.run(synthetic_cfg,
                         use_evidently=False)
    assert mon.alert.iloc[-1]  # la ventana con drift inducido dispara la alerta
    meta = figures.run(synthetic_cfg)
    assert (synthetic_cfg.figures_dir / "qini_curves.png").exists()
    assert meta["surrogate_arm"] == "ads"
    assert (synthetic_cfg.figures_dir / "surrogate_index.png").exists()
