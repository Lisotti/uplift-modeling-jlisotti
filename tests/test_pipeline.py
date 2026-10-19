"""Tests unitarios del pipeline de uplift (sobre la config de Hillstrom)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from uplift.data import DataValidationError, load_raw, sample, split, treatment_index, validate
from uplift.dataset import available_configs, load_config
from uplift.features import FeatureBuilder, add_segments
from uplift.learners import HurdleRegressor, SLearner, TLearner, XLearner, make_base_learner
from uplift.metrics import auuc_normalized, qini_coefficient, qini_relative, uplift_at_k
from uplift.policy import assign, evaluate_action, ipw_value, value_matrix

CFG = load_config("hillstrom")
FEATURES = ["recency", "history", "log_history", "mens", "womens", "both_categories", "newbie",
            "zip_rural", "zip_surburban", "zip_urban", "channel_multichannel", "channel_phone", "channel_web"]


@pytest.fixture(scope="module")
def raw():
    return load_raw(CFG)


def synthetic(n=6000, seed=0):
    """Datos sintéticos con efecto heterogéneo conocido: sólo los clientes `mens=1`
    responden al tratamiento 1 y sólo `womens=1` al tratamiento 2."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "recency": rng.integers(1, 13, n), "history": rng.gamma(2, 120, n),
        "mens": rng.integers(0, 2, n), "womens": rng.integers(0, 2, n),
        "zip_code": rng.choice(["Rural", "Surburban", "Urban"], n), "newbie": rng.integers(0, 2, n),
        "channel": rng.choice(["Multichannel", "Phone", "Web"], n),
    })
    t = rng.integers(0, 3, n)
    p = 0.10 + 0.25 * ((t == 1) & (X.mens == 1)) + 0.25 * ((t == 2) & (X.womens == 1))
    y = rng.binomial(1, p)
    return FeatureBuilder.from_config(CFG).transform(X), t, y, X


# ── Configs ───────────────────────────────────────────────────────
def test_configs_load():
    assert {"hillstrom", "criteo"} <= set(available_configs())
    c = load_config("criteo")
    assert c.n_treatments == 1 and c.outcome_names == ["visit", "conversion"]
    assert "exposure" in c.excluded_columns and "exposure" not in c.raw_features


# ── Datos ─────────────────────────────────────────────────────────
def test_raw_contract(raw):
    assert raw.shape == (64000, 12)
    assert set(raw[CFG.treatment_col].unique()) == {a.value for a in CFG.arms}


def test_validate_rejects_bad_recency(raw):
    bad = raw.head(10).copy()
    bad.loc[0, "recency"] = 99
    with pytest.raises(DataValidationError):
        validate(bad, CFG)


def test_validate_rejects_logic_violation(raw):
    bad = raw.head(10).copy()
    bad.loc[0, ["conversion", "visit"]] = [1, 0]
    with pytest.raises(DataValidationError):
        validate(bad, CFG)


def test_split_is_stratified(raw):
    tr, te = split(raw, CFG)
    assert len(te) == pytest.approx(0.3 * len(raw), abs=1)
    assert tr.conversion.mean() == pytest.approx(te.conversion.mean(), abs=5e-4)
    for arm in CFG.arms:
        assert (tr.segment == arm.value).mean() == pytest.approx((te.segment == arm.value).mean(), abs=2e-3)


def test_sample_keeps_proportions(raw):
    s = sample(raw, CFG, n=10_000)
    assert len(s) == 10_000
    assert s.conversion.mean() == pytest.approx(raw.conversion.mean(), abs=1e-3)


def test_treatment_index():
    s = pd.Series(["No E-Mail", "Mens E-Mail", "Womens E-Mail"])
    assert treatment_index(s, CFG).tolist() == [0, 1, 2]


# ── Features ──────────────────────────────────────────────────────
def test_feature_builder(raw):
    fb = FeatureBuilder.from_config(CFG)
    X = fb.fit_transform(raw.head(100))
    assert list(X.columns) == FEATURES == list(fb.get_feature_names_out())
    assert "history_segment" not in X.columns
    assert np.allclose(X.log_history, np.log1p(raw.head(100).history))
    assert (X[[c for c in X if c.startswith("zip_")]].sum(axis=1) == 1).all()


def test_feature_builder_handles_single_row():
    one = pd.DataFrame([dict(recency=3, history=100.0, mens=1, womens=0, zip_code="Rural",
                             newbie=1, channel="Phone")])
    assert FeatureBuilder.from_config(CFG).transform(one).shape == (1, len(FEATURES))


def test_segments(raw):
    d = add_segments(raw.head(500), CFG)
    assert set(d.categoria.unique()) <= {"Solo hombre", "Solo mujer", "Ambas"}


# ── Métricas ──────────────────────────────────────────────────────
def test_metrics_match_sklift():
    sk = pytest.importorskip("sklift.metrics")
    rng = np.random.default_rng(1)
    y, w = rng.integers(0, 2, 3000), rng.integers(0, 2, 3000)
    s = rng.normal(size=3000) + 0.5 * y * w
    assert auuc_normalized(y, w, s) == pytest.approx(sk.uplift_auc_score(y, s, w))
    assert qini_coefficient(y, w, s) == pytest.approx(sk.qini_auc_score(y, s, w))
    assert uplift_at_k(y, w, s, 0.3) == pytest.approx(sk.uplift_at_k(y, s, w, "overall", k=0.3))


def test_qini_relative_random_is_near_zero():
    rng = np.random.default_rng(2)
    w = rng.integers(0, 2, 20000)
    y = rng.binomial(1, 0.10 + 0.05 * w)  # efecto homogéneo: ningún ranking debería ganarle al azar
    assert abs(qini_relative(y, w, rng.normal(size=20000))) < 0.1


# ── Learners ──────────────────────────────────────────────────────
@pytest.mark.parametrize("Learner", [SLearner, TLearner, XLearner])
def test_learners_recover_heterogeneous_effect(Learner):
    X, t, y, raw = synthetic()
    u = Learner(base_learner=make_base_learner(True), n_treatments=2).fit(X, t, y).predict_uplift(X)
    assert u.shape == (len(X), 2)
    assert u[raw.mens == 1, 0].mean() - u[raw.mens == 0, 0].mean() > 0.12
    assert u[raw.womens == 1, 1].mean() - u[raw.womens == 0, 1].mean() > 0.12


def test_single_treatment_learner():
    X, t, y, raw = synthetic()
    keep = t < 2  # un único tratamiento (como Criteo)
    u = TLearner(base_learner=make_base_learner(True), n_treatments=1).fit(X[keep], t[keep], y[keep]).predict_uplift(X)
    assert u.shape == (len(X), 1)
    assert u[raw.mens == 1, 0].mean() > u[raw.mens == 0, 0].mean()


def test_linear_s_learner_with_interactions():
    X, t, y, raw = synthetic()
    u = SLearner(base_learner=make_base_learner(True, "linear"), n_treatments=2,
                 interactions=True).fit(X, t, y).predict_uplift(X)
    assert u[raw.mens == 1, 0].mean() > u[raw.mens == 0, 0].mean()


def test_hurdle_regressor_non_negative():
    X, _t, y, _ = synthetic(3000)
    spend = y * np.random.default_rng(0).gamma(2, 50, len(y))
    pred = HurdleRegressor().fit(X, spend).predict(X)
    assert (pred >= 0).all()
    assert pred.mean() == pytest.approx(spend.mean(), rel=0.35)


# ── Política ──────────────────────────────────────────────────────
def test_assign_respects_budget_and_threshold():
    u = np.array([[0.3, 0.1], [-0.1, -0.2], [0.05, 0.2], [0.4, 0.5]])
    assert assign(u).tolist() == [1, 0, 2, 2]
    assert assign(u, budget=0.5).tolist() == [1, 0, 0, 2]


def test_ipw_value_equals_arm_mean_for_constant_policy():
    rng = np.random.default_rng(3)
    t = rng.integers(0, 3, 9000)
    y = rng.binomial(1, 0.1 + 0.05 * (t == 1))
    assert ipw_value(y, t, np.ones_like(t)) == pytest.approx(y[t == 1].mean(), rel=1e-9)


def test_evaluate_action_ci_contains_point(raw):
    t = treatment_index(raw[CFG.treatment_col], CFG)
    r = evaluate_action(np.ones(len(t), dtype=int), raw, t, CFG)
    assert r["inc_visit_lo"] < r["inc_visit_per_1000"] < r["inc_visit_hi"]
    assert r["inc_visit_per_1000"] == pytest.approx(1000 * (raw.visit[t == 1].mean() - raw.visit[t == 0].mean()))
    assert "profit_per_1000" in r


def test_value_matrix_profit():
    u = {"visit": np.zeros((2, 2)), "conversion": np.zeros((2, 2)), "spend": np.array([[1.0, 0.0], [0.0, 2.0]])}
    v = value_matrix(u, CFG, "profit", {"spend": 0.5}, 0.1)
    assert np.allclose(v, [[0.4, -0.1], [-0.1, 0.9]])
