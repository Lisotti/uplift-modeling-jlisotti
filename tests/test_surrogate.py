"""Tests del índice sustituto (visita valorizada con el embudo de conversión)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from uplift.dataset import load_config
from uplift.features import FeatureBuilder
from uplift.policy import evaluate_action, value_matrix
from uplift.surrogate import SurrogateIndex, _beta_prior_strength, _shrink_means

CFG = load_config("hillstrom")


def funnel(n=60_000, seed=0):
    """Datos sintéticos que cumplen el supuesto del embudo, con parámetros conocidos.

    P(visita) depende del brazo y de `mens`; P(compra | visita) depende sólo del brazo
    (0,05 / 0,07 / 0,06) y el ticket es exponencial con media 100.
    """
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "recency": rng.integers(1, 13, n), "history": rng.gamma(2, 120, n),
        "mens": rng.integers(0, 2, n), "womens": rng.integers(0, 2, n),
        "zip_code": rng.choice(["Rural", "Surburban", "Urban"], n), "newbie": rng.integers(0, 2, n),
        "channel": rng.choice(["Multichannel", "Phone", "Web"], n),
    })
    t = rng.integers(0, 3, n)
    p_visit = 0.10 + 0.08 * ((t == 1) & (X.mens == 1)) + 0.04 * (t == 2)
    visit = rng.binomial(1, p_visit)
    c = np.array([0.05, 0.07, 0.06])[t]
    conv = visit * rng.binomial(1, c)
    spend = conv * rng.exponential(100, n)
    df = X.assign(segment=np.array(["No E-Mail", "Mens E-Mail", "Womens E-Mail"])[t],
                  visit=visit, conversion=conv, spend=spend)
    return df, t, FeatureBuilder.from_config(CFG).transform(X)


@pytest.fixture(scope="module")
def fitted():
    df, t, X = funnel()
    return SurrogateIndex.from_config(CFG).fit(df, t, X, CFG), df, t, X


def test_from_config_and_validation():
    s = SurrogateIndex.from_config(CFG)
    assert (s.proxy, s.conversion, s.value, s.strata) == ("visit", "conversion", "spend", "categoria")
    bad = type(CFG)(raw={**CFG.raw, "surrogate": {**CFG.raw["surrogate"], "proxy": "no_existe"}})
    with pytest.raises(ValueError):
        SurrogateIndex.from_config(bad)
    assert SurrogateIndex.from_config(type(CFG)(raw={k: v for k, v in CFG.raw.items() if k != "surrogate"})) is None


def test_fit_recovers_funnel(fitted):
    s, *_ = fitted
    rates = s.rates_.groupby("arm")["rate_arm"].first()
    assert rates.to_numpy() == pytest.approx([0.05, 0.07, 0.06], abs=0.006)
    # sin heterogeneidad real entre estratos el encogimiento lleva cada tasa al promedio del brazo
    r = s.rates_
    assert np.abs(r.rate - r.rate_arm).max() < 0.004
    assert s.tickets_.ticket.to_numpy() == pytest.approx(100, rel=0.12)


def test_value_uplift_formula(fitted):
    s, df, t, X = fitted
    strata = s.stratum(df.head(3), CFG)
    R, T = s._lookup(strata)
    tau, mu0 = np.array([[0.1, 0.02]] * 3), np.full(3, 0.1)
    v = s.value_uplift(tau, mu0, strata)
    expected_mens = (0.1 * R[:, 1] + 0.1 * (R[:, 1] - R[:, 0])) * T
    assert v.shape == (3, 2) and v[:, 0] == pytest.approx(expected_mens)


def test_surrogate_outcome_matches_observed_effect(fitted):
    s, df, t, X = fitted
    chk = s.check(df, t, CFG).set_index(["treatment", "estimate"])
    for arm in ("mens", "womens"):
        lo, hi = chk.loc[(arm, "diferencia"), ["ci_lo", "ci_hi"]]
        assert lo < 0 < hi  # el supuesto se cumple por construcción
        # el sustituto estima el mismo efecto con mucha menos varianza
        assert chk.loc[(arm, "sustituto"), "se"] < 0.5 * chk.loc[(arm, "observado"), "se"]


def test_policy_evaluation_with_surrogate(fitted):
    s, df, t, X = fitted
    ys = s.outcome(df, t, CFG)
    r = evaluate_action(np.ones(len(t), dtype=int), df, t, CFG, surrogate=ys)
    assert r["profit_sur_lo"] < r["profit_sur_per_1000"] < r["profit_sur_hi"]
    assert (r["profit_sur_hi"] - r["profit_sur_lo"]) < (r["profit_hi"] - r["profit_lo"])
    u = {"surrogate": np.array([[2.0, 0.0]])}
    assert np.allclose(value_matrix(u, CFG, "profit_surrogate", {"spend": 0.4}, 0.1), [[0.7, -0.1]])


def test_shrinkage_helpers():
    # homogéneo ⇒ encogimiento total; heterogéneo ⇒ prior finito
    assert np.isinf(_beta_prior_strength(np.array([50, 51, 49]), np.array([1000, 1000, 1000])))
    assert np.isfinite(_beta_prior_strength(np.array([10, 100, 200]), np.array([1000, 1000, 1000])))
    shr = _shrink_means(np.array([50.0, 150.0]), np.array([900.0, 900.0]), np.array([5, 500]), 100.0)
    assert abs(shr[0] - 100) < abs(50 - 100) and abs(shr[1] - 150) < 5


def test_ranking_gain_is_paired_and_exact(fitted):
    from uplift.policy import ranking_gain
    s, df, t, X = fitted
    ys = s.outcome(df, t, CFG)
    rng = np.random.default_rng(1)
    value = np.column_stack([rng.normal(size=len(t)), rng.normal(size=len(t)) - 5])  # ranking sin información
    g = ranking_gain(value, df, t, CFG, ys, weight=0.4, cost=0.1, budgets=[0.2, 0.5])
    # el azar con el mejor brazo vale exactamente b × el envío masivo
    full = ranking_gain(value, df, t, CFG, ys, weight=0.4, cost=0.1, budgets=[1.0]).random_best_arm_per_1000.iloc[0]
    assert g.random_best_arm_per_1000.to_numpy() == pytest.approx([0.2 * full, 0.5 * full])
    # un ranking aleatorio no le gana significativamente al azar
    assert (g.gain_lo < 0).all() and (g.gain_hi > 0).all()
    assert set(g.columns) >= {"budget", "model_per_1000", "gain_per_1000", "gain_lo", "gain_hi", "best_arm"}
