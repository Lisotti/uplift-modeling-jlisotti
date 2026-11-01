"""Política de targeting multi-tratamiento y evaluación off-policy sobre el RCT.

Como la asignación fue aleatoria con probabilidades conocidas p(T=k), el valor
esperado de cualquier política π se estima sin sesgo con *Inverse Propensity
Weighting* (IPW):

    V(π) = (1/n) Σ_i  Y_i · 1{T_i = π(x_i)} / p(T_i)

El incremental frente a "no contactar" es la media de las contribuciones
individuales ``Y_i·(1{T_i=π(x_i)}/p(T_i) − 1{T_i=0}/p(0))``; al ser una media de
términos independientes, su intervalo de confianza al 95 % se obtiene con la
aproximación normal (media ± 1,96·d.e./√n).

Beneficio = Σ_o valor_o × incremental_o − costo por contacto × contactados, con
los valores por unidad de outcome definidos en ``business.values`` de la config.

El presupuesto de la política es un hiperparámetro: se elige sobre un conjunto de
validación (predicciones out-of-fold de train, ver ``train.cross_validate``) con
``select_budget`` y recién después se evalúa, ya fijo, en test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import RANDOM_STATE
from .dataset import DatasetConfig

Z95 = 1.959964


def assign(uplift: np.ndarray, budget: float = 1.0, min_uplift: float = 0.0) -> np.ndarray:
    """Asigna a cada cliente el tratamiento de mayor uplift, o 0 (no contactar).

    * ``budget``: fracción máxima de la base a contactar (0–1). Se priorizan los
      clientes con mayor uplift máximo.
    * ``min_uplift``: umbral mínimo de uplift para que convenga contactar.
    """
    uplift = np.asarray(uplift, dtype=float)
    if uplift.ndim == 1:
        uplift = uplift[:, None]
    best_k = uplift.argmax(axis=1) + 1
    best_u = uplift.max(axis=1)
    action = np.where(best_u > min_uplift, best_k, 0)
    n_max = int(np.floor(budget * len(uplift)))
    if (action > 0).sum() > n_max:
        order = np.argsort(-best_u, kind="mergesort")
        keep = np.zeros(len(uplift), dtype=bool)
        keep[order[:n_max]] = True
        action = np.where(keep, action, 0)
    return action


def propensities(t: np.ndarray) -> np.ndarray:
    return np.bincount(np.asarray(t, dtype=int)) / len(t)


def ipw_value(y, t, action, propensity=None) -> float:
    """Valor medio por cliente de la política ``action`` estimado por IPW."""
    y, t, action = np.asarray(y, float), np.asarray(t, int), np.asarray(action, int)
    p = propensities(t) if propensity is None else np.asarray(propensity)
    return float(np.mean(y * (t == action) / p[t]))


def ipw_contrib(y, t, action, propensity=None) -> np.ndarray:
    """Contribución individual al incremental de la política frente a no contactar."""
    y, t, action = np.asarray(y, float), np.asarray(t, int), np.asarray(action, int)
    p = propensities(t) if propensity is None else np.asarray(propensity)
    return y * ((t == action) / p[t] - (t == 0) / p[0])


def mean_ci(contrib: np.ndarray) -> tuple[float, float, float]:
    m = float(np.mean(contrib))
    h = Z95 * float(np.std(contrib, ddof=1)) / np.sqrt(len(contrib))
    return m, m - h, m + h


def evaluate_action(action, df: pd.DataFrame, t, cfg: DatasetConfig,
                    values: dict | None = None, cost: float | None = None,
                    surrogate: np.ndarray | None = None) -> dict:
    """KPIs incrementales por cada 1.000 clientes de una política, con IC 95 %.

    ``surrogate`` (opcional) es el resultado sustituto observado Y* de cada cliente
    (``SurrogateIndex.outcome``): agrega el incremental y el beneficio estimados con
    el índice sustituto (``inc_surrogate_*`` y ``profit_sur_*``).
    """
    values = cfg.value_defaults if values is None else values
    cost = cfg.business["cost_per_contact"] if cost is None else cost
    action = np.asarray(action, dtype=int)
    p = propensities(t)
    r = {"share_contacted": float((action > 0).mean())}
    for k, arm in enumerate(cfg.treatments, start=1):
        r[f"share_{arm.label}"] = float((action == k).mean())
    value_contrib = np.zeros(len(action))
    for o in cfg.outcome_names:
        c = 1000 * ipw_contrib(df[o].to_numpy(), t, action, p)
        r[f"inc_{o}_per_1000"], r[f"inc_{o}_lo"], r[f"inc_{o}_hi"] = mean_ci(c)
        if o in values:
            value_contrib += values[o] * c
    r["cost_per_1000"] = 1000 * r["share_contacted"] * cost
    if values:
        v, lo, hi = mean_ci(value_contrib)
        r["profit_per_1000"] = v - r["cost_per_1000"]
        r["profit_lo"], r["profit_hi"] = lo - r["cost_per_1000"], hi - r["cost_per_1000"]
        r["roi"] = r["profit_per_1000"] / r["cost_per_1000"] if r["cost_per_1000"] > 0 else np.nan
    if surrogate is not None:
        c = 1000 * ipw_contrib(surrogate, t, action, p)
        r["inc_surrogate_per_1000"], r["inc_surrogate_lo"], r["inc_surrogate_hi"] = mean_ci(c)
        w = values.get(cfg.raw["surrogate"]["value"], 0.0) if values else 0.0
        if w:
            v, lo, hi = mean_ci(w * c)
            r["profit_sur_per_1000"] = v - r["cost_per_1000"]
            r["profit_sur_lo"], r["profit_sur_hi"] = lo - r["cost_per_1000"], hi - r["cost_per_1000"]
    return r


def budget_curve(uplift: np.ndarray, df: pd.DataFrame, t, cfg: DatasetConfig,
                 budgets=np.linspace(0, 1, 21), values=None, cost=None, surrogate=None) -> pd.DataFrame:
    """Simula la política del modelo para distintos presupuestos (fracción contactada)."""
    rows = []
    for b in budgets:
        rows.append({"budget": float(b), **evaluate_action(assign(uplift, budget=b), df, t, cfg, values, cost,
                                                           surrogate)})
    return pd.DataFrame(rows)


def budget_criterion(cfg: DatasetConfig, outcome: str) -> str:
    """Columna de ``budget_curve`` que se maximiza al elegir el presupuesto.

    Beneficio neto si la config valoriza algún outcome; si no, el incremental del
    outcome con el que se rankea.
    """
    return "profit_per_1000" if cfg.business["values"] else f"inc_{outcome}_per_1000"


def select_budget(curve: pd.DataFrame, criterion: str) -> float:
    """Presupuesto que maximiza ``criterion`` en una curva calculada sobre VALIDACIÓN.

    Nunca debe aplicarse a la curva de test: elegir el máximo sobre los mismos
    datos en los que se reporta sesga el resultado hacia arriba (data leakage).
    Ante empates se toma el presupuesto más chico (menos contactos).
    """
    return float(curve.loc[curve[criterion].idxmax(), "budget"])


def benchmark_policies(uplift: np.ndarray, df: pd.DataFrame, t, cfg: DatasetConfig,
                       budget: float, values=None, cost=None,
                       selected_budget: float | None = None, surrogate=None) -> pd.DataFrame:
    """Compara la política del modelo con políticas de referencia.

    ``selected_budget`` (opcional) agrega, como última fila, la política del modelo
    con el presupuesto elegido en validación.
    """
    n = len(t)
    rng = np.random.default_rng(RANDOM_STATE)
    n_send = int(np.floor(budget * n))
    rand_sel = np.zeros(n, dtype=int)
    idx = rng.choice(n, n_send, replace=False)
    rand_sel[idx] = rng.integers(1, cfg.n_treatments + 1, n_send)

    policies = {"No enviar": np.zeros(n, dtype=int)}
    for k, arm in enumerate(cfg.treatments, start=1):
        policies[f"Enviar a todos — {arm.name}"] = np.full(n, k)
    policies[f"Aleatoria ({budget:.0%})"] = rand_sel
    policies[f"Modelo ({budget:.0%})"] = assign(uplift, budget=budget)
    policies["Modelo (sin restricción, uplift>0)"] = assign(uplift, budget=1.0)
    if selected_budget is not None:
        policies[f"Modelo (presupuesto de validación, {selected_budget:.0%})"] = assign(uplift, budget=selected_budget)
    rows = [{"policy": name, **evaluate_action(a, df, t, cfg, values, cost, surrogate)}
            for name, a in policies.items()]
    return pd.DataFrame(rows)


def ranking_gain(value: np.ndarray, df: pd.DataFrame, t, cfg: DatasetConfig, outcome: np.ndarray,
                 weight: float, cost: float, budgets=np.round(np.arange(0.1, 1.01, 0.1), 2)) -> pd.DataFrame:
    """Valor del ranking: política del modelo vs. el mejor tratamiento enviado a una fracción al azar.

    Para cada presupuesto b compara (a) la política del modelo, que elige a quién y qué
    tratamiento, con (b) enviar el tratamiento que más rinde en el envío masivo a una
    fracción b de la base elegida al azar. Ambas contactan a la misma cantidad de
    clientes; la diferencia es el valor de elegir *a quién*.

    El valor esperado de (b) es exactamente ``b ×`` el del envío masivo (cada cliente
    entra con probabilidad b), así que la comparación es pareada y sin sorteos: la
    contribución individual de (b) es ``b · c_i(masivo)``. ``outcome`` es el resultado
    que se valoriza (p. ej. el sustituto Y*) y ``weight`` su valor por unidad.
    """
    t = np.asarray(t, dtype=int)
    p = propensities(t)
    n = len(t)

    def contrib(action):
        action = np.asarray(action, dtype=int)
        return weight * 1000 * ipw_contrib(outcome, t, action, p) - 1000 * cost * (action > 0)

    mass = {k: contrib(np.full(n, k)) for k in range(1, cfg.n_treatments + 1)}
    best_k = max(mass, key=lambda k: mass[k].mean())
    rows = []
    for b in budgets:
        a = assign(value, budget=float(b))
        cm, cr = contrib(a), b * mass[best_k]
        d, lo, hi = mean_ci(cm - cr)
        row = {"budget": float(b), "share_contacted": float((a > 0).mean())}
        row.update({f"share_{arm.label}": float((a == k).mean()) for k, arm in enumerate(cfg.treatments, start=1)})
        row.update({"model_per_1000": float(cm.mean()), "random_best_arm_per_1000": float(cr.mean()),
                    "best_arm": cfg.treatments[best_k - 1].label, "gain_per_1000": d, "gain_lo": lo, "gain_hi": hi,
                    "gain_pct": d / cr.mean() if cr.mean() else np.nan})
        rows.append(row)
    return pd.DataFrame(rows)


def value_matrix(uplifts: dict[str, np.ndarray], cfg: DatasetConfig, objective: str,
                 values: dict | None = None, cost: float | None = None) -> np.ndarray:
    """Matriz n × K con el valor esperado de contactar con cada tratamiento.

    * ``objective == "profit"``: Σ_o valor_o × uplift_o − costo por contacto.
    * ``objective == "profit_surrogate"``: valor × ingreso incremental del índice
      sustituto (``uplifts["surrogate"]``) − costo por contacto.
    * otro: el uplift del outcome indicado.
    """
    if objective not in ("profit", "profit_surrogate"):
        return uplifts[objective]
    values = cfg.value_defaults if values is None else values
    cost = cfg.business["cost_per_contact"] if cost is None else cost
    if objective == "profit_surrogate":
        return values.get(cfg.raw["surrogate"]["value"], 0.0) * uplifts["surrogate"] - cost
    first = next(iter(uplifts.values()))
    v = np.zeros_like(first, dtype=float)
    for o, w in values.items():
        if o not in uplifts:
            continue
        v = v + w * uplifts[o]
    return v - cost
