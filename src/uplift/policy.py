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
                    values: dict | None = None, cost: float | None = None) -> dict:
    """KPIs incrementales por cada 1.000 clientes de una política, con IC 95 %."""
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
    return r


def budget_curve(uplift: np.ndarray, df: pd.DataFrame, t, cfg: DatasetConfig,
                 budgets=np.linspace(0, 1, 21), values=None, cost=None) -> pd.DataFrame:
    """Simula la política del modelo para distintos presupuestos (fracción contactada)."""
    rows = []
    for b in budgets:
        rows.append({"budget": float(b), **evaluate_action(assign(uplift, budget=b), df, t, cfg, values, cost)})
    return pd.DataFrame(rows)


def benchmark_policies(uplift: np.ndarray, df: pd.DataFrame, t, cfg: DatasetConfig,
                       budget: float, values=None, cost=None) -> pd.DataFrame:
    """Compara la política del modelo con políticas de referencia."""
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
    rows = [{"policy": name, **evaluate_action(a, df, t, cfg, values, cost)} for name, a in policies.items()]
    return pd.DataFrame(rows)


def value_matrix(uplifts: dict[str, np.ndarray], cfg: DatasetConfig, objective: str,
                 values: dict | None = None, cost: float | None = None) -> np.ndarray:
    """Matriz n × K con el valor esperado de contactar con cada tratamiento.

    * ``objective == "profit"``: Σ_o valor_o × uplift_o − costo por contacto.
    * otro: el uplift del outcome indicado.
    """
    if objective != "profit":
        return uplifts[objective]
    values = cfg.value_defaults if values is None else values
    cost = cfg.business["cost_per_contact"] if cost is None else cost
    first = next(iter(uplifts.values()))
    v = np.zeros_like(first, dtype=float)
    for o, w in values.items():
        v = v + w * uplifts[o]
    return v - cost
