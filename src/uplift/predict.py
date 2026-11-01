"""Scoring de nuevos clientes con el bundle entrenado de un dataset."""

from __future__ import annotations

from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from .data import validate
from .dataset import DatasetConfig, load_config
from .policy import value_matrix


@lru_cache(maxsize=8)
def _load_bundle_path(path: str, mtime: float) -> dict:
    return joblib.load(path)


def load_bundle(dataset: str | DatasetConfig = "hillstrom") -> dict:
    """Carga (con caché, invalidada si el archivo cambia) el bundle entrenado de un dataset."""
    cfg = dataset if isinstance(dataset, DatasetConfig) else load_config(dataset)
    if not cfg.bundle_path.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo de '{cfg.name}' en {cfg.bundle_path}. "
            f"Ejecutá primero `make train DATASET={cfg.name}`."
        )
    return _load_bundle_path(str(cfg.bundle_path), cfg.bundle_path.stat().st_mtime)


def objectives(cfg: DatasetConfig) -> list[str]:
    """Objetivos de optimización disponibles para un dataset.

    ``profit_surrogate`` (beneficio con la visita valorizada por el índice sustituto)
    está disponible si la config declara ``surrogate`` y valoriza su outcome ``value``.
    """
    objs = ["profit"] if cfg.business["values"] else []
    sur = cfg.raw.get("surrogate")
    if sur and sur.get("value") in cfg.business["values"]:
        objs.append("profit_surrogate")
    return objs + cfg.outcome_names


def predict_uplifts(customers: pd.DataFrame, bundle: dict, cfg: DatasetConfig | None = None) -> dict[str, np.ndarray]:
    """Uplift por outcome; con ``cfg`` y un bundle con índice sustituto agrega ``surrogate``
    (ingreso incremental esperado vía visita valorizada)."""
    X = bundle["feature_builder"].transform(customers)
    out = {o: m.predict_uplift(X) for o, m in bundle["models"].items()}
    sur = bundle.get("surrogate")
    if sur is not None and cfg is not None:
        out["surrogate"] = sur.value_uplift(out[sur.proxy], sur.mu0(X), sur.stratum(customers, cfg))
    return out


def score(customers: pd.DataFrame, cfg: DatasetConfig, bundle: dict | None = None,
          objective: str = "profit", values: dict | None = None, cost: float | None = None) -> pd.DataFrame:
    """Uplift estimado por outcome y tratamiento, y la acción recomendada.

    ``objective``:
      * ``profit``: tratamiento con mayor beneficio esperado (Σ valor × uplift − costo),
        sólo si es positivo. Requiere ``business.values`` en la config.
      * ``profit_surrogate``: ídem, con el ingreso estimado por el índice sustituto.
      * nombre de un outcome: tratamiento con mayor uplift en ese outcome, si es positivo.
    """
    if objective not in objectives(cfg):
        raise ValueError(f"objective debe ser uno de {objectives(cfg)}")
    bundle = bundle or load_bundle(cfg)
    if objective == "profit_surrogate" and bundle.get("surrogate") is None:
        raise ValueError(f"El modelo de '{cfg.name}' no tiene índice sustituto: reentrená con "
                         f"`make train DATASET={cfg.name}`.")
    validate(customers, cfg, require_outcomes=False)
    uplifts = predict_uplifts(customers, bundle, cfg)

    out = pd.DataFrame(index=customers.index)
    for o, u in uplifts.items():
        prefix = "expected_value_surrogate" if o == "surrogate" else f"uplift_{o}"
        for k, arm in enumerate(cfg.treatments):
            out[f"{prefix}_{arm.label}"] = u[:, k]
    value = value_matrix(uplifts, cfg, objective, values, cost)
    best = value.argmax(axis=1)
    labels = np.array([a.label for a in cfg.treatments])
    out["recommended_treatment"] = np.where(value.max(axis=1) > 0, labels[best], cfg.none_action)
    out["expected_value"] = value.max(axis=1)
    if cfg.business["values"]:
        out["expected_profit_usd"] = value_matrix(uplifts, cfg, "profit", values, cost).max(axis=1)
    return out
