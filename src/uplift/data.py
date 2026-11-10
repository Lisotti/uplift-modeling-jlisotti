"""Ingesta, validación del contrato de datos, muestreo y partición (genérico por config)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import RANDOM_STATE
from .dataset import DatasetConfig


class DataValidationError(ValueError):
    """Se lanza cuando el dataset no cumple el contrato de datos esperado."""


def load_raw(cfg: DatasetConfig, validate_data: bool = True) -> pd.DataFrame:
    """Carga el archivo original. Para datasets grandes guarda una caché parquet."""
    cache = cfg.data_dir / "raw.parquet"
    if cache.exists() and cache.stat().st_mtime >= cfg.source_path.stat().st_mtime:
        df = pd.read_parquet(cache)
    else:
        if not cfg.source_path.exists():
            manual = f" o a mano desde {cfg.manual_url}," if cfg.manual_url else ""
            raise FileNotFoundError(
                f"No se encontró {cfg.source_path}. Descargalo con `make download DATASET={cfg.name}`"
                f"{manual} y guardá el CSV en data/raw/."
            )
        df = pd.read_csv(cfg.source_path, dtype=cfg.dtypes)
        if len(df) > 500_000:  # la caché sólo vale la pena en datasets grandes
            cfg.data_dir.mkdir(parents=True, exist_ok=True)
            df.to_parquet(cache, index=False)
    if validate_data:
        validate(df, cfg)
    return df


def validate(df: pd.DataFrame, cfg: DatasetConfig, require_outcomes: bool = True) -> None:
    """Valida esquema, dominios y restricciones lógicas declaradas en la config."""
    cols = list(cfg.raw_features)
    if require_outcomes:
        cols += [cfg.treatment_col, *cfg.outcome_names]
    missing = set(cols) - set(df.columns)
    if missing:
        raise DataValidationError(f"Columnas faltantes: {sorted(missing)}")
    if df[cols].isna().any().any():
        raise DataValidationError("El dataset contiene valores faltantes")
    for spec in cfg.feature_specs:
        name, kind = spec["name"], spec["type"]
        if kind == "binary" and not df[name].isin([0, 1]).all():
            raise DataValidationError(f"{name} no es binaria")
        if kind == "categorical" and spec.get("categories") and not df[name].isin(spec["categories"]).all():
            raise DataValidationError(f"{name} con categorías desconocidas")
    for col, (lo, hi) in (cfg.validation.get("ranges") or {}).items():
        if col not in df.columns:
            continue
        if lo is not None and (df[col] < lo).any():
            raise DataValidationError(f"{col} por debajo de {lo}")
        if hi is not None and (df[col] > hi).any():
            raise DataValidationError(f"{col} por encima de {hi}")
    if not require_outcomes:
        return
    arm_values = [a.value for a in cfg.arms]
    if not df[cfg.treatment_col].isin(arm_values).all():
        raise DataValidationError(f"{cfg.treatment_col} con grupos desconocidos")
    for o in cfg.outcomes:
        if o.binary and not df[o.name].isin([0, 1]).all():
            raise DataValidationError(f"outcome {o.name} no es binario")
    for a, b in cfg.validation.get("implies") or []:
        if ((df[a] == 1) & (df[b] == 0)).any():
            raise DataValidationError(f"{a}=1 sin {b}=1")
    for a, b in cfg.validation.get("positive_iff") or []:
        if ((df[a] > 0) != (df[b] == 1)).any():
            raise DataValidationError(f"{a}>0 debe equivaler a {b}=1")


def treatment_index(segment: pd.Series, cfg: DatasetConfig) -> np.ndarray:
    """Codifica el grupo experimental: 0 = control, 1..K = tratamientos (orden de la config)."""
    mapping = {a.value: i for i, a in enumerate(cfg.arms)}
    return segment.map(mapping).to_numpy(dtype=int)


def _strata(df: pd.DataFrame, cfg: DatasetConfig) -> pd.Series:
    return df[cfg.treatment_col].astype(str) + "_" + df[cfg.stratify_outcome].astype(str)


def sample(df: pd.DataFrame, cfg: DatasetConfig, n: int | None = None,
           random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """Muestra estratificada por tratamiento × outcome de estratificación (mantiene proporciones)."""
    n = n if n is not None else cfg.modeling["sample_size"]
    if not n or n >= len(df):
        return df.reset_index(drop=True)
    _, s = train_test_split(df, test_size=n, random_state=random_state, stratify=_strata(df, cfg))
    return s.reset_index(drop=True)


def split(df: pd.DataFrame, cfg: DatasetConfig, test_size: float | None = None,
          random_state: int = RANDOM_STATE) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Partición train/test estratificada por tratamiento × outcome raro.

    Estratificar por la combinación garantiza que el evento raro (p. ej. 0,9 % de
    conversiones en Hillstrom) quede representado de forma proporcional en cada brazo.
    """
    test_size = test_size or cfg.modeling["test_size"]
    train, test = train_test_split(df, test_size=test_size, random_state=random_state,
                                   stratify=_strata(df, cfg))
    return train.reset_index(drop=True), test.reset_index(drop=True)


def save_splits(train: pd.DataFrame, test: pd.DataFrame, cfg: DatasetConfig) -> None:
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    fmt = "parquet" if len(train) > 200_000 else "csv"
    for name, d in (("train", train), ("test", test)):
        path = cfg.data_dir / f"{name}.{fmt}"
        d.to_parquet(path, index=False) if fmt == "parquet" else d.to_csv(path, index=False)


def load_split(name: str, cfg: DatasetConfig) -> pd.DataFrame:
    """Lee train/test/reference guardados por el entrenamiento (csv o parquet)."""
    for ext in ("parquet", "csv"):
        p = cfg.data_dir / f"{name}.{ext}"
        if p.exists():
            return pd.read_parquet(p) if ext == "parquet" else pd.read_csv(p)
    raise FileNotFoundError(f"No existe {name} para {cfg.name}. Ejecutá `make train DATASET={cfg.name}`.")
