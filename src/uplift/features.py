"""Feature engineering declarativo (definido en ``features`` de la config del dataset).

Tipos soportados:
* ``numeric``: se usa tal cual; con ``log1p: true`` agrega ``log_<nombre>``.
* ``binary``: 0/1.
* ``categorical``: one-hot con categorías fijas (estable ante lotes con categorías
  ausentes); nombre ``<prefix>_<categoría en minúsculas>``.
* ``interaction``: producto de dos o más binarias (``of: [a, b]``).

El transformador no aprende parámetros de los datos (``fit`` es no-op), lo que
elimina por construcción el riesgo de *data leakage* entre train y test.

Para Hillstrom reproduce exactamente las decisiones del EDA (Entrega 2 §8):
``history_segment`` excluida, ``log1p(history)``, one-hot de ``zip_code`` y
``channel`` y la interacción ``both_categories``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .dataset import DatasetConfig


class FeatureBuilder(BaseEstimator, TransformerMixin):
    def __init__(self, specs: list[dict] | None = None):
        self.specs = specs

    @classmethod
    def from_config(cls, cfg: DatasetConfig) -> "FeatureBuilder":
        return cls(specs=list(cfg.feature_specs))

    def fit(self, X: pd.DataFrame, y=None):  # noqa: D401 - API sklearn
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = {}
        for s in self.specs:
            name, kind = s["name"], s["type"]
            if kind == "numeric":
                v = X[name].astype(float).to_numpy()
                out[name] = v
                if s.get("log1p"):
                    out[f"log_{name}"] = np.log1p(v)
            elif kind == "binary":
                out[name] = X[name].astype(int).to_numpy()
            elif kind == "categorical":
                prefix = s.get("prefix", name)
                for c in s["categories"]:
                    out[f"{prefix}_{str(c).lower()}"] = (X[name] == c).astype(int).to_numpy()
            elif kind == "interaction":
                v = np.ones(len(X), dtype=int)
                for col in s["of"]:
                    v = v & X[col].astype(int).to_numpy()
                out[name] = v
            else:
                raise ValueError(f"Tipo de feature desconocido: {kind}")
        return pd.DataFrame(out, index=X.index)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        names = []
        for s in self.specs:
            if s["type"] == "numeric":
                names += [s["name"]] + ([f"log_{s['name']}"] if s.get("log1p") else [])
            elif s["type"] == "categorical":
                names += [f"{s.get('prefix', s['name'])}_{str(c).lower()}" for c in s["categories"]]
            else:
                names.append(s["name"])
        return np.array(names)


def add_segments(df: pd.DataFrame, cfg: DatasetConfig) -> pd.DataFrame:
    """Agrega las columnas de segmentos derivados declarados en ``segments`` de la config."""
    df = df.copy()
    for seg, rules in cfg.segments.items():
        col = pd.Series("Otro", index=df.index, dtype=object)
        for label, expr in rules.items():
            col[df.eval(expr)] = label
        df[seg] = pd.Categorical(col, categories=[*rules.keys(), "Otro"])
        if (df[seg] == "Otro").sum() == 0:
            df[seg] = df[seg].cat.remove_categories("Otro")
    return df


def quantile_bins(v: pd.Series, q: int = 5, max_categories: int = 8) -> tuple[pd.Series, list[str]]:
    """Tramos ordenados para segmentar una variable (EDA, figuras y dashboard).

    * Categóricas o con pocos valores: cada valor es un tramo.
    * Numéricas: cuantiles "Q1 [a; b]" … Si la variable concentra mucha masa en un
      único valor (frecuente en features anonimizadas como las de Criteo) y los
      cuantiles colapsan a menos de 2 tramos, se separa "= moda" / "< moda" / "> moda".
    """
    if str(v.dtype) == "category" or v.dtype == object or v.nunique() <= max_categories:
        b = v.astype(str)
        try:
            order = [str(x) for x in sorted(v.dropna().unique())]
        except TypeError:
            order = sorted(b.unique())
        return b, order
    codes, edges = pd.qcut(v, q, labels=False, retbins=True, duplicates="drop")
    if len(edges) - 1 >= 2:
        labels = [f"Q{i + 1} [{edges[i]:.3g}; {edges[i + 1]:.3g}]" for i in range(len(edges) - 1)]
        return pd.Series(np.array(labels, dtype=object)[codes.to_numpy().astype(int)], index=v.index), labels
    mode = v.mode().iloc[0]
    lab = np.where(v == mode, f"= moda ({mode:.3g})", np.where(v < mode, "< moda", "> moda"))
    order = [x for x in ("< moda", f"= moda ({mode:.3g})", "> moda") if x in set(lab)]
    return pd.Series(lab, index=v.index, dtype=object), order
