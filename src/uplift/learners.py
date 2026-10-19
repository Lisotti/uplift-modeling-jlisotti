"""Meta-learners multi-tratamiento para estimar el CATE (Künzel et al., 2019).

Convención: ``t`` es un vector entero con 0 = control y 1..K = tratamientos.
Todos los learners exponen ``predict_uplift(X) -> ndarray (n, K)`` donde la
columna ``k-1`` es el efecto estimado del tratamiento ``k`` frente al control:

    τ_k(x) = E[Y | X=x, T=k] − E[Y | X=x, T=0]

Se implementan aquí (y no con una librería externa) porque scikit-uplift sólo
soporta tratamiento binario y Hillstrom tiene dos campañas alternativas; con
un único tratamiento (p. ej. Criteo) los mismos learners funcionan con K = 1.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.base import BaseEstimator, RegressorMixin, clone, is_classifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import RANDOM_STATE

# ── Base learners ──────────────────────────────────────────────────
# Hiperparámetros fuertemente regularizados: la señal de uplift es pequeña
# (efectos de 0,3–8 pp) y los árboles profundos sobreajustan el ruido. Se
# fijaron a partir de una comparación preliminar por validación cruzada sobre train
# de tres niveles de complejidad (num_leaves 15/7/4), realizada durante el desarrollo
# y no incluida en el pipeline final (ver informe, sección «Modelos»).
LGBM_PARAMS = dict(
    n_estimators=200,
    learning_rate=0.03,
    num_leaves=7,
    min_child_samples=500,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=5.0,
    random_state=RANDOM_STATE,
    verbose=-1,
    n_jobs=2,
)


def set_n_jobs(n_jobs: int) -> None:
    """Hilos de LightGBM (``modeling.n_jobs`` de la config; -1 = todos los núcleos)."""
    LGBM_PARAMS["n_jobs"] = n_jobs


class HurdleRegressor(BaseEstimator, RegressorMixin):
    """Modelo Hurdle (dos partes) para targets inflados de ceros como ``spend``.

    E[Y|x] = P(Y>0|x) · E[Y | Y>0, x]
    """

    def __init__(self, classifier=None, regressor=None):
        self.classifier = classifier
        self.regressor = regressor

    def fit(self, X, y):
        y = np.asarray(y, dtype=float)
        pos = y > 0
        self.classifier_ = clone(self.classifier or LGBMClassifier(**LGBM_PARAMS))
        self.classifier_.fit(X, pos.astype(int))
        # Pocos compradores (≈400 en train): regresor muy simple sobre log1p(spend)
        self.regressor_ = clone(
            self.regressor
            or LGBMRegressor(**{**LGBM_PARAMS, "n_estimators": 100, "min_child_samples": 30})
        )
        if pos.sum() >= 10:
            z = np.log1p(y[pos])
            self.regressor_.fit(X[pos], z)
            # Factor de smearing de Duan (1983) para corregir el sesgo de retransformación
            self.smearing_ = float(np.mean(np.exp(z - self.regressor_.predict(X[pos]))))
            self.mean_pos_ = float(y[pos].mean())
        else:  # salvaguarda para tests con pocos positivos
            self.regressor_ = None
            self.mean_pos_ = float(y[pos].mean()) if pos.any() else 0.0
        return self

    def predict(self, X):
        p = self.classifier_.predict_proba(X)[:, 1]
        if self.regressor_ is None:
            return p * self.mean_pos_
        amount = np.exp(self.regressor_.predict(X)) * self.smearing_ - 1.0
        amount = np.clip(amount, 0.0, None)
        return p * amount


def make_base_learner(binary: bool, kind: str = "lgbm"):
    """Fábrica de base learners según el tipo de outcome.

    * ``lgbm``: gradient boosting (captura no linealidades e interacciones).
    * ``linear``: regresión logística / ridge estandarizada (interpretable).
    * ``hurdle``: modelo en dos partes para ``spend``.
    """
    if kind == "hurdle":
        return HurdleRegressor()
    if kind == "linear":
        if binary:
            return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
        return make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    if binary:
        return LGBMClassifier(**LGBM_PARAMS)
    return LGBMRegressor(**LGBM_PARAMS)


def _predict_mean(model, X) -> np.ndarray:
    """E[Y|X]: probabilidad positiva para clasificadores, predicción para regresores."""
    if is_classifier(model):
        return model.predict_proba(X)[:, 1]
    return model.predict(X)


def _as_frame(X) -> pd.DataFrame:
    return X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)


# ── Meta-learners ──────────────────────────────────────────────────
@dataclass
class BaseMetaLearner:
    base_learner: object
    n_treatments: int = 2
    name: str = "base"

    def fit(self, X, t, y):  # pragma: no cover - interfaz
        raise NotImplementedError

    def predict_uplift(self, X) -> np.ndarray:  # pragma: no cover - interfaz
        raise NotImplementedError


@dataclass
class ResponseModel(BaseMetaLearner):
    """Baseline clásico (no causal): modela P(Y | X, T=k) sólo con los tratados.

    Es el enfoque habitual de marketing ("¿quién va a responder?"). Su score
    ordena clientes por probabilidad de respuesta, no por efecto incremental.
    """

    name: str = "Baseline (response)"
    models_: list = field(default_factory=list)

    def fit(self, X, t, y):
        X, t, y = _as_frame(X), np.asarray(t), np.asarray(y)
        self.models_ = []
        for k in range(1, self.n_treatments + 1):
            m = clone(self.base_learner).fit(X[t == k], y[t == k])
            self.models_.append(m)
        return self

    def predict_uplift(self, X) -> np.ndarray:
        X = _as_frame(X)
        return np.column_stack([_predict_mean(m, X) for m in self.models_])


@dataclass
class SLearner(BaseMetaLearner):
    """Un único modelo f(X, T) con el tratamiento como feature (one-hot)."""

    name: str = "S-Learner"
    interactions: bool = False  # agrega términos T_k × X (necesario para modelos lineales)
    model_: object = None

    def _augment(self, X: pd.DataFrame, t: np.ndarray) -> pd.DataFrame:
        Xa = X.copy()
        base_cols = list(X.columns)
        for k in range(1, self.n_treatments + 1):
            tk = (t == k).astype(int)
            Xa[f"_treat_{k}"] = tk
            if self.interactions:
                for c in base_cols:
                    Xa[f"_treat_{k}_x_{c}"] = X[c].to_numpy() * tk
        return Xa

    def fit(self, X, t, y):
        X, t = _as_frame(X), np.asarray(t)
        self.model_ = clone(self.base_learner).fit(self._augment(X, t), np.asarray(y))
        return self

    def predict_uplift(self, X) -> np.ndarray:
        X = _as_frame(X)
        n = len(X)
        mu0 = _predict_mean(self.model_, self._augment(X, np.zeros(n, dtype=int)))
        cols = []
        for k in range(1, self.n_treatments + 1):
            muk = _predict_mean(self.model_, self._augment(X, np.full(n, k)))
            cols.append(muk - mu0)
        return np.column_stack(cols)


@dataclass
class TLearner(BaseMetaLearner):
    """Un modelo independiente por brazo: τ_k(x) = μ_k(x) − μ_0(x)."""

    name: str = "T-Learner"
    models_: list = field(default_factory=list)

    def fit(self, X, t, y):
        X, t, y = _as_frame(X), np.asarray(t), np.asarray(y)
        self.models_ = [
            clone(self.base_learner).fit(X[t == k], y[t == k])
            for k in range(0, self.n_treatments + 1)
        ]
        return self

    def predict_outcomes(self, X) -> np.ndarray:
        X = _as_frame(X)
        return np.column_stack([_predict_mean(m, X) for m in self.models_])

    def predict_uplift(self, X) -> np.ndarray:
        mu = self.predict_outcomes(X)
        return mu[:, 1:] - mu[:, [0]]


@dataclass
class XLearner(BaseMetaLearner):
    """X-Learner (Künzel et al., 2019) con propensity conocida por diseño (RCT).

    Etapa 1: μ_0 y μ_k como en el T-Learner.
    Etapa 2: efectos imputados D¹ = Y¹ − μ_0(X¹) en tratados y D⁰ = μ_k(X⁰) − Y⁰
             en controles; se ajustan regresores τ_1 y τ_0 sobre ellos.
    Etapa 3: τ(x) = e·τ_0(x) + (1 − e)·τ_1(x), con e = P(T=k | T∈{0,k}).
    """

    effect_learner: object = None
    name: str = "X-Learner"
    mu_: list = field(default_factory=list)
    tau1_: list = field(default_factory=list)
    tau0_: list = field(default_factory=list)
    propensity_: list = field(default_factory=list)

    def fit(self, X, t, y):
        X, t, y = _as_frame(X), np.asarray(t), np.asarray(y, dtype=float)
        eff = self.effect_learner or LGBMRegressor(**LGBM_PARAMS)
        self.mu_ = [
            clone(self.base_learner).fit(X[t == k], y[t == k])
            for k in range(0, self.n_treatments + 1)
        ]
        self.tau1_, self.tau0_, self.propensity_ = [], [], []
        X0, y0 = X[t == 0], y[t == 0]
        for k in range(1, self.n_treatments + 1):
            Xk, yk = X[t == k], y[t == k]
            d1 = yk - _predict_mean(self.mu_[0], Xk)
            d0 = _predict_mean(self.mu_[k], X0) - y0
            self.tau1_.append(clone(eff).fit(Xk, d1))
            self.tau0_.append(clone(eff).fit(X0, d0))
            self.propensity_.append(len(yk) / (len(yk) + len(y0)))
        return self

    def predict_uplift(self, X) -> np.ndarray:
        X = _as_frame(X)
        cols = []
        for k in range(self.n_treatments):
            e = self.propensity_[k]
            cols.append(e * self.tau0_[k].predict(X) + (1 - e) * self.tau1_[k].predict(X))
        return np.column_stack(cols)


def build_learners(binary: bool, n_treatments: int, hurdle: bool = False) -> dict[str, BaseMetaLearner]:
    """Conjunto de modelos comparados para un outcome.

    Se combinan tres meta-learners (S, T, X) con dos familias de base learner
    (LightGBM y lineal regularizado) y el baseline de respuesta clásico; para
    outcomes continuos inflados de ceros (``hurdle: true``) se agrega el T-Learner Hurdle.
    """
    gbm = make_base_learner(binary, "lgbm")
    lin = make_base_learner(binary, "linear")
    k = n_treatments
    lin_effect = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    learners: dict[str, BaseMetaLearner] = {
        "baseline": ResponseModel(base_learner=gbm, n_treatments=k),
        "s_learner_lgbm": SLearner(base_learner=gbm, n_treatments=k, name="S-Learner (LGBM)"),
        "t_learner_lgbm": TLearner(base_learner=gbm, n_treatments=k, name="T-Learner (LGBM)"),
        "x_learner_lgbm": XLearner(base_learner=gbm, n_treatments=k, name="X-Learner (LGBM)"),
        "s_learner_linear": SLearner(base_learner=lin, n_treatments=k, interactions=True, name="S-Learner (lineal)"),
        "t_learner_linear": TLearner(base_learner=lin, n_treatments=k, name="T-Learner (lineal)"),
        "x_learner_linear": XLearner(base_learner=lin, n_treatments=k, effect_learner=lin_effect, name="X-Learner (lineal)"),
    }
    if hurdle and not binary:
        learners["t_learner_hurdle"] = TLearner(base_learner=HurdleRegressor(), n_treatments=k,
                                                name="T-Learner (Hurdle)")
    return learners


LEARNER_LABELS = {
    "baseline": "Baseline (response)",
    "s_learner_lgbm": "S-Learner (LGBM)",
    "t_learner_lgbm": "T-Learner (LGBM)",
    "x_learner_lgbm": "X-Learner (LGBM)",
    "s_learner_linear": "S-Learner (lineal)",
    "t_learner_linear": "T-Learner (lineal)",
    "x_learner_linear": "X-Learner (lineal)",
    "t_learner_hurdle": "T-Learner (Hurdle)",
}
