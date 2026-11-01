"""Índice sustituto: valorización de las visitas a partir del embudo de conversión.

Con pocos compradores (≈ 170 en test en Hillstrom) el efecto sobre el gasto se
estima con mucho ruido. La visita tiene diez veces más eventos y precede a toda
compra, así que se usa como *proxy* (Athey, Chetty, Imbens & Kang, 2019) y se
valoriza con el embudo, estimado sólo con datos de entrenamiento:

    E[gasto | T=k, x] = P(visita | k, x) · c_{k,s} · t_s

con ``c_{k,s} = P(compra | visita, T=k, estrato s)`` y ``t_s`` el ticket medio de una
compra en el estrato ``s``. El ingreso incremental esperado del tratamiento k es

    ΔI_k(x) = [ τ_k(x) · c_{k,s} + μ_0(x) · (c_{k,s} − c_{0,s}) ] · t_s

* ``τ_k(x)``: CATE en visita del learner seleccionado (la parte heterogénea).
* ``μ_0(x)``: probabilidad de visita sin tratamiento (modelo del grupo control).
* El segundo término recoge que el tratamiento también puede cambiar la tasa de
  compra de quienes visitan (en Hillstrom el email Mens la sube de 5,4 % a 6,9 %).

Las tasas y los tickets por estrato se *encogen* hacia el promedio (Bayes empírico)
para que un estrato con pocas compras no produzca valores extremos.

Para evaluar políticas, el resultado sustituto observado de cada cliente es
``Y*_i = visita_i · c_{T_i, s_i} · t_{s_i}``: su IPW estima el valor de la política
con mucha menos varianza que el IPW sobre el gasto observado, a cambio del supuesto
de que el embudo (``c``, ``t``) estimado en train vale para los clientes evaluados.
``check`` contrasta ese supuesto comparando el efecto promedio sobre el gasto
observado y sobre el sustituto.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold

from .config import RANDOM_STATE
from .dataset import DatasetConfig
from .features import add_segments

Z95 = 1.959964
ALL = "Todos"


def _beta_prior_strength(x: np.ndarray, n: np.ndarray) -> float:
    """Fuerza del prior Beta (método de momentos sobre una beta-binomial).

    Devuelve M tal que la tasa encogida es (x + M·p) / (n + M). Si la variabilidad
    entre estratos no supera la binomial esperada, el encogimiento es total (M = ∞).
    """
    x, n = np.asarray(x, float), np.asarray(n, float)
    keep = n > 0
    x, n = x[keep], n[keep]
    if len(n) < 2:
        return np.inf
    p = x.sum() / n.sum()
    if p <= 0 or p >= 1:
        return np.inf
    r = x / n
    w = n / n.sum()
    s2 = float(np.sum(w * (r - p) ** 2))          # varianza ponderada observada entre estratos
    binom = p * (1 - p) * float(np.mean(1 / n))    # parte esperada por azar binomial
    between = s2 - binom
    if between <= 0:
        return np.inf
    return max(p * (1 - p) / between - 1, 0.0)


def _shrink_means(means: np.ndarray, var: np.ndarray, n: np.ndarray, grand: float) -> np.ndarray:
    """Encogimiento James-Stein / Bayes empírico de medias de estrato hacia la media global."""
    means, var, n = (np.asarray(a, float) for a in (means, var, n))
    keep = n > 1
    se2 = np.where(keep, var / np.maximum(n, 1), np.inf)
    if keep.sum() < 2:
        return np.full_like(means, grand)
    tau2 = max(float(np.var(means[keep], ddof=1) - np.mean(se2[keep])), 0.0)
    w = np.where(keep, tau2 / (tau2 + se2), 0.0)
    return w * means + (1 - w) * grand


@dataclass
class SurrogateIndex:
    """Embudo estimado en train para valorizar la visita (ver docstring del módulo)."""

    proxy: str
    conversion: str
    value: str
    strata: str | None = None
    shrinkage: bool = True
    n_arms: int = 3
    rates_: pd.DataFrame = field(default=None, repr=False)      # brazo × estrato
    tickets_: pd.DataFrame = field(default=None, repr=False)    # estrato
    mu0_model_: object = field(default=None, repr=False)
    levels_: list = field(default_factory=list)

    # ── Configuración ──────────────────────────────────────────────
    @classmethod
    def from_config(cls, cfg: DatasetConfig) -> "SurrogateIndex | None":
        s = cfg.raw.get("surrogate")
        if not s:
            return None
        names = cfg.outcome_names
        for key in ("proxy", "conversion", "value"):
            if s.get(key) not in names:
                raise ValueError(f"surrogate.{key} debe ser un outcome de la config ({names})")
        if s.get("strata") and s["strata"] not in cfg.segments:
            raise ValueError(f"surrogate.strata '{s['strata']}' no está en segments de la config")
        return cls(proxy=s["proxy"], conversion=s["conversion"], value=s["value"], strata=s.get("strata"),
                   shrinkage=s.get("shrinkage", True), n_arms=cfg.n_treatments + 1)

    def stratum(self, df: pd.DataFrame, cfg: DatasetConfig) -> np.ndarray:
        if not self.strata:
            return np.full(len(df), ALL, dtype=object)
        return add_segments(df, cfg)[self.strata].astype(str).to_numpy()

    # ── Ajuste ─────────────────────────────────────────────────────
    def fit(self, df: pd.DataFrame, t: np.ndarray, X: pd.DataFrame, cfg: DatasetConfig) -> "SurrogateIndex":
        s = self.stratum(df, cfg)
        v = df[self.proxy].to_numpy() > 0
        conv = df[self.conversion].to_numpy() > 0
        val = df[self.value].to_numpy(dtype=float)
        self.levels_ = sorted(set(s))
        rows = []
        for k in range(self.n_arms):
            mk = (t == k) & v
            x = np.array([(conv & mk & (s == lv)).sum() for lv in self.levels_])
            n = np.array([(mk & (s == lv)).sum() for lv in self.levels_])
            p = x.sum() / max(n.sum(), 1)
            m = _beta_prior_strength(x, n) if self.shrinkage else 0.0
            shrunk = np.full(len(n), p) if np.isinf(m) else (x + m * p) / np.maximum(n + m, 1e-12)
            for lv, xi, ni, ri in zip(self.levels_, x, n, shrunk):
                rows.append({"arm": k, "stratum": lv, "visits": int(ni), "conversions": int(xi),
                             "rate_raw": xi / ni if ni else np.nan, "rate": float(ri),
                             "rate_arm": p, "prior_strength": m})
        self.rates_ = pd.DataFrame(rows)
        buyers = conv & (val > 0) if self.value != self.conversion else conv
        tk = pd.DataFrame({"s": s[buyers], "v": val[buyers]})
        g = tk.groupby("s")["v"].agg(["count", "mean", "var"]).reindex(self.levels_)
        g = g.fillna({"count": 0, "var": 0.0})
        grand = float(tk["v"].mean()) if len(tk) else 0.0
        g["mean"] = g["mean"].fillna(grand)
        shrunk_t = _shrink_means(g["mean"], g["var"], g["count"], grand) if self.shrinkage else g["mean"].to_numpy()
        self.tickets_ = pd.DataFrame({"stratum": self.levels_, "buyers": g["count"].astype(int).to_numpy(),
                                      "ticket_raw": g["mean"].to_numpy(), "ticket": shrunk_t,
                                      "ticket_global": grand})
        # P(visita | control, x): modelo del grupo control (learner-agnóstico)
        from .learners import LGBM_PARAMS
        self.mu0_model_ = LGBMClassifier(**LGBM_PARAMS).fit(X[t == 0], v[t == 0].astype(int))
        return self

    def mu0_oof(self, X: pd.DataFrame, t: np.ndarray, df: pd.DataFrame, folds: int = 5) -> np.ndarray:
        """P(visita | control, x) out-of-fold para todos los clientes de train (cross-fitting)."""
        from .learners import LGBM_PARAMS
        v = df[self.proxy].to_numpy() > 0
        out = np.zeros(len(X))
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
        for tr, va in skf.split(X, t * 2 + v):
            ctrl = tr[t[tr] == 0]
            m = clone(LGBMClassifier(**LGBM_PARAMS)).fit(X.iloc[ctrl], v[ctrl].astype(int))
            out[va] = m.predict_proba(X.iloc[va])[:, 1]
        return out

    # ── Uso ────────────────────────────────────────────────────────
    def _lookup(self, s: np.ndarray):
        rate = self.rates_.pivot(index="stratum", columns="arm", values="rate")
        tick = self.tickets_.set_index("stratum")["ticket"]
        fallback_r = self.rates_.groupby("arm")["rate_arm"].first()
        R = np.column_stack([rate[k].reindex(s).fillna(fallback_r[k]).to_numpy() for k in range(self.n_arms)])
        T = tick.reindex(s).fillna(self.tickets_["ticket_global"].iloc[0]).to_numpy()
        return R, T

    def mu0(self, X: pd.DataFrame) -> np.ndarray:
        return self.mu0_model_.predict_proba(X)[:, 1]

    def value_uplift(self, tau_visit: np.ndarray, mu0: np.ndarray, s: np.ndarray) -> np.ndarray:
        """Ingreso (o valor del outcome ``value``) incremental esperado, matriz n × K."""
        R, T = self._lookup(s)
        tau = np.atleast_2d(np.asarray(tau_visit, float).T).T
        cols = [(tau[:, k - 1] * R[:, k] + mu0 * (R[:, k] - R[:, 0])) * T for k in range(1, self.n_arms)]
        return np.column_stack(cols)

    def outcome(self, df: pd.DataFrame, t: np.ndarray, cfg: DatasetConfig) -> np.ndarray:
        """Resultado sustituto observado Y* = visita · c_{T,s} · t_s (para evaluar políticas por IPW)."""
        s = self.stratum(df, cfg)
        R, T = self._lookup(s)
        v = (df[self.proxy].to_numpy() > 0).astype(float)
        return v * R[np.arange(len(df)), np.asarray(t, int)] * T

    def check(self, df: pd.DataFrame, t: np.ndarray, cfg: DatasetConfig) -> pd.DataFrame:
        """Validación del supuesto: efecto promedio sobre el valor observado vs. sobre el sustituto.

        La diferencia de ambos efectos es el efecto del tratamiento sobre (Y − Y*); su IC
        al 95 % (aproximación normal) indica si el sustituto reproduce el efecto real.
        """
        y = df[self.value].to_numpy(dtype=float)
        ys = self.outcome(df, t, cfg)
        rows = []
        for k, arm in enumerate(cfg.treatments, start=1):
            a, b = t == k, t == 0
            for name, z in (("observado", y), ("sustituto", ys), ("diferencia", y - ys)):
                d = z[a].mean() - z[b].mean()
                se = np.sqrt(z[a].var(ddof=1) / a.sum() + z[b].var(ddof=1) / b.sum())
                rows.append({"treatment": arm.label, "estimate": name, "ate": d,
                             "ci_lo": d - Z95 * se, "ci_hi": d + Z95 * se, "se": se})
        return pd.DataFrame(rows)

    def summary(self) -> pd.DataFrame:
        """Tabla de parámetros del embudo (tasas por brazo × estrato y ticket por estrato)."""
        return self.rates_.merge(self.tickets_, on="stratum", how="left")
