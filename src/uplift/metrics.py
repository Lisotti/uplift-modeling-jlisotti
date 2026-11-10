"""Métricas de evaluación de uplift.

Se implementan de forma vectorizada (válidas para outcomes binarios y continuos)
y se contrastan en los tests contra scikit-uplift para el caso binario.

Notación, para un ranking descendente por score y cada corte ``i``:
    N_t(i), N_c(i): tratados y controles dentro del top-i
    R_t(i), R_c(i): suma del outcome en tratados y controles del top-i

* Curva de uplift  U(i) = (R_t/N_t − R_c/N_c) · (N_t + N_c)
* Curva Qini       Q(i) = R_t − R_c · N_t / N_c
* AUUC normalizado = (A_modelo − A_random) / (A_perfecto − A_random) [sólo binario]
* Qini relativo    = (A_modelo − A_random) / (N · |Q(N)|): área sobre el azar,
  normalizada por el efecto total; comparable entre outcomes (incluido spend).

Significancia estadística (outcomes raros, p. ej. ≈ 170 conversiones en test):

* ``bootstrap_metric``: IC percentil de una métrica.
* ``permutation_pvalue``: H0 «el ranking del modelo no es mejor que uno al azar».
* ``paired_bootstrap_diff``: diferencia entre dos modelos con el MISMO remuestreo
  (comparación pareada: el ruido común a ambos se cancela).
* ``corrected_resampled_ttest``: t-test de Nadeau & Bengio (2003) sobre los folds
  de una CV repetida, con la varianza corregida por el solapamiento de los train.
* ``event_counts`` / ``ate_mde``: tamaño efectivo de la muestra y efecto mínimo
  detectable, para leer los resultados a la luz de la potencia disponible.
"""

from __future__ import annotations

import numpy as np


def _cumulative(y, w, score):
    y, w, score = np.asarray(y, float), np.asarray(w, int), np.asarray(score, float)
    order = np.argsort(-score, kind="mergesort")
    y, w, score = y[order], w[order], score[order]
    # Agrupar empates de score para que el orden dentro del empate no sesgue la curva
    distinct = np.where(np.diff(score))[0]
    idx = np.r_[distinct, len(score) - 1]
    nt = np.cumsum(w)[idx]
    nc = np.cumsum(1 - w)[idx]
    rt = np.cumsum(y * w)[idx]
    rc = np.cumsum(y * (1 - w))[idx]
    n = idx + 1
    return n, nt, nc, rt, rc


def uplift_curve(y, w, score):
    n, nt, nc, rt, rc = _cumulative(y, w, score)
    with np.errstate(divide="ignore", invalid="ignore"):
        u = np.nan_to_num(rt / nt) - np.nan_to_num(rc / nc)
    return np.r_[0, n], np.r_[0, u * n]


def qini_curve(y, w, score):
    n, nt, nc, rt, rc = _cumulative(y, w, score)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = rt - np.nan_to_num(rc * nt / nc)
    return np.r_[0, n], np.r_[0, q]


def _area(x, y):
    return float(np.trapezoid(y, x)) if hasattr(np, "trapezoid") else float(np.trapz(y, x))


def _perfect_uplift_score(y, w):
    """Ranking del modelo perfecto para la curva de uplift (misma definición que sklift)."""
    y, w = np.asarray(y, int), np.asarray(w, int)
    cr = np.sum((y == 1) & (w == 0))  # control responders
    tn = np.sum((y == 0) & (w == 1))  # treated non-responders
    summand = y if cr > tn else w
    return 2 * (y == w) + summand


def _perfect_qini_score(y, w):
    """Ranking perfecto Qini con efectos negativos: persuadables primero, sleeping dogs al final."""
    y, w = np.asarray(y, int), np.asarray(w, int)
    return y * w - y * (1 - w)


def auuc_normalized(y, w, score) -> float:
    """AUUC normalizado (equivalente a ``sklift.metrics.uplift_auc_score``). Sólo binario."""
    x, u = uplift_curve(y, w, score)
    xp, up = uplift_curve(y, w, _perfect_uplift_score(y, w))
    a_rand = _area([0, x[-1]], [0, u[-1]])
    a_model, a_perf = _area(x, u), _area(xp, up)
    return (a_model - a_rand) / (a_perf - a_rand)


def qini_coefficient(y, w, score) -> float:
    """Coeficiente Qini normalizado (equivalente a ``sklift.metrics.qini_auc_score``). Sólo binario."""
    y, w = np.asarray(y), np.asarray(w)
    x, q = qini_curve(y, w, score)
    # Curva perfecta Qini sin efectos negativos (negative_effect=True de sklift)
    xp, qp = qini_curve(y, w, _perfect_qini_score(y, w))
    a_rand = _area([0, x[-1]], [0, q[-1]])
    return (_area(x, q) - a_rand) / (_area(xp, qp) - a_rand)


def qini_relative(y, w, score) -> float:
    """Área de la curva Qini por encima del azar normalizada por N·|Q(N)|."""
    x, q = qini_curve(y, w, score)
    a_rand = _area([0, x[-1]], [0, q[-1]])
    denom = x[-1] * abs(q[-1])
    return (_area(x, q) - a_rand) / denom if denom > 0 else 0.0


def uplift_at_k(y, w, score, k: float = 0.3) -> float:
    """Uplift observado (media tratados − media control) en el top-k del ranking."""
    y, w, score = np.asarray(y, float), np.asarray(w, int), np.asarray(score, float)
    n_top = max(int(np.ceil(k * len(y))), 1)
    top = np.argsort(-score, kind="mergesort")[:n_top]
    yt, wt = y[top], w[top]
    if wt.sum() == 0 or (1 - wt).sum() == 0:
        return float("nan")
    return float(yt[wt == 1].mean() - yt[wt == 0].mean())


def uplift_by_decile(y, w, score, bins: int = 10):
    """Uplift observado por decil de score (decil 1 = mayor score)."""
    import pandas as pd

    df = pd.DataFrame({"y": y, "w": w, "s": score})
    df["decile"] = pd.qcut(df["s"].rank(method="first", ascending=False), bins, labels=False) + 1
    g = df.groupby("decile")
    out = pd.DataFrame(
        {
            "n_treated": g.apply(lambda d: int((d.w == 1).sum()), include_groups=False),
            "n_control": g.apply(lambda d: int((d.w == 0).sum()), include_groups=False),
            "rate_treated": g.apply(lambda d: d.y[d.w == 1].mean(), include_groups=False),
            "rate_control": g.apply(lambda d: d.y[d.w == 0].mean(), include_groups=False),
        }
    )
    out["uplift"] = out["rate_treated"] - out["rate_control"]
    return out.reset_index()


def evaluate(y, w, score, binary: bool) -> dict:
    """Conjunto estándar de métricas para un par tratamiento-control."""
    res = {
        "qini_relative": qini_relative(y, w, score),
        "uplift_at_10": uplift_at_k(y, w, score, 0.10),
        "uplift_at_30": uplift_at_k(y, w, score, 0.30),
    }
    if binary:
        res["auuc"] = auuc_normalized(y, w, score)
        res["qini_coef"] = qini_coefficient(y, w, score)
    return res


def bootstrap_metric(fn, y, w, score, n_boot: int = 200, seed: int = 0, alpha: float = 0.05):
    """Intervalo de confianza bootstrap (percentil) para una métrica de uplift."""
    rng = np.random.default_rng(seed)
    y, w, score = np.asarray(y), np.asarray(w), np.asarray(score)
    n = len(y)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        vals.append(fn(y[i], w[i], score[i]))
    lo, hi = np.nanpercentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def permutation_pvalue(fn, y, w, score, n_perm: int = 500, seed: int = 0) -> float:
    """p-valor (una cola) de H0: el ranking del modelo no supera a un ranking aleatorio.

    Se mantienen fijos ``y`` y ``w`` (y por lo tanto el efecto total) y se permuta el
    score: la distribución resultante es la de la métrica bajo un modelo sin
    capacidad de ordenar. p = (1 + #{perm ≥ obs}) / (1 + n_perm).
    """
    rng = np.random.default_rng(seed)
    y, w, score = np.asarray(y), np.asarray(w), np.asarray(score)
    obs = fn(y, w, score)
    null = np.array([fn(y, w, rng.permutation(score)) for _ in range(n_perm)])
    return float((1 + np.sum(null >= obs)) / (1 + n_perm))


def paired_bootstrap_diff(fn, y, w, score_a, score_b, n_boot: int = 200, seed: int = 0,
                          alpha: float = 0.05) -> dict:
    """Diferencia ``fn(a) − fn(b)`` con IC bootstrap pareado y p-valor de una cola (H0: a ≤ b)."""
    rng = np.random.default_rng(seed)
    y, w = np.asarray(y), np.asarray(w)
    sa, sb = np.asarray(score_a), np.asarray(score_b)
    n = len(y)
    diffs = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        diffs.append(fn(y[i], w[i], sa[i]) - fn(y[i], w[i], sb[i]))
    diffs = np.asarray(diffs)
    lo, hi = np.nanpercentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"diff": float(fn(y, w, sa) - fn(y, w, sb)), "ci95": [float(lo), float(hi)],
            "p_value": float((1 + np.sum(diffs <= 0)) / (1 + n_boot))}


def corrected_resampled_ttest(diffs, n_train: float, n_test: float) -> tuple[float, float]:
    """t-test corregido de Nadeau & Bengio (2003) para diferencias por fold de una CV repetida.

    Los folds comparten datos de entrenamiento, así que las diferencias no son
    independientes y el t-test clásico subestima la varianza. La corrección usa
    Var = (1/J + n_test/n_train) · s². Devuelve (t, p bilateral) con J − 1 g.l.
    """
    from scipy import stats

    d = np.asarray(diffs, dtype=float)
    j = len(d)
    var = (1 / j + n_test / n_train) * np.var(d, ddof=1)
    if j < 2 or var <= 0:
        return float("nan"), float("nan")
    t = float(np.mean(d) / np.sqrt(var))
    return t, float(2 * stats.t.sf(abs(t), df=j - 1))


def event_counts(y, w) -> dict:
    """Tamaño por grupo y cantidad de eventos (y > 0) en tratados y controles."""
    y, w = np.asarray(y, float), np.asarray(w, int)
    return {"n_treated": int(w.sum()), "n_control": int((1 - w).sum()),
            "events_treated": int(((y > 0) & (w == 1)).sum()),
            "events_control": int(((y > 0) & (w == 0)).sum())}


def ate_mde(y, w, alpha: float = 0.05, power: float = 0.80) -> float:
    """Efecto promedio mínimo detectable (test bilateral) con los tamaños y varianzas observados."""
    from scipy import stats

    y, w = np.asarray(y, float), np.asarray(w, int)
    se = np.sqrt(y[w == 1].var(ddof=1) / w.sum() + y[w == 0].var(ddof=1) / (1 - w).sum())
    return float((stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)) * se)
