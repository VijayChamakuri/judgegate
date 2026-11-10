"""Error-rate upper bounds: what evidence can certify an agent's error rate?

Y = 1 when the agent's booking was wrong (human truth); f = the judge's probability it is wrong.

wilson_upper       Human audits drawn at random; Wilson score bound. Valid, but cannot certify
                   anything much below ~3/n after n error-free audits (TrustGate's rule).
ppi_upper_plain    Textbook prediction-powered inference (Angelopoulos et al., Science 2023;
                   PPI++ power tuning, arXiv:2311.01453) with random audits. Unbiased, but its
                   normal-approximation bound collapses towards 0 when the audited sample happens
                   to contain no errors, so it under-covers exactly when errors are rare.
active_ppi_upper   Judge-guided audits (active statistical inference, Zrnic & Candes, ICML 2024).
                   Each item is audited with probability pi_i proportional to the judge's risk
                   score (plus a floor so every item can be picked). The estimate
                       theta = mean_i [ f_i + (Y_i - f_i) * S_i / pi_i ]
                   is unbiased for the true error rate whatever the judge's quality; audits land
                   where errors are, so the bound tightens without losing coverage.

`wilson_interval` is adapted from TrustGate (github.com/VijayChamakuri/trustgate).
"""

from __future__ import annotations

import math

import numpy as np

Z95 = 1.959963984540054
FLOOR = 0.2  # audit-propensity floor, as a fraction of the mean judge risk


def wilson_interval(errors: int, n: int, z: float = Z95) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = errors / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def wilson_upper(y: np.ndarray, z: float = Z95) -> float:
    y = np.asarray(y, float)
    return wilson_interval(int(y.sum()), len(y), z)[1]


def ppi_estimate(y_lab: np.ndarray, f_lab: np.ndarray, f_all: np.ndarray) -> tuple[float, float]:
    """PPI++ point estimate and standard error of the mean of Y (random audits)."""
    y_lab, f_lab, f_all = (np.asarray(a, float) for a in (y_lab, f_lab, f_all))
    n, N = len(y_lab), len(f_all)
    if n < 2 or N < 2:
        return (float(y_lab.mean()) if n else 0.5), 0.5
    vf = f_all.var(ddof=1)
    lam = 0.0 if vf <= 0 else float(np.cov(y_lab, f_lab, ddof=1)[0, 1] / ((1 + n / N) * vf))
    lam = min(1.0, max(0.0, lam))
    est = lam * f_all.mean() + (y_lab - lam * f_lab).mean()
    var = lam**2 * f_all.var(ddof=1) / N + (y_lab - lam * f_lab).var(ddof=1) / n
    return float(est), float(math.sqrt(max(var, 0.0)))


def ppi_upper_plain(y_lab: np.ndarray, f_lab: np.ndarray, f_all: np.ndarray, z: float = Z95) -> float:
    est, se = ppi_estimate(y_lab, f_lab, f_all)
    return min(1.0, max(est, 0.0) + z * se)


def platt_fit(f: np.ndarray, y: np.ndarray) -> tuple[float, float] | None:
    """Recalibrate judge scores: logistic regression of the human truth on logit(judge score).
    Returns (slope, intercept), or None if the audits do not yet contain both outcomes."""
    from sklearn.linear_model import LogisticRegression

    y = np.asarray(y, int)
    if len(np.unique(y)) < 2:
        return None
    m = LogisticRegression(C=100.0).fit(_logit(np.asarray(f, float))[:, None], y)
    # rounded so Linux CI and macOS reach identical gate decisions (last-bit solver noise cannot flip a threshold)
    return round(float(m.coef_[0, 0]), 6), round(float(m.intercept_[0]), 6)


def platt_apply(f: np.ndarray, params: tuple[float, float] | None) -> np.ndarray:
    f = np.asarray(f, float)
    if params is None:
        return f
    a, b = params
    return 1.0 / (1.0 + np.exp(-(a * _logit(f) + b)))


def _logit(f: np.ndarray) -> np.ndarray:
    f = np.clip(f, 1e-4, 1 - 1e-4)
    return np.log(f / (1 - f))


def audit_propensity(f: np.ndarray, expected_audits: float, floor: float = FLOOR) -> np.ndarray:
    """Inclusion probability per item: proportional to judge risk plus a floor, capped at 1."""
    f = np.asarray(f, float)
    w = f + floor * max(f.mean(), 1e-6)
    return np.minimum(1.0, expected_audits * w / w.sum())


def active_ppi_terms(y: np.ndarray, f: np.ndarray, audited: np.ndarray, pi: np.ndarray, lam: float) -> np.ndarray:
    """Per-item terms whose mean is unbiased for the error rate, for any fixed lam in [0, 1].
    lam = 0 is plain inverse-probability weighting of the audits; lam = 1 fully trusts the judge
    as a control variate."""
    y = np.where(audited, np.asarray(y, float), 0.0)
    return lam * f + np.where(audited, (y - lam * f) / pi, 0.0)


def active_ppi_upper(
    y: np.ndarray, f: np.ndarray, audited: np.ndarray, pi: np.ndarray, z: float = Z95, tune: bool = False
) -> float:
    """Upper bound from judge-guided audits. `y` is only read where `audited` is True.

    The gate uses lam = 1 (no tuning). `tune=True` picks lam on a grid to minimise the variance
    of the same terms it then reports; on small, error-free audit samples that selection makes
    the bound too optimistic (see results/coverage.json), so it is kept only for comparison."""
    f, audited, pi = np.asarray(f, float), np.asarray(audited, bool), np.asarray(pi, float)
    if len(f) < 2:
        return 1.0
    lams = np.linspace(0, 1, 11) if tune else np.array([1.0])
    terms = [active_ppi_terms(y, f, audited, pi, lam) for lam in lams]
    zi = min(terms, key=lambda t: t.var(ddof=1))
    est, se = zi.mean(), zi.std(ddof=1) / math.sqrt(len(zi))
    return min(1.0, max(est, 0.0) + z * se)


def ece(prob: np.ndarray, outcome: np.ndarray, bins: int = 10) -> float:
    prob, outcome = np.asarray(prob, float), np.asarray(outcome, float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(prob, edges) - 1, 0, bins - 1)
    return float(
        sum(abs(prob[idx == b].mean() - outcome[idx == b].mean()) * (idx == b).mean() for b in range(bins) if (idx == b).any())
    )


def bootstrap_ci(values: np.ndarray, stat=np.mean, n: int = 2000, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    v = np.asarray(values)
    draws = [stat(v[rng.integers(0, len(v), len(v))]) for _ in range(n)]
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]
