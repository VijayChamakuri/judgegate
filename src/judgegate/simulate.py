"""Coverage simulation: does each "95% upper bound" really cover the true error rate 95% of the time?

A synthetic population of N items has a true error rate p. A judge scores every item with
strength `sep` (how much higher its risk score is on real errors) plus noise. Each method spends
the same expected number of human audits n. We record coverage (how often the bound is at or
above the truth) and tightness (mean bound). A method that looks tight by under-covering would
grant autonomy that was never earned.
"""

from __future__ import annotations

import numpy as np

from .stats import active_ppi_upper, audit_propensity, ppi_upper_plain, wilson_upper

METHODS = ("wilson_random_audits", "ppi_plain_random_audits", "active_ppi_judge_guided_audits", "active_ppi_tuned_same_sample")


def draw(rng: np.random.Generator, N: int, p_err: float, sep: float) -> tuple[np.ndarray, np.ndarray]:
    y = (rng.random(N) < p_err).astype(float)
    f = np.clip(0.05 + sep * y + rng.normal(0, 0.12, N), 0.001, 0.999)
    return y, f


def one(rng: np.random.Generator, y: np.ndarray, f: np.ndarray, n: int) -> dict[str, float]:
    lab = rng.choice(len(y), n, replace=False)
    pi = audit_propensity(f, n)
    s = rng.random(len(y)) < pi
    return {
        "wilson_random_audits": wilson_upper(y[lab]),
        "ppi_plain_random_audits": ppi_upper_plain(y[lab], f[lab], f),
        "active_ppi_judge_guided_audits": active_ppi_upper(y, f, s, pi),
        "active_ppi_tuned_same_sample": active_ppi_upper(y, f, s, pi, tune=True),
    }


def coverage(p_err: float, n: int, sep: float, N: int = 3000, reps: int = 500, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    cov = dict.fromkeys(METHODS, 0)
    up = dict.fromkeys(METHODS, 0.0)
    for _ in range(reps):
        y, f = draw(rng, N, p_err, sep)
        truth = y.mean()
        for k, u in one(rng, y, f, n).items():
            cov[k] += int(truth <= u)
            up[k] += u
    return {k: {"coverage": cov[k] / reps, "mean_upper": up[k] / reps} for k in METHODS}


def run(reps: int = 500) -> dict:
    grid = [
        {"judge_strength": sep, "true_error": p, "audits": n, **coverage(p, n, sep, reps=reps)}
        for sep in (0.3, 0.6, 0.9)
        for p in (0.01, 0.03, 0.10)
        for n in (40, 120)
    ]
    return {"reps": reps, "population": 3000, "grid": grid}


def real_coverage(y: np.ndarray, f: np.ndarray, n: int, reps: int = 1000, seed: int = 0) -> dict:
    """Same check on a real population (truth known for every line): resample only the audits."""
    rng = np.random.default_rng(seed)
    truth = float(np.mean(y))
    cov = dict.fromkeys(METHODS, 0)
    up = dict.fromkeys(METHODS, 0.0)
    for _ in range(reps):
        for k, u in one(rng, y, f, n).items():
            cov[k] += int(truth <= u)
            up[k] += u
    return {
        "true_error": truth,
        "population": int(len(y)),
        "audits": n,
        **{k: {"coverage": cov[k] / reps, "mean_upper": up[k] / reps} for k in METHODS},
    }
