"""The statistics the gate relies on: coverage is the property that matters."""

import numpy as np

from judgegate.simulate import coverage
from judgegate.stats import active_ppi_upper, audit_propensity, ppi_estimate, wilson_interval, wilson_upper


def test_wilson_known_values():
    lo, hi = wilson_interval(0, 100)
    assert lo < 1e-12 and 0.035 < hi < 0.040
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_ppi_is_unbiased_even_with_a_biased_judge():
    rng = np.random.default_rng(0)
    errs = []
    for _ in range(400):
        y = (rng.random(2000) < 0.1).astype(float)
        f = np.clip(0.5 + 0.3 * y + rng.normal(0, 0.1, 2000), 0, 1)  # judge badly over-predicts errors
        lab = rng.choice(2000, 200, replace=False)
        est, _ = ppi_estimate(y[lab], f[lab], f)
        errs.append(est - y.mean())
    assert abs(np.mean(errs)) < 0.005


def test_judge_guided_audits_keep_coverage_where_textbook_ppi_fails():
    """Rare errors + few audits: the regime an autonomy gate actually operates in."""
    res = coverage(p_err=0.01, n=40, sep=0.6, reps=300, seed=1)
    assert res["active_ppi_judge_guided_audits"]["coverage"] >= 0.93
    assert res["wilson_random_audits"]["coverage"] >= 0.93
    assert res["ppi_plain_random_audits"]["coverage"] < 0.80  # documented failure, see results/coverage.json
    assert res["active_ppi_tuned_same_sample"]["coverage"] < 0.93  # tuning on the same audits also fails


def test_judge_guided_audits_tighten_the_bound():
    res = coverage(p_err=0.01, n=40, sep=0.6, reps=300, seed=2)
    assert res["active_ppi_judge_guided_audits"]["mean_upper"] < res["wilson_random_audits"]["mean_upper"]


def test_active_estimate_is_unbiased_with_a_useless_judge():
    rng = np.random.default_rng(0)
    errs = []
    for _ in range(400):
        y = (rng.random(2000) < 0.05).astype(float)
        f = rng.uniform(0, 1, 2000)  # judge carries no information at all
        pi = audit_propensity(f, 100)
        s = rng.random(2000) < pi
        z = f + np.where(s, (y - f) / pi, 0.0)
        errs.append(z.mean() - y.mean())
    assert abs(np.mean(errs)) < 0.005


def test_upper_bounds_are_probabilities():
    y = np.zeros(1000)
    f = np.full(1000, 0.2)
    pi = audit_propensity(f, 40)
    assert 0 < active_ppi_upper(y, f, np.zeros(1000, bool), pi) <= 1
    assert 0 < wilson_upper(y[:40]) <= 1
