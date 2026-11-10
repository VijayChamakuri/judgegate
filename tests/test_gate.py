import numpy as np
import pandas as pd

from judgegate.gate import assign_audits, rollout


def world(n_months=18, per_month=150, seed=0, judge_noise=0.05):
    rng = np.random.default_rng(seed)
    rows = []
    for m in pd.date_range("2020-01-01", periods=n_months, freq="MS"):
        conf = rng.uniform(0.3, 1.0, per_month)
        y = (rng.random(per_month) < (1 - conf) * 0.3).astype(int)  # higher confidence, fewer errors
        f = np.clip(0.8 * y + rng.normal(0.1, judge_noise, per_month), 0, 1)
        rows.append(
            pd.DataFrame(
                {"statement_month": m, "conf": conf, "y": y, "f": f, "p_correct": 1 - f, "amount": rng.uniform(5, 500, per_month)}
            )
        )
    return pd.concat(rows, ignore_index=True)


def test_audits_are_per_month_and_reproducible():
    df = world()
    a1, a2 = assign_audits(df, "human_wilson", 20, seed=3), assign_audits(df, "human_wilson", 20, seed=3)
    assert a1.audited.equals(a2.audited)
    assert a1[a1.audited].groupby("statement_month").size().eq(20).all()


def test_judge_guided_audits_spend_the_same_budget_on_riskier_lines():
    df = world()
    act = assign_audits(df, "active_ppi", 20, seed=1)
    per_month = act[act.audited].groupby("statement_month").size()
    assert abs(per_month.mean() - 20) < 3
    assert act[act.audited].y.mean() > df.y.mean()  # audits land where errors are


def test_gate_uses_only_past_evidence():
    """Changing labels in a month must not change that month's threshold."""
    df = world()
    base = rollout(df, "human_wilson", 0.10, 40)
    df2 = df.copy()
    last = df2.statement_month.max()
    df2.loc[df2.statement_month == last, "y"] = 1
    tampered = rollout(df2, "human_wilson", 0.10, 40)
    assert base["by_month"][-1]["tau"] == tampered["by_month"][-1]["tau"]


def test_judge_recalibration_uses_only_earlier_months():
    df = world()
    last = df.statement_month.max()
    a = assign_audits(df, "active_ppi", 20, seed=2)
    tampered = df.copy()
    tampered.loc[tampered.statement_month == last, "y"] = 1 - tampered.loc[tampered.statement_month == last, "y"]
    b = assign_audits(tampered, "active_ppi", 20, seed=2)
    assert np.allclose(a.f, b.f)  # month m's calibrated scores ignore month m's own truth
    assert not np.allclose(a.f, a.f_raw)  # and the calibration did change the scores
