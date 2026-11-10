"""Monthly autonomy rollout: which evidence lets the gate earn autonomy safely?

Every month the gate picks the lowest agent-confidence threshold whose 95% upper bound on the
error rate, computed only from earlier months, fits the budget; lines at or above it are
auto-posted. Policies differ only in the evidence behind that bound:

  human_wilson   k random human audits a month; Wilson bound (TrustGate's rule).
  judge_raw      the LLM judge's verdict on every past line treated as truth; no humans.
  ppi_plain      judge on every line + k random human audits; textbook PPI++ bound.
  active_ppi     judge on every line + k *judge-guided* human audits (more likely where the judge
                 sees risk), inverse-probability weighted; active PPI bound.

For the two PPI policies the judge's raw score is first recalibrated, month by month, on the
human audits from *earlier* months only (Platt scaling). A small local judge is badly calibrated;
its raw probability is a poor baseline for the correction term. Because each month's calibration
uses only earlier audits, it is fixed before that month's audits are drawn and the estimators
stay unbiased.

Every policy with humans spends the same expected number of audits. Outcomes are always scored
against the human-booked truth.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import SHADOW_MONTHS
from .stats import active_ppi_upper, audit_propensity, platt_apply, platt_fit, ppi_upper_plain, wilson_interval, wilson_upper

GRID = np.round(np.concatenate([np.arange(0.30, 0.95, 0.01), np.arange(0.95, 1.0001, 0.0025)]), 4)
MIN_EVIDENCE = 20
POLICIES = ("human_wilson", "judge_raw", "ppi_plain", "active_ppi")


def assign_audits(df: pd.DataFrame, policy: str, per_month: int, seed: int = 0) -> pd.DataFrame:
    """Adds `audited` and `pi` (inclusion probability) columns, month by month. For the PPI
    policies `f` is replaced by the judge score recalibrated on earlier months' audits (`f_raw`
    keeps the original)."""
    rng = np.random.default_rng(seed)
    out = df.sort_values("statement_month", kind="stable").copy()
    out["audited"], out["pi"], out["f_raw"] = False, 1.0, out.f
    if policy == "judge_raw":
        return out
    for month, idx in out.groupby("statement_month").groups.items():
        idx = np.asarray(idx)
        if policy in ("ppi_plain", "active_ppi"):
            prior = out[(out.statement_month < month) & out.audited]
            out.loc[idx, "f"] = platt_apply(
                out.loc[idx, "f_raw"].to_numpy(), platt_fit(prior.f_raw.to_numpy(), prior.y.to_numpy())
            )
        if policy == "active_ppi":
            pi = audit_propensity(out.loc[idx, "f"].to_numpy(), per_month)
            out.loc[idx, "pi"] = pi
            out.loc[idx, "audited"] = rng.random(len(idx)) < pi
        else:
            k = min(per_month, len(idx))
            out.loc[rng.choice(idx, size=k, replace=False), "audited"] = True
            out.loc[idx, "pi"] = k / len(idx)
    return out


def bound(policy: str, sel: pd.DataFrame) -> tuple[float, int]:
    """95% upper bound on the error rate of the lines in `sel`, and the evidence count behind it."""
    lab = sel[sel.audited]
    if policy == "human_wilson":
        return wilson_upper(lab.y.to_numpy()), len(lab)
    if policy == "judge_raw":
        return wilson_upper((sel.p_correct < 0.5).to_numpy().astype(float)), len(sel)
    if policy == "ppi_plain":
        return ppi_upper_plain(lab.y.to_numpy(), lab.f.to_numpy(), sel.f.to_numpy()), len(lab)
    return active_ppi_upper(sel.y.to_numpy(), sel.f.to_numpy(), sel.audited.to_numpy(), sel.pi.to_numpy()), len(lab)


def choose_threshold(policy: str, past: pd.DataFrame, budget: float) -> float | None:
    for tau in GRID:
        upper, n = bound(policy, past[past.conf >= tau])
        if n < MIN_EVIDENCE:
            return None
        if upper <= budget:
            return float(tau)
    return None


def rollout(df: pd.DataFrame, policy: str, budget: float, audits_per_month: int, seed: int = 0) -> dict:
    """df columns: statement_month, conf, y (1 = agent wrong), f (judge P(wrong)), p_correct, amount."""
    d = assign_audits(df.reset_index(drop=True), policy, audits_per_month, seed)
    months = sorted(d.statement_month.unique())
    live = months[SHADOW_MONTHS:]
    rows, auto_all = [], pd.Series(False, index=d.index)
    for m in live:
        tau = choose_threshold(policy, d[d.statement_month < m], budget)
        cur = d[d.statement_month == m]
        auto = (cur.conf >= tau) if tau is not None else pd.Series(False, index=cur.index)
        auto_all.loc[cur.index] = auto.to_numpy()
        a = cur[auto]
        rows.append({"month": str(pd.Timestamp(m).date()), "tau": tau, "n": len(cur), "auto": len(a), "errors": int(a.y.sum())})
    r = pd.DataFrame(rows)
    n, a, e = int(r.n.sum()), int(r.auto.sum()), int(r.errors.sum())
    lo, hi = wilson_interval(e, a)
    livedf = d[d.statement_month.isin(live)]
    wrong_auto = livedf[auto_all.loc[livedf.index] & (livedf.y == 1)]
    return {
        "policy": policy,
        "budget": budget,
        "audits_per_month": 0 if policy == "judge_raw" else audits_per_month,
        "automation_rate": a / max(n, 1),
        "auto_error_rate": e / max(a, 1),
        "auto_error_ci": [lo, hi],
        "auto": a,
        "auto_errors": e,
        "live_lines": n,
        "live_months": len(live),
        "months_over_budget": int(((r.errors / r.auto.clip(lower=1)) > budget).sum()),
        "months_with_autonomy": int((r.auto > 0).sum()),
        "misposted_gbp": float(wrong_auto.amount.sum()),
        "human_labels": int(d.audited.sum()),
        "by_month": r.to_dict("records"),
    }
