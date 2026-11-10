"""Compute every reported number from the committed cache (no model needed).

Writes results/judgegate.json and results/coverage.json.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import cohen_kappa_score, roc_auc_score

from . import simulate
from .data import RESULTS, SHADOW_MONTHS, load_lines
from .gate import POLICIES, rollout
from .generate import SETTINGS, agent_file, load_judge
from .stats import bootstrap_ci, ece, platt_apply, platt_fit, wilson_interval

JUDGES = {"card_feed": ["llama"], "upstream": ["llama", "qwen"]}
BUDGETS = (0.02, 0.05, 0.10)
AUDITS = (10, 20, 40)


def frame(setting: str, judge: str | None) -> pd.DataFrame:
    lines = load_lines()[
        [
            "card_line_id",
            "statement_month",
            "merchant_raw",
            "department",
            "service_category",
            "merchant_category",
            "amount",
            "gl_account",
        ]
    ]
    props = pd.read_csv(agent_file(setting))[["card_line_id", "pred", "conf"]]
    df = props.merge(lines, on="card_line_id", how="inner")
    df["y"] = (df.pred != df.gl_account).astype(int)
    if judge:
        df = df.merge(load_judge(setting, judge), on="card_line_id", how="inner")
        df["f"] = 1.0 - df.p_correct
    return df.sort_values(["statement_month", "card_line_id"]).reset_index(drop=True)


def judge_quality(df: pd.DataFrame) -> dict[str, Any]:
    y, f, passed = df.y.to_numpy(), df.f.to_numpy(), (df.p_correct >= 0.5).to_numpy()
    wrong, right = y == 1, y == 0
    auc_ci = bootstrap_ci(
        np.arange(len(df)), stat=lambda idx: roc_auc_score(y[idx], f[idx]) if 0 < y[idx].sum() < len(idx) else 0.5, n=500
    )
    return {
        "n": int(len(df)),
        "agent_error_rate": float(y.mean()),
        "auc": float(roc_auc_score(y, f)),
        "auc_ci": auc_ci,
        "kappa_at_0_5": float(cohen_kappa_score(y, (~passed).astype(int))),
        "false_pass_rate": float(passed[wrong].mean()),
        "false_pass_ci": list(wilson_interval(int(passed[wrong].sum()), int(wrong.sum()))),
        "false_fail_rate": float((~passed)[right].mean()),
        "false_fail_ci": list(wilson_interval(int((~passed)[right].sum()), int(right.sum()))),
        "judge_pass_rate": float(passed.mean()),
        "ece": ece(f, y),
        "calibration": [
            {
                "bin_lo": float(lo),
                "mean_pred_wrong": float(f[(f >= lo) & (f < lo + 0.1)].mean()),
                "observed_wrong": float(y[(f >= lo) & (f < lo + 0.1)].mean()),
                "n": int(((f >= lo) & (f < lo + 0.1)).sum()),
            }
            for lo in np.arange(0, 1, 0.1)
            if ((f >= lo) & (f < lo + 0.1)).sum() >= 20
        ],
    }


REVIEW_SHARES = (0.1, 0.2, 0.3, 0.5)


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def incremental_value(df: pd.DataFrame) -> dict[str, float]:
    """Does the judge tell us anything the agent's own confidence does not?

    Cross-validated AUC for predicting the agent's errors from (a) its confidence alone and
    (b) confidence + judge score, and a likelihood-ratio test of the judge's coefficient."""
    from scipy.stats import chi2
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    y = df.y.to_numpy()
    xa = _logit(df.conf.to_numpy())[:, None]
    xb = np.column_stack([xa[:, 0], _logit(df.f.to_numpy())])
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    lr = LogisticRegression(C=1e4, max_iter=1000)
    pa = cross_val_predict(lr, xa, y, cv=cv, method="predict_proba")[:, 1]
    pb = cross_val_predict(lr, xb, y, cv=cv, method="predict_proba")[:, 1]
    lla = -log_loss(y, lr.fit(xa, y).predict_proba(xa)[:, 1], normalize=False)
    llb = -log_loss(y, lr.fit(xb, y).predict_proba(xb)[:, 1], normalize=False)
    stat = max(0.0, 2 * (llb - lla))
    return {
        "auc_agent_confidence": float(roc_auc_score(y, pa)),
        "auc_agent_plus_judge": float(roc_auc_score(y, pb)),
        "auc_judge_alone": float(roc_auc_score(y, df.f)),
        "lr_stat": float(stat),
        "lr_p_value": float(chi2.sf(stat, 1)),
    }


def triage(df: pd.DataFrame) -> list[dict[str, float]]:
    """If reviewers check lines in the judge's risk order, what share of the agent's errors do they
    find after reviewing a given share of lines? (Random order finds the same share as reviewed.)"""
    order = df.sort_values(["f", "card_line_id"], ascending=[False, True])
    found = order.y.cumsum().to_numpy() / max(int(order.y.sum()), 1)
    return [{"review_share": r, "errors_found": float(found[max(int(round(r * len(order))) - 1, 0)])} for r in REVIEW_SHARES]


def mcnemar(a: pd.DataFrame, b: pd.DataFrame) -> dict[str, Any]:
    """Do two judges agree with humans equally often? Exact McNemar test on discordant pairs."""
    m = a[["card_line_id", "y", "p_correct"]].merge(b[["card_line_id", "p_correct"]], on="card_line_id", suffixes=("_a", "_b"))
    ok_a = (m.p_correct_a >= 0.5) == (m.y == 0)
    ok_b = (m.p_correct_b >= 0.5) == (m.y == 0)
    only_a, only_b = int((ok_a & ~ok_b).sum()), int((~ok_a & ok_b).sum())
    p = binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
    return {
        "n": int(len(m)),
        "accuracy_a": float(ok_a.mean()),
        "accuracy_b": float(ok_b.mean()),
        "only_a_right": only_a,
        "only_b_right": only_b,
        "p_value": float(p),
    }


def error_breakdown(df: pd.DataFrame, k: int = 5) -> dict[str, Any]:
    """Which staff accounts the agent's errors (and the judge's false passes) belong to."""
    err = df[df.y == 1]
    fp = err[err.p_correct >= 0.5]

    def top(d: pd.DataFrame) -> list[dict[str, Any]]:
        return [
            {"account": a, "share": float(c / max(len(d), 1)), "n": int(c)}
            for a, c in d.gl_account.value_counts().head(k).items()
        ]

    return {
        "agent_errors": int(len(err)),
        "agent_error_accounts": top(err),
        "false_passes": int(len(fp)),
        "false_pass_accounts": top(fp),
        "account_share_of_all_lines": {
            a: float((df.gl_account == a).mean()) for a in err.gl_account.value_counts().head(k).index
        },
    }


def judge_mistakes(df: pd.DataFrame, k: int = 25) -> list[dict]:
    """The judge's most confident mistakes: wrong bookings it passed, right bookings it failed."""
    fp = df[df.y == 1].sort_values(["p_correct", "card_line_id"], ascending=[False, True]).head(k).assign(kind="false_pass")
    ff = df[df.y == 0].sort_values(["p_correct", "card_line_id"]).head(k).assign(kind="false_fail")
    return [
        {
            "kind": r.kind,
            "month": str(pd.Timestamp(r.statement_month).date())[:7],
            "merchant": r.merchant_raw,
            "department": r.department,
            "service": r.service_category,
            "amount": round(float(r.amount), 2),
            "proposed": r.pred,
            "booked_by_staff": r.gl_account,
            "judge_p_correct": round(float(r.p_correct), 3),
        }
        for r in pd.concat([fp, ff]).itertuples()
    ]


REAL_AUDITS = (40, 400)
REAL_SLICES = (0.0, 0.1, 0.2, 0.5, 0.9)  # keep lines with agent confidence >= this quantile


def real_data_coverage(judge: str, setting: str, reps: int = 1000) -> list[dict[str, Any]]:
    """Coverage of each bound on real populations whose true error rate is known.

    Populations: live-month lines with agent confidence at or above a quantile (all lines down to
    the most confident tenth), so the true error rate ranges from rare to common. The judge's
    scores are recalibrated once, on the shadow months, so the calibration map is fixed and
    independent of the population being estimated. Only the audits are redrawn. 400 audits is
    about what the gate has accumulated by mid-rollout at 20 a month."""
    df = frame(setting, judge)
    months = sorted(df.statement_month.unique())
    shadow, live = df[df.statement_month < months[SHADOW_MONTHS]], df[df.statement_month >= months[SHADOW_MONTHS]]
    params = platt_fit(shadow.f.to_numpy(), shadow.y.to_numpy())
    out = []
    for q in REAL_SLICES:
        sl = live[live.conf >= live.conf.quantile(q)]
        f = platt_apply(sl.f.to_numpy(), params)
        for n in (n for n in REAL_AUDITS if n <= len(sl) // 2):  # audits must stay a sample
            out.append(
                {
                    "judge": judge,
                    "setting": setting,
                    "confidence_quantile": q,
                    **simulate.real_coverage(sl.y.to_numpy(), f, n, reps=reps),
                }
            )
    return out


SEEDS = 20  # independent audit draws per rollout; a single draw is mostly noise at these audit levels


def _rollout_seed(args: tuple[pd.DataFrame, str, float, int, int]) -> dict[str, Any]:
    df, pol, budget, k, seed = args
    r = rollout(df, pol, budget, k, seed=seed)
    r.pop("by_month")
    return r


def summarise_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Average a policy's rollout over independent audit draws."""
    auto = np.array([r["automation_rate"] for r in runs])
    over = np.array([r["months_over_budget"] for r in runs])
    n_auto, n_err = sum(r["auto"] for r in runs), sum(r["auto_errors"] for r in runs)
    r0 = runs[0]
    return {
        "policy": r0["policy"],
        "budget": r0["budget"],
        "audits_per_month": r0["audits_per_month"],
        "seeds": len(runs),
        "automation_rate": float(auto.mean()),
        "automation_rate_sd": float(auto.std(ddof=1)) if len(runs) > 1 else 0.0,
        "automation_rate_range": [float(auto.min()), float(auto.max())],
        "auto_error_rate": n_err / max(n_auto, 1),
        "months_over_budget": float(over.mean()),
        "runs_over_budget": float((over > 0).mean()),
        "misposted_gbp": float(np.mean([r["misposted_gbp"] for r in runs])),
        "human_labels": float(np.mean([r["human_labels"] for r in runs])),
        "live_lines": r0["live_lines"],
        "live_months": r0["live_months"],
    }


def run_rollouts(frames: dict[str, pd.DataFrame], workers: int | None = None) -> list[dict[str, Any]]:
    """Every (setting, judge, budget, policy, audits) rollout over SEEDS audit draws, in parallel.
    Human-only rollouts do not depend on the judge, so they are computed once per setting."""
    import os
    from concurrent.futures import ProcessPoolExecutor

    jobs: dict[tuple, list[tuple]] = {}
    for key, df in frames.items():
        j, s = key.split("/")
        for budget in BUDGETS:
            for pol in POLICIES:
                for k in (0,) if pol == "judge_raw" else AUDITS:
                    owner = f"any/{s}" if pol == "human_wilson" else key
                    seeds = 1 if pol == "judge_raw" else SEEDS
                    jobs.setdefault((owner, budget, pol, k), [(df, pol, budget, k, seed) for seed in range(seeds)])
    flat = [(cfg, a) for cfg, args in jobs.items() for a in args]
    with ProcessPoolExecutor(max_workers=workers or min(4, os.cpu_count() or 1)) as ex:
        results = list(ex.map(_rollout_seed, [a for _, a in flat], chunksize=4))
    grouped: dict[tuple, list[dict[str, Any]]] = {}
    for (cfg, _), r in zip(flat, results, strict=True):
        grouped.setdefault(cfg, []).append(r)
    out = []
    for key in frames:
        j, s = key.split("/")
        for (owner, *_), runs in grouped.items():
            if owner in (key, f"any/{s}"):
                out.append({"setting": s, "judge": j, **summarise_runs(runs)})
    return sorted(out, key=lambda r: (r["setting"], r["judge"], r["budget"], POLICIES.index(r["policy"]), r["audits_per_month"]))


def run(sim_reps: int = 500) -> dict[str, Any]:
    out: dict[str, Any] = {"agent": {}, "judges": {}, "rollouts": [], "judge_comparison": {}, "judge_mistakes": {}}
    frames: dict[str, pd.DataFrame] = {}
    for s in SETTINGS:
        a = frame(s, None)
        acc = 1 - a.y.mean()
        lo, hi = wilson_interval(int((a.y == 0).sum()), len(a))
        out["agent"][s] = {"lines": int(len(a)), "accuracy": float(acc), "accuracy_ci": [lo, hi]}
        for j in JUDGES[s]:
            df = frame(s, j)
            if df.empty:
                continue
            key = f"{j}/{s}"
            out["judges"][key] = judge_quality(df) | {
                "triage": triage(df),
                "triage_agent_confidence": triage(df.assign(f=1 - df.conf)),
                "incremental": incremental_value(df),
                "error_breakdown": error_breakdown(df),
            }
            out["judge_mistakes"][key] = judge_mistakes(df)
            frames[key] = df
    out["rollouts"] = run_rollouts(frames)
    if "llama/upstream" in out["judges"] and "qwen/upstream" in out["judges"]:
        out["judge_comparison"]["upstream_llama_vs_qwen"] = mcnemar(frame("upstream", "llama"), frame("upstream", "qwen"))
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "judgegate.json").write_text(json.dumps(out, indent=2, default=float))
    cov = simulate.run(reps=sim_reps)
    cov["real_data"] = [g for key in out["judges"] for g in real_data_coverage(*key.split("/"))]
    (RESULTS / "coverage.json").write_text(json.dumps(cov, indent=2))
    return out
