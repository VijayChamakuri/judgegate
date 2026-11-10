"""Render README.md from docs/README.template.md and results/*.json.

Placeholders look like {{name}}. Every number in the README comes from a results file, and
tests/test_readme.py fails if README.md differs from a fresh render, so prose and evidence
cannot drift apart.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .data import RESULTS, ROOT, load_lines

TEMPLATE = ROOT / "docs" / "README.template.md"
README = ROOT / "README.md"

JUDGE_LABEL = {"llama": "Llama-3.2-3B", "qwen": "Qwen2.5-3B"}
SETTING_LABEL = {"card_feed": "Card feed", "upstream": "Upstream"}
POLICY_LABEL = {
    "human_wilson": "Human audits only (Wilson bound)",
    "judge_raw": "Raw judge, no humans",
    "ppi_plain": "Judge + random audits (plain PPI)",
    "active_ppi": "Judge + judge-guided audits (active PPI)",
}
METHOD_LABEL = {
    "wilson_random_audits": "Wilson, random audits",
    "ppi_plain_random_audits": "Plain PPI, random audits",
    "active_ppi_judge_guided_audits": "Active PPI, judge-guided audits",
    "active_ppi_tuned_same_sample": "Active PPI, tuned on the same sample",
}
HEAD: dict[str, Any] = {"setting": "upstream", "judge": "llama", "budget": 0.05, "audits": 20}


def pct(x: float | None, d: int = 1) -> str:
    return "–" if x is None else f"{100 * x:.{d}f}%"


def _j(name: str) -> dict:
    return json.loads((RESULTS / name).read_text())


def _roll(
    res: dict,
    policy: str,
    audits: int,
    setting: str = HEAD["setting"],
    judge: str = HEAD["judge"],
    budget: float = HEAD["budget"],
) -> dict:
    k = 0 if policy == "judge_raw" else audits
    return next(
        r
        for r in res["rollouts"]
        if r["setting"] == setting
        and r["judge"] == judge
        and r["budget"] == budget
        and r["policy"] == policy
        and r["audits_per_month"] == k
    )


def judge_rows(res: dict) -> str:
    out = [
        "| Judge | Setting | Lines | Agent wrong | AUC (95% CI) | Cohen's κ at 0.5 | False pass | False fail | ECE |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|",
    ]
    for key, q in res["judges"].items():
        j, s = key.split("/")
        out.append(
            f"| {JUDGE_LABEL[j]} | {SETTING_LABEL[s]} | {q['n']:,} | {pct(q['agent_error_rate'])} | "
            f"{q['auc']:.2f} ({q['auc_ci'][0]:.2f}–{q['auc_ci'][1]:.2f}) | {q['kappa_at_0_5']:.2f} | "
            f"{pct(q['false_pass_rate'])} | {pct(q['false_fail_rate'])} | {q['ece']:.3f} |"
        )
    return "\n".join(out)


def rollout_rows(res: dict, setting: str = HEAD["setting"], judge: str = HEAD["judge"], budget: float = HEAD["budget"]) -> str:
    out = [
        "| Policy | Audits / month | Human labels | Auto-posted (mean ± sd) | Error when auto-posted | Months over budget (mean) | Draws with any month over | Mis-posted £ (mean) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    rows = [r for r in res["rollouts"] if r["setting"] == setting and r["judge"] == judge and r["budget"] == budget]
    for pol in POLICY_LABEL:
        for r in sorted((r for r in rows if r["policy"] == pol), key=lambda r: r["audits_per_month"]):
            err = pct(r["auto_error_rate"], 2) if r["automation_rate"] > 0 else "–"
            out.append(
                f"| {POLICY_LABEL[pol]} | {r['audits_per_month']} | {r['human_labels']:,.0f} | "
                f"{pct(r['automation_rate'])} ± {pct(r['automation_rate_sd'])} | {err} | "
                f"{r['months_over_budget']:.1f} / {r['live_months']} | {pct(r['runs_over_budget'], 0)} | £{r['misposted_gbp']:,.0f} |"
            )
    return "\n".join(out)


def budget_rows(res: dict, audits: int = HEAD["audits"]) -> str:
    """Every setting x judge x budget for the raw judge and the two human-backed policies, at one audit level."""
    out = [
        "| Setting | Judge | Budget | Raw judge: auto / months over | Human only: auto / months over | Active PPI: auto / months over |",
        "|---|---|---:|---|---|---|",
    ]
    for key in res["judges"]:
        j, s = key.split("/")
        for b in sorted({r["budget"] for r in res["rollouts"]}):
            cell = []
            for pol in ("judge_raw", "human_wilson", "active_ppi"):
                r = _roll(res, pol, audits, s, j, b)
                cell.append(f"{pct(r['automation_rate'])} / {r['months_over_budget']:.1f}")
            out.append(f"| {SETTING_LABEL[s]} | {JUDGE_LABEL[j]} | {pct(b, 0)} | " + " | ".join(cell) + " |")
    return "\n".join(out)


RARE = (0.0, 0.10)  # README shows real-data slices whose true error is in (0, 10%): the regime a gate decides in


def coverage_rows(cov: dict) -> str:
    out = [
        "| Judge | Setting | Lines kept (agent confidence) | True error | Audits | " + " | ".join(METHOD_LABEL.values()) + " |",
        "|---|---|---|---:|---:|" + "---|" * len(METHOD_LABEL),
    ]
    for g in cov["real_data"]:
        if not RARE[0] < g["true_error"] < RARE[1]:
            continue
        cells = [f"{pct(g[m]['coverage'])} / {pct(g[m]['mean_upper'], 2)}" for m in METHOD_LABEL]
        out.append(
            f"| {JUDGE_LABEL[g['judge']]} | {SETTING_LABEL[g['setting']]} | top {pct(1 - g['confidence_quantile'], 0)} | "
            f"{pct(g['true_error'], 2)} | {g['audits']} | " + " | ".join(cells) + " |"
        )
    return "\n".join(out)


def sim_rows(cov: dict, audits: int = 40) -> str:
    out = [
        "| Judge strength | True error | " + " | ".join(METHOD_LABEL.values()) + " |",
        "|---:|---:|" + "---|" * len(METHOD_LABEL),
    ]
    for g in cov["grid"]:
        if g["audits"] != audits:
            continue
        cells = [f"{pct(g[m]['coverage'])} / {pct(g[m]['mean_upper'], 2)}" for m in METHOD_LABEL]
        out.append(f"| {g['judge_strength']} | {pct(g['true_error'], 0)} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def mistake_rows(res: dict, key: str = "llama/upstream", kind: str = "false_pass", k: int = 6) -> str:
    out = ["| Merchant | Department | £ | Agent proposed | Staff booked | Judge P(correct) |", "|---|---|---:|---|---|---:|"]
    seen: set[tuple] = set()
    rows = []
    for m in res["judge_mistakes"].get(key, []):
        sig = (m["merchant"], m["amount"], m["proposed"], m["booked_by_staff"])
        if m["kind"] == kind and sig not in seen:
            seen.add(sig)
            rows.append(m)
    for m in rows[:k]:
        out.append(
            f"| {m['merchant']} | {m['department']} | {m['amount']:,.2f} | {m['proposed']} | {m['booked_by_staff']} | "
            f"{m['judge_p_correct']:.3f} |"
        )
    return "\n".join(out)


def values() -> dict[str, str]:
    res, cov = _j("judgegate.json"), _j("coverage.json")
    a = HEAD["audits"]
    hj = f"{HEAD['judge']}/{HEAD['setting']}"
    q = res["judges"][hj]
    raw, human, plain, active = (_roll(res, p, a) for p in ("judge_raw", "human_wilson", "ppi_plain", "active_ppi"))
    agent = res["agent"]
    v = {
        "lines": f"{len(load_lines()):,}",
        "live_lines": f"{raw['live_lines']:,}",
        "live_months": str(raw["live_months"]),
        "audits": str(a),
        "budget": pct(HEAD["budget"], 0),
        "headline_judge": JUDGE_LABEL[HEAD["judge"]],
        "agent_acc_up": pct(agent["upstream"]["accuracy"]),
        "agent_acc_cf": pct(agent["card_feed"]["accuracy"]),
        "judge_auc": f"{q['auc']:.2f}",
        "judge_kappa": f"{q['kappa_at_0_5']:.2f}",
        "judge_false_pass": pct(q["false_pass_rate"]),
        "judge_false_fail": pct(q["false_fail_rate"]),
        "judge_pass_rate": pct(q["judge_pass_rate"]),
        "raw_auto": pct(raw["automation_rate"]),
        "raw_err": pct(raw["auto_error_rate"], 2),
        "raw_over": f"{raw['months_over_budget']:.1f}/{raw['live_months']}",
        "raw_gbp": f"£{raw['misposted_gbp']:,.0f}",
        "human_auto": pct(human["automation_rate"]),
        "human_err": pct(human["auto_error_rate"], 2),
        "human_over": f"{human['months_over_budget']:.1f}/{human['live_months']}",
        "human_labels": f"{human['human_labels']:,.0f}",
        "plain_auto": pct(plain["automation_rate"]),
        "plain_err": pct(plain["auto_error_rate"], 2),
        "plain_over": f"{plain['months_over_budget']:.1f}/{plain['live_months']}",
        "active_auto": pct(active["automation_rate"]),
        "active_err": pct(active["auto_error_rate"], 2),
        "active_over": f"{active['months_over_budget']:.1f}/{active['live_months']}",
        "active_labels": f"{active['human_labels']:,.0f}",
        "sim_reps": str(cov["reps"]),
        "seeds": str(raw.get("seeds", 1) if raw["policy"] != "judge_raw" else human["seeds"]),
        "active_cov_min": pct(min(g["active_ppi_judge_guided_audits"]["coverage"] for g in cov["real_data"]), 0),
        "plain_cov_min": pct(min(g["ppi_plain_random_audits"]["coverage"] for g in cov["real_data"]), 0),
        "cf_plain_over": f"{_roll(res, 'ppi_plain', 40, 'card_feed', 'llama', 0.10)['months_over_budget']:.1f}",
        "cf_human_over": f"{_roll(res, 'human_wilson', 40, 'card_feed', 'llama', 0.10)['months_over_budget']:.1f}",
        "cf_plain_err": pct(_roll(res, "ppi_plain", 40, "card_feed", "llama", 0.10)["auto_error_rate"]),
    }
    if "upstream_llama_vs_qwen" in res["judge_comparison"]:
        c = res["judge_comparison"]["upstream_llama_vs_qwen"]
        v |= {
            "mc_llama": pct(c["accuracy_a"]),
            "mc_qwen": pct(c["accuracy_b"]),
            "mc_p": f"{c['p_value']:.2g}",
            "mc_only_a": str(c["only_a_right"]),
            "mc_only_b": str(c["only_b_right"]),
        }
    return v


def _p(x: float) -> str:
    """p-value for prose; exact tests on thousands of lines can underflow to 0."""
    return "< 1e-300" if x == 0 else f"= {x:.0e}" if x < 1e-3 else f"= {x:.3f}"


def prose() -> dict[str, str]:
    """Narrative sections. Every number is read from results; the wording was written after the
    results were in and states only what they show."""
    res, cov = _j("judgegate.json"), _j("coverage.json")
    v = values()
    a = HEAD["audits"]
    R = lambda pol, b, judge="llama", setting="upstream", k=a: _roll(res, pol, k, setting, judge, b)  # noqa: E731
    rare = [g for g in cov["real_data"] if RARE[0] < g["true_error"] < RARE[1]]
    plain_cov40 = [g["ppi_plain_random_audits"]["coverage"] for g in rare if g["audits"] == 40]
    plain_cov400 = [g["ppi_plain_random_audits"]["coverage"] for g in rare if g["audits"] == 400]
    active_cov = [g["active_ppi_judge_guided_audits"]["coverage"] for g in cov["real_data"]]
    wilson_cov = [g["wilson_random_audits"]["coverage"] for g in cov["real_data"]]
    inc = {k: res["judges"][k]["incremental"] for k in res["judges"]}
    tri = res["judges"]["llama/upstream"]
    t20 = next(t for t in tri["triage"] if t["review_share"] == 0.2)["errors_found"]
    a20 = next(t for t in tri["triage_agent_confidence"] if t["review_share"] == 0.2)["errors_found"]
    ql, qq = res["judges"]["llama/upstream"], res["judges"]["qwen/upstream"]
    eb = ql["error_breakdown"]
    covid = next((x for x in eb["agent_error_accounts"] if x["account"] == "COVID 19"), None)
    seeds = R("human_wilson", 0.05)["seeds"]

    plain_english = (
        f"A finance agent books council card payments to ledger accounts. A *gate* decides how much of that work the "
        f"agent may do unchecked. It needs evidence, usually human spot-checks, that the agent's error rate is under a "
        f'budget. Human checks are slow and costly, so the tempting move is to let a cheaper AI "judge" do the checking.\n\n'
        f"JudgeGate tests that move on {v['lines']} real bookings, where council staff's own coding is the right answer, "
        f"with two free local judges. Both can *rank* bookings (wrong ones score as riskier), but both are too strict to "
        f"believe as-is, so on their own they unlock nothing. The textbook statistical repair (PPI) makes a judge look like "
        f"a big win: {pct(R('ppi_plain', 0.02)['automation_rate'], 0)} of bookings automated at a 2% error budget instead of "
        f"{pct(R('human_wilson', 0.02)['automation_rate'], 0)}. But the safety margin it reports is too small when mistakes "
        f"are rare, which is exactly when a gate is deciding. A version that is honest about its uncertainty does exist. "
        f"With the better-calibrated judge it roughly matches human checks; with the worse one it loses to them.\n\n"
        f"So the deliverable is less a judge than a **test harness**: before any judge-backed number is allowed to grant "
        f"autonomy, check its coverage on real data where the truth is known, and average over audit draws rather than "
        f"trusting one lucky run."
    )

    key_results = "\n".join(
        [
            f"1. **Judges rank well but can't be taken at their word.** Llama-3.2-3B and Qwen2.5-3B rank wrong bookings above "
            f"right ones about equally well (AUC {ql['auc']:.2f} and {qq['auc']:.2f}), but at the natural 0.5 cut-off they reject "
            f"{pct(ql['false_fail_rate'], 0)} and {pct(qq['false_fail_rate'], 0)} of *correct* bookings. Qwen is far better "
            f"calibrated (ECE {qq['ece']:.2f} vs {ql['ece']:.2f}; it agrees with staff on {v['mc_qwen']} of lines vs {v['mc_llama']}, "
            f"exact McNemar p {_p(float(res['judge_comparison']['upstream_llama_vs_qwen']['p_value']))}). Used as the gate's only "
            f"evidence, neither judge grants **any autonomy** at any budget.",
            f"2. **The obvious statistical fix looks like a free lunch, and isn't.** Plain PPI (judge on every line, recalibrated "
            f"on earlier months, corrected with {a} random audits a month) let the gate auto-post "
            f"{pct(R('ppi_plain', 0.02)['automation_rate'])} of live upstream lines at a 2% budget, against "
            f"{pct(R('human_wilson', 0.02)['automation_rate'])} for the same audits and no judge (mean of {seeds} audit draws). "
            f'But on real slices where errors are rare (0–10%), its "95%" bound covered the true error rate as little as '
            f"{pct(min(plain_cov40), 0)} of the time with 40 audits and {pct(min(plain_cov400), 0)} with 400, against a "
            f"promised 95%. Its extra autonomy rests on a safety margin that is too small: on this easy data the agent's "
            f"confident lines happen to be nearly error-free, so nothing broke, but the bound would not have warned you if they "
            f"weren't.",
            f"3. **An honest judge-assisted bound exists, and its value depends on the judge.** Active PPI (audits drawn where "
            f"the judge sees risk, inverse-probability weighted) covered the truth at least {pct(min(active_cov), 0)} of the "
            f"time on every real slice (Wilson: {pct(min(wilson_cov), 0)}). At a {v['budget']} budget with {a} audits a month "
            f"it auto-posted {pct(R('active_ppi', 0.05, 'qwen')['automation_rate'])} with Qwen, "
            f"{pct(R('active_ppi', 0.05)['automation_rate'])} with Llama, and {pct(R('human_wilson', 0.05)['automation_rate'])} "
            f"with humans alone (draw-to-draw sd about {pct(R('active_ppi', 0.05, 'qwen')['automation_rate_sd'], 0)}). "
            f"Judge-guided audits flow to risky lines the gate never automates, so only a well-calibrated judge earns them back.",
            f"4. **The judge's extra information is real but small.** Added to the agent's own confidence, Llama lifts "
            f"cross-validated error-detection AUC from {inc['llama/upstream']['auc_agent_confidence']:.3f} to "
            f"{inc['llama/upstream']['auc_agent_plus_judge']:.3f} upstream (Qwen: {inc['qwen/upstream']['auc_agent_plus_judge']:.3f}) "
            f"and from {inc['llama/card_feed']['auc_agent_confidence']:.3f} to {inc['llama/card_feed']['auc_agent_plus_judge']:.3f} "
            f"on card feed (likelihood-ratio test p {_p(inc['llama/upstream']['lr_p_value'])} upstream, "
            f"p {_p(inc['llama/card_feed']['lr_p_value'])} card feed). On its own the judge is *worse* than the agent's confidence at "
            f"triage: the 20% of upstream lines Llama flags hold {pct(t20, 0)} of the errors, against {pct(a20, 0)} for the "
            f"agent's least-confident 20%.",
            f"5. **Single runs mislead.** With {a} audits a month, plain PPI's automation rate at a 2% budget has a standard "
            f"deviation of {100 * R('ppi_plain', 0.02)['automation_rate_sd']:.0f} percentage points between audit draws. Every rollout "
            f"number here is a mean over {seeds} draws.",
        ]
    )
    key_results += (
        "\n\n**What I would ship:** human audits stay the evidence for autonomy unless a judge-assisted bound passes a "
        "coverage test like section 3 on real, rare-error data, and then only with a well-calibrated judge. The judge, "
        "combined with the agent's confidence, orders the human review queue."
    )

    failure = (
        f"{pct(covid['share'], 0)} of the agent's upstream errors are lines staff booked to the **COVID 19** account "
        f"({pct(eb['account_share_of_all_lines']['COVID 19'], 1)} of all lines): a pandemic-era account that no merchant, "
        f'department or precedent predicts, so neither agent nor judge can see it coming. Many other "errors" are '
        f"near-synonym accounts (Travelling Expenses vs Expenses and Allowances)."
        if covid
        else ""
    )

    limitations = "\n".join(
        [
            "- **One organisation, one task.** Stockport's coding culture and 21-account chart are not a company's.",
            "- **Small local judges.** 3B models on a laptop. A frontier judge would probably be better calibrated. The "
            "harness, not the judge, is the point: it would tell you whether that judge's numbers can be trusted.",
            '- **Staff labels are not perfect.** Some "false passes" may be inconsistent staff coding. Examples are on the site.',
            "- **The upstream setting is easy by design.** Its procurement category nearly fixes the account, and the agent "
            "makes almost no errors on its confident half, so coverage there is trivially 100%. The informative slices are "
            "shown above.",
            "- **The gate itself is deliberately simple.** It takes the lowest confidence threshold whose bound fits (a search "
            "that favours optimistic bounds) and has no novelty guard, so on card-feed data even human-only gating ran over "
            "budget in some months. [TrustGate](https://github.com/VijayChamakuri/trustgate) adds those guards. Here they are "
            "left out so that only the evidence differs between policies.",
            f"- **Simulation settings are stylised.** Judge noise and error rates are chosen to bracket the real data, not "
            f"fitted to it. {v['sim_reps']} repeats per cell.",
        ]
    )
    comparison = (
        f"Llama vs Qwen on the same upstream lines (exact McNemar test on the lines where they disagree at 0.5): Llama "
        f"right and Qwen wrong on {v['mc_only_a']} lines, the reverse on {v['mc_only_b']}, p "
        f"{_p(float(res['judge_comparison']['upstream_llama_vs_qwen']['p_value']))}."
    )
    return {
        "plain_english": plain_english,
        "key_results": key_results,
        "limitations": limitations,
        "judge_comparison": comparison,
        "failure_analysis": failure,
    }


def tables() -> dict[str, str]:
    res, cov = _j("judgegate.json"), _j("coverage.json")
    return {
        "judge_table": judge_rows(res),
        "rollout_table": rollout_rows(res),
        "budget_table": budget_rows(res),
        "coverage_table": coverage_rows(cov),
        "sim_table": sim_rows(cov),
        "false_pass_table": mistake_rows(res),
        "false_fail_table": mistake_rows(res, kind="false_fail"),
    }


def render_text() -> str:
    text = TEMPLATE.read_text()
    subs = values() | tables() | prose()

    def sub(m: re.Match[str]) -> str:
        key = m.group(1)
        if key not in subs:
            raise KeyError(f"README template placeholder {{{{{key}}}}} has no value")
        return subs[key]

    return re.sub(r"\{\{(\w+)\}\}", sub, text)


def render(write: bool = True) -> str:
    text = render_text()
    if write:
        README.write_text(text)
    return text
