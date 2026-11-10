"""PNG charts for the README and site, drawn only from results/*.json."""

from __future__ import annotations

import json
from typing import Any

import matplotlib.pyplot as plt

from . import chartstyle as cs
from .data import RESULTS

POLICY_STYLE = {  # fixed order; colour follows the policy in every chart
    "human_wilson": (cs.BLUE, "o", "Human audits only (Wilson)"),
    "ppi_plain": (cs.ORANGE, "s", "Judge + random audits (plain PPI)"),
    "active_ppi": (cs.PURPLE, "D", "Judge + judge-guided audits (active PPI)"),
    "judge_raw": (cs.INK2, "X", "Judge alone, no people"),
}
METHOD_LABEL = {
    "wilson_random_audits": "Wilson, random audits",
    "ppi_plain_random_audits": "Plain PPI, random audits",
    "active_ppi_judge_guided_audits": "Active PPI, judge-guided",
    "active_ppi_tuned_same_sample": "Active PPI, tuned on same sample",
}
METHOD_COLOUR = dict(zip(METHOD_LABEL, cs.SERIES, strict=True))
SETTING = {"upstream": "upstream", "card_feed": "card feed"}
NICE = {
    "llama/card_feed": "Llama-3.2-3B, card feed",
    "llama/upstream": "Llama-3.2-3B, upstream",
    "qwen/upstream": "Qwen2.5-3B, upstream",
}
LINE: dict[str, Any] = {"linewidth": 1.8, "markersize": 6, "markeredgecolor": "white", "markeredgewidth": 1.2}


def load() -> tuple[dict, dict]:
    return (json.loads((RESULTS / "judgegate.json").read_text()), json.loads((RESULTS / "coverage.json").read_text()))


def headline_chart(res: dict, cov: dict) -> str:
    """Left: autonomy each policy appears to earn (upstream, 2% budget, mean of audit draws).
    Right: how often each policy's "95%" bound really covers the truth on real rare-error slices."""
    cs.apply()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    ax = axes[0]
    rows = [r for r in res["rollouts"] if r["setting"] == "upstream" and r["judge"] == "llama" and r["budget"] == 0.02]
    for pol, (colour, marker, label) in POLICY_STYLE.items():
        pts = sorted((r["audits_per_month"], 100 * r["automation_rate"]) for r in rows if r["policy"] == pol)
        if pts:
            xs, ys = zip(*pts, strict=True)
            ax.plot(xs, ys, color=colour, marker=marker, label=label, linestyle="--" if pol == "ppi_plain" else "-", **LINE)
    ax.set_xticks([0, 10, 20, 40])
    ax.set_xlabel("Human audits per month")
    ax.set_ylabel("Live lines auto-posted (%)")
    ax.set_title("Autonomy each policy earns (upstream, 2% budget, Llama judge)")
    handles, labels = ax.get_legend_handles_labels()

    ax = axes[1]
    sl = sorted(
        (g for g in cov["real_data"] if g["judge"] == "llama" and g["audits"] == 400 and 0 < g["true_error"] < 0.1),
        key=lambda g: g["true_error"],
    )
    pos = list(range(len(sl)))
    for m, pol in (
        ("wilson_random_audits", "human_wilson"),
        ("ppi_plain_random_audits", "ppi_plain"),
        ("active_ppi_judge_guided_audits", "active_ppi"),
    ):
        colour, marker, _ = POLICY_STYLE[pol]
        ax.plot(
            pos,
            [100 * g[m]["coverage"] for g in sl],
            color=colour,
            marker=marker,
            linestyle="--" if pol == "ppi_plain" else "-",
            **LINE,
        )
    ax.axhline(95, color=cs.INK2, linewidth=0.9, linestyle=":")
    ax.annotate("promised 95%", (pos[0], 95), xytext=(4, -12), textcoords="offset points", fontsize=8, color=cs.INK2)
    ax.set_xticks(pos, [f"{100 * g['true_error']:.2f}%\n{SETTING[g['setting']]}" for g in sl])
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(50, 101)
    ax.set_xlabel("True error rate of the real slice")
    ax.set_ylabel("How often the 95% bound held (%)")
    ax.set_title("Does the bound hold? (400 audits, 1,000 redraws)")
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 0.0), ncol=4, handlelength=2.4)
    fig.tight_layout(w_pad=3)
    out = RESULTS / "headline.png"
    fig.savefig(out)
    plt.close(fig)
    return out.name


def coverage_chart(cov: dict, sep: float = 0.6, audits: int = 40) -> str:
    cs.apply()
    grid = [g for g in cov["grid"] if g["judge_strength"] == sep and g["audits"] == audits]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.9))
    ps = [g["true_error"] for g in grid]
    width = 0.2
    for ax, key, title in (
        (axes[0], "coverage", "How often the bound held (%), target 95"),
        (axes[1], "mean_upper", "Average upper bound (%), lower is tighter"),
    ):
        for i, m in enumerate(METHOD_LABEL):
            vals = [100 * g[m][key] for g in grid]
            ax.bar(
                [j + (i - 1.5) * width for j in range(len(ps))],
                vals,
                width=width - 0.04,
                color=METHOD_COLOUR[m],
                label=METHOD_LABEL[m],
            )
        ax.set_xticks(range(len(ps)), [f"{p:.0%} true error" for p in ps])
        ax.tick_params(axis="x", length=0)
        ax.set_title(title)
    axes[0].axhline(95, color=cs.INK2, linewidth=0.9, linestyle="--")
    axes[0].set_ylim(0, 100)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower left", bbox_to_anchor=(0.01, -0.08), ncol=4, handlelength=1)
    fig.tight_layout(w_pad=3)
    out = RESULTS / "coverage.png"
    fig.savefig(out)
    plt.close(fig)
    return out.name


def calibration_chart(res: dict) -> str:
    cs.apply()
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.grid(axis="x")
    ax.plot([0, 1], [0, 1], color=cs.NEUTRAL, linewidth=1, linestyle="--", label="Perfect calibration")
    for k, c in zip(res["judges"], cs.SERIES, strict=False):
        cal = res["judges"][k]["calibration"]
        ax.plot(
            [b["mean_pred_wrong"] for b in cal],
            [b["observed_wrong"] for b in cal],
            color=c,
            marker="o",
            label=f"{NICE[k]} (ECE {res['judges'][k]['ece']:.2f})",
            **LINE,
        )
    ax.set_xlabel("Judge's probability the booking is wrong")
    ax.set_ylabel("Share actually wrong (staff booking)")
    ax.set_title("Judge calibration")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left")
    fig.tight_layout()
    out = RESULTS / "calibration.png"
    fig.savefig(out)
    plt.close(fig)
    return out.name


def build_all() -> list[str]:
    res, cov = load()
    return [headline_chart(res, cov), coverage_chart(cov), calibration_chart(res)]
