"""PNG charts for the README, drawn only from results/*.json."""

from __future__ import annotations

import json
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .data import RESULTS  # noqa: E402

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
POLICY_STYLE = {  # fixed categorical order; colour follows the policy everywhere
    "human_wilson": ("#2a78d6", "o", "Human audits only (Wilson)"),
    "ppi_plain": ("#eb6834", "s", "Judge + random audits (plain PPI)"),
    "active_ppi": ("#1baf7a", "D", "Judge + judge-guided audits (active PPI)"),
    "judge_raw": ("#e34948", "X", "Raw judge, no humans"),
}
METHOD_LABEL = {
    "wilson_random_audits": "Wilson, random audits",
    "ppi_plain_random_audits": "Plain PPI, random audits",
    "active_ppi_judge_guided_audits": "Active PPI, judge-guided",
    "active_ppi_tuned_same_sample": "Active PPI, tuned on same sample",
}
METHOD_COLOUR = dict(zip(METHOD_LABEL, ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"], strict=True))


def _style(ax: Any) -> None:
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def load() -> tuple[dict, dict]:
    return (json.loads((RESULTS / "judgegate.json").read_text()), json.loads((RESULTS / "coverage.json").read_text()))


def headline_chart(res: dict, cov: dict) -> str:
    """Left: autonomy each policy appears to earn (upstream, 2% budget, mean of audit draws).
    Right: how often each policy's "95%" bound really covers the truth on real rare-error slices."""
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), facecolor=SURFACE)
    ax = axes[0]
    _style(ax)
    rows = [r for r in res["rollouts"] if r["setting"] == "upstream" and r["judge"] == "llama" and r["budget"] == 0.02]
    for pol, (colour, marker, label) in POLICY_STYLE.items():
        pts = sorted((r["audits_per_month"], 100 * r["automation_rate"]) for r in rows if r["policy"] == pol)
        if pts:
            xs, ys = zip(*pts, strict=True)
            ax.plot(
                xs,
                ys,
                color=colour,
                linewidth=2,
                marker=marker,
                markersize=8,
                label=label,
                markeredgecolor=SURFACE,
                markeredgewidth=1.5,
                linestyle="--" if pol == "ppi_plain" else "-",
            )
    ax.set_xticks([0, 10, 20, 40])
    ax.set_xlabel("Human audits per month", color=INK, fontsize=10)
    ax.set_ylabel("Live lines auto-posted (%)", color=INK, fontsize=10)
    ax.set_title("Autonomy earned (upstream, 2% budget, Llama judge)", color=INK, fontsize=10.5, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper left")

    ax = axes[1]
    _style(ax)
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
        colour, marker, label = POLICY_STYLE[pol]
        ax.plot(
            pos,
            [100 * g[m]["coverage"] for g in sl],
            color=colour,
            linewidth=2,
            marker=marker,
            markersize=8,
            markeredgecolor=SURFACE,
            markeredgewidth=1.5,
            linestyle="--" if pol == "ppi_plain" else "-",
        )
    ax.axhline(95, color=INK, linewidth=1, linestyle=":")
    ax.annotate("promised 95%", (pos[0], 95), xytext=(4, -12), textcoords="offset points", ha="left", fontsize=8, color=INK)
    ax.set_xticks(pos, [f"{100 * g['true_error']:.2f}%\n{SETTING[g['setting']]}" for g in sl])
    ax.set_ylim(50, 101)
    ax.set_xlabel("True error rate of the real slice", color=INK, fontsize=10)
    ax.set_ylabel("Coverage of the 95% bound (%)", color=INK, fontsize=10)
    ax.set_title("Is the bound honest? (400 audits, 1,000 redraws)", color=INK, fontsize=10.5, loc="left")
    fig.suptitle(
        "Plain PPI (dashed) looks best on the left because its bound is too optimistic on the right.",
        color=MUTED,
        fontsize=9,
        x=0.01,
        ha="left",
    )
    fig.tight_layout()
    out = RESULTS / "headline.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out.name


SETTING = {"upstream": "upstream", "card_feed": "card feed"}
NICE = {
    "llama/card_feed": "Llama-3.2-3B, card feed",
    "llama/upstream": "Llama-3.2-3B, upstream",
    "qwen/upstream": "Qwen2.5-3B, upstream",
}


def coverage_chart(cov: dict, sep: float = 0.6, audits: int = 40) -> str:
    grid = [g for g in cov["grid"] if g["judge_strength"] == sep and g["audits"] == audits]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), facecolor=SURFACE)
    ps = [g["true_error"] for g in grid]
    width = 0.2
    for ax, key, title in (
        (axes[0], "coverage", "Coverage (target 95%)"),
        (axes[1], "mean_upper", "Mean upper bound (lower = tighter)"),
    ):
        _style(ax)
        for i, m in enumerate(METHOD_LABEL):
            vals = [100 * g[m][key] for g in grid]
            ax.bar(
                [j + (i - 1.5) * width for j in range(len(ps))],
                vals,
                width=width - 0.03,
                color=METHOD_COLOUR[m],
                label=METHOD_LABEL[m],
            )
        ax.set_xticks(range(len(ps)), [f"{p:.0%} true error" for p in ps])
        ax.set_title(title, color=INK, fontsize=10, loc="left")
        ax.set_ylabel("%", color=MUTED)
    axes[0].axhline(95, color=INK, linewidth=1, linestyle="--")
    axes[0].set_ylim(0, 100)
    axes[1].legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.suptitle(
        f"Simulation: {cov['reps']} repeats, population {cov['population']}, {audits} expected audits, judge strength {sep}",
        color=MUTED,
        fontsize=9,
        x=0.01,
        ha="left",
    )
    fig.tight_layout()
    out = RESULTS / "coverage.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out.name


def calibration_chart(res: dict) -> str:
    keys = list(res["judges"])
    colours = ["#2a78d6", "#eb6834", "#1baf7a"]
    fig, ax = plt.subplots(figsize=(5.2, 4.6), facecolor=SURFACE)
    _style(ax)
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle="--", label="Perfect calibration")
    for k, c in zip(keys, colours, strict=False):
        cal = res["judges"][k]["calibration"]
        ax.plot(
            [b["mean_pred_wrong"] for b in cal],
            [b["observed_wrong"] for b in cal],
            color=c,
            linewidth=2,
            marker="o",
            markersize=7,
            markeredgecolor=SURFACE,
            label=f"{NICE[k]} (ECE {res['judges'][k]['ece']:.2f})",
        )
    ax.set_xlabel("Judge's probability the booking is wrong", color=INK, fontsize=10)
    ax.set_ylabel("Share actually wrong (staff truth)", color=INK, fontsize=10)
    ax.set_title("Judge calibration", color=INK, fontsize=11, loc="left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout()
    out = RESULTS / "calibration.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out.name


def build_all() -> list[str]:
    res, cov = load()
    return [headline_chart(res, cov), coverage_chart(cov), calibration_chart(res)]
