# JudgeGate

**Can an LLM judge replace the human audits that decide when an AI agent may act on its own?**
Tested on {{lines}} real purchase-card bookings that council finance staff coded by hand, with free local models (Llama-3.2-3B and Qwen2.5-3B on Apple Silicon, no API).

[![ci](https://github.com/VijayChamakuri/judgegate/actions/workflows/ci.yml/badge.svg)](https://github.com/VijayChamakuri/judgegate/actions/workflows/ci.yml)
[![site](https://img.shields.io/badge/site-live-1baf7a)](https://vijaychamakuri.github.io/judgegate/)
[![licence: MIT](https://img.shields.io/badge/licence-MIT-blue)](LICENSE)

**[Live dashboard and judge-mistake browser →](https://vijaychamakuri.github.io/judgegate/)** · Companion to [TrustGate](https://github.com/VijayChamakuri/trustgate), which gates an agent's autonomy with human audits.

## In plain English

{{plain_english}}

## Key results

{{key_results}}

![Left: autonomy each policy appears to earn (upstream, 2% budget). Right: months over budget on card-feed data (10% budget)](results/headline.png)

## Terms used below

| Term | Meaning |
|---|---|
| Agent | The model that proposes a GL account for each card line (TrustGate's coder, refit every 3 months on earlier months only) |
| Judge | A local LLM that sees the line, the agent's proposal and similar earlier bookings, and gives the probability the proposal is correct. It never sees the staff answer |
| Gate | Each month, auto-posts the agent's lines above a confidence threshold, chosen so that a 95% upper bound on their error rate, from earlier months only, fits the budget |
| Error budget | The highest error rate allowed among auto-posted lines (2%, 5% or 10%) |
| Audit | A human checking one line. Here the staff booking plays the human |
| False pass / false fail | The judge approving a wrong booking / rejecting a correct one |
| AUC | How well the judge's scores rank wrong bookings above right ones: 0.5 = chance, 1 = perfect |
| Calibration (ECE) | Whether "70% likely correct" means right 70% of the time. ECE is the average gap |
| PPI | Prediction-powered inference: estimate from the judge's scores on every line, then correct the judge's bias with the human audits |
| Active PPI | PPI where audits are drawn more often where the judge sees risk, and each audit is up-weighted by 1/(chance it was drawn) so the estimate stays unbiased |
| Coverage | How often a "95% upper bound" really sits above the true error rate. Below 95% means the bound is over-optimistic |
| Months over budget | Live months in which the auto-posted lines' real error rate exceeded the budget |

## 1 · How good are the judges?

{{judge_table}}

{{judge_comparison}}

![Judge calibration](results/calibration.png)

## 2 · Does the judge earn autonomy safely?

{{live_lines}} live lines over {{live_months}} months. Upstream setting, {{headline_judge}} judge, {{budget}} error budget. Every policy with humans spends the same expected number of audits, and every row is a mean over {{seeds}} independent audit draws.

{{rollout_table}}

Across budgets and judges, with {{audits}} audits a month (cells: share auto-posted / mean months over budget):

{{budget_table}}

## 3 · Are the error bounds honest?

**Simulation** ({{sim_reps}} repeats per cell, population 3,000, 40 expected audits; cells: coverage / mean bound):

{{sim_table}}

![Coverage simulation](results/coverage.png)

**Real data**: live-month Stockport lines, kept from the agent's most confident down, so the true error rate (known for every line) ranges from common to rare. Judge scores are recalibrated once on the shadow months, and only the audits are redrawn, 1,000 times. Shown: slices with a true error between 0% and 10%, the regime a gate decides in. Every slice is on the [site](https://vijaychamakuri.github.io/judgegate/) and in `results/coverage.json`.

{{coverage_table}}

## 4 · What the judge gets wrong

{{failure_analysis}}

Wrong bookings the {{headline_judge}} judge was most sure were correct (upstream):

{{false_pass_table}}

Correct bookings it was most sure were wrong:

{{false_fail_table}}

Browse every judge's mistakes on the [site](https://vijaychamakuri.github.io/judgegate/). Some "false passes" may be staff labelling inconsistencies rather than judge errors. See the data card.

## How it works

```
Stockport card lines + staff-booked accounts (TrustGate's dbt warehouse)
   ├─ agent: expanding-window coder → proposed account + confidence per line
   ├─ judge: local LLM, one forward pass → P(proposal correct) per line
   └─ gate: monthly threshold from a 95% upper bound on error, earlier months only
         evidence = human audits | raw judge | judge + random audits (PPI) | judge + judge-guided audits (active PPI)
```

Details are in [docs/METHODOLOGY.md](docs/METHODOLOGY.md), the reasoning behind each choice in [docs/DESIGN_DECISIONS.md](docs/DESIGN_DECISIONS.md), and the data in [docs/DATA_CARD.md](docs/DATA_CARD.md).

## Reproduce

```bash
uv sync --all-groups
uv run judgegate reproduce   # every number, chart, the site and this README, from committed model outputs (any OS, minutes)
uv run judgegate generate    # optional: re-run the models (Apple Silicon + MLX, about 4 hours on an 8 GB M2, resumable)
make test                    # ruff, mypy, pytest (includes "README matches results")
```

CI recomputes `results/*.json` from the committed judge verdicts and fails if they differ, or if this README is out of date.

## Repository layout

```
src/judgegate/   agent.py · llm.py (MLX letter scoring) · prompts.py · retrieval.py · generate.py
                 stats.py (Wilson, PPI++, active PPI, calibration) · gate.py (monthly rollout) · simulate.py
                 analyze.py · charts.py · site.py · readme.py · cli.py
data/            stockport_labelled_lines.csv.gz · generations/ (agent proposals, judge verdicts)
results/         judgegate.json · coverage.json · charts
docs/            METHODOLOGY · DESIGN_DECISIONS · DATA_CARD · README.template.md
tests/           bounds and coverage, no-leak guards for retrieval, audits and calibration, README drift
```

## Limitations

{{limitations}}

## Citation and licence

Code: MIT ([LICENSE](LICENSE)); please cite with [CITATION.cff](CITATION.cff). Data: Stockport Metropolitan Borough Council purchase-card transactions. *Contains public sector information licensed under the Open Government Licence v3.0.*

Methods: Angelopoulos, Bates, Fannjiang, Jordan, Zrnic, "Prediction-powered inference", *Science* 2023 · Angelopoulos, Duchi, Zrnic, "PPI++", arXiv:2311.01453 · Zrnic & Candès, "Active statistical inference", ICML 2024.
