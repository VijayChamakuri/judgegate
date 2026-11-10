# Methodology

## The question

An AI agent books card transactions to general-ledger accounts. A gate decides each month which bookings it may post without a person checking them. The gate needs evidence that, above some confidence threshold, the agent's error rate is under a budget (2%, 5% or 10%). That evidence costs human audits.

An LLM judge is cheap: it can check every booking. **Can its verdicts replace the human audits, wholly or partly, without the gate granting autonomy it has not earned?**

## Pipeline

```
Stockport lines (staff-booked accounts)
   │
   ├─ agent: expanding-window coder, refit every 3 months on earlier months only
   │      → proposed account + confidence for every line from 2020-01
   │
   ├─ judge: local LLM sees the line, the proposal and 5 similar *earlier* bookings
   │      → P(proposal correct), read from the probabilities of the answer letters
   │
   └─ gate: each live month, pick the lowest confidence threshold whose 95% upper bound
          on error (computed from earlier months only) fits the budget; auto-post above it
          → score every auto-posted line against the staff booking
```

The agent is the TrustGate coder (char TF-IDF + one-hot fields + embeddings + logistic regression). A 3B local LLM was tried as the agent first and was much weaker (see DESIGN_DECISIONS). The judges are where the LLMs earn their keep.

**No leakage.** The judge's retrieved precedents come only from months before the line's month (`tests/test_retrieval.py`). The gate's threshold for month *m* uses only lines from before *m* (`tests/test_gate.py`). The judge never sees the staff label.

**Judge score.** One forward pass. The prompt ends with "A) correct / B) wrong … Answer:", and the judge's score is softmax(logit_A, logit_B)[A]. The shared part of every prompt is kept in a key/value cache, so each call only processes the new evidence. This was checked against an uncached pass: they agree within 0.002.

## Four kinds of evidence for the bound

Write *Y* = 1 if the agent's booking is wrong (staff truth) and *f* = the judge's probability that it is wrong.

| Policy | Evidence | Bound |
|---|---|---|
| **Human audits only** | *k* random audits a month | Wilson score upper bound on audited errors (TrustGate's rule) |
| **Raw judge** | judge verdict on every line, treated as truth; no humans | Wilson bound on judge-flagged lines |
| **Plain PPI** | judge on every line + *k* random audits | Prediction-powered inference (Angelopoulos et al., *Science* 2023) with PPI++ power tuning (arXiv:2311.01453); normal-approximation bound |
| **Active PPI** | judge on every line + *k* **judge-guided** audits | Active statistical inference (Zrnic & Candès, ICML 2024) |

**Recalibration (both PPI policies).** A 3B judge's raw probabilities are far off; it calls most correct bookings wrong. Before month *m*, the judge's scores are recalibrated with Platt scaling (logistic regression of the truth on logit(score)), fitted only on the human audits from months before *m*. The map is fixed before month *m*'s audits are drawn, so the estimators stay unbiased; `tests/test_gate.py` checks that changing month *m*'s truth leaves its calibrated scores unchanged. Audit selection depends only on the score, so the audited lines give an unbiased view of P(wrong | score) without reweighting.

### Prediction-powered inference, in one paragraph
Average the judge's scores over all lines (cheap, but biased if the judge is off), then correct that average using the audited lines: on those, compare what the judge said with the truth and add the mean gap back. The correction removes the judge's bias, so the estimate is right on average **whatever the judge's quality**. When the judge tracks the truth, the gap has little variance, so the confidence interval is narrower than with audits alone.

### Why plain PPI is not enough here
Errors among confident bookings are rare. A sample of 40 random audits often contains zero errors. The PPI correction term then has zero sample variance and the normal-approximation bound collapses to roughly the judge's average, with almost no width. In the simulation and on real data, that bound sits below the truth far more often than 5% of the time ("under-covers"). It would unlock autonomy that was never earned.

### Active PPI: send the auditors where the judge sees risk
Each line *i* is audited with probability π*ᵢ* ∝ *fᵢ* + 0.2·mean(*f*) (the floor keeps every line auditable), scaled so the expected number of audits per month is still *k*. With *Sᵢ* = 1 if audited, the estimate is

  θ̂ = mean*ᵢ* [ *fᵢ* + (*Yᵢ* − *fᵢ*)·*Sᵢ* / π*ᵢ* ]

Its expectation is the true error rate for any judge, because each audit is up-weighted by 1/π. The upper bound is θ̂ + 1.96 × standard error. Audits land on the lines the judge flags, which is where errors are, so the audited sample is rarely error-free by accident. For estimating the error rate of a whole population this lowers the variance.

**Why it still loses inside the gate.** The gate's question is not "what is the overall error rate?" but "what is the error rate among the agent's *confident* lines?". Those are the lines the judge is least worried about, so judge-guided sampling sends them the fewest audits. Each of those audits then carries a large weight, and the bound on exactly the lines being decided is wider than with random audits. On this data that cost outweighs the control-variate gain. Allocating by the judge's uncertainty, √(f(1−f)), instead of its risk did not help on synthetic tests either. We use the control-variate weight λ = 1 (fully trusting *f* as the baseline). Choosing λ to minimise the variance of the same sample makes the bound optimistic, so that variant is reported only as a cautionary row.

### How each bound is validated
1. **Simulation** (`results/coverage.json` → `grid`): synthetic populations of 3,000 lines with true error 1%, 3% or 10%, judges of three strengths, and 40 or 120 expected audits. Each method is repeated 500 times, recording coverage (how often bound ≥ truth; target 95%) and mean bound (tightness).
2. **Real data** (`real_data`): the truth is known for every Stockport line. Populations are the live-month lines with agent confidence at or above the 0/10/20/50/90th percentile, so the true error rate ranges from common to zero. Judge scores are recalibrated once, on the six shadow months, so the calibration does not depend on the population being estimated. Only the audits are redrawn, 1,000 times, at 40 and 400 audits (400 is roughly what the gate has accumulated by mid-rollout).
3. **Rollouts**: realised error on the lines each policy actually auto-posted, months over budget, and pounds mis-posted.

## Judge quality metrics
- **AUC:** does the judge rank wrong bookings above right ones? It has a 95% bootstrap CI (500 resamples).
- **Cohen's κ at 0.5:** agreement with staff beyond chance when the judge's verdict is read as pass/fail.
- **False-pass rate:** the share of *wrong* bookings the judge passed. This is what lets errors through. **False-fail rate:** the share of *right* bookings it failed. This is what wastes reviewer time. Both have Wilson CIs.
- **ECE:** expected calibration error over 10 bins. A judge can rank well (high AUC) and still be badly calibrated.
- **McNemar (exact):** do two judges agree with staff equally often on the same lines?
- **Incremental value:** does the judge add anything beyond the agent's own confidence? This compares 5-fold cross-validated AUC for predicting the agent's errors from confidence alone against confidence + judge score, with a likelihood-ratio test of the judge's coefficient.
- **Triage:** if a reviewer checks lines in the judge's risk order, what share of errors do they find after 10/20/30/50% of lines? The same is computed for ordering by the agent's own (inverted) confidence, which is the baseline the judge must beat.

## Reproducing
```bash
uv sync --all-groups
uv run judgegate reproduce        # every number, chart, the site and README from committed outputs
uv run judgegate generate         # optional: re-run the models (Apple Silicon, ~4 h on an 8 GB M2)
```
