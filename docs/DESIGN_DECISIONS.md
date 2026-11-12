# Design decisions

**Staff bookings as human labels, rather than hand-labelling a sample.** Council finance staff booked every line to an account. That gives 14,851 real human labels at no labelling cost, and the judge can be scored on every line rather than on a few hundred.

**The agent is a classical model; the LLMs are judges.** The first version made a local 3B LLM the agent (choose one of 21 lettered accounts, with retrieved precedents). In a 200-line pilot it was right only about a third of the time, well below TrustGate's coder on the same lines. Gating a weak agent says little. So the agent is TrustGate's coder, refit every three months on earlier months only, and the LLMs do what companies actually ask of them here: review another system's output.

**Local models only.** These are Llama-3.2-3B and Qwen2.5-3B in 4-bit, run on MLX on an 8 GB M2. There are no API keys and no cost, and anyone with a Mac can re-run them. A frontier judge would probably be stronger. The point of the project is the method for deciding *how much* to trust whatever judge you have, and that method does not depend on judge size.

**One forward pass per verdict.** The judge's score is read from the probabilities of the answer letters, not parsed from generated text. That gives a continuous score on every line (needed for AUC, calibration and PPI), with no parsing failures and no sampling noise.

**Memory on an 8 GB machine.** MLX keeps freed GPU buffers for reuse. With a different prompt length on every call, that cache grew until the machine swapped (5 GB+ resident, calls taking a minute). Capping the cache at 256 MB and applying the output head to the last position only (not all ~400 positions × 128k vocabulary) brought the peak to 2.4 GB at about 0.9 s per line, with the same scores (within 0.002 of an uncached full pass).

**Recalibrate the judge on earlier months, and decide that before seeing real results.** A dry run with synthetic judge scores (before any real verdict existed) showed raw scores make a poor control variate: a judge that says "20% likely wrong" about correct bookings adds noise to every audit, and a judge that is too strict made active PPI earn nothing. Platt scaling fitted month by month on earlier audits fixed most of this. It was adopted then. The same dry run compared risk-based and uncertainty-based audit allocation; risk-based was as good or better in 10 of 12 cells, so the planned rule was kept.

**Claims follow the data, not the plan.** The plan expected a *lenient* judge, and active PPI as the method that safely cuts human labels. The real judge is *strict*, plain PPI's apparent gains come from an over-optimistic bound, and active PPI is valid but earns less than human audits in the gate. The README reports that, and the harness that caught it is the contribution.

**Why test judge-guided audits at all.** Plain PPI with random audits is the textbook method, but its normal-approximation bound under-covers when errors are rare, which is exactly the regime of confident bookings. Active PPI keeps the same expected audit budget, sends audits where the judge sees risk, and stays unbiased through inverse-probability weights. We considered a "plus-four" correction to plain PPI. It restored coverage but was so loose it earned less than human audits alone, so it was dropped.

**No power tuning of λ on the evaluation sample.** Picking λ to minimise the variance of the same terms you then report is a form of selection, and on small error-free samples it makes the bound optimistic. The gate uses λ = 1. The tuned variant is kept in the coverage tables as a warning.

**Same gate as TrustGate, except for the evidence.** The threshold grid, minimum evidence of 20 audits, six shadow months, monthly cadence and the "earlier months only" rule are all unchanged. Any difference in outcome comes from the evidence policy.

**Determinism.** Audit draws, bootstraps and simulations are seeded. CI recomputes `results/*.json` from the committed model outputs and fails if they differ, and fails if `README.md` or `docs/RESULTS.md` does not match a fresh render.
