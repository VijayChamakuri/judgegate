# Changelog

## 1.0.0 (2026-11-10)

- Agent proposals (expanding-window coder) and local LLM-judge verdicts (Llama-3.2-3B, Qwen2.5-3B via MLX) for every live Stockport line, committed under `data/generations/`.
- Judge evaluation against staff bookings: AUC, Cohen's κ, false-pass and false-fail rates, calibration, McNemar.
- Monthly autonomy rollouts comparing human-only, raw-judge, plain PPI and active (judge-guided) PPI evidence.
- Coverage checks of every bound, in simulation and on the real data.
