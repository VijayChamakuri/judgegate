# Data card

## Stockport MBC purchase-card lines with staff-booked accounts (real)

| | |
|---|---|
| Publisher | Stockport Metropolitan Borough Council (UK local authority), via data.gov.uk `purchase-card-transactions` |
| Licence | Open Government Licence v3.0. *Contains public sector information licensed under the Open Government Licence v3.0.* |
| File | `data/stockport_labelled_lines.csv.gz` (14,851 lines, 2017-01 to 2022-06, 21 GL accounts) |
| Built by | [TrustGate](https://github.com/VijayChamakuri/trustgate)'s dbt warehouse, model `int_stockport__labelled_lines`: raw monthly files are downloaded with checksums, cleaned, and each line's free-text "purpose of expenditure" is mapped to one of 21 canonical accounts by a reviewed seed |
| Label | `gl_account`: the account council finance staff booked. JudgeGate treats it as the human ground truth, for the agent **and** for the judges |
| Columns | `card_line_id, statement_month, purchase_date, merchant_raw, merchant_key, merchant_category, department, service_category, amount, gl_account` |

**Window used here.** Agent proposals and judge verdicts cover every line from 2020-01 on (the expanding-window agent trains on everything earlier). The gate's first six months are a shadow period, and the remaining months are "live".

**Two settings (as in TrustGate).**
- **Card feed:** what a card network sends (merchant string, department, service, amount). This is hard, because the same merchant can be booked to several accounts.
- **Upstream:** adds the council's procurement category, which in these years was assigned by staff and nearly fixes the account. This stands in for an agent that has an approved purchase order or a requisition.

**Known limitations.**
- Staff labels contain their own errors and inconsistencies, so a "false pass" may sometimes be a judge that is right and a label that is wrong. The browser on the site shows real examples to judge by eye.
- Only one organisation is covered, along with its coding culture.
- The procurement category in the upstream setting partly leaks the label, by design (see above).

## Model outputs (committed)

| File | What |
|---|---|
| `data/generations/agent_{setting}.csv` | Agent's proposed account and confidence per line |
| `data/generations/judge_{judge}_{setting}.jsonl` | Judge's probability that the proposal is correct per line |

These are produced by `judgegate generate` (Apple Silicon, MLX, several hours) and committed, so `judgegate analyze` reproduces every number on any OS in minutes. CI checks that it does.

## Models

| Role | Model | Licence |
|---|---|---|
| Judge (headline) | `mlx-community/Llama-3.2-3B-Instruct-4bit` | Llama 3.2 Community Licence |
| Judge (second family, upstream only) | `mlx-community/Qwen2.5-3B-Instruct-4bit` | Qwen Research Licence |
| Retrieval | `BAAI/bge-small-en-v1.5` via fastembed | MIT |
