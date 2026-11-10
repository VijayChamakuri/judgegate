"""Real human labels: Stockport MBC purchase-card lines with the GL account staff booked.

`data/stockport_labelled_lines.csv.gz` is exported from TrustGate's governed dbt model
`int_stockport__labelled_lines` (github.com/VijayChamakuri/trustgate): 14,851 lines, 21 canonical
accounts, Jan 2017 to Jun 2022. Contains public sector information licensed under the Open
Government Licence v3.0 (Stockport Metropolitan Borough Council).
"""

from __future__ import annotations

import string
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
LINES_CSV = DATA / "stockport_labelled_lines.csv.gz"

WINDOW_START = pd.Timestamp("2020-01-01")  # first month the agent runs (6 shadow months, then live)
SHADOW_MONTHS = 6


def load_lines() -> pd.DataFrame:
    df = pd.read_csv(LINES_CSV, parse_dates=["statement_month", "purchase_date"])
    for c in ("department", "service_category", "merchant_category"):
        df[c] = df[c].fillna("unknown")
    df["amount"] = df.amount.astype(float)
    return df.sort_values(["purchase_date", "card_line_id"]).reset_index(drop=True)


def accounts(df: pd.DataFrame) -> list[str]:
    return sorted(df.gl_account.unique())


def letters(n: int) -> list[str]:
    if n > 26:
        raise ValueError("at most 26 options")
    return list(string.ascii_uppercase[:n])


def window(df: pd.DataFrame) -> pd.DataFrame:
    return df[df.statement_month >= WINDOW_START].copy()
