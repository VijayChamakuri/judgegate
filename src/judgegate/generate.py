"""Produce the cached inputs every statistic is computed from.

1. Agent proposals (TrustGate's coder, expanding-window refits) for every line from 2020-01 on.
2. LLM-judge verdicts on each proposal, from local models via MLX.

Everything is written under data/generations/ and committed, so `judgegate analyze` reproduces
all results on any machine without running a model. Judge runs are resumable: lines already in a
cache file are skipped.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from .agent import propose
from .data import DATA, load_lines
from .prompts import JUDGE_SYSTEM, judge_prefix, judge_suffix
from .retrieval import past_neighbours

GEN = DATA / "generations"
SETTINGS = ("card_feed", "upstream")


def agent_file(setting: str) -> Path:
    return GEN / f"agent_{setting}.csv"


def judge_file(setting: str, judge: str) -> Path:
    return GEN / f"judge_{judge}_{setting}.jsonl"


def run_agent(setting: str) -> Path:
    GEN.mkdir(parents=True, exist_ok=True)
    props = propose(load_lines(), setting)
    props["conf"] = props.conf.round(5)
    props.to_csv(agent_file(setting), index=False)
    return agent_file(setting)


def load_judge(setting: str, judge: str) -> pd.DataFrame:
    path = judge_file(setting, judge)
    rows = [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
    return pd.DataFrame(rows, columns=["card_line_id", "p_correct"])


def run_judge(setting: str, judge: str, log: Callable[[str], None] = print) -> Path:
    from .llm import JUDGE_MODELS, LetterScorer  # MLX: Apple Silicon only, so imported lazily

    df = load_lines()
    by_id = df.set_index("card_line_id")
    props = pd.read_csv(agent_file(setting))
    done = set(load_judge(setting, judge).card_line_id)
    todo = props[~props.card_line_id.isin(done)]
    neigh = past_neighbours(df, df[df.card_line_id.isin(todo.card_line_id)])
    scorer = LetterScorer(JUDGE_MODELS[judge])
    path = judge_file(setting, judge)
    t0, n = time.time(), 0
    with path.open("a") as f:
        for r in todo.itertuples():
            line = by_id.loc[r.card_line_id].to_dict()
            nb = df.iloc[neigh[r.card_line_id]].to_dict("records")
            p = scorer.letter_probs(JUDGE_SYSTEM, judge_prefix(), judge_suffix(line, nb, r.pred, setting), ["A", "B"])
            f.write(json.dumps({"card_line_id": r.card_line_id, "p_correct": round(float(p[0]), 5)}) + "\n")
            n += 1
            if n % 100 == 0:
                f.flush()
                log(f"[judge {judge} {setting}] {n}/{len(todo)} at {(time.time() - t0) / n:.2f}s/line")
    log(f"[judge {judge} {setting}] done: {n} new verdicts in {time.time() - t0:.0f}s")
    return path
