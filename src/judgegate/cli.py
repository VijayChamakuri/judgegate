"""`judgegate` command line.

judgegate generate     agent proposals + local LLM judge verdicts (Apple Silicon, MLX; hours, resumable)
judgegate analyze      every statistic, from the committed cache (any OS, minutes)
judgegate charts       results/*.png
judgegate site         site/ (GitHub Pages)
judgegate readme       README.md from docs/README.template.md + results (CI checks it is current)
judgegate reproduce    analyze + charts + site + readme
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from typing import Any


def _step(name: str, fn: Callable[..., Any], *a: Any) -> Any:
    t = time.time()
    out = fn(*a)
    print(f"  {name:<40} {time.time() - t:7.1f}s")
    return out


def cmd_generate(_: argparse.Namespace) -> None:
    from .generate import SETTINGS, run_agent, run_judge

    for s in SETTINGS:
        _step(f"agent proposals ({s})", run_agent, s)
    for s, j in (("upstream", "llama"), ("card_feed", "llama"), ("upstream", "qwen")):
        _step(f"judge {j} ({s})", run_judge, s, j)


def cmd_analyze(_: argparse.Namespace) -> None:
    from .analyze import run

    _step("analyze (judges, rollouts, coverage)", run)


def cmd_charts(_: argparse.Namespace) -> None:
    from .charts import build_all

    _step("charts", build_all)


def cmd_site(_: argparse.Namespace) -> None:
    from .site import build

    _step("site", build)


def cmd_readme(_: argparse.Namespace) -> None:
    from .readme import render

    _step("README.md", render, True)


def cmd_reproduce(a: argparse.Namespace) -> None:
    for fn in (cmd_analyze, cmd_charts, cmd_site, cmd_readme):
        fn(a)


def main() -> None:
    p = argparse.ArgumentParser(prog="judgegate", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in (
        ("generate", cmd_generate),
        ("analyze", cmd_analyze),
        ("charts", cmd_charts),
        ("site", cmd_site),
        ("readme", cmd_readme),
        ("reproduce", cmd_reproduce),
    ):
        sub.add_parser(name).set_defaults(fn=fn)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
