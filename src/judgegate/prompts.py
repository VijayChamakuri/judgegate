"""Prompts for the agent (books a GL account) and the judge (decides if a booking is right).

Both see the same evidence: the transaction and its most similar past bookings. The judge also
sees the agent's proposed account. Neither ever sees the human label for the line being scored.

Layout matters for speed: everything that is identical across calls (role, account menu, answer
format) comes first, so the local model processes it once and reuses it (prefix KV cache); only
the per-transaction evidence is new on each call.
"""

from __future__ import annotations

N_PRECEDENTS = 5

AGENT_SYSTEM = (
    "You are a finance assistant at a UK local council. You assign each corporate-card "
    "transaction to exactly one general-ledger (GL) account, using similar past bookings as precedent."
)

JUDGE_SYSTEM = (
    "You are a finance reviewer at a UK local council. An assistant has proposed a general-ledger "
    "(GL) account for a corporate-card transaction. Decide whether it is the account the council's "
    "finance staff would book, using similar past bookings as precedent."
)


def describe(line: dict, setting: str) -> str:
    parts = [f'merchant "{line["merchant_raw"]}"', f'dept "{line["department"]}"', f'service "{line["service_category"]}"']
    if setting == "upstream":
        parts.append(f'category "{line["merchant_category"]}"')
    parts.append(f"£{line['amount']:.2f}")
    return ", ".join(parts)


def agent_prefix(accounts: list[str], letters: list[str]) -> str:
    menu = "\n".join(f"{ltr}) {acc}" for ltr, acc in zip(letters, accounts, strict=True))
    return f"Accounts:\n{menu}\n\nReply with the letter of one account only.\n\n"


def agent_suffix(line: dict, neighbours: list[dict], setting: str) -> str:
    rows = "\n".join(f"- {describe(n, setting)} -> {n['gl_account']}" for n in neighbours[:N_PRECEDENTS])
    return f"Similar past bookings:\n{rows}\n\nTransaction to book: {describe(line, setting)}"


def judge_prefix() -> str:
    return "Reply A if the proposed account is correct, or B if it is wrong.\n\n"


def judge_suffix(line: dict, neighbours: list[dict], proposed: str, setting: str) -> str:
    rows = "\n".join(f"- {describe(n, setting)} -> {n['gl_account']}" for n in neighbours[:N_PRECEDENTS])
    return (
        f"Similar past bookings:\n{rows}\n\nTransaction: {describe(line, setting)}\n"
        f"Proposed account: {proposed}\n\nA) correct\nB) wrong"
    )
