import numpy as np
import pandas as pd

from judgegate.retrieval import past_neighbours


class FakeEmbedder:
    def encode(self, texts):
        rng = np.random.default_rng(len(texts))
        v = rng.normal(size=(len(texts), 8))
        return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_neighbours_come_only_from_earlier_months():
    months = pd.date_range("2020-01-01", periods=4, freq="MS")
    df = pd.DataFrame(
        {
            "card_line_id": [f"l{i}" for i in range(40)],
            "statement_month": np.repeat(months, 10),
            "merchant_raw": [f"M{i % 7}" for i in range(40)],
            "department": "d",
            "service_category": "s",
        }
    )
    targets = df[df.statement_month >= months[1]]
    neigh = past_neighbours(df, targets, k=5, emb=FakeEmbedder())
    month_of = df.statement_month.to_numpy()
    for cid, idxs in neigh.items():
        m = df.loc[df.card_line_id == cid, "statement_month"].iloc[0]
        assert all(month_of[i] < m for i in idxs)
    assert all(c not in neigh for c in df[df.statement_month == months[0]].card_line_id)  # nothing before month 1
