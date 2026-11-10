"""The agent being gated: TrustGate's GL coder (char TF-IDF + local embeddings + logistic regression).

Adapted from TrustGate (github.com/VijayChamakuri/trustgate, `realgl.LinearCoder`). It is refit
every quarter on all months strictly before the month being coded (expanding window), exactly as
in TrustGate's real-data rollout, so its proposals and confidences are what a deployed coder would
have produced at the time.

Settings: `card_feed` uses merchant string, department, service and amount; `upstream` also uses
the council's procurement category (assigned by staff, which makes the task much easier).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

from .data import WINDOW_START
from .retrieval import Embedder

CATS = {"card_feed": ["department", "service_category"], "upstream": ["department", "service_category", "merchant_category"]}


class LinearCoder:
    def __init__(self, setting: str, emb: Embedder):
        self.cats, self.emb = CATS[setting], emb

    def _X(self, df: pd.DataFrame, fit: bool):
        if fit:
            self.ohe = OneHotEncoder(handle_unknown="ignore", min_frequency=2).fit(df[self.cats])
            self.tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True).fit(df.merchant_raw)
        return sparse.hstack(
            [
                self.ohe.transform(df[self.cats]),
                self.tfidf.transform(df.merchant_raw),
                sparse.csr_matrix(self.emb.encode(df.merchant_raw.tolist()) * 2.0),
                sparse.csr_matrix(np.log1p(df.amount.clip(lower=0).to_numpy())[:, None] / 5),
            ]
        ).tocsr()

    def fit(self, df: pd.DataFrame) -> LinearCoder:
        self.model = LogisticRegression(C=6.0, max_iter=2000).fit(self._X(df, True), df.gl_account)
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self._X(df, False))


def propose(df: pd.DataFrame, setting: str, retrain_every: int = 3, emb: Embedder | None = None) -> pd.DataFrame:
    """Expanding-window proposals for every line from WINDOW_START on."""
    emb = emb or Embedder()
    months = sorted(m for m in df.statement_month.unique() if m >= WINDOW_START)
    out, model = [], None
    for i, m in enumerate(months):
        if model is None or i % retrain_every == 0:
            model = LinearCoder(setting, emb).fit(df[df.statement_month < m])
        te = df[df.statement_month == m]
        P = model.predict_proba(te)
        k = P.argmax(1)
        out.append(
            pd.DataFrame(
                {
                    "card_line_id": te.card_line_id.to_numpy(),
                    "statement_month": m,
                    "pred": model.model.classes_[k],
                    "conf": P.max(1),
                }
            )
        )
    return pd.concat(out, ignore_index=True)
