"""Aggregate sentence probabilities into document probabilities."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .weights import length_weight

POOLING_METHODS = ("mean", "sqrt_length")


def pool_documents(df: pd.DataFrame, sentence_proba, method: str) -> pd.DataFrame:
    if method not in POOLING_METHODS:
        raise ValueError(f"method must be one of {POOLING_METHODS}")

    p = np.asarray(sentence_proba, dtype=float)
    if len(p) != len(df):
        raise ValueError("sentence_proba length does not match feature table")

    weights = np.ones(len(df), dtype=float)
    if method == "sqrt_length":
        weights = length_weight(df["n_words"])

    tmp = pd.DataFrame(
        {
            "document_id": df["document_id"].to_numpy(),
            "weighted_p": weights * p,
            "weight": weights,
        },
        index=df.index,
    )
    g = tmp.groupby("document_id", sort=False)
    doc_proba = g["weighted_p"].sum() / g["weight"].sum()

    meta = df.groupby("document_id", sort=False)
    return pd.DataFrame(
        {
            "doc_proba": doc_proba,
            "label": meta["label"].first(),
            "group_key": meta["group_key"].first(),
            "n_sentences": meta.size(),
            "n_words": meta["n_words"].sum(),
        }
    )
