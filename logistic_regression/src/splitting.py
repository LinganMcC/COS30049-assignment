"""Leakage-safe grouped CV created at DOCUMENT level."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .data_utils import document_table


def make_document_group_folds(
    sentence_df: pd.DataFrame,
    n_splits: int,
    random_state: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return sentence-row indices for CV folds defined on one-row-per-document data.

    Stratification therefore balances document labels rather than sentence counts,
    while group_key keeps related prompt/question groups wholly inside one fold.
    """
    docs = document_table(sentence_df)
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    folds: list[tuple[np.ndarray, np.ndarray]] = []
    dummy_x = np.zeros((len(docs), 1))
    for train_doc_pos, valid_doc_pos in splitter.split(
        dummy_x,
        y=docs["label"].to_numpy(),
        groups=docs["group_key"].to_numpy(),
    ):
        train_docs = set(docs.iloc[train_doc_pos]["document_id"])
        valid_docs = set(docs.iloc[valid_doc_pos]["document_id"])
        train_groups = set(docs.iloc[train_doc_pos]["group_key"])
        valid_groups = set(docs.iloc[valid_doc_pos]["group_key"])

        assert train_docs.isdisjoint(valid_docs)
        assert train_groups.isdisjoint(valid_groups)

        train_idx = sentence_df.index[sentence_df["document_id"].isin(train_docs)].to_numpy()
        valid_idx = sentence_df.index[sentence_df["document_id"].isin(valid_docs)].to_numpy()
        folds.append((train_idx, valid_idx))

    return folds


def fold_summary(sentence_df: pd.DataFrame, folds) -> pd.DataFrame:
    docs = document_table(sentence_df).set_index("document_id")
    rows = []
    for fold_id, (train_idx, valid_idx) in enumerate(folds, start=1):
        for split_name, idx in (("train", train_idx), ("validation", valid_idx)):
            doc_ids = sentence_df.loc[idx, "document_id"].drop_duplicates()
            sub = docs.loc[doc_ids]
            rows.append(
                {
                    "fold": fold_id,
                    "split": split_name,
                    "n_documents": int(len(sub)),
                    "n_groups": int(sub["group_key"].nunique()),
                    "human_documents": int((sub["label"] == 0).sum()),
                    "ai_documents": int((sub["label"] == 1).sum()),
                    "ai_share": float(sub["label"].mean()),
                }
            )
    return pd.DataFrame(rows)
