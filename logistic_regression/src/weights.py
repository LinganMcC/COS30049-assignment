"""Training sample weights for sentence-supervised Logistic Regression."""
from __future__ import annotations

import numpy as np
import pandas as pd

WEIGHTINGS = ("uniform", "doc_equal", "sqrt_doc")
CLASS_BALANCE_MODES = ("none", "doc_balanced")


def length_weight(n_words) -> np.ndarray:
    return np.sqrt(np.maximum(np.asarray(n_words, dtype=float), 1.0))


def compute_training_weights(
    df: pd.DataFrame,
    weighting: str,
    class_balance: str,
) -> np.ndarray:
    """Build one sample_weight per sentence.

    uniform:
        every sentence receives equal initial weight.

    doc_equal:
        each document has total weight 1; sentences within it are equal.

    sqrt_doc:
        each document has total weight 1; within a document, sentence weight is
        proportional to sqrt(n_words), so longer sentences matter more without
        allowing long documents to dominate.

    doc_balanced:
        after the document/sentence weighting above, rescale Human and AI to have
        equal total training weight. This avoids stacking sklearn class_weight on
        top of custom sample weights.
    """
    if weighting not in WEIGHTINGS:
        raise ValueError(f"weighting must be one of {WEIGHTINGS}")
    if class_balance not in CLASS_BALANCE_MODES:
        raise ValueError(f"class_balance must be one of {CLASS_BALANCE_MODES}")

    if weighting == "sqrt_doc":
        raw = pd.Series(length_weight(df["n_words"]), index=df.index, dtype=float)
    else:
        raw = pd.Series(1.0, index=df.index, dtype=float)

    if weighting in {"doc_equal", "sqrt_doc"}:
        totals = raw.groupby(df["document_id"]).transform("sum")
        raw = raw / totals

    weights = raw.to_numpy(dtype=float, copy=True)
    y = df["label"].to_numpy(dtype=int)

    if class_balance == "doc_balanced":
        original_total = float(weights.sum())
        for label in (0, 1):
            mask = y == label
            if not mask.any():
                raise ValueError(f"Training fold contains no examples of class {label}")
            target_total = original_total / 2.0
            weights[mask] *= target_total / weights[mask].sum()

    # Keep average sample weight = 1 so C remains comparable across schemes.
    weights *= len(weights) / weights.sum()
    return weights
