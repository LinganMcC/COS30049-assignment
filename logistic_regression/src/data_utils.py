"""Load, validate and inspect sentence-level feature tables."""
from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from config import FINAL_FEATURES

META_REQUIRED = ["document_id", "label", "group_key", "n_words"]


def load_feature_table(path: Path, name: str = "") -> pd.DataFrame:
    """Load one feature table and enforce structural invariants.

    Missing/infinite model features are retained as NaN so the TRAIN-fitted
    median imputer can handle them, but a warning is emitted first. This gives
    robustness without silently hiding feature-extraction problems.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{name or 'Feature table'} not found: {path}")

    df = pd.read_csv(path)
    required = list(dict.fromkeys(META_REQUIRED + list(FINAL_FEATURES)))
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"{path.name} is missing required columns: {missing_cols}")

    df["label"] = pd.to_numeric(df["label"], errors="raise").astype(int)
    if not set(df["label"].unique()).issubset({0, 1}):
        raise ValueError(f"{path.name}: label must contain only 0 and 1")

    numeric = df[FINAL_FEATURES].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    df[FINAL_FEATURES] = numeric

    if df.groupby("document_id")["label"].nunique().max() > 1:
        raise ValueError(f"{path.name}: one or more documents contain mixed labels")
    if df.groupby("document_id")["group_key"].nunique().max() > 1:
        raise ValueError(f"{path.name}: one or more documents span multiple group_key values")

    missing_by_feature = numeric.isna().sum()
    missing_by_feature = missing_by_feature[missing_by_feature > 0]
    if len(missing_by_feature):
        detail = ", ".join(f"{k}={int(v)}" for k, v in missing_by_feature.items())
        warnings.warn(
            f"{path.name}: missing/non-finite feature values found ({detail}). "
            "They will be median-imputed using TRAIN-fitted medians.",
            RuntimeWarning,
        )

    if "sentence_index" in df.columns:
        df = df.sort_values(["document_id", "sentence_index"], kind="stable")
    return df.reset_index(drop=True)


def document_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per document, which is the evaluation unit."""
    g = df.groupby("document_id", sort=False)
    docs = pd.DataFrame(
        {
            "document_id": g.size().index,
            "label": g["label"].first().to_numpy(),
            "group_key": g["group_key"].first().to_numpy(),
            "n_sentences": g.size().to_numpy(),
            "n_words": g["n_words"].sum().to_numpy(),
        }
    )
    return docs.reset_index(drop=True)


def inspect_table(df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    docs = document_table(df)
    feats = df[FINAL_FEATURES]
    group_label_counts = docs.groupby("group_key")["label"].nunique()

    summary = {
        "n_sentences": int(len(df)),
        "n_documents": int(len(docs)),
        "n_groups": int(docs["group_key"].nunique()),
        "sentence_class_counts": {
            int(k): int(v) for k, v in df["label"].value_counts().sort_index().items()
        },
        "document_class_counts": {
            int(k): int(v) for k, v in docs["label"].value_counts().sort_index().items()
        },
        "ai_share_sentences": float(df["label"].mean()),
        "ai_share_documents": float(docs["label"].mean()),
        "groups_with_both_labels": int((group_label_counts > 1).sum()),
        "missing_feature_values": int(feats.isna().sum().sum()),
        "missing_by_feature": {
            k: int(v) for k, v in feats.isna().sum().items() if int(v) > 0
        },
        "zero_word_sentences": int((df["n_words"] <= 0).sum()),
        "sentences_per_document": docs["n_sentences"].describe().to_dict(),
    }

    ranges = feats.describe(percentiles=[0.01, 0.5, 0.99]).T
    ranges["missing"] = feats.isna().sum()
    return summary, ranges
