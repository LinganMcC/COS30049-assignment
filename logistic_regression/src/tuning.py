"""Grouped-CV model tuning utilities."""
from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd

from .metrics import compute_metrics
from .modeling import fit_pipeline, predict_sentence_proba
from .pooling import pool_documents


def evaluate_grid(
    df: pd.DataFrame,
    folds,
    *,
    weightings,
    class_balances,
    c_values,
    poolings,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate every model configuration on exactly the same grouped folds.

    The model is fit once per (weighting, class_balance, C, fold). Both pooling
    methods then reuse the same held-out sentence probabilities.
    """
    fold_rows = []

    for weighting, class_balance, c_value in product(weightings, class_balances, c_values):
        for fold_id, (train_idx, valid_idx) in enumerate(folds, start=1):
            train = df.loc[train_idx].copy()
            valid = df.loc[valid_idx].copy()
            pipe, fit_info = fit_pipeline(
                train,
                weighting=weighting,
                class_balance=class_balance,
                c_value=float(c_value),
            )
            sent_p = predict_sentence_proba(pipe, valid)

            for pooling in poolings:
                docs = pool_documents(valid, sent_p, pooling)
                metrics = compute_metrics(docs["label"], docs["doc_proba"], 0.5)
                fold_rows.append(
                    {
                        "weighting": weighting,
                        "class_balance": class_balance,
                        "C": float(c_value),
                        "pooling": pooling,
                        "fold": fold_id,
                        "n_iter": fit_info["n_iter"],
                        "n_documents": metrics["n_documents"],
                        "roc_auc": metrics["roc_auc"],
                        "average_precision": metrics["average_precision"],
                        # Diagnostic only; configuration ranking does not use 0.5 metrics.
                        "accuracy_at_0_5": metrics["accuracy"],
                        "f1_at_0_5": metrics["f1"],
                    }
                )

    folds_df = pd.DataFrame(fold_rows)
    group_cols = ["weighting", "class_balance", "C", "pooling"]
    summary = (
        folds_df.groupby(group_cols, as_index=False)
        .agg(
            mean_roc_auc=("roc_auc", "mean"),
            std_roc_auc=("roc_auc", "std"),
            mean_average_precision=("average_precision", "mean"),
            std_average_precision=("average_precision", "std"),
            mean_accuracy_at_0_5=("accuracy_at_0_5", "mean"),
            mean_f1_at_0_5=("f1_at_0_5", "mean"),
            mean_iterations=("n_iter", "mean"),
            max_iterations=("n_iter", "max"),
        )
    )

    # Transparent recommendation rule: equal emphasis on the two
    # threshold-independent ranking metrics. This is a recommendation, not a
    # silent freeze decision.
    summary["selection_score"] = (
        summary["mean_roc_auc"] + summary["mean_average_precision"]
    ) / 2.0
    summary = summary.sort_values(
        ["selection_score", "mean_roc_auc", "mean_average_precision"],
        ascending=False,
    ).reset_index(drop=True)
    summary["rank"] = np.arange(1, len(summary) + 1)
    return folds_df, summary


def recommendation_from_summary(summary: pd.DataFrame) -> dict:
    row = summary.iloc[0]
    return {
        "weighting": str(row["weighting"]),
        "class_balance": str(row["class_balance"]),
        "C": float(row["C"]),
        "pooling": str(row["pooling"]),
        "mean_roc_auc": float(row["mean_roc_auc"]),
        "mean_average_precision": float(row["mean_average_precision"]),
        "selection_score": float(row["selection_score"]),
        "note": "Recommendation only. Inspect the full CV tables before freezing.",
    }


def fine_c_grid(center_c: float, multipliers) -> list[float]:
    values = sorted({float(center_c) * float(m) for m in multipliers})
    return [float(f"{v:.10g}") for v in values if v > 0]
