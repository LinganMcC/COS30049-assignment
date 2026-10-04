"""Document-level metrics, threshold selection and group bootstrap CIs."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

import config

CORE_METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc", "average_precision")


def compute_metrics(y_true, proba, threshold: float = 0.5) -> dict:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(proba, dtype=float)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    two_classes = len(np.unique(y)) == 2

    return {
        "threshold": float(threshold),
        "n_documents": int(len(y)),
        "accuracy": float((tp + tn) / len(y)) if len(y) else float("nan"),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)) if two_classes else float("nan"),
        "average_precision": float(average_precision_score(y, p)) if two_classes else float("nan"),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def threshold_sweep(y_true, proba, grid=None) -> pd.DataFrame:
    grid = config.THRESHOLD_GRID if grid is None else grid
    rows = [compute_metrics(y_true, proba, float(t)) for t in grid]
    cols = [
        "threshold",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "macro_f1",
        "tn",
        "fp",
        "fn",
        "tp",
    ]
    return pd.DataFrame(rows)[cols]


def select_threshold(sweep: pd.DataFrame) -> pd.Series:
    """Maximise macro-F1; break ties by choosing the threshold closest to 0.5."""
    best_value = sweep[config.THRESHOLD_OBJECTIVE].max()
    candidates = sweep[np.isclose(sweep[config.THRESHOLD_OBJECTIVE], best_value)].copy()
    candidates["distance_from_0_5"] = (candidates["threshold"] - 0.5).abs()
    return candidates.sort_values(["distance_from_0_5", "threshold"]).iloc[0]


def group_bootstrap_ci(
    doc_scores: pd.DataFrame,
    *,
    threshold: float,
    n_bootstrap: int,
    ci_level: float,
    random_state: int,
) -> pd.DataFrame:
    """Cluster bootstrap by group_key.

    Each bootstrap draw samples group_key values with replacement and carries all
    documents in the sampled group. This respects prompt/question clustering.
    """
    required = {"label", "doc_proba", "group_key"}
    missing = required - set(doc_scores.columns)
    if missing:
        raise ValueError(f"doc_scores missing columns: {sorted(missing)}")

    grouped = [g for _, g in doc_scores.groupby("group_key", sort=False)]
    if len(grouped) < 2:
        raise ValueError("At least two group_key clusters are needed for bootstrap CI")

    rng = np.random.default_rng(random_state)
    samples = {m: [] for m in CORE_METRICS}

    for _ in range(int(n_bootstrap)):
        picked = rng.integers(0, len(grouped), size=len(grouped))
        boot = pd.concat([grouped[i] for i in picked], ignore_index=True)
        metrics = compute_metrics(boot["label"], boot["doc_proba"], threshold)
        for metric in CORE_METRICS:
            value = metrics[metric]
            if np.isfinite(value):
                samples[metric].append(value)

    alpha = (1.0 - ci_level) / 2.0
    point = compute_metrics(doc_scores["label"], doc_scores["doc_proba"], threshold)
    rows = []
    for metric in CORE_METRICS:
        values = np.asarray(samples[metric], dtype=float)
        rows.append(
            {
                "metric": metric,
                "estimate": point[metric],
                "ci_low": float(np.quantile(values, alpha)) if len(values) else np.nan,
                "ci_high": float(np.quantile(values, 1 - alpha)) if len(values) else np.nan,
                "valid_bootstrap_draws": int(len(values)),
            }
        )
    return pd.DataFrame(rows)
