"""Leakage-safe dataset splitting and HC3 external-test construction.

This file answers one question:
    "How should the cleaned documents be divided for model development/testing?"

Important project rule:
    DRCAT is the development dataset.
    HC3 stays separate as an external unseen evaluation dataset.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def _split_score(
    splits: Iterable[tuple[np.ndarray, float]],
    row_counts: np.ndarray,
    ai_counts: np.ndarray,
    total_rows: int,
    overall_ai_rate: float,
) -> float:
    """Give one candidate DRCAT group split a balance score.

    Why this helps:
        We want train/validation/test to be close to the target sizes AND to
        have AI/human ratios reasonably close to the full DRCAT dataset.
        Lower score = better candidate.

    Example idea:
        Dataset is 50% AI overall.
        Candidate test split is 15% of rows and 51% AI -> good.
        Candidate test split is 15% of rows and 95% AI -> poor score.
    """
    score = 0.0

    for positions, target_fraction in splits:
        rows = row_counts[positions].sum()
        if rows == 0:
            return float("inf")

        ai_rate = ai_counts[positions].sum() / rows

        # Penalise wrong split size.
        score += abs(rows / total_rows - target_fraction)

        # Penalise label imbalance. 1.5 gives label balance a little more weight.
        score += 1.5 * abs(ai_rate - overall_ai_rate)

    return float(score)


def balanced_group_split_drcat(
    df: pd.DataFrame,
    *,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
    search_trials: int = 30000,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split DRCAT by whole prompt groups while keeping class ratios balanced.

    Why this helps:
        Multiple essays can share the same prompt. If the same prompt appears in
        both training and testing, the model may exploit topic/prompt clues and
        evaluation can become over-optimistic. Therefore each group_key goes to
        exactly ONE partition.

    Because DRCAT has relatively few prompt groups, a simple random group split
    can also produce a very human-heavy or AI-heavy validation/test set. This
    function tries many group assignments and keeps the best-balanced one.

    Example:
        group_key="drcat::Car-free cities"
        -> ALL documents with that prompt go to train OR validation OR test,
           never more than one of them.
    """
    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError("train_size + val_size + test_size must equal 1.0")

    # For every prompt group, count total documents and AI documents.
    stats = df.groupby("group_key")["label"].agg(["count", "sum"])
    groups = stats.index.to_numpy()
    row_counts = stats["count"].to_numpy()
    ai_counts = stats["sum"].to_numpy()
    n_groups = len(groups)

    if n_groups < 3:
        raise ValueError("At least three group_key values are required.")

    # Convert requested fractions into an approximate number of whole groups.
    n_test_groups = max(1, round(n_groups * test_size))
    n_val_groups = max(1, round(n_groups * val_size))
    if n_test_groups + n_val_groups >= n_groups:
        n_test_groups = 1
        n_val_groups = 1

    total_rows = int(row_counts.sum())
    overall_ai_rate = float(ai_counts.sum() / total_rows)
    rng = np.random.default_rng(seed)

    best: tuple | None = None

    # Search many random whole-group assignments. This is still reproducible
    # because the random-number generator uses a fixed seed by default.
    for _ in range(search_trials):
        perm = rng.permutation(n_groups)
        test_pos = perm[:n_test_groups]
        val_pos = perm[n_test_groups : n_test_groups + n_val_groups]
        train_pos = perm[n_test_groups + n_val_groups :]

        score = _split_score(
            [
                (train_pos, train_size),
                (val_pos, val_size),
                (test_pos, test_size),
            ],
            row_counts,
            ai_counts,
            total_rows,
            overall_ai_rate,
        )

        if best is None or score < best[0]:
            best = (score, train_pos.copy(), val_pos.copy(), test_pos.copy())

    assert best is not None
    _, train_pos, val_pos, test_pos = best

    train_groups = set(groups[train_pos])
    val_groups = set(groups[val_pos])
    test_groups = set(groups[test_pos])

    train = df[df["group_key"].isin(train_groups)].reset_index(drop=True)
    val = df[df["group_key"].isin(val_groups)].reset_index(drop=True)
    test = df[df["group_key"].isin(test_groups)].reset_index(drop=True)

    # Safety checks: if any assertion fails, prompt leakage exists.
    assert set(train["group_key"]).isdisjoint(val["group_key"])
    assert set(train["group_key"]).isdisjoint(test["group_key"])
    assert set(val["group_key"]).isdisjoint(test["group_key"])

    return train, val, test


def build_balanced_hc3_external_test(
    hc3: pd.DataFrame,
    *,
    n_questions: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Build a matched HC3 external test with one human + one AI answer/question.

    Why this helps:
        Some HC3 questions contain more human answers than AI answers. If we used
        every answer directly, questions with many responses could dominate the
        external score. Sampling one human and one ChatGPT answer per question
        gives each selected question equal representation and perfect 50/50
        class balance.

    Example:
        Question A: 4 human answers + 1 AI answer
        Question B: 2 human answers + 1 AI answer

        External test keeps:
            Question A -> 1 human + 1 AI
            Question B -> 1 human + 1 AI
    """
    # Only use questions that still contain at least one human and one AI answer
    # after cleaning/deduplication/overlap removal.
    label_variety = hc3.groupby("group_key")["label"].nunique()
    eligible_groups = label_variety[label_variety >= 2].index.to_numpy()

    if len(eligible_groups) == 0:
        raise ValueError("No HC3 questions contain both human and AI answers.")

    rng = np.random.default_rng(seed)
    rng.shuffle(eligible_groups)

    if n_questions <= 0:
        selected = set(eligible_groups)
    else:
        selected = set(eligible_groups[: min(n_questions, len(eligible_groups))])

    subset = hc3[hc3["group_key"].isin(selected)].copy()

    # Select exactly one answer of each label from every selected question.
    human = (
        subset[subset["label"] == 0]
        .groupby("group_key", group_keys=False)
        .sample(n=1, random_state=seed)
    )
    ai = (
        subset[subset["label"] == 1]
        .groupby("group_key", group_keys=False)
        .sample(n=1, random_state=seed + 1)
    )

    result = pd.concat([human, ai], ignore_index=True)

    # Shuffle final rows so human/AI examples are not stored in obvious blocks.
    return result.sample(frac=1, random_state=seed).reset_index(drop=True)
