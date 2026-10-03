"""Evaluate the frozen 15 non-LM sentence features on full DRCAT TRAIN.

Uses:
- 5-fold StratifiedGroupKFold by group_key
- document-balanced sentence sample weights
- Logistic Regression
- mean pooling of sentence probabilities to document level
- document-level Accuracy / Precision / Recall / F1 / ROC-AUC / PR-AUC

This script performs evaluation only. It does NOT run GPT-2.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


FEATURES_15 = [
    # Stylometry
    "n_words",
    "long_word_ratio",
    "func_word_ratio",
    "unique_word_ratio",
    "comma_rate",
    "exclaim",
    "apostrophe_rate",
    "starts_with_transition",
    "rel_len_deviation",

    # POS
    "noun_ratio",
    "adjective_ratio",
    "adverb_ratio",
    "pronoun_ratio",
    "auxiliary_ratio",
    "conjunction_ratio",
]


def document_balanced_weights(df: pd.DataFrame) -> np.ndarray:
    """Give each document approximately equal total weight."""
    sentence_count = (
        df.groupby("document_id")["document_id"]
        .transform("size")
        .astype(float)
    )
    return (1.0 / sentence_count).to_numpy()


def pool_to_documents(
    df: pd.DataFrame,
    sentence_probabilities: np.ndarray,
) -> pd.DataFrame:
    """Mean-pool sentence probabilities to one score per document."""
    temp = df[["document_id", "label"]].copy()
    temp["prob_ai"] = sentence_probabilities

    if (temp.groupby("document_id")["label"].nunique() > 1).any():
        raise ValueError("A document has inconsistent labels.")

    return (
        temp.groupby("document_id", as_index=False)
        .agg(
            label=("label", "first"),
            prob_ai=("prob_ai", "mean"),
        )
    )


def document_metrics(doc_df: pd.DataFrame) -> dict[str, float]:
    y_true = doc_df["label"].to_numpy(dtype=int)
    probabilities = doc_df["prob_ai"].to_numpy(dtype=float)
    predictions = (probabilities >= 0.5).astype(int)

    return {
        "n_documents": int(len(doc_df)),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, predictions, zero_division=0)
        ),
        "f1": float(
            f1_score(y_true, predictions, zero_division=0)
        ),
        "roc_auc": float(
            roc_auc_score(y_true, probabilities)
        ),
        "pr_auc": float(
            average_precision_score(y_true, probabilities)
        ),
    }


def main(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)

    required = {
        "document_id",
        "label",
        "group_key",
        *FEATURES_15,
    }

    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            "Input file is missing required columns:\n"
            + "\n".join(f"  - {name}" for name in missing)
        )

    df = df.copy()
    df["label"] = pd.to_numeric(
        df["label"],
        errors="raise",
    ).astype(int)

    print(f"Sentence rows: {len(df):,}")
    print(f"Documents: {df['document_id'].nunique():,}")
    print(f"Groups: {df['group_key'].nunique():,}")
    print(f"Features: {len(FEATURES_15)}")

    splitter = StratifiedGroupKFold(
        n_splits=args.cv_folds,
        shuffle=True,
        random_state=args.seed,
    )

    fold_rows: list[dict] = []

    for fold, (train_idx, valid_idx) in enumerate(
        splitter.split(
            df,
            y=df["label"],
            groups=df["group_key"],
        ),
        start=1,
    ):
        train_df = df.iloc[train_idx].copy()
        valid_df = df.iloc[valid_idx].copy()

        overlap = (
            set(train_df["group_key"])
            & set(valid_df["group_key"])
        )
        if overlap:
            raise AssertionError(
                f"group_key leakage detected in fold {fold}"
            )

        model = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                random_state=args.seed,
            ),
        )

        model.fit(
            train_df[FEATURES_15],
            train_df["label"],
            logisticregression__sample_weight=(
                document_balanced_weights(train_df)
            ),
        )

        sentence_probabilities = model.predict_proba(
            valid_df[FEATURES_15]
        )[:, 1]

        doc_df = pool_to_documents(
            valid_df,
            sentence_probabilities,
        )

        metrics = document_metrics(doc_df)

        fold_rows.append({
            "fold": fold,
            **metrics,
        })

        print(
            f"Fold {fold}: "
            f"AUROC={metrics['roc_auc']:.4f} | "
            f"PR-AUC={metrics['pr_auc']:.4f} | "
            f"F1={metrics['f1']:.4f} | "
            f"Accuracy={metrics['accuracy']:.4f}"
        )

    folds = pd.DataFrame(fold_rows)

    folds_path = output_dir / "full_train_15_features_folds.csv"
    folds.to_csv(folds_path, index=False)

    metric_names = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
    ]

    summary = {
        "n_features": len(FEATURES_15),
        "n_sentence_rows": len(df),
        "n_documents": df["document_id"].nunique(),
        "n_groups": df["group_key"].nunique(),
        "cv_folds": args.cv_folds,
    }

    for metric in metric_names:
        summary[f"{metric}_mean"] = float(
            folds[metric].mean()
        )
        summary[f"{metric}_std"] = float(
            folds[metric].std(ddof=1)
        )

    summary_df = pd.DataFrame([summary])

    summary_path = output_dir / "full_train_15_features_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\n=== FULL TRAIN 15-FEATURE RESULT ===")
    for metric in metric_names:
        print(
            f"{metric:10s}: "
            f"{summary[f'{metric}_mean']:.4f} "
            f"+/- {summary[f'{metric}_std']:.4f}"
        )

    print(f"\nSaved:\n  {folds_path}\n  {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/features/drcat_train_sentence_features_research.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="results/feature_analysis/full_train_15_features",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    main(parser.parse_args())
