"""Training-only feature analysis for the AI-generated content detection report.

Run from the project root after feature extraction:
    python analyze_features_for_report.py \
        --input data/features/drcat_train_final_sentence_features.csv \
        --output-dir results/feature_analysis/report

Important: use DRCAT TRAIN only. Do not use validation/test/HC3 to decide features.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FEATURES = [
    "n_words",
    "long_word_ratio",
    "func_word_ratio",
    "unique_word_ratio",
    "comma_rate",
    "exclaim",
    "apostrophe_rate",
    "starts_with_transition",
    "rel_len_deviation",
    "noun_ratio",
    "adjective_ratio",
    "adverb_ratio",
    "pronoun_ratio",
    "auxiliary_ratio",
    "conjunction_ratio",
]


def pooled_cohens_d(human: pd.Series, ai: pd.Series) -> float:
    """Return Cohen's d using AI mean - Human mean."""
    human = pd.to_numeric(human, errors="coerce").dropna()
    ai = pd.to_numeric(ai, errors="coerce").dropna()
    n_h, n_a = len(human), len(ai)
    if n_h < 2 or n_a < 2:
        return float("nan")
    s_h, s_a = human.std(ddof=1), ai.std(ddof=1)
    pooled = np.sqrt(((n_h - 1) * s_h**2 + (n_a - 1) * s_a**2) / (n_h + n_a - 2))
    if pooled == 0:
        return 0.0
    return float((ai.mean() - human.mean()) / pooled)


def analyse(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in ["label", *FEATURES] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    rows = []
    for feature in FEATURES:
        human = df.loc[df["label"] == 0, feature]
        ai = df.loc[df["label"] == 1, feature]
        rows.append(
            {
                "feature": feature,
                "human_mean": pd.to_numeric(human, errors="coerce").mean(),
                "ai_mean": pd.to_numeric(ai, errors="coerce").mean(),
                "human_median": pd.to_numeric(human, errors="coerce").median(),
                "ai_median": pd.to_numeric(ai, errors="coerce").median(),
                "cohen_d_ai_minus_human": pooled_cohens_d(human, ai),
            }
        )
    return pd.DataFrame(rows)


def save_effect_size_plot(summary: pd.DataFrame, out_path: Path) -> None:
    plot_df = summary.sort_values("cohen_d_ai_minus_human")
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(plot_df["feature"], plot_df["cohen_d_ai_minus_human"])
    ax.axvline(0, linewidth=1)
    ax.set_xlabel("Cohen's d (AI mean - Human mean)")
    ax.set_ylabel("Feature")
    ax.set_title("DRCAT TRAIN: standardised Human-AI feature differences")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_correlation_heatmap(df: pd.DataFrame, out_path: Path) -> None:
    corr = df[FEATURES].apply(pd.to_numeric, errors="coerce").corr()
    fig, ax = plt.subplots(figsize=(11, 9))
    image = ax.imshow(corr.values, aspect="auto", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(FEATURES)))
    ax.set_yticks(np.arange(len(FEATURES)))
    ax.set_xticklabels(FEATURES, rotation=70, ha="right", fontsize=8)
    ax.set_yticklabels(FEATURES, fontsize=8)
    ax.set_title("DRCAT TRAIN: correlation among final sentence features")
    fig.colorbar(image, ax=ax, label="Pearson correlation")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="DRCAT TRAIN final sentence feature CSV")
    parser.add_argument("--output-dir", default="results/feature_analysis/report")
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    summary = analyse(df)
    summary.to_csv(out_dir / "feature_class_summary.csv", index=False)

    save_effect_size_plot(summary, out_dir / "feature_effect_sizes.png")
    save_correlation_heatmap(df, out_dir / "feature_correlation_heatmap.png")

    print(f"Rows: {len(df):,}")
    if "document_id" in df.columns:
        print(f"Documents: {df['document_id'].nunique():,}")
    print("\nFeatures ranked by absolute Cohen's d:")
    ranked = summary.iloc[summary["cohen_d_ai_minus_human"].abs().argsort()[::-1]]
    print(ranked.to_string(index=False))
    print(f"\nOutputs written to: {out_dir}")


if __name__ == "__main__":
    main()
