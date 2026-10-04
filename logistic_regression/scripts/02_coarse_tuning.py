from __future__ import annotations

import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from src.data_utils import load_feature_table
from src.reporting import markdown_table, write_json, write_text
from src.splitting import fold_summary, make_document_group_folds
from src.tuning import evaluate_grid, recommendation_from_summary


def main() -> None:
    out = config.stage_dir("02_coarse_tuning")
    train = load_feature_table(config.DATASETS["train"], "TRAIN")
    folds = make_document_group_folds(train, config.N_SPLITS, config.RANDOM_STATE)

    fold_info = fold_summary(train, folds)
    fold_info.to_csv(out / "cv_fold_structure.csv", index=False)

    fold_results, summary = evaluate_grid(
        train,
        folds,
        weightings=config.SENTENCE_WEIGHTING_OPTIONS,
        class_balances=config.CLASS_BALANCE_OPTIONS,
        c_values=config.COARSE_C_GRID,
        poolings=config.POOLING_OPTIONS,
    )
    fold_results.to_csv(out / "all_fold_results.csv", index=False)
    summary.to_csv(out / "configuration_summary.csv", index=False)

    recommendation = recommendation_from_summary(summary)
    write_json(out / "recommended_config.json", recommendation)

    report = "\n".join(
        [
            "# Logistic Regression coarse TRAIN-CV tuning",
            "",
            "CV folds were created on one-row-per-document data and grouped by group_key.",
            "The recommendation uses only threshold-independent ROC-AUC and Average Precision.",
            "It is a recommendation, not a frozen decision.",
            "",
            "## Fold structure",
            "",
            markdown_table(fold_info),
            "",
            "## Top 20 configurations",
            "",
            markdown_table(summary.head(20)),
            "",
            "## Recommended starting point for fine C search",
            "",
            "```json",
            json.dumps(recommendation, indent=2),
            "```",
        ]
    )
    write_text(out / "coarse_tuning_report.md", report)

    print("Coarse tuning complete.")
    print("Top recommendation (inspect results before accepting):")
    print(json.dumps(recommendation, indent=2))
    print(f"Full results: {out / 'configuration_summary.csv'}")


if __name__ == "__main__":
    main()
