from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from src.data_utils import load_feature_table
from src.reporting import markdown_table, write_json, write_text
from src.splitting import make_document_group_folds
from src.tuning import evaluate_grid, fine_c_grid, recommendation_from_summary


def load_coarse_recommendation() -> dict:
    if not config.COARSE_RECOMMENDATION_PATH.exists():
        raise SystemExit("Run scripts/02_coarse_tuning.py first.")
    return json.loads(config.COARSE_RECOMMENDATION_PATH.read_text(encoding="utf-8"))


def main(args) -> None:
    out = config.stage_dir("03_fine_tuning")
    train = load_feature_table(config.DATASETS["train"], "TRAIN")
    folds = make_document_group_folds(train, config.N_SPLITS, config.RANDOM_STATE)

    coarse = load_coarse_recommendation()
    weighting = args.weighting or coarse["weighting"]
    class_balance = args.class_balance or coarse["class_balance"]
    pooling = args.pooling or coarse["pooling"]
    center_c = args.center_c if args.center_c is not None else float(coarse["C"])
    c_values = args.c_values or fine_c_grid(center_c, config.FINE_C_MULTIPLIERS)

    folds_df, summary = evaluate_grid(
        train,
        folds,
        weightings=[weighting],
        class_balances=[class_balance],
        c_values=c_values,
        poolings=[pooling],
    )
    folds_df.to_csv(out / "all_fold_results.csv", index=False)
    summary.to_csv(out / "configuration_summary.csv", index=False)
    recommendation = recommendation_from_summary(summary)
    write_json(out / "recommended_config.json", recommendation)

    report = "\n".join(
        [
            "# Fine C tuning",
            "",
            f"Fixed weighting: `{weighting}`",
            f"Fixed class balance: `{class_balance}`",
            f"Fixed pooling: `{pooling}`",
            f"C values: `{c_values}`",
            "",
            markdown_table(summary),
            "",
            "## Recommendation",
            "",
            "```json",
            json.dumps(recommendation, indent=2),
            "```",
        ]
    )
    write_text(out / "fine_tuning_report.md", report)

    print("Fine tuning complete.")
    print(json.dumps(recommendation, indent=2))
    print(f"Inspect: {out / 'configuration_summary.csv'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weighting", choices=config.SENTENCE_WEIGHTING_OPTIONS)
    ap.add_argument("--class-balance", choices=config.CLASS_BALANCE_OPTIONS)
    ap.add_argument("--pooling", choices=config.POOLING_OPTIONS)
    ap.add_argument("--center-c", type=float)
    ap.add_argument("--c-values", nargs="+", type=float)
    main(ap.parse_args())
