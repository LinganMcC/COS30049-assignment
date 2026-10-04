from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

import config
from src.data_utils import load_feature_table
from src.metrics import compute_metrics, select_threshold, threshold_sweep
from src.modeling import coefficient_table, fit_pipeline, save_frozen, score_documents
from src.plots import plot_threshold_sweep, plot_top_coefficients
from src.reporting import markdown_table, write_json, write_text


def load_recommendation() -> dict:
    path = config.FINE_RECOMMENDATION_PATH if config.FINE_RECOMMENDATION_PATH.exists() else config.COARSE_RECOMMENDATION_PATH
    if not path.exists():
        raise SystemExit("No CV recommendation found. Run coarse/fine tuning first.")
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_config(args) -> dict:
    if args.use_cv_recommendation:
        rec = load_recommendation()
        return {
            "weighting": rec["weighting"],
            "class_balance": rec["class_balance"],
            "C": float(rec["C"]),
            "pooling": rec["pooling"],
        }

    missing = [
        name
        for name, value in {
            "--weighting": args.weighting,
            "--class-balance": args.class_balance,
            "--c": args.c_value,
            "--pooling": args.pooling,
        }.items()
        if value is None
    ]
    if missing:
        raise SystemExit(
            "Choose the TRAIN-CV configuration explicitly or pass --use-cv-recommendation. "
            f"Missing: {', '.join(missing)}"
        )
    return {
        "weighting": args.weighting,
        "class_balance": args.class_balance,
        "C": float(args.c_value),
        "pooling": args.pooling,
    }


def main(args) -> None:
    out = config.stage_dir("04_validation")
    selected = resolve_config(args)

    train = load_feature_table(config.DATASETS["train"], "TRAIN")
    val = load_feature_table(config.DATASETS["validation"], "VALIDATION")

    pipe, fit_info = fit_pipeline(
        train,
        weighting=selected["weighting"],
        class_balance=selected["class_balance"],
        c_value=selected["C"],
    )
    _, val_docs = score_documents(pipe, val, selected["pooling"])

    sweep = threshold_sweep(val_docs["label"], val_docs["doc_proba"])
    best = select_threshold(sweep)
    threshold = float(best["threshold"])
    metrics = compute_metrics(val_docs["label"], val_docs["doc_proba"], threshold)

    val_docs.reset_index().to_csv(out / "validation_document_scores.csv", index=False)
    sweep.to_csv(out / "threshold_sweep.csv", index=False)
    coefficients = coefficient_table(pipe)
    coefficients.to_csv(out / "coefficients.csv", index=False)
    plot_threshold_sweep(sweep, out / "threshold_sweep.png")
    plot_top_coefficients(coefficients, out / "coefficients.png")

    frozen_cfg = {
        "features": list(config.FINAL_FEATURES),
        "weighting": selected["weighting"],
        "class_balance": selected["class_balance"],
        "C": selected["C"],
        "pooling": selected["pooling"],
        "threshold": threshold,
        "threshold_objective": config.THRESHOLD_OBJECTIVE,
        "solver": config.LR_BASE_PARAMS["solver"],
        "max_iter": config.LR_BASE_PARAMS["max_iter"],
        "random_state": config.RANDOM_STATE,
        "fit_info": fit_info,
        "validation_metrics": metrics,
    }
    write_json(out / "candidate_config.json", frozen_cfg)

    report = "\n".join(
        [
            "# Validation and threshold selection",
            "",
            "Model-family choices below came from TRAIN CV; VALIDATION is used only for threshold selection.",
            "",
            "```json",
            json.dumps(selected, indent=2),
            "```",
            "",
            f"Selected threshold by macro-F1: **{threshold:.2f}**",
            "",
            "## Validation metrics",
            "",
            markdown_table(pd.DataFrame([metrics])),
            "",
            "## Coefficients",
            "",
            markdown_table(coefficients),
        ]
    )
    write_text(out / "validation_report.md", report)

    if args.no_freeze:
        print("Validation complete. --no-freeze was used, so nothing was frozen.")
    else:
        save_frozen(pipe, frozen_cfg)
        print(f"Frozen model: {config.FROZEN_MODEL_PATH}")
        print(f"Frozen config: {config.FROZEN_CONFIG_PATH}")
    print(f"Validation threshold: {threshold:.2f}")
    print(f"Inspect: {out / 'threshold_sweep.csv'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--use-cv-recommendation", action="store_true")
    ap.add_argument("--weighting", choices=config.SENTENCE_WEIGHTING_OPTIONS)
    ap.add_argument("--class-balance", choices=config.CLASS_BALANCE_OPTIONS)
    ap.add_argument("--c", dest="c_value", type=float)
    ap.add_argument("--pooling", choices=config.POOLING_OPTIONS)
    ap.add_argument("--no-freeze", action="store_true")
    main(ap.parse_args())
