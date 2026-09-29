"""Run the complete document-level preprocessing pipeline for DRCAT + HC3.

This is the ONLY preprocessing file you normally run directly.
The detailed jobs are separated into smaller modules:

    data_adapters.py     -> make DRCAT and HC3 use the same columns
    data_cleaning.py     -> clean text, remove duplicates and overlap
    data_splitting.py    -> build leakage-safe DRCAT splits + HC3 external test
    preprocess_report.py -> create audit/report evidence

Data locations stay the same:
    input:  data/raw/
    output: data/processed/

Typical run:
    python preprocess.py \\
        --drcat data/raw/train_v2_drcat_02.csv \\
        --hc3 data/raw/hc3_all.jsonl

Because --out-dir defaults to data/processed, you do not need to type it unless
wanting a different location.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from data_adapters import OUTPUT_COLUMNS, adapt_drcat, adapt_hc3
from data_cleaning import (
    clean_frame,
    deduplicate_labeled,
    remove_hc3_overlap_with_drcat,
)
from data_splitting import balanced_group_split_drcat, build_balanced_hc3_external_test
from preprocess_report import (
    describe_raw_drcat,
    describe_raw_hc3,
    summarize_dataset,
    write_report,
    write_transformation_examples,
)


def build_datasets(
    *,
    drcat_path: str | Path,
    hc3_path: str | Path,
    out_dir: str | Path = "data/processed",
    min_chars: int = 30,
    max_chars: int = 20000,
    hc3_questions: int = 1000,
    seed: int = 42,
) -> dict[str, Path]:
    """Run the complete preprocessing workflow and save all processed files.

    Why this helps:
        This function is the coordinator. It does not contain the detailed
        cleaning/splitting algorithms anymore; it simply calls each small module
        in the correct order.

    Pipeline:
        1. adapt raw DRCAT/HC3 into the same columns
        2. clean and deduplicate each dataset
        3. remove HC3 text that overlaps DRCAT
        4. save a unified inspection file
        5. split DRCAT by prompt group
        6. build balanced HC3 external unseen test
        7. save transformation examples + preprocessing report

    Example result:
        data/processed/drcat_train.csv
        data/processed/drcat_validation.csv
        data/processed/drcat_test.csv
        data/processed/hc3_external_test.csv
        data/processed/unified_documents.csv
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report_lines: list[str] = []

    # ------------------------------------------------------------------
    # 1) DRCAT: raw shape -> common schema -> cleaned documents
    # ------------------------------------------------------------------
    report_lines += describe_raw_drcat(drcat_path)

    drcat = adapt_drcat(drcat_path)
    drcat, drcat_clean_stats = clean_frame(
        drcat,
        min_chars=min_chars,
        max_chars=max_chars,
    )
    drcat, drcat_dedup_stats = deduplicate_labeled(drcat, "DRCAT")

    report_lines += [
        "=== DRCAT CLEANING ===",
        str(drcat_clean_stats),
        str(drcat_dedup_stats),
        f"After preprocessing: {summarize_dataset(drcat)}",
        "",
    ]

    # ------------------------------------------------------------------
    # 2) HC3: answer-list shape -> common schema -> cleaned documents
    # ------------------------------------------------------------------
    report_lines += describe_raw_hc3(hc3_path)

    hc3 = adapt_hc3(hc3_path)
    hc3, hc3_clean_stats = clean_frame(
        hc3,
        min_chars=min_chars,
        max_chars=max_chars,
    )
    hc3, hc3_dedup_stats = deduplicate_labeled(hc3, "HC3")

    # Keep the external dataset genuinely separate from DRCAT wherever exact
    # text overlap can be detected.
    hc3, overlap_removed = remove_hc3_overlap_with_drcat(drcat, hc3)

    report_lines += [
        "=== HC3 CLEANING ===",
        str(hc3_clean_stats),
        str(hc3_dedup_stats),
        f"HC3 rows removed because exact text also existed in DRCAT: {overlap_removed:,}",
        f"After preprocessing: {summarize_dataset(hc3)}",
        "",
    ]

    # ------------------------------------------------------------------
    # 3) Unified document file: useful for inspection/EDA, NOT random split
    # ------------------------------------------------------------------
    unified = pd.concat([drcat, hc3], ignore_index=True)[OUTPUT_COLUMNS]
    unified_path = out_dir / "unified_documents.csv"
    unified.to_csv(unified_path, index=False)

    report_lines += [
        "=== UNIFIED DOCUMENT FILE ===",
        f"Saved: {unified_path}",
        f"Unified rows: {len(unified):,}",
        "Purpose: audit / analysis / visualisation. Do NOT randomly train/test split this whole file.",
        "DRCAT remains the development dataset; HC3 remains external unseen evaluation.",
        "",
    ]

    # ------------------------------------------------------------------
    # 4) DRCAT development split: whole prompt groups stay together
    # ------------------------------------------------------------------
    drcat_train, drcat_val, drcat_test = balanced_group_split_drcat(
        drcat,
        seed=seed,
    )

    train_path = out_dir / "drcat_train.csv"
    val_path = out_dir / "drcat_validation.csv"
    test_path = out_dir / "drcat_test.csv"

    drcat_train.to_csv(train_path, index=False)
    drcat_val.to_csv(val_path, index=False)
    drcat_test.to_csv(test_path, index=False)

    report_lines += [
        "=== DRCAT PROMPT-GROUP SPLIT ===",
        f"Train: {summarize_dataset(drcat_train)}",
        f"Validation: {summarize_dataset(drcat_val)}",
        f"Test: {summarize_dataset(drcat_test)}",
        "Leakage rule: the same DRCAT prompt/group_key never appears in more than one split.",
        "",
    ]

    # ------------------------------------------------------------------
    # 5) HC3 stays external: one human + one ChatGPT answer/question
    # ------------------------------------------------------------------
    hc3_external = build_balanced_hc3_external_test(
        hc3,
        n_questions=hc3_questions,
        seed=seed,
    )

    hc3_external_path = out_dir / "hc3_external_test.csv"
    hc3_external.to_csv(hc3_external_path, index=False)

    report_lines += [
        "=== HC3 EXTERNAL UNSEEN TEST ===",
        f"External test: {summarize_dataset(hc3_external)}",
        f"Requested question groups: {hc3_questions if hc3_questions > 0 else 'ALL'}",
        "Sampling rule: one human answer + one ChatGPT answer per selected question.",
        "HC3 is not used to tune features/model if you want it to remain a true external test.",
        "",
    ]

    # ------------------------------------------------------------------
    # 6) Evidence files for your team/report
    # ------------------------------------------------------------------
    examples_path = out_dir / "transformation_examples.csv"
    write_transformation_examples(drcat, hc3, examples_path)

    report_path = out_dir / "preprocess_report.txt"
    write_report(report_lines, report_path)

    paths = {
        "unified": unified_path,
        "drcat_train": train_path,
        "drcat_validation": val_path,
        "drcat_test": test_path,
        "hc3_external_test": hc3_external_path,
        "examples": examples_path,
        "report": report_path,
    }

    print("\nPreprocessing complete.\n")
    for name, path in paths.items():
        print(f"{name:22s} -> {path}")

    print(
        "\nIMPORTANT: train/tune on DRCAT train+validation. "
        "Keep HC3 external until final evaluation."
    )
    return paths


def main() -> None:
    """Read command-line options and start build_datasets().

    Example:
        python preprocess.py \\
            --drcat data/raw/train_v2_drcat_02.csv \\
            --hc3 data/raw/hc3_all.jsonl

    The output directory defaults to data/processed, so the command above keeps
    exactly the folder organisation you requested.
    """
    parser = argparse.ArgumentParser(description="Preprocess DRCAT + HC3")

    parser.add_argument(
        "--drcat",
        required=True,
        help="Path to train_v2_drcat_02.csv",
    )
    parser.add_argument(
        "--hc3",
        required=True,
        help="Path to HC3 all.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        default="data/processed",
        help="Where processed CSV/report files are saved",
    )
    parser.add_argument("--min-chars", type=int, default=30)
    parser.add_argument("--max-chars", type=int, default=20000)
    parser.add_argument(
        "--hc3-questions",
        type=int,
        default=1000,
        help="Number of HC3 question groups in external test; <=0 means all eligible questions.",
    )
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    build_datasets(
        drcat_path=args.drcat,
        hc3_path=args.hc3,
        out_dir=args.out_dir,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        hc3_questions=args.hc3_questions,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
