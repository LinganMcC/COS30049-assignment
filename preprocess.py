"""Run the complete document-level preprocessing pipeline for DRCAT + HC3.

This is the ONLY preprocessing file you normally run directly.

Detailed jobs are separated into modules:

    data_adapters.py
        -> convert DRCAT and HC3 into the same document schema

    data_cleaning.py
        -> minimal text cleaning
        -> HC3 document-length filtering
        -> same-label duplicate removal

    data_splitting.py
        -> leakage-safe DRCAT train/validation/test split
        -> balanced HC3 external test

    preprocess_report.py
        -> preprocessing/transformation audit evidence


Current preprocessing decisions:

DRCAT:
    - clean text
    - remove same-label duplicates
    - do NOT filter documents by length

HC3:
    - clean text
    - remove documents outside the allowed length range
    - remove same-label duplicates

Not performed:
    - conflicting-label duplicate removal
    - DRCAT/HC3 exact-overlap removal


Data locations:

    input:
        data/raw/

    output:
        data/processed/


Typical run:

    python preprocess.py \
        --drcat data/raw/train_v2_drcat_02.csv \
        --hc3 data/raw/hc3_all.jsonl
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from data_adapters import (
    OUTPUT_COLUMNS,
    adapt_drcat,
    adapt_hc3,
)

from data_cleaning import (
    clean_frame,
    filter_document_length,
    deduplicate_same_label,
)

from data_splitting import (
    balanced_group_split_drcat,
    build_balanced_hc3_external_test,
)

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
    """Run the complete document-level preprocessing workflow."""

    out_dir = Path(out_dir)

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_lines: list[str] = []

    # ==============================================================
    # 1) DRCAT
    # ==============================================================

    report_lines += describe_raw_drcat(
        drcat_path
    )

    drcat = adapt_drcat(
        drcat_path
    )

    # Minimal formatting/text normalisation only.
    #
    # IMPORTANT:
    # DRCAT is NOT filtered according to document length.
    drcat = clean_frame(
        drcat
    )

    # Remove only repeated copies having the same text AND label.
    drcat, drcat_dedup_stats = deduplicate_same_label(
        drcat,
        "DRCAT",
    )

    report_lines += [
        "=== DRCAT CLEANING ===",
        "Minimal text cleaning applied.",
        "Document-length filtering: NOT applied.",
        (
            "Same-label duplicate rows removed: "
            f"{drcat_dedup_stats['same_label_duplicate_rows_removed']:,}"
        ),
        (
            "After preprocessing: "
            f"{summarize_dataset(drcat)}"
        ),
        "",
    ]

    # ==============================================================
    # 2) HC3
    # ==============================================================

    report_lines += describe_raw_hc3(
        hc3_path
    )

    hc3 = adapt_hc3(
        hc3_path
    )

    # First perform the same minimal text cleaning as DRCAT.
    hc3 = clean_frame(
        hc3
    )

    # HC3 only:
    # remove empty/tiny/extremely large answer documents.
    hc3, hc3_length_stats = filter_document_length(
        hc3,
        min_chars=min_chars,
        max_chars=max_chars,
    )

    # Remove repeated copies having the same text AND label.
    hc3, hc3_dedup_stats = deduplicate_same_label(
        hc3,
        "HC3",
    )

    report_lines += [
        "=== HC3 CLEANING ===",
        "Minimal text cleaning applied.",
        (
            "Document-length filter: "
            f"{min_chars:,} to {max_chars:,} characters"
        ),
        (
            "Rows before length filtering: "
            f"{hc3_length_stats['before']:,}"
        ),
        (
            "Rows after length filtering: "
            f"{hc3_length_stats['after']:,}"
        ),
        (
            "Rows dropped because of document length/empty text: "
            f"{hc3_length_stats['dropped_length_or_empty']:,}"
        ),
        (
            "Same-label duplicate rows removed: "
            f"{hc3_dedup_stats['same_label_duplicate_rows_removed']:,}"
        ),
        (
            "After preprocessing: "
            f"{summarize_dataset(hc3)}"
        ),
        "",
    ]

    # ==============================================================
    # 3) Unified document file
    # ==============================================================

    unified = pd.concat(
        [
            drcat,
            hc3,
        ],
        ignore_index=True,
    )[OUTPUT_COLUMNS]

    unified_path = (
        out_dir
        / "unified_documents.csv"
    )

    unified.to_csv(
        unified_path,
        index=False,
    )

    report_lines += [
        "=== UNIFIED DOCUMENT FILE ===",
        f"Saved: {unified_path}",
        f"Unified rows: {len(unified):,}",
        (
            "Purpose: audit / analysis / visualisation. "
            "Do NOT randomly train/test split this whole file."
        ),
        (
            "DRCAT remains the development dataset; "
            "HC3 remains external evaluation."
        ),
        "",
    ]

    # ==============================================================
    # 4) DRCAT train / validation / internal test
    # ==============================================================

    (
        drcat_train,
        drcat_val,
        drcat_test,
    ) = balanced_group_split_drcat(
        drcat,
        seed=seed,
    )

    train_path = (
        out_dir
        / "drcat_train.csv"
    )

    val_path = (
        out_dir
        / "drcat_validation.csv"
    )

    test_path = (
        out_dir
        / "drcat_test.csv"
    )

    drcat_train.to_csv(
        train_path,
        index=False,
    )

    drcat_val.to_csv(
        val_path,
        index=False,
    )

    drcat_test.to_csv(
        test_path,
        index=False,
    )

    report_lines += [
        "=== DRCAT PROMPT-GROUP SPLIT ===",
        (
            "Train: "
            f"{summarize_dataset(drcat_train)}"
        ),
        (
            "Validation: "
            f"{summarize_dataset(drcat_val)}"
        ),
        (
            "Test: "
            f"{summarize_dataset(drcat_test)}"
        ),
        (
            "Leakage rule: the same DRCAT "
            "prompt/group_key never appears in "
            "more than one split."
        ),
        "",
    ]

    # ==============================================================
    # 5) HC3 external unseen test
    # ==============================================================

    hc3_external = (
        build_balanced_hc3_external_test(
            hc3,
            n_questions=hc3_questions,
            seed=seed,
        )
    )

    hc3_external_path = (
        out_dir
        / "hc3_external_test.csv"
    )

    hc3_external.to_csv(
        hc3_external_path,
        index=False,
    )

    report_lines += [
        "=== HC3 EXTERNAL UNSEEN TEST ===",
        (
            "External test: "
            f"{summarize_dataset(hc3_external)}"
        ),
        (
            "Requested question groups: "
            f"{hc3_questions if hc3_questions > 0 else 'ALL'}"
        ),
        (
            "Sampling rule: one human answer + "
            "one ChatGPT answer per selected question."
        ),
        (
            "HC3 is not used for model training or "
            "hyperparameter tuning if it is being kept "
            "as a true external test."
        ),
        "",
    ]

    # ==============================================================
    # 6) Transformation evidence
    # ==============================================================

    examples_path = (
        out_dir
        / "transformation_examples.csv"
    )

    write_transformation_examples(
        drcat,
        hc3,
        examples_path,
    )

    # ==============================================================
    # 7) Preprocessing report
    # ==============================================================

    report_path = (
        out_dir
        / "preprocess_report.txt"
    )

    write_report(
        report_lines,
        report_path,
    )

    # ==============================================================
    # 8) Return generated file locations
    # ==============================================================

    paths = {
        "unified": unified_path,
        "drcat_train": train_path,
        "drcat_validation": val_path,
        "drcat_test": test_path,
        "hc3_external_test": hc3_external_path,
        "examples": examples_path,
        "report": report_path,
    }

    print(
        "\nPreprocessing complete.\n"
    )

    for name, path in paths.items():
        print(
            f"{name:22s} -> {path}"
        )

    return paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess DRCAT + HC3"
    )

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
        help="Directory for processed datasets and reports",
    )

    parser.add_argument(
        "--min-chars",
        type=int,
        default=30,
        help=(
            "Minimum HC3 document length. "
            "This filter is NOT applied to DRCAT."
        ),
    )

    parser.add_argument(
        "--max-chars",
        type=int,
        default=20000,
        help=(
            "Maximum HC3 document length. "
            "This filter is NOT applied to DRCAT."
        ),
    )

    parser.add_argument(
        "--hc3-questions",
        type=int,
        default=1000,
        help=(
            "Number of HC3 question groups in the external test. "
            "<=0 means use all eligible question groups."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

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