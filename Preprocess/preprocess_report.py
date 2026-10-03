"""Small reporting helpers for the preprocessing pipeline.

This file does NOT change the training data.

It only records what happened so the preprocessing and dataset
transformation can be inspected and explained.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def summarize_dataset(df: pd.DataFrame) -> str:
    """Return a compact human-readable summary of a processed dataset.

    Example:

        rows=44,864,
        AI=17,497,
        human=27,367,
        AI%=39.00%,
        groups=15

    AI% simply means:

        number of AI rows / total rows * 100
    """
    if len(df) == 0:
        return "rows=0"

    ai_count = int(df["label"].sum())

    human_count = int(
        (df["label"] == 0).sum()
    )

    ai_percentage = (
        ai_count / len(df)
    ) * 100

    group_count = df["group_key"].nunique()

    return (
        f"rows={len(df):,}, "
        f"AI={ai_count:,}, "
        f"human={human_count:,}, "
        f"AI%={ai_percentage:.2f}%, "
        f"groups={group_count:,}"
    )


def describe_raw_drcat(
    path: str | Path,
) -> list[str]:
    """Describe DRCAT before adaptation.

    This records the original shape and explains how the DRCAT
    fields are mapped into the project's common document schema.
    """
    raw = pd.read_csv(path)

    return [
        "=== DRCAT RAW -> COMMON SCHEMA ===",
        f"Raw rows: {len(raw):,}",
        f"Raw columns: {list(raw.columns)}",
        (
            "Mapping: text->text, label->label, "
            "prompt_name->group_key, "
            "source->generator for AI rows"
        ),
        (
            "domain is set to 'student_essay'; "
            "human generator is set to 'human'."
        ),
        "",
    ]


def describe_raw_hc3(
    path: str | Path,
) -> list[str]:
    """Describe HC3 before adaptation.

    HC3 stores multiple answers inside human_answers and
    chatgpt_answers lists.

    The adapter flattens those lists so every answer becomes
    one document row.
    """
    raw = pd.read_json(
        path,
        lines=True,
    )

    human_answer_count = sum(
        len(value)
        if isinstance(value, list)
        else 0
        for value in raw["human_answers"]
    )

    ai_answer_count = sum(
        len(value)
        if isinstance(value, list)
        else 0
        for value in raw["chatgpt_answers"]
    )

    return [
        "=== HC3 RAW -> COMMON SCHEMA ===",
        f"Raw question rows: {len(raw):,}",
        f"Raw columns: {list(raw.columns)}",
        (
            "Human answers stored inside lists: "
            f"{human_answer_count:,}"
        ),
        (
            "ChatGPT answers stored inside lists: "
            f"{ai_answer_count:,}"
        ),
        (
            "Transformation: flatten each list item "
            "into its own document row."
        ),
        (
            "All answers to the same question receive "
            "the same group_key."
        ),
        (
            "source->domain; generator becomes "
            "'human' or 'chatgpt'."
        ),
        "",
    ]


def write_transformation_examples(
    drcat: pd.DataFrame,
    hc3: pd.DataFrame,
    out_path: str | Path,
    n_each: int = 4,
) -> None:
    """Save example transformed rows for inspection.

    This file is evidence/documentation only.

    It is NOT used for:
        - training
        - validation
        - testing
        - feature extraction

    It simply gives concrete examples showing that DRCAT and HC3
    ended in the same common schema.
    """
    out_path = Path(out_path)

    examples = pd.concat(
        [
            drcat.head(n_each).assign(
                transformation_note=(
                    "DRCAT: existing document row "
                    "mapped to common schema"
                )
            ),
            hc3.head(n_each).assign(
                transformation_note=(
                    "HC3: answer-list item flattened "
                    "into one document row"
                )
            ),
        ],
        ignore_index=True,
    )

    examples.to_csv(
        out_path,
        index=False,
    )


def write_report(
    report_lines: list[str],
    out_path: str | Path,
) -> None:
    """Write preprocessing information into preprocess_report.txt."""
    Path(out_path).write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )