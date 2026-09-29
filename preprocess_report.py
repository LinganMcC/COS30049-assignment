"""Small reporting helpers for the preprocessing pipeline.

This file does NOT change the training data. It records what happened to explain and document the preprocessing/transformation steps.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def summarize_dataset(df: pd.DataFrame) -> str:
    """Return a compact human-readable summary of one processed dataset.

    Why this helps:
        Instead of printing a full DataFrame, the report can quickly show the
        number of rows, human/AI balance, and number of groups.

    Example output:
        rows=10,000, AI=5,200, human=4,800, pct_ai=0.5200, groups=15
    """
    if len(df) == 0:
        return "rows=0"

    return (
        f"rows={len(df):,}, AI={df['label'].sum():,}, "
        f"human={(df['label'] == 0).sum():,}, pct_ai={df['label'].mean():.4f}, "
        f"groups={df['group_key'].nunique():,}"
    )


def describe_raw_drcat(path: str | Path) -> list[str]:
    """Create report lines describing DRCAT before adaptation.

    Why this helps:
        It gives evidence of the original column shape and documents how DRCAT
        is mapped into the common schema.
    """
    raw = pd.read_csv(path)

    return [
        "=== DRCAT RAW -> COMMON SCHEMA ===",
        f"Raw rows: {len(raw):,}",
        f"Raw columns: {list(raw.columns)}",
        "Mapping: text->text, label->label, prompt_name->group_key, source->generator for AI rows",
        "domain is set to 'student_essay'; human generator is set to 'human'.",
        "",
    ]


def describe_raw_hc3(path: str | Path) -> list[str]:
    """Create report lines showing why HC3 needs a real transformation.

    Why this helps:
        HC3 stores answers inside lists, unlike DRCAT. Counting those list items
        makes the before/after transformation easier to explain in the report.

    Example:
        Raw rows may be 24,000 questions, but there can be many more answer
        documents after human_answers/chatgpt_answers are flattened.
    """
    raw = pd.read_json(path, lines=True)

    human_answer_count = sum(
        len(x) if isinstance(x, list) else 0 for x in raw["human_answers"]
    )
    ai_answer_count = sum(
        len(x) if isinstance(x, list) else 0 for x in raw["chatgpt_answers"]
    )

    return [
        "=== HC3 RAW -> COMMON SCHEMA ===",
        f"Raw question rows: {len(raw):,}",
        f"Raw columns: {list(raw.columns)}",
        f"Human answers stored inside lists: {human_answer_count:,}",
        f"ChatGPT answers stored inside lists: {ai_answer_count:,}",
        "Transformation: flatten each list item into its own document row.",
        "All answers to the same question receive the same group_key.",
        "source->domain; generator becomes 'human' or 'chatgpt'.",
        "",
    ]


def write_transformation_examples(
    drcat: pd.DataFrame,
    hc3: pd.DataFrame,
    out_path: str | Path,
    n_each: int = 4,
) -> None:
    """Save a few processed rows as evidence of the two dataset transformations.

    Why this helps:
        transformation_examples.csv is easy for a tutor/team member to inspect.
        It demonstrates that both datasets end in the same column format even
        though HC3 started as question rows containing answer lists.

    Example:
        first 4 DRCAT processed rows + first 4 HC3 processed rows are saved with
        a transformation_note explaining where each type came from.
    """
    out_path = Path(out_path)

    examples = pd.concat(
        [
            drcat.head(n_each).assign(
                transformation_note="DRCAT: one document row -> common schema"
            ),
            hc3.head(n_each).assign(
                transformation_note="HC3: answer-list item flattened -> one answer row"
            ),
        ],
        ignore_index=True,
    )

    examples.to_csv(out_path, index=False)


def write_report(report_lines: list[str], out_path: str | Path) -> None:
    """Write all collected preprocessing notes into one plain-text report.

    Why this helps:
        preprocess.py can focus on running the pipeline while this helper handles
        the final evidence file. The report stays in data/processed/ with the
        datasets it describes.
    """
    Path(out_path).write_text("\n".join(report_lines), encoding="utf-8")
