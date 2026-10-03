"""Convert raw DRCAT and HC3 data into one common document format.

This file answers one question:
    "The datasets arrive in different shapes. How do we make them look the same?"

Both adapters return these columns:
    document_id, text, label, source_dataset, group_key, domain, generator

Labels:
    0 = human-written text
    1 = AI-generated text
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd


# Every processed document uses exactly these columns.
# Keeping the schema in one place prevents DRCAT and HC3 from slowly drifting
# into different formats as the project grows.
OUTPUT_COLUMNS = [
    "document_id",
    "text",
    "label",
    "source_dataset",
    "group_key",
    "domain",
    "generator",
]


def _stable_hash(value: str) -> str:
    """Create a repeatable short identifier from text.

    Why this helps:
        A long essay/question is awkward to use inside an ID. A hash converts it
        into a fixed-length value, and the same input always produces the same
        result. This hash is only for IDs/grouping, not for password security.

    Example:
        question = "What is Python?"
        _stable_hash(question) -> something like "a1b2c3..."
        group_key -> "hc3::a1b2c3..."
    """
    return hashlib.md5(str(value).encode("utf-8")).hexdigest()


def adapt_drcat(path: str | Path) -> pd.DataFrame:
    """Convert the raw DRCAT CSV into the project's common document schema.

    Why this helps:
        DRCAT already has one document per row, but its column meanings are not
        exactly the same as the format we want to use everywhere. This adapter
        gives it standard names and adds metadata used later for safe splitting.

    Important mapping:
        text        -> text
        label       -> label
        prompt_name -> group_key (same prompt stays together during splitting)
        source      -> generator for AI rows

    Example:
        BEFORE (DRCAT):
            text="Essay...", label=1, prompt_name="Car-free cities",
            source="chat_gpt_moth"

        AFTER:
            text="Essay...", label=1,
            source_dataset="drcat_v2",
            group_key="drcat::Car-free cities",
            domain="student_essay",
            generator="chat_gpt_moth"
    """
    df = pd.read_csv(path)

    # Fail early if somebody supplies a different/incorrect DRCAT file.
    required = {"text", "label", "prompt_name", "source"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DRCAT is missing required columns: {sorted(missing)}")

    # Convert labels to integer 0/1 and reject unexpected values.
    labels = pd.to_numeric(df["label"], errors="raise").astype(int)
    if not set(labels.unique()).issubset({0, 1}):
        raise ValueError("DRCAT label must contain only 0/1.")

    # group_key is used to prevent prompt leakage. If prompt_name is missing,
    # give that row its own fallback group instead of grouping blanks together.
    prompts = df["prompt_name"].fillna("").astype(str).str.strip()
    fallback_groups = pd.Series(
        [f"row_{i}" for i in range(len(df))], index=df.index, dtype="object"
    )
    prompts = prompts.where(prompts.ne(""), fallback_groups)

    # Human documents use generator="human". For AI documents, keep DRCAT's
    # source field because it can tell us which generator/source produced them.
    generator = pd.Series("human", index=df.index, dtype="object")
    generator.loc[labels == 1] = (
        df.loc[labels == 1, "source"].fillna("unknown_ai").astype(str)
    )

    # document_id identifies one exact row/document.
    # group_key instead identifies documents that belong to the same prompt.
    document_ids = [
        f"drcat::{_stable_hash(f'{i}|{text}')[:16]}"
        for i, text in enumerate(df["text"].fillna("").astype(str))
    ]

    return pd.DataFrame(
        {
            "document_id": document_ids,
            "text": df["text"],
            "label": labels,
            "source_dataset": "drcat_v2",
            "group_key": "drcat::" + prompts,
            "domain": "student_essay",
            "generator": generator,
        },
        columns=OUTPUT_COLUMNS,
    )


def adapt_hc3(path: str | Path) -> pd.DataFrame:
    """Flatten HC3 answer lists into one document per row.

    Why this helps:
        HC3 has a genuinely different structure from DRCAT. One HC3 row stores
        a question plus LISTS of human and ChatGPT answers. A classifier expects
        one piece of text per row, so the lists must be flattened first.

    Example:
        BEFORE (one HC3 row):
            question = "What is Python?"
            human_answers = ["Human answer A", "Human answer B"]
            chatgpt_answers = ["AI answer A"]

        AFTER (three rows):
            text="Human answer A", label=0, generator="human"
            text="Human answer B", label=0, generator="human"
            text="AI answer A",    label=1, generator="chatgpt"

        All three answers receive the SAME group_key because they answer the
        same question. That lets us keep related texts together when needed.
    """
    df = pd.read_json(path, lines=True)

    required = {"question", "human_answers", "chatgpt_answers"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"HC3 is missing required columns: {sorted(missing)}")

    rows: list[dict] = []

    for raw_index, row in df.iterrows():
        question = str(row.get("question", "") or "").strip()
        domain = str(row.get("source", "unknown") or "unknown")

        # Use the question to form the group. If a question is missing, keep
        # that raw HC3 row isolated rather than incorrectly merging it elsewhere.
        if question:
            question_hash = _stable_hash(question)
            group_key = f"hc3::{question_hash}"
        else:
            question_hash = f"row_{raw_index}"
            group_key = f"hc3::{question_hash}"

        human_answers = row.get("human_answers", [])
        if not isinstance(human_answers, list):
            human_answers = []

        # Each item in human_answers becomes its own document row.
        for answer_index, answer in enumerate(human_answers):
            rows.append(
                {
                    "document_id": f"hc3::{question_hash}::human::{answer_index}",
                    "text": answer,
                    "label": 0,
                    "source_dataset": "hc3",
                    "group_key": group_key,
                    "domain": domain,
                    "generator": "human",
                }
            )

        ai_answers = row.get("chatgpt_answers", [])
        if not isinstance(ai_answers, list):
            ai_answers = []

        # Each item in chatgpt_answers becomes its own AI document row.
        for answer_index, answer in enumerate(ai_answers):
            rows.append(
                {
                    "document_id": f"hc3::{question_hash}::chatgpt::{answer_index}",
                    "text": answer,
                    "label": 1,
                    "source_dataset": "hc3",
                    "group_key": group_key,
                    "domain": domain,
                    "generator": "chatgpt",
                }
            )

    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
