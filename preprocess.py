"""Preprocess DRCAT + HC3 for the AI-generated content detection project.

Design:
- DRCAT is the development dataset (train / validation / internal test).
- HC3 is transformed into the same schema, but kept as an external unseen test set.
- Splitting happens at DOCUMENT level before sentence splitting.
- `group_key` is used only to prevent leakage; it is NOT a model feature.

Typical run:
    python preprocess.py \
        --drcat data/raw/train_v2_drcat_02.csv \
        --hc3 data/raw/hc3_all.jsonl \
        --out-dir data/processed
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import re
import unicodedata
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


OUTPUT_COLUMNS = [
    "document_id",
    "text",
    "label",
    "source_dataset",
    "group_key",
    "domain",
    "generator",
]

# Conservative HTML removal. It removes normal-looking tags such as <p>, <div>,
# <br>, etc., while leaving URL-like angle-bracket strings such as
# <http://example.com> alone.
HTML_TAG_RE = re.compile(
    r"</?(?:html|body|p|div|span|br|a|strong|b|em|i|ul|ol|li|h[1-6]|"
    r"blockquote|pre|code|table|thead|tbody|tr|td|th)(?:\s+[^<>]*?)?\s*/?>",
    flags=re.IGNORECASE,
)


def _stable_hash(value: str) -> str:
    """Stable MD5 used only for IDs / duplicate checks, not security."""
    return hashlib.md5(str(value).encode("utf-8")).hexdigest()


def _duplicate_hash(text: str) -> str:
    """Normalised hash used for exact-ish duplicate detection."""
    return _stable_hash(str(text).strip().lower())


def clean_text(text) -> str:
    """Minimal document cleaning.

    We deliberately DO NOT lowercase, remove punctuation, remove stopwords,
    stem, or lemmatise because writing-style signals are useful for this task.

    Sentence-boundary spacing repair is intentionally NOT done here. Your
    tutor-approved repair belongs in sentence_splitter.py immediately before
    spaCy segmentation.
    """
    if not isinstance(text, str):
        return ""

    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = HTML_TAG_RE.sub("", text)
    return text.strip()


def adapt_drcat(path: str | Path) -> pd.DataFrame:
    """Transform train_v2_drcat_02.csv into the common document schema."""
    df = pd.read_csv(path)

    required = {"text", "label", "prompt_name", "source"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DRCAT is missing required columns: {sorted(missing)}")

    labels = pd.to_numeric(df["label"], errors="raise").astype(int)
    if not set(labels.unique()).issubset({0, 1}):
        raise ValueError("DRCAT label must contain only 0/1.")

    prompts = df["prompt_name"].fillna("").astype(str).str.strip()
    fallback_groups = pd.Series(
        [f"row_{i}" for i in range(len(df))], index=df.index, dtype="object"
    )
    prompts = prompts.where(prompts.ne(""), fallback_groups)

    generator = pd.Series("human", index=df.index, dtype="object")
    generator.loc[labels == 1] = (
        df.loc[labels == 1, "source"].fillna("unknown_ai").astype(str)
    )

    # Unique document ID is different from group_key:
    # - document_id identifies one document.
    # - group_key groups documents sharing the same prompt/topic.
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
    """Flatten HC3 from one-question-with-answer-lists into one answer per row.

    HC3 raw shape:
        question
        human_answers: [answer1, answer2, ...]
        chatgpt_answers: [answer1, ...]
        source

    Common shape:
        one answer per row with label 0/1.

    Every response to the same question receives the same group_key.
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

        if question:
            question_hash = _stable_hash(question)
            group_key = f"hc3::{question_hash}"
        else:
            question_hash = f"row_{raw_index}"
            group_key = f"hc3::{question_hash}"

        human_answers = row.get("human_answers", [])
        if not isinstance(human_answers, list):
            human_answers = []

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


def clean_frame(
    df: pd.DataFrame,
    *,
    min_chars: int = 30,
    max_chars: int = 20000,
) -> tuple[pd.DataFrame, dict]:
    """Apply shared cleaning and length filtering."""
    out = df.copy()
    before = len(out)
    out["text"] = out["text"].apply(clean_text)

    lengths = out["text"].str.len()
    valid = (lengths >= min_chars) & (lengths <= max_chars)
    out = out.loc[valid].reset_index(drop=True)

    stats = {
        "before": before,
        "after": len(out),
        "dropped_length_or_empty": before - len(out),
    }
    return out, stats


def deduplicate_labeled(df: pd.DataFrame, dataset_name: str) -> tuple[pd.DataFrame, dict]:
    """Remove conflicting duplicates, then ordinary duplicates.

    If the same cleaned text appears with both labels, all conflicting copies
    are removed rather than arbitrarily trusting whichever row appears first.
    """
    out = df.copy()
    out["_text_hash"] = out["text"].apply(_duplicate_hash)

    label_counts = out.groupby("_text_hash")["label"].nunique()
    conflicting_hashes = set(label_counts[label_counts > 1].index)

    conflict_rows = int(out["_text_hash"].isin(conflicting_hashes).sum())
    if conflicting_hashes:
        out = out.loc[~out["_text_hash"].isin(conflicting_hashes)].copy()

    before_exact = len(out)
    out = out.drop_duplicates(subset="_text_hash", keep="first")
    exact_duplicate_rows = before_exact - len(out)

    out = out.drop(columns="_text_hash").reset_index(drop=True)

    return out, {
        "dataset": dataset_name,
        "conflicting_duplicate_rows_removed": conflict_rows,
        "same_label_duplicate_rows_removed": exact_duplicate_rows,
    }


def remove_hc3_overlap_with_drcat(
    drcat: pd.DataFrame, hc3: pd.DataFrame
) -> tuple[pd.DataFrame, int]:
    """Protect the external HC3 test from exact text overlap with DRCAT."""
    seen = set(drcat["text"].apply(_duplicate_hash))
    hc3_work = hc3.copy()
    hc3_work["_text_hash"] = hc3_work["text"].apply(_duplicate_hash)
    overlap_mask = hc3_work["_text_hash"].isin(seen)
    removed = int(overlap_mask.sum())
    hc3_work = hc3_work.loc[~overlap_mask].drop(columns="_text_hash")
    return hc3_work.reset_index(drop=True), removed


def _split_score(
    splits: Iterable[tuple[np.ndarray, float]],
    row_counts: np.ndarray,
    ai_counts: np.ndarray,
    total_rows: int,
    overall_ai_rate: float,
) -> float:
    """Score a candidate group split by size balance + label balance."""
    score = 0.0
    for positions, target_fraction in splits:
        rows = row_counts[positions].sum()
        if rows == 0:
            return float("inf")
        ai_rate = ai_counts[positions].sum() / rows
        score += abs(rows / total_rows - target_fraction)
        score += 1.5 * abs(ai_rate - overall_ai_rate)
    return float(score)


def balanced_group_split_drcat(
    df: pd.DataFrame,
    *,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
    search_trials: int = 30000,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split DRCAT by whole prompt groups while keeping label ratios similar.

    DRCAT has relatively few prompt groups, so a plain random group split can
    accidentally make validation/test heavily human or AI. This searches many
    group assignments and keeps the best size/label-balanced candidate.
    """
    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError("train_size + val_size + test_size must equal 1.0")

    stats = df.groupby("group_key")["label"].agg(["count", "sum"])
    groups = stats.index.to_numpy()
    row_counts = stats["count"].to_numpy()
    ai_counts = stats["sum"].to_numpy()
    n_groups = len(groups)

    if n_groups < 3:
        raise ValueError("At least three group_key values are required.")

    n_test_groups = max(1, round(n_groups * test_size))
    n_val_groups = max(1, round(n_groups * val_size))
    if n_test_groups + n_val_groups >= n_groups:
        n_test_groups = 1
        n_val_groups = 1

    total_rows = int(row_counts.sum())
    overall_ai_rate = float(ai_counts.sum() / total_rows)
    rng = np.random.default_rng(seed)

    best: tuple | None = None

    for _ in range(search_trials):
        perm = rng.permutation(n_groups)
        test_pos = perm[:n_test_groups]
        val_pos = perm[n_test_groups : n_test_groups + n_val_groups]
        train_pos = perm[n_test_groups + n_val_groups :]

        score = _split_score(
            [
                (train_pos, train_size),
                (val_pos, val_size),
                (test_pos, test_size),
            ],
            row_counts,
            ai_counts,
            total_rows,
            overall_ai_rate,
        )

        if best is None or score < best[0]:
            best = (score, train_pos.copy(), val_pos.copy(), test_pos.copy())

    assert best is not None
    _, train_pos, val_pos, test_pos = best

    train_groups = set(groups[train_pos])
    val_groups = set(groups[val_pos])
    test_groups = set(groups[test_pos])

    train = df[df["group_key"].isin(train_groups)].reset_index(drop=True)
    val = df[df["group_key"].isin(val_groups)].reset_index(drop=True)
    test = df[df["group_key"].isin(test_groups)].reset_index(drop=True)

    # Leakage checks.
    assert set(train["group_key"]).isdisjoint(val["group_key"])
    assert set(train["group_key"]).isdisjoint(test["group_key"])
    assert set(val["group_key"]).isdisjoint(test["group_key"])

    return train, val, test


def build_balanced_hc3_external_test(
    hc3: pd.DataFrame,
    *,
    n_questions: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Create a fair external test: one human + one ChatGPT answer/question.

    This prevents questions with many human answers from dominating and keeps
    human/AI text matched to the same sampled question set.
    """
    label_variety = hc3.groupby("group_key")["label"].nunique()
    eligible_groups = label_variety[label_variety >= 2].index.to_numpy()

    if len(eligible_groups) == 0:
        raise ValueError("No HC3 questions contain both human and AI answers.")

    rng = np.random.default_rng(seed)
    rng.shuffle(eligible_groups)

    if n_questions <= 0:
        selected = set(eligible_groups)
    else:
        selected = set(eligible_groups[: min(n_questions, len(eligible_groups))])

    subset = hc3[hc3["group_key"].isin(selected)].copy()

    human = (
        subset[subset["label"] == 0]
        .groupby("group_key", group_keys=False)
        .sample(n=1, random_state=seed)
    )
    ai = (
        subset[subset["label"] == 1]
        .groupby("group_key", group_keys=False)
        .sample(n=1, random_state=seed + 1)
    )

    result = pd.concat([human, ai], ignore_index=True)
    return result.sample(frac=1, random_state=seed).reset_index(drop=True)


def _summary(df: pd.DataFrame) -> str:
    if len(df) == 0:
        return "rows=0"
    return (
        f"rows={len(df):,}, AI={df['label'].sum():,}, "
        f"human={(df['label'] == 0).sum():,}, pct_ai={df['label'].mean():.4f}, "
        f"groups={df['group_key'].nunique():,}"
    )


def write_transformation_examples(
    drcat: pd.DataFrame,
    hc3: pd.DataFrame,
    out_path: Path,
    n_each: int = 4,
) -> None:
    examples = pd.concat(
        [
            drcat.head(n_each).assign(transformation_note="DRCAT: one document row -> common schema"),
            hc3.head(n_each).assign(
                transformation_note="HC3: answer-list item flattened -> one answer row"
            ),
        ],
        ignore_index=True,
    )
    examples.to_csv(out_path, index=False)


def build_datasets(
    *,
    drcat_path: str | Path,
    hc3_path: str | Path,
    out_dir: str | Path,
    min_chars: int = 30,
    max_chars: int = 20000,
    hc3_questions: int = 1000,
    seed: int = 42,
) -> dict[str, Path]:
    """Run the whole document-level preprocessing pipeline."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report_lines: list[str] = []

    # -----------------------------
    # DRCAT
    # -----------------------------
    raw_drcat = pd.read_csv(drcat_path)
    report_lines += [
        "=== DRCAT RAW -> COMMON SCHEMA ===",
        f"Raw rows: {len(raw_drcat):,}",
        f"Raw columns: {list(raw_drcat.columns)}",
        "Mapping: text->text, label->label, prompt_name->group_key, source->generator for AI rows",
        "domain is set to 'student_essay'; human generator is set to 'human'.",
        "",
    ]

    drcat = adapt_drcat(drcat_path)
    drcat, drcat_clean_stats = clean_frame(
        drcat, min_chars=min_chars, max_chars=max_chars
    )
    drcat, drcat_dedup_stats = deduplicate_labeled(drcat, "DRCAT")

    report_lines += [
        "=== DRCAT CLEANING ===",
        str(drcat_clean_stats),
        str(drcat_dedup_stats),
        f"After preprocessing: {_summary(drcat)}",
        "",
    ]

    # -----------------------------
    # HC3
    # -----------------------------
    raw_hc3 = pd.read_json(hc3_path, lines=True)
    human_answer_count = sum(
        len(x) if isinstance(x, list) else 0 for x in raw_hc3["human_answers"]
    )
    ai_answer_count = sum(
        len(x) if isinstance(x, list) else 0 for x in raw_hc3["chatgpt_answers"]
    )

    report_lines += [
        "=== HC3 RAW -> COMMON SCHEMA ===",
        f"Raw question rows: {len(raw_hc3):,}",
        f"Raw columns: {list(raw_hc3.columns)}",
        f"Human answers stored inside lists: {human_answer_count:,}",
        f"ChatGPT answers stored inside lists: {ai_answer_count:,}",
        "Transformation: flatten each list item into its own document row.",
        "All answers to the same question receive the same group_key.",
        "source->domain; generator becomes 'human' or 'chatgpt'.",
        "",
    ]

    hc3 = adapt_hc3(hc3_path)
    hc3, hc3_clean_stats = clean_frame(hc3, min_chars=min_chars, max_chars=max_chars)
    hc3, hc3_dedup_stats = deduplicate_labeled(hc3, "HC3")
    hc3, overlap_removed = remove_hc3_overlap_with_drcat(drcat, hc3)

    report_lines += [
        "=== HC3 CLEANING ===",
        str(hc3_clean_stats),
        str(hc3_dedup_stats),
        f"HC3 rows removed because exact text also existed in DRCAT: {overlap_removed:,}",
        f"After preprocessing: {_summary(hc3)}",
        "",
    ]

    # -----------------------------
    # Unified file for inspection
    # -----------------------------
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

    # -----------------------------
    # DRCAT leakage-safe split
    # -----------------------------
    drcat_train, drcat_val, drcat_test = balanced_group_split_drcat(
        drcat, seed=seed
    )

    train_path = out_dir / "drcat_train.csv"
    val_path = out_dir / "drcat_validation.csv"
    test_path = out_dir / "drcat_test.csv"

    drcat_train.to_csv(train_path, index=False)
    drcat_val.to_csv(val_path, index=False)
    drcat_test.to_csv(test_path, index=False)

    report_lines += [
        "=== DRCAT PROMPT-GROUP SPLIT ===",
        f"Train: {_summary(drcat_train)}",
        f"Validation: {_summary(drcat_val)}",
        f"Test: {_summary(drcat_test)}",
        "Leakage rule: the same DRCAT prompt/group_key never appears in more than one split.",
        "",
    ]

    # -----------------------------
    # HC3 external set
    # -----------------------------
    hc3_external = build_balanced_hc3_external_test(
        hc3, n_questions=hc3_questions, seed=seed
    )
    hc3_external_path = out_dir / "hc3_external_test.csv"
    hc3_external.to_csv(hc3_external_path, index=False)

    report_lines += [
        "=== HC3 EXTERNAL UNSEEN TEST ===",
        f"External test: {_summary(hc3_external)}",
        f"Requested question groups: {hc3_questions if hc3_questions > 0 else 'ALL'}",
        "Sampling rule: one human answer + one ChatGPT answer per selected question.",
        "HC3 is not used to tune features/model if you want it to remain a true external test.",
        "",
    ]

    # -----------------------------
    # Evidence files
    # -----------------------------
    examples_path = out_dir / "transformation_examples.csv"
    write_transformation_examples(drcat, hc3, examples_path)

    report_path = out_dir / "preprocess_report.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

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

    print("\nIMPORTANT: train/tune on DRCAT train+validation. Keep HC3 external until final evaluation.")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess DRCAT + HC3")
    parser.add_argument("--drcat", required=True, help="Path to train_v2_drcat_02.csv")
    parser.add_argument("--hc3", required=True, help="Path to HC3 all.jsonl")
    parser.add_argument("--out-dir", default="data/processed")
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
