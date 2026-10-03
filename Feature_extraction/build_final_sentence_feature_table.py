"""Build the frozen 15-feature sentence table.

Expected input columns:
    sentence_id, document_id, sentence_index, text, label, group_key

The script does not re-split data and does not run any language model.
It extracts only the frozen 9 stylometric/lexical + 6 spaCy POS features.

Example:
    python build_final_sentence_feature_table.py \
      --input data/sentences/drcat_train_sentences.csv \
      --output data/features/drcat_train_final_sentence_features.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import spacy

from feature_config import FINAL_FEATURES
from features import count_words, extract_pos_features, extract_stylometry


REQUIRED_COLUMNS = {
    "sentence_id",
    "document_id",
    "sentence_index",
    "text",
    "label",
    "group_key",
}


def load_spacy_model(model_name: str):
    """Load the POS tagger while disabling components not used here."""
    try:
        return spacy.load(
            model_name,
            disable=["ner", "lemmatizer"],
        )
    except OSError as exc:
        raise SystemExit(
            f"spaCy model '{model_name}' is missing.\n"
            f"Run:\n"
            f"    python -m spacy download {model_name}"
        ) from exc


def build_feature_table(
    df: pd.DataFrame,
    *,
    spacy_model: str = "en_core_web_sm",
    spacy_batch_size: int = 256,
    spacy_processes: int = 1,
) -> pd.DataFrame:
    """Return the input sentence table plus the frozen 15 features."""
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            "Input sentence CSV is missing columns: "
            f"{sorted(missing)}"
        )

    df = df.copy().reset_index(drop=True)
    df["text"] = df["text"].fillna("").astype(str)
    df["label"] = pd.to_numeric(df["label"], errors="raise").astype(int)

    if not set(df["label"].unique()).issubset({0, 1}):
        raise ValueError("label must contain only 0/1")

    # ----------------------------------------------------------
    # 1. Stylometric/lexical features
    # ----------------------------------------------------------
    print("Calculating 9 stylometric/lexical features ...")

    actual_word_count = df["text"].map(count_words)
    doc_avg_sentence_words = (
        pd.DataFrame(
            {
                "document_id": df["document_id"],
                "word_count": actual_word_count,
            }
        )
        .groupby("document_id")["word_count"]
        .transform("mean")
    )

    stylometry_df = pd.DataFrame(
        [
            extract_stylometry(text, float(doc_avg))
            for text, doc_avg in zip(
                df["text"],
                doc_avg_sentence_words,
            )
        ]
    )

    # ----------------------------------------------------------
    # 2. POS-profile features
    # ----------------------------------------------------------
    print(f"spaCy POS scoring {len(df):,} sentences ...")
    nlp = load_spacy_model(spacy_model)
    pos_rows: list[dict[str, float]] = []

    for i, doc in enumerate(
        nlp.pipe(
            df["text"].tolist(),
            batch_size=spacy_batch_size,
            n_process=spacy_processes,
        ),
        start=1,
    ):
        pos_rows.append(extract_pos_features(doc))

        if i % 25000 == 0:
            print(f"spaCy scored {i:,}/{len(df):,}")

    pos_df = pd.DataFrame(pos_rows)

    # ----------------------------------------------------------
    # 3. Final table + safety checks
    # ----------------------------------------------------------
    result = pd.concat(
        [
            df.reset_index(drop=True),
            stylometry_df.reset_index(drop=True),
            pos_df.reset_index(drop=True),
        ],
        axis=1,
    )

    missing_final = [name for name in FINAL_FEATURES if name not in result.columns]
    if missing_final:
        raise ValueError(f"Missing final features: {missing_final}")

    numeric = result[FINAL_FEATURES].apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=float)

    inf_count = int(np.isinf(values).sum())
    nan_count = int(numeric.isna().sum().sum())

    if inf_count:
        raise ValueError(f"Found {inf_count} infinite model-feature values")
    if nan_count:
        raise ValueError(f"Found {nan_count} missing model-feature values")

    return result


def main(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    result = build_feature_table(
        df,
        spacy_model=args.spacy_model,
        spacy_batch_size=args.spacy_batch_size,
        spacy_processes=args.spacy_processes,
    )

    result.to_csv(output_path, index=False)

    print("\n=== FINAL 15-FEATURE TABLE ===")
    print(f"Rows: {len(result):,}")
    print(f"Documents: {result['document_id'].nunique():,}")
    print(f"Groups: {result['group_key'].nunique():,}")
    print(f"Model features: {len(FINAL_FEATURES)}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build the frozen 15-feature sentence table."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--spacy-model", default="en_core_web_sm")
    parser.add_argument("--spacy-batch-size", type=int, default=256)
    parser.add_argument("--spacy-processes", type=int, default=1)
    args = parser.parse_args()

    if args.spacy_batch_size < 1:
        parser.error("--spacy-batch-size must be >= 1")
    if args.spacy_processes < 1:
        parser.error("--spacy-processes must be >= 1")

    main(args)
