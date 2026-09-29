"""Sentence splitting for the processed DRCAT + HC3 document files.

IMPORTANT PROJECT RULE:
Document-level train/validation/test splitting happens BEFORE sentence splitting.
That prevents sentences from the same document/prompt leaking across partitions.

Tutor-approved repair:
Before spaCy segmentation we identify missing whitespace after sentence-ending
punctuation and repair patterns such as:
    Hello.World   -> Hello. World
    Hello .World  -> Hello. World
This targeted repair is kept intentionally because it helps spaCy recognise the
intended sentence boundary.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
import spacy


REQUIRED_DOCUMENT_COLUMNS = {
    "document_id",
    "text",
    "label",
    "source_dataset",
    "group_key",
    "domain",
    "generator",
}


def load_spacy_model():
    try:
        return spacy.load(
            "en_core_web_sm",
            disable=["ner", "lemmatizer", "attribute_ruler"],
        )
    except OSError as exc:
        raise SystemExit(
            "spaCy model 'en_core_web_sm' is not installed.\n"
            "Run:\n"
            "    python -m spacy download en_core_web_sm\n"
            "Then run this script again."
        ) from exc


NLP = load_spacy_model()


def normalize_text(text: str) -> str:
    """Repair targeted punctuation-spacing errors before spaCy splitting.

    Tutor-discussed examples:
        "Hello.World"  -> "Hello. World"
        "Hello .World" -> "Hello. World"

    We do NOT lowercase, remove punctuation, remove stopwords, stem, or
    lemmatise. This function exists only to make intended boundaries easier
    for spaCy to identify.
    """
    return re.sub(
        r"(?<=[a-z])\s*([.!?])(?=[A-Z])",
        r"\1 ",
        text,
    )


def split_sentences(text: str) -> list[str]:
    """Return spaCy sentence strings after the targeted spacing repair."""
    if text is None:
        return []

    text = str(text).strip()
    if not text:
        return []

    text = normalize_text(text)
    doc = NLP(text)

    return [
        sentence.text.strip()
        for sentence in doc.sents
        if sentence.text.strip()
    ]


def documents_to_sentences(df: pd.DataFrame) -> pd.DataFrame:
    """Convert a processed document dataframe into a sentence dataframe.

    Sentence labels are inherited from document labels. They are therefore
    weak/document-inherited labels, not independently annotated sentence-level
    ground truth.
    """
    missing = REQUIRED_DOCUMENT_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Input file is missing columns: {sorted(missing)}")

    rows: list[dict] = []

    for _, row in df.iterrows():
        sentences = split_sentences(row["text"])

        for sentence_index, sentence in enumerate(sentences):
            rows.append(
                {
                    "sentence_id": f"{row['document_id']}::s{sentence_index}",
                    "document_id": row["document_id"],
                    "sentence_index": sentence_index,
                    "text": sentence,
                    "label": int(row["label"]),
                    "label_origin": "document_inherited",
                    "source_dataset": row["source_dataset"],
                    "group_key": row["group_key"],
                    "domain": row["domain"],
                    "generator": row["generator"],
                }
            )

    return pd.DataFrame(rows)


def split_file(input_path: str | Path, output_path: str | Path) -> None:
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    documents = pd.read_csv(input_path)
    sentences = documents_to_sentences(documents)
    sentences.to_csv(output_path, index=False)

    print(
        f"{input_path.name}: {len(documents):,} documents -> "
        f"{len(sentences):,} sentences -> {output_path}"
    )


def split_processed_directory(
    processed_dir: str | Path,
    out_dir: str | Path,
) -> None:
    processed_dir = Path(processed_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = [
        "drcat_train.csv",
        "drcat_validation.csv",
        "drcat_test.csv",
        "hc3_external_test.csv",
    ]

    for filename in files:
        source = processed_dir / filename
        if not source.exists():
            print(f"Skipping missing file: {source}")
            continue

        target = out_dir / f"{source.stem}_sentences.csv"
        split_file(source, target)


def main() -> None:
    parser = argparse.ArgumentParser(description="Split processed documents with spaCy")

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--input", help="One processed document CSV")
    mode.add_argument(
        "--processed-dir",
        help="Directory containing drcat_train/validation/test and hc3_external_test",
    )

    parser.add_argument("--output", help="Output CSV when using --input")
    parser.add_argument("--out-dir", default="data/sentences")
    args = parser.parse_args()

    if args.input:
        if not args.output:
            parser.error("--output is required when using --input")
        split_file(args.input, args.output)
    else:
        split_processed_directory(args.processed_dir, args.out_dir)


if __name__ == "__main__":
    main()
