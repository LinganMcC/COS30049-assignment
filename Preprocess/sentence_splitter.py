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

ALPHABETIC_RE = re.compile(r"[A-Za-z]")


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
            "    python -m spacy download en_core_web_sm"
        ) from exc


NLP = load_spacy_model()


def normalize_text(text: str) -> str:
    """Repair targeted missing whitespace after sentence-ending punctuation."""
    return re.sub(
        r"(?<=[a-z])\s*([.!?])(?=[A-Z])",
        r"\1 ",
        str(text),
    )


def is_usable_sentence(text: str) -> bool:
    """
    Keep a sentence only if it contains at least one alphabetic character.

    Examples removed:
        2013.
        163)
        <
        🌎
        💰🚀
    """
    text = str(text).strip()
    return bool(text and ALPHABETIC_RE.search(text))


def split_sentences(text: str, *, return_removed: bool = False):
    if text is None or not str(text).strip():
        return ([], []) if return_removed else []

    doc = NLP(normalize_text(str(text).strip()))
    kept = []
    removed = []

    for span in doc.sents:
        sentence = span.text.strip()
        if not sentence:
            continue

        if is_usable_sentence(sentence):
            kept.append(sentence)
        else:
            removed.append(sentence)

    return (kept, removed) if return_removed else kept


def documents_to_sentences(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    missing = REQUIRED_DOCUMENT_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Input file is missing columns: {sorted(missing)}")

    rows = []
    removed_rows = []

    for _, row in df.iterrows():
        sentences, removed = split_sentences(
            row["text"],
            return_removed=True,
        )

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

        for fragment in removed:
            removed_rows.append(
                {
                    "document_id": row["document_id"],
                    "fragment_text": fragment,
                    "label": int(row["label"]),
                    "source_dataset": row["source_dataset"],
                    "group_key": row["group_key"],
                    "domain": row["domain"],
                    "generator": row["generator"],
                    "removal_reason": "no_alphabetic_character",
                }
            )

    return pd.DataFrame(rows), pd.DataFrame(removed_rows)


def split_file(input_path: str | Path, output_path: str | Path) -> None:
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    documents = pd.read_csv(input_path)
    sentences, removed = documents_to_sentences(documents)

    sentences.to_csv(output_path, index=False)

    removed_path = output_path.parent / f"{output_path.stem}_removed_fragments.csv"
    removed.to_csv(removed_path, index=False)

    print(
        f"{input_path.name}: "
        f"{len(documents):,} documents -> "
        f"{len(sentences):,} usable sentences -> "
        f"{len(removed):,} removed non-lexical fragments"
    )
    print(f"  sentences -> {output_path}")
    print(f"  audit     -> {removed_path}")


def split_processed_directory(
    processed_dir: str | Path,
    out_dir: str | Path,
) -> None:
    processed_dir = Path(processed_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    filenames = [
        "drcat_train.csv",
        "drcat_validation.csv",
        "drcat_test.csv",
        "hc3_external_test.csv",
    ]

    for filename in filenames:
        source = processed_dir / filename
        if not source.exists():
            print(f"Skipping missing file: {source}")
            continue

        target = out_dir / f"{source.stem}_sentences.csv"
        split_file(source, target)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Split processed documents with spaCy and filter non-lexical fragments."
        )
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--input", help="One processed document CSV")
    mode.add_argument(
        "--processed-dir",
        help="Directory containing processed split CSV files",
    )

    parser.add_argument(
        "--output",
        help="Output CSV when using --input",
    )
    parser.add_argument(
        "--out-dir",
        default="data/sentences",
    )

    args = parser.parse_args()

    if args.input:
        if not args.output:
            parser.error("--output is required when using --input")
        split_file(args.input, args.output)
    else:
        split_processed_directory(args.processed_dir, args.out_dir)


if __name__ == "__main__":
    main()
