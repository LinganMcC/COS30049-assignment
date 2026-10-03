"""Cleaning helpers for DRCAT and HC3.

The preprocessing is intentionally conservative because punctuation, casing,
sentence structure and other writing-style signals may be useful for
AI-generated text detection.

Current project decisions:
    - DRCAT:
        * clean text
        * remove same-label duplicate documents
        * DO NOT filter by document length

    - HC3:
        * clean text
        * remove documents outside the allowed length range
        * remove same-label duplicate documents

We do NOT:
    - remove conflicting-label duplicates
    - check/remove overlap between DRCAT and HC3
    - lowercase text
    - remove punctuation
    - remove stopwords
    - stem or lemmatise
    - repair sentence boundaries here
"""

from __future__ import annotations

import hashlib
import re
import unicodedata

import pandas as pd


HTML_TAG_RE = re.compile(
    r"</?(?:html|body|p|div|span|br|a|strong|b|em|i|ul|ol|li|h[1-6]|"
    r"blockquote|pre|code|table|thead|tbody|tr|td|th)(?:\s+[^<>]*?)?\s*/?>",
    flags=re.IGNORECASE,
)


def _duplicate_hash(text: str) -> str:
    """Create a stable hash used only for duplicate detection.

    Text is stripped and lowercased only for the duplicate comparison.

    This does NOT modify the text stored in the dataset.

    Example:
        "Hello world"
        "  hello world  "

    produce the same duplicate hash.
    """
    normalised = str(text).strip().lower()

    return hashlib.md5(
        normalised.encode("utf-8")
    ).hexdigest()


def clean_text(text) -> str:
    """Perform minimal text normalisation.

    What we do:
        - Unicode NFKC normalisation
        - normalise line endings
        - collapse repeated spaces/tabs
        - collapse excessive blank lines
        - remove conservative HTML tags
        - trim leading/trailing whitespace

    What we do NOT do:
        - lowercase
        - remove punctuation
        - remove stopwords
        - stem
        - lemmatise
        - remove documents by length
        - remove duplicate documents
        - repair sentence boundaries

    Example:
        "<p>Hello   world</p>\\r\\n\\r\\n\\r\\nNext"

        becomes:

        "Hello world\\n\\nNext"
    """
    if not isinstance(text, str):
        return ""

    text = unicodedata.normalize("NFKC", text)

    text = text.replace("\r\n", "\n").replace("\r", "\n")

    text = re.sub(r"[ \t]+", " ", text)

    text = re.sub(r"\n{3,}", "\n\n", text)

    text = HTML_TAG_RE.sub("", text)

    return text.strip()


def clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Apply clean_text() to every document.

    This function does NOT remove rows.

    It is used for both DRCAT and HC3.
    """
    out = df.copy()

    out["text"] = out["text"].apply(clean_text)

    return out


def filter_document_length(
    df: pd.DataFrame,
    *,
    min_chars: int = 30,
    max_chars: int = 20000,
) -> tuple[pd.DataFrame, dict]:
    """Remove documents outside the allowed character-length range.

    Current project decision:
        This function is used for HC3 only.

        DRCAT is the main development dataset and is NOT filtered
        by document length.

    Example:
        length 10
            -> removed

        length 1,000
            -> kept

        length 25,000
            -> removed

    The returned statistics are used in preprocess_report.txt.
    """
    out = df.copy()

    before = len(out)

    lengths = out["text"].str.len()

    valid = (
        (lengths >= min_chars)
        & (lengths <= max_chars)
    )

    out = out.loc[valid].reset_index(drop=True)

    stats = {
        "before": before,
        "after": len(out),
        "dropped_length_or_empty": before - len(out),
    }

    return out, stats


def deduplicate_same_label(
    df: pd.DataFrame,
    dataset_name: str,
) -> tuple[pd.DataFrame, dict]:
    """Remove repeated copies of the same text with the same label.

    Example:

        "Example text", label=0
        "Example text", label=0

    becomes:

        "Example text", label=0


    Important:
        This function does NOT search for or remove conflicting labels.

    Therefore:

        "Example text", label=0
        "Example text", label=1

    would both remain.

    We currently do not need special conflicting-label handling because
    no conflicting duplicate rows were found in the current DRCAT/HC3 data.
    """
    out = df.copy()

    out["_text_hash"] = out["text"].apply(_duplicate_hash)

    before = len(out)

    # A duplicate must have BOTH:
    #     same text
    #     same label
    #
    # Therefore a text with label=0 and label=1 is not removed here.
    out = out.drop_duplicates(
        subset=["_text_hash", "label"],
        keep="first",
    )

    removed = before - len(out)

    out = (
        out
        .drop(columns="_text_hash")
        .reset_index(drop=True)
    )

    stats = {
        "dataset": dataset_name,
        "same_label_duplicate_rows_removed": removed,
    }

    return out, stats