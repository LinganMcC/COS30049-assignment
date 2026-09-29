"""Shared cleaning and duplicate checks for processed document data.

This file answers one question:
    "Now that DRCAT and HC3 have the same columns, how do we clean them safely?"

The cleaning is deliberately conservative because punctuation, casing and other
writing-style signals may later become useful AI-detection features.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata

import pandas as pd


# Remove normal HTML tags but do NOT blindly remove every <...> pattern.
# For example, <p>Hello</p> is HTML and can be removed, while a string such as
# <http://example.com> should not be treated as an HTML tag.
HTML_TAG_RE = re.compile(
    r"</?(?:html|body|p|div|span|br|a|strong|b|em|i|ul|ol|li|h[1-6]|"
    r"blockquote|pre|code|table|thead|tbody|tr|td|th)(?:\s+[^<>]*?)?\s*/?>",
    flags=re.IGNORECASE,
)


def _duplicate_hash(text: str) -> str:
    """Create a normalised fingerprint used only for duplicate comparison.

    Why this helps:
        Two rows such as "Hello world" and "  hello world  " should count as
        the same exact-ish text for duplicate checking. We lowercase and trim
        ONLY inside this comparison helper; the real text is not lowercased.

    Example:
        "Hello World" -> same fingerprint as " hello world "
    """
    normalised = str(text).strip().lower()
    return hashlib.md5(normalised.encode("utf-8")).hexdigest()


def clean_text(text) -> str:
    """Perform minimal document-level text cleaning.

    Why this helps:
        Raw datasets can contain inconsistent Unicode, Windows line endings,
        repeated spaces or HTML tags. Cleaning those makes later processing more
        consistent without destroying writing-style information.

    What we DO:
        - Unicode NFKC normalisation
        - normalise line endings
        - collapse repeated spaces/tabs
        - collapse 3+ blank lines to 2
        - remove conservative HTML tags

    What we intentionally DO NOT do:
        - lowercase the text
        - remove punctuation
        - remove stopwords
        - stem or lemmatise
        - repair sentence punctuation spacing

    The sentence-boundary repair discussed with your tutor stays in
    sentence_splitter.py immediately before spaCy segmentation.

    Example:
        "<p>Hello   world</p>\r\n\r\n\r\nNext"
        -> "Hello world\n\nNext"
    """
    if not isinstance(text, str):
        return ""

    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = HTML_TAG_RE.sub("", text)
    return text.strip()


def clean_frame(
    df: pd.DataFrame,
    *,
    min_chars: int = 30,
    max_chars: int = 20000,
) -> tuple[pd.DataFrame, dict]:
    """Clean every document and remove texts outside the allowed length range.

    Why this helps:
        Empty/tiny records often do not contain enough writing evidence, while
        extremely large records can make feature extraction unnecessarily slow.
        The function also returns counts so the preprocessing report can show
        exactly how much data was removed.

    Example:
        text length 10     -> removed when min_chars=30
        text length 1,000  -> kept
        text length 25,000 -> removed when max_chars=20,000
    """
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


def deduplicate_labeled(
    df: pd.DataFrame,
    dataset_name: str,
) -> tuple[pd.DataFrame, dict]:
    """Remove conflicting duplicate texts, then ordinary same-label duplicates.

    Why this helps:
        Duplicate examples can make evaluation look better than it really is.
        Worse, if exactly the same text appears once as human and once as AI,
        there is no trustworthy label to learn from. In that case we remove all
        conflicting copies instead of arbitrarily choosing one label.

    Example 1 - same label:
        "Example text", label=0
        "Example text", label=0
        -> keep one row

    Example 2 - conflicting labels:
        "Example text", label=0
        "Example text", label=1
        -> remove both rows
    """
    out = df.copy()
    out["_text_hash"] = out["text"].apply(_duplicate_hash)

    # Find texts associated with more than one label.
    label_counts = out.groupby("_text_hash")["label"].nunique()
    conflicting_hashes = set(label_counts[label_counts > 1].index)

    conflict_rows = int(out["_text_hash"].isin(conflicting_hashes).sum())
    if conflicting_hashes:
        out = out.loc[~out["_text_hash"].isin(conflicting_hashes)].copy()

    # After conflicts are gone, ordinary repeated copies can safely be reduced
    # to one row because they all represent the same cleaned text/label.
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
    drcat: pd.DataFrame,
    hc3: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    """Remove HC3 texts that already occur in DRCAT.

    Why this helps:
        HC3 is being used as an external unseen test set. If an HC3 text is also
        present in DRCAT, the model may already have seen that exact document
        during development. Removing overlap makes the external test cleaner.

    Example:
        DRCAT contains "The sky is blue..."
        HC3 also contains exactly "The sky is blue..."
        -> remove that HC3 row before building the external test.
    """
    seen = set(drcat["text"].apply(_duplicate_hash))

    hc3_work = hc3.copy()
    hc3_work["_text_hash"] = hc3_work["text"].apply(_duplicate_hash)

    overlap_mask = hc3_work["_text_hash"].isin(seen)
    removed = int(overlap_mask.sum())

    hc3_work = hc3_work.loc[~overlap_mask].drop(columns="_text_hash")
    return hc3_work.reset_index(drop=True), removed
