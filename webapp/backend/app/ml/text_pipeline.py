"""Text preparation: evasion normalisation, cleaning and sentence splitting.

Order of operations for one request:

    raw text
      -> normalise_for_evasion()   (web-app defence; NOT part of training)
      -> clean_text()              (copied from Assignment 2 Preprocess/data_cleaning.py)
      -> normalize_text()          (copied from Assignment 2 Preprocess/sentence_splitter.py)
      -> spaCy sentence split + is_usable_sentence()  (same as sentence_splitter.py)

Everything after the first step is exactly what the training data went
through, so the model sees sentences prepared the same way it was trained on.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# 1. Evasion normalisation (Assignment 3, section 2.8 "detection evasion")
# ---------------------------------------------------------------------------
# The feature extractor only counts ASCII letters ([A-Za-z']+). An attacker can
# keep a text looking identical to a reader while breaking every long word in
# two for the model, by inserting invisible characters or swapping Latin
# letters for look-alikes from other alphabets. These shorter "words" pull
# long_word_ratio (the strongest AI-direction feature) down, so the score
# falls toward "human". We undo those tricks before the model sees the text.

# Characters that render as nothing. Unicode category Cf ("format") covers
# zero-width space/joiners, word joiner, BOM, soft hyphen and bidi marks.
# The extra ones are invisible but sit in other categories.
_EXTRA_INVISIBLE = {
    "͏",  # combining grapheme joiner
    "ᅟ", "ᅠ", "ㅤ", "ﾠ",  # Hangul fillers
}

# Latin look-alikes from Cyrillic, Greek and other scripts. NFKC (applied later
# by clean_text) already folds full-width and "mathematical" letters, so only
# true cross-script look-alikes need listing here.
HOMOGLYPHS = {
    # Cyrillic lower case
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x",
    "ѕ": "s", "і": "i", "ј": "j", "ԁ": "d", "һ": "h", "ԛ": "q", "ԝ": "w",
    "ӏ": "l", "ь": "b", "ѵ": "v",
    # Cyrillic upper case
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H", "О": "O",
    "Р": "P", "С": "C", "Т": "T", "Х": "X", "У": "Y", "Ѕ": "S", "І": "I",
    "Ј": "J", "Ԁ": "D", "Ԛ": "Q", "Ԝ": "W", "Ӏ": "I",
    # Greek
    "ο": "o", "ν": "v", "ρ": "p", "ι": "i", "κ": "k", "α": "a", "ϲ": "c",
    "Α": "A", "Β": "B", "Ε": "E", "Ζ": "Z", "Η": "H", "Ι": "I", "Κ": "K",
    "Μ": "M", "Ν": "N", "Ο": "O", "Ρ": "P", "Τ": "T", "Υ": "Y", "Χ": "X",
    # Latin extensions and others that look like plain letters
    "ɡ": "g", "ı": "i", "ȷ": "j", "ɑ": "a", "ʏ": "y", "օ": "o", "ս": "u",
}


@dataclass
class NormalisationReport:
    invisible_removed: int = 0
    homoglyphs_replaced: int = 0
    marks_removed: int = 0

    @property
    def changed(self) -> bool:
        return bool(self.invisible_removed or self.homoglyphs_replaced or self.marks_removed)


def _is_invisible(ch: str) -> bool:
    if ch in ("\n", "\t"):
        return False
    return unicodedata.category(ch) == "Cf" or ch in _EXTRA_INVISIBLE or (
        "︀" <= ch <= "️"  # variation selectors
    )


def normalise_for_evasion(text: str) -> tuple[str, NormalisationReport]:
    """Undo encoding tricks without changing genuine text.

    Works on the text exactly as received, BEFORE any Unicode composition:
      - invisible characters are removed;
      - look-alike letters from other alphabets are mapped to Latin;
      - a separate combining accent placed straight after a plain a-z letter is
        removed. Keyboards produce accented letters as single characters
        ("é"), so genuine text is untouched; the trick of typing "a" followed
        by U+0301 to split words for the model is undone.

    Genuine accented letters (café, résumé, Việt) pass through unchanged, so the
    model sees exactly what the Assignment 2 pipeline would have produced.
    Cost: genuine Greek or Cyrillic letters are transliterated. README section 8.
    """
    report = NormalisationReport()
    out: list[str] = []
    for ch in text:
        if _is_invisible(ch):
            report.invisible_removed += 1
            continue
        if unicodedata.category(ch) == "Mn" and out and out[-1].isascii() and out[-1].isalpha():
            report.marks_removed += 1
            continue
        mapped = HOMOGLYPHS.get(ch)
        if mapped is not None:
            report.homoglyphs_replaced += 1
            out.append(mapped)
            continue
        out.append(ch)
    return "".join(out), report


# ---------------------------------------------------------------------------
# 2. Cleaning (from Assignment 2 Preprocess/data_cleaning.py)
# ---------------------------------------------------------------------------
_HTML_TAGS = (
    r"html|body|p|div|span|br|a|strong|b|em|i|ul|ol|li|h[1-6]|"
    r"blockquote|pre|code|table|thead|tbody|tr|td|th"
)

# The Assignment 2 pattern. "\s+[^<>]*?\s*" lets a run of whitespace be split
# between three parts in every possible way, so when no ">" follows, the
# regex engine tries them all: cubic time. "<p" followed by 1,500 form feeds
# takes ~6 s; 12,000 characters would take most of an hour. Kept ONLY so the
# denial-of-service demo can switch it back on (DOS_GUARD_ENABLED=false).
HTML_TAG_RE_ASSIGNMENT2 = re.compile(
    r"</?(?:" + _HTML_TAGS + r")(?:\s+[^<>]*?)?\s*/?>", flags=re.IGNORECASE
)

# Equivalent pattern without the ambiguity. Both accept exactly
#   tag + ( ">" | "/>" | whitespace + anything except < or > + ">" ),
# and gave identical output on 200,000 random tag-like strings (see tests),
# so cleaning is unchanged for every input, but matching is linear.
HTML_TAG_RE = re.compile(
    r"</?(?:" + _HTML_TAGS + r")(?:/|\s[^<>]*)?>", flags=re.IGNORECASE
)


def clean_text(text, *, linear_regex: bool = True) -> str:
    """Minimal normalisation used on every training document."""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    pattern = HTML_TAG_RE if linear_regex else HTML_TAG_RE_ASSIGNMENT2
    text = pattern.sub("", text)
    return text.strip()


# ---------------------------------------------------------------------------
# 3. Sentence splitting (copied from Assignment 2 Preprocess/sentence_splitter.py)
# ---------------------------------------------------------------------------
ALPHABETIC_RE = re.compile(r"[A-Za-z]")


def normalize_text(text: str) -> str:
    """Repair targeted missing whitespace after sentence-ending punctuation."""
    return re.sub(r"(?<=[a-z])\s*([.!?])(?=[A-Z])", r"\1 ", str(text))


def is_usable_sentence(text: str) -> bool:
    """Keep a sentence only if it contains at least one alphabetic character."""
    text = str(text).strip()
    return bool(text and ALPHABETIC_RE.search(text))


@dataclass
class Segment:
    """One piece of the displayed passage: a scored sentence or the text between."""

    text: str
    sentence_index: int | None = None  # None for gaps and unscored fragments


@dataclass
class PreparedText:
    display_text: str
    sentences: list[str] = field(default_factory=list)
    segments: list[Segment] = field(default_factory=list)
    unscored_fragments: int = 0
    normalisation: NormalisationReport = field(default_factory=NormalisationReport)


def prepare_text(
    raw: str, split_nlp, *, evasion_defence: bool = True, linear_regex: bool = True
) -> PreparedText:
    """Run the full preparation and keep character positions for highlighting.

    The passage shown to the user is rebuilt from the same string spaCy split,
    so every sentence lines up exactly with its score.
    """
    report = NormalisationReport()
    if evasion_defence:
        raw, report = normalise_for_evasion(raw)

    display = normalize_text(clean_text(raw, linear_regex=linear_regex).strip())
    prepared = PreparedText(display_text=display, normalisation=report)
    if not display:
        return prepared

    doc = split_nlp(display)
    cursor = 0
    for span in doc.sents:
        raw_span = span.text
        stripped = raw_span.strip()
        if not stripped:
            continue
        start = span.start_char + (len(raw_span) - len(raw_span.lstrip()))
        end = start + len(stripped)

        if start > cursor:
            prepared.segments.append(Segment(display[cursor:start]))

        if is_usable_sentence(stripped):
            prepared.segments.append(Segment(stripped, len(prepared.sentences)))
            prepared.sentences.append(stripped)
        else:
            prepared.segments.append(Segment(stripped))
            prepared.unscored_fragments += 1
        cursor = end

    if cursor < len(display):
        prepared.segments.append(Segment(display[cursor:]))
    return prepared
