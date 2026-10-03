"""Final 15 sentence-level features used by the AI-content detector.

This module contains only the frozen production feature definitions:
    - 9 stylometric/lexical features
    - 6 spaCy POS-profile features

No GPT-2, perplexity, NLL, or GLTR computation is performed.
"""

from __future__ import annotations

import re

from feature_config import (
    SELECTED_POS_FEATURES,
    SELECTED_STYLOMETRY_FEATURES,
)


TRANSITION_WORDS = {
    "additionally", "moreover", "furthermore", "however", "therefore",
    "consequently", "overall", "in conclusion", "in summary", "ultimately",
    "nonetheless", "nevertheless", "thus", "hence", "meanwhile",
    "in addition", "as a result", "for instance", "for example",
    "on the other hand", "in contrast", "notably", "importantly",
}

FUNCTION_WORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "so", "because",
    "of", "in", "on", "at", "to", "for", "with", "as", "by", "that",
    "this", "these", "those", "is", "are", "was", "were", "be", "been",
    "i", "you", "he", "she", "it", "we", "they", "my", "your", "his",
    "her", "its", "our", "their",
}

WORD_RE = re.compile(r"[A-Za-z']+")


def words(text: str) -> list[str]:
    """Tokenise with the same lightweight rule used during feature research."""
    return WORD_RE.findall(str(text))


def count_words(text: str) -> int:
    """Return the number of project-defined word tokens."""
    return len(words(text))


def extract_stylometry(
    sentence: str,
    doc_avg_sent_len: float = 0.0,
) -> dict[str, float]:
    """Extract the nine frozen stylometric/lexical sentence features."""
    sentence = str(sentence)
    tokens = words(sentence)
    lower_tokens = [token.lower() for token in tokens]

    word_count = len(tokens)
    denominator = max(word_count, 1)

    long_word_count = sum(len(token) >= 7 for token in tokens)
    function_word_count = sum(token in FUNCTION_WORDS for token in lower_tokens)
    unique_word_count = len(set(lower_tokens))

    lower_sentence = sentence.lower().lstrip()
    starts_transition = any(
        lower_sentence.startswith(transition)
        for transition in TRANSITION_WORDS
    )

    if doc_avg_sent_len > 0:
        rel_len_deviation = abs(word_count - doc_avg_sent_len) / doc_avg_sent_len
    else:
        rel_len_deviation = 0.0

    row = {
        "n_words": float(word_count),
        "long_word_ratio": float(long_word_count / denominator),
        "func_word_ratio": float(function_word_count / denominator),
        "unique_word_ratio": float(unique_word_count / denominator),
        "comma_rate": float(sentence.count(",") / denominator),
        "exclaim": 1.0 if "!" in sentence else 0.0,
        "apostrophe_rate": float(sentence.count("'") / denominator),
        "starts_with_transition": 1.0 if starts_transition else 0.0,
        "rel_len_deviation": float(rel_len_deviation),
    }

    return {name: row[name] for name in SELECTED_STYLOMETRY_FEATURES}


def extract_pos_features(doc) -> dict[str, float]:
    """Extract the six frozen POS-profile features from one spaCy Doc."""
    alpha_tokens = [token for token in doc if token.is_alpha]
    denominator = max(len(alpha_tokens), 1)

    def ratio(pos_values: set[str]) -> float:
        count = sum(token.pos_ in pos_values for token in alpha_tokens)
        return float(count / denominator)

    row = {
        "noun_ratio": ratio({"NOUN", "PROPN"}),
        "adjective_ratio": ratio({"ADJ"}),
        "adverb_ratio": ratio({"ADV"}),
        "pronoun_ratio": ratio({"PRON"}),
        "auxiliary_ratio": ratio({"AUX"}),
        "conjunction_ratio": ratio({"CCONJ", "SCONJ"}),
    }

    return {name: row[name] for name in SELECTED_POS_FEATURES}
