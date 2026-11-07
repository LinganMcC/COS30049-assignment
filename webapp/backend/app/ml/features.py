"""The 15 sentence-level features, copied from Assignment 2.

Source: Feature_extraction/features.py and Feature_extraction/feature_config.py.
The feature definitions below are unchanged. The only edit is that the two
Assignment 2 files are merged into one module so the API has no dependency on
the training folders. If a definition here ever drifts from training, the
model receives features it was never trained on, so do not edit these
without retraining.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Frozen feature lists (feature_config.py)
# ---------------------------------------------------------------------------
SELECTED_STYLOMETRY_FEATURES = [
    "n_words",
    "long_word_ratio",
    "func_word_ratio",
    "unique_word_ratio",
    "comma_rate",
    "exclaim",
    "apostrophe_rate",
    "starts_with_transition",
    "rel_len_deviation",
]

SELECTED_POS_FEATURES = [
    "noun_ratio",
    "adjective_ratio",
    "adverb_ratio",
    "pronoun_ratio",
    "auxiliary_ratio",
    "conjunction_ratio",
]

FINAL_FEATURES = SELECTED_STYLOMETRY_FEATURES + SELECTED_POS_FEATURES

FEATURE_MEANING = {
    "n_words": "Number of alphabetic/apostrophe word tokens in the sentence.",
    "long_word_ratio": "Fraction of words with at least 7 characters.",
    "func_word_ratio": "Fraction of words in the configured function-word list.",
    "unique_word_ratio": "Fraction of distinct lower-cased word tokens in the sentence.",
    "comma_rate": "Number of commas divided by sentence word count.",
    "exclaim": "1 when the sentence contains an exclamation mark, otherwise 0.",
    "apostrophe_rate": "Number of apostrophes divided by sentence word count.",
    "starts_with_transition": "1 when the sentence starts with a configured transition phrase.",
    "rel_len_deviation": "Absolute sentence word-count deviation from its document mean, divided by that mean.",
    "noun_ratio": "Fraction of alphabetic tokens tagged NOUN or PROPN by spaCy.",
    "adjective_ratio": "Fraction of alphabetic tokens tagged ADJ by spaCy.",
    "adverb_ratio": "Fraction of alphabetic tokens tagged ADV by spaCy.",
    "pronoun_ratio": "Fraction of alphabetic tokens tagged PRON by spaCy.",
    "auxiliary_ratio": "Fraction of alphabetic tokens tagged AUX by spaCy.",
    "conjunction_ratio": "Fraction of alphabetic tokens tagged CCONJ or SCONJ by spaCy.",
}

# Short names for the interface. Added for the web app; not used by the model.
FEATURE_LABEL = {
    "n_words": "Sentence length",
    "long_word_ratio": "Long words",
    "func_word_ratio": "Function words",
    "unique_word_ratio": "Word variety",
    "comma_rate": "Commas",
    "exclaim": "Exclamation mark",
    "apostrophe_rate": "Apostrophes",
    "starts_with_transition": "Opens with a transition",
    "rel_len_deviation": "Length vs. other sentences",
    "noun_ratio": "Nouns",
    "adjective_ratio": "Adjectives",
    "adverb_ratio": "Adverbs",
    "pronoun_ratio": "Pronouns",
    "auxiliary_ratio": "Auxiliary verbs",
    "conjunction_ratio": "Conjunctions",
}

# ---------------------------------------------------------------------------
# Feature extraction (features.py)
# ---------------------------------------------------------------------------
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


def extract_stylometry(sentence: str, doc_avg_sent_len: float = 0.0) -> dict[str, float]:
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
    starts_transition = any(lower_sentence.startswith(t) for t in TRANSITION_WORDS)

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
