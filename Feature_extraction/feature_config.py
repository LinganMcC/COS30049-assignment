"""Frozen final sentence-feature configuration for the detector.

Labels:
    0 = human
    1 = AI

The final production representation contains 15 sentence-level features:
    - 9 stylometric/lexical features
    - 6 spaCy POS-profile features

The feature list was frozen using DRCAT TRAIN only. Validation, DRCAT test,
and HC3 external test must not be used to revise this list.
"""

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

FINAL_FEATURES = (
    SELECTED_STYLOMETRY_FEATURES
    + SELECTED_POS_FEATURES
)

FEATURE_FAMILY = {
    **{name: "stylometry" for name in SELECTED_STYLOMETRY_FEATURES},
    **{name: "pos" for name in SELECTED_POS_FEATURES},
}

# Human-readable descriptions can later be reused by the explanation/UI layer.
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
