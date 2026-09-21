"""
features_updated.py
Sentence-level + document-level stylometric feature extraction.
"""

import re
from collections import Counter
import numpy as np

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

SENTENCE_FEATURE_NAMES = [
    "n_words", "long_word_ratio", "func_word_ratio",
    "unique_word_ratio", "comma_rate", "semicolon_rate", "colon_rate",
    "exclaim", "question", "apostrophe_rate", "starts_with_transition",
    "starts_lowercase", "has_semicolon_or_colon", "rel_len_deviation",
]

DOCUMENT_FEATURE_NAMES = [
    "sentence_length_std",
    "sentence_length_cv",
    "document_ttr",
    "hapax_rate",
    "flesch_reading_ease",
]


def split_sentences(text: str):
    text = str(text).strip()
    raw = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9"\'])', text)
    return [s.strip() for s in raw if s.strip()]


def _words(text: str):
    return re.findall(r"[A-Za-z']+", str(text))


def _safe_div(num, den):
    return float(num) / float(den) if den else 0.0


def _count_syllables(word: str) -> int:
    word = re.sub(r"[^a-z]", "", word.lower())
    if not word:
        return 0
    if len(word) <= 3:
        return 1

    vowels = "aeiouy"
    count = 0
    prev_is_vowel = False

    for ch in word:
        is_vowel = ch in vowels
        if is_vowel and not prev_is_vowel:
            count += 1
        prev_is_vowel = is_vowel

    if word.endswith("e") and not word.endswith(("le", "ye")) and count > 1:
        count -= 1

    if word.endswith(("es", "ed")) and len(word) > 4 and count > 1:
        count -= 1

    return max(count, 1)


def flesch_reading_ease(text: str) -> float:
    sentences = split_sentences(text)
    words = _words(text)

    if not sentences or not words:
        return 0.0

    syllables = sum(_count_syllables(w) for w in words)
    words_per_sentence = len(words) / len(sentences)
    syllables_per_word = syllables / len(words)

    return (
        206.835
        - 1.015 * words_per_sentence
        - 84.6 * syllables_per_word
    )


def sentence_features(sentence: str, doc_avg_sent_len: float = 0.0) -> dict:
    words = _words(sentence)
    n = max(len(words), 1)
    lower_words = [w.lower() for w in words]

    long_words = sum(1 for w in words if len(w) >= 7)
    func_words = sum(1 for w in lower_words if w in FUNCTION_WORDS)
    unique_words = len(set(lower_words))

    first_two = " ".join(lower_words[:2])
    lower_sentence = sentence.lower()

    starts_transition = any(
        lower_sentence.startswith(tw) or first_two.startswith(tw)
        for tw in TRANSITION_WORDS
    )

    starts_lower = sentence[:1].islower() if sentence else False

    if doc_avg_sent_len > 0:
        rel_len_deviation = abs(n - doc_avg_sent_len) / doc_avg_sent_len
    else:
        rel_len_deviation = 0.0

    return {
        "n_words": float(n),
        "long_word_ratio": long_words / n,
        "func_word_ratio": func_words / n,
        "unique_word_ratio": unique_words / n,
        "comma_rate": sentence.count(",") / n,
        "semicolon_rate": sentence.count(";") / n,
        "colon_rate": sentence.count(":") / n,
        "exclaim": 1.0 if "!" in sentence else 0.0,
        "question": 1.0 if "?" in sentence else 0.0,
        "apostrophe_rate": sentence.count("'") / n,
        "starts_with_transition": 1.0 if starts_transition else 0.0,
        "starts_lowercase": 1.0 if starts_lower else 0.0,
        "has_semicolon_or_colon": 1.0 if (";" in sentence or ":" in sentence) else 0.0,
        "rel_len_deviation": rel_len_deviation,
    }


def featurize_passage(text: str):
    sentences = split_sentences(text)

    if not sentences:
        return [], np.zeros((0, len(SENTENCE_FEATURE_NAMES)), dtype=float)

    sentence_lengths = [len(_words(s)) for s in sentences]
    doc_avg_sent_len = float(np.mean(sentence_lengths))

    rows = [
        sentence_features(s, doc_avg_sent_len=doc_avg_sent_len)
        for s in sentences
    ]

    X = np.array(
        [[row[name] for name in SENTENCE_FEATURE_NAMES] for row in rows],
        dtype=float,
    )

    return sentences, X


def document_features(text: str) -> dict:
    text = str(text)
    sentences = split_sentences(text)
    words = _words(text)
    lower_words = [w.lower() for w in words]

    sentence_lengths = np.array(
        [len(_words(s)) for s in sentences],
        dtype=float,
    )

    if len(sentence_lengths) > 0:
        mean_len = float(np.mean(sentence_lengths))
        sentence_length_std = float(np.std(sentence_lengths))
        sentence_length_cv = _safe_div(sentence_length_std, mean_len)
    else:
        sentence_length_std = 0.0
        sentence_length_cv = 0.0

    n_words = len(lower_words)

    if n_words > 0:
        counts = Counter(lower_words)
        document_ttr = len(counts) / n_words
        hapax_count = sum(1 for freq in counts.values() if freq == 1)
        hapax_rate = hapax_count / n_words
    else:
        document_ttr = 0.0
        hapax_rate = 0.0

    return {
        "sentence_length_std": sentence_length_std,
        "sentence_length_cv": sentence_length_cv,
        "document_ttr": document_ttr,
        "hapax_rate": hapax_rate,
        "flesch_reading_ease": flesch_reading_ease(text),
    }


def featurize_document(text: str):
    features = document_features(text)
    return np.array(
        [features[name] for name in DOCUMENT_FEATURE_NAMES],
        dtype=float,
    )


def build_document_feature_matrix(texts):
    rows = [featurize_document(text) for text in texts]

    if not rows:
        return np.zeros((0, len(DOCUMENT_FEATURE_NAMES)), dtype=float)

    return np.vstack(rows)


# Backward compatibility with your current train.py.
FEATURE_NAMES = SENTENCE_FEATURE_NAMES
