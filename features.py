"""
Sentence-level stylometric feature extraction for AI-vs-human detection.

Every feature describes HOW something is written, not WHAT it is about.
No bag-of-words / TF-IDF / topic words are used, on purpose: a feature
that leans on vocabulary will memorize this dataset's essay topics and
fail on a stranger's pasted text about something else. Style features
(rhythm, punctuation habits, transition-word usage) transfer much better.
"""

import re
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

# NOTE: avg_word_len was removed after feature_correlation.py found it
# 85% correlated with long_word_ratio (r=0.854) -- both were measuring
# essentially the same thing (vocabulary "heaviness"). Keeping only
# long_word_ratio, since it's less sensitive to a single unusually long
# word skewing a short sentence's average.
FEATURE_NAMES = [
    "n_words", "long_word_ratio", "func_word_ratio",
    "unique_word_ratio", "comma_rate", "semicolon_rate", "colon_rate",
    "exclaim", "question", "apostrophe_rate", "starts_with_transition",
    "starts_lowercase", "has_semicolon_or_colon", "rel_len_deviation",
]


def split_sentences(text: str):
    text = str(text).strip()
    raw = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9"\'])', text)
    return [s.strip() for s in raw if s.strip()]


def _words(s: str):
    return re.findall(r"[A-Za-z']+", s)


def sentence_features(sentence: str, doc_avg_sent_len: float = 0.0) -> dict:
    """Compute the numeric style-feature vector for a single sentence.

    doc_avg_sent_len: the *document's own* average sentence length (in
    words). Used to compute how much this sentence deviates from the
    passage's own rhythm -- one of the strongest signals in the dataset.
    """
    words = _words(sentence)
    n = max(len(words), 1)
    lower_words = [w.lower() for w in words]
    word_lengths = [len(w) for w in words]

    long_words = sum(1 for w in words if len(w) >= 7)
    func_words = sum(1 for w in lower_words if w in FUNCTION_WORDS)
    unique = len(set(lower_words))

    first_two = " ".join(lower_words[:2])
    lower_sentence = sentence.lower()
    starts_transition = any(
        lower_sentence.startswith(tw) or first_two.startswith(tw)
        for tw in TRANSITION_WORDS
    )
    starts_lower = sentence[:1].islower() if sentence else False

    return {
        "n_words": n,
        "long_word_ratio": long_words / n,
        "func_word_ratio": func_words / n,
        "unique_word_ratio": unique / n,
        "comma_rate": sentence.count(",") / n,
        "semicolon_rate": sentence.count(";") / n,
        "colon_rate": sentence.count(":") / n,
        "exclaim": 1.0 if "!" in sentence else 0.0,
        "question": 1.0 if "?" in sentence else 0.0,
        "apostrophe_rate": sentence.count("'") / n,
        "starts_with_transition": 1.0 if starts_transition else 0.0,
        "starts_lowercase": 1.0 if starts_lower else 0.0,
        "has_semicolon_or_colon": 1.0 if (";" in sentence or ":" in sentence) else 0.0,
        "rel_len_deviation": abs(n - doc_avg_sent_len) / max(doc_avg_sent_len, 1) if doc_avg_sent_len else 0.0,
    }


def featurize_passage(text: str):
    """Split a passage into sentences and return (sentences, feature_matrix).
    feature_matrix is an (n_sentences, n_features) numpy array, column
    order matching FEATURE_NAMES."""
    sents = split_sentences(text)
    if not sents:
        return [], np.zeros((0, len(FEATURE_NAMES)))
    lens = [len(_words(s)) for s in sents]
    avg_len = float(np.mean(lens)) if lens else 0.0
    rows = [sentence_features(s, doc_avg_sent_len=avg_len) for s in sents]
    X = np.array([[r[k] for k in FEATURE_NAMES] for r in rows])
    return sents, X
