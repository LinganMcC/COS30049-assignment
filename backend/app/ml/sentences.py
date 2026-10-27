"""Sentence splitting with the same rules used to build the training data.

Mirrors Preprocess/sentence_splitter.py and train_transformer.py so the model
sees sentences exactly as it did in training:
    spaCy en_core_web_sm split -> keep sentences containing a letter ->
    only sentences with MIN_WORDS+ words are scored.
"""
import re

import spacy

MIN_WORDS = 5   # A1 outline: sentences under five words are excluded from scoring

_ALPHABETIC_RE = re.compile(r'[A-Za-z]')
_WORD_RE = re.compile(r"[A-Za-z']+")
_MISSING_SPACE_RE = re.compile(r'(?<=[a-z])\s*([.!?])(?=[A-Z])')


def load_spacy():
    # Same disabled components as training: only the parser's sentence
    # boundaries are needed, and skipping the rest makes splitting faster.
    return spacy.load('en_core_web_sm', disable=['ner', 'lemmatizer', 'attribute_ruler'])


def count_words(text: str) -> int:
    """Same word definition as training, so the 5-word rule matches."""
    return len(_WORD_RE.findall(text))


def split_sentences(nlp, text: str) -> list[str]:
    """Split text into sentences, dropping fragments with no letters ('2013.', '🚀')."""
    # Repairs "end.Start" -> "end. Start" so spaCy sees the boundary.
    repaired = _MISSING_SPACE_RE.sub(r'\1 ', text.strip())
    sentences = []
    for span in nlp(repaired).sents:
        sentence = span.text.strip()
        if sentence and _ALPHABETIC_RE.search(sentence):
            sentences.append(sentence)
    return sentences
