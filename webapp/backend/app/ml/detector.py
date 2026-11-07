"""Load the frozen Assignment 2 model once and score passages with it.

Pipeline for one passage (mirrors Assignment 2 training and evaluation):

    prepare_text()           clean + split into sentences (text_pipeline.py)
    sentence features        9 stylometric + 6 POS features (features.py)
    pipe.predict_proba()     one AI-probability per sentence
    pooled score             sqrt(word count)-weighted mean  (logistic_regression/src/pooling.py)
    confidence range         team formula from transformer/train_transformer.py
    explanations             exact per-feature logit contributions (src/modeling.py)
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import warnings
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import spacy

from .features import (
    FEATURE_LABEL,
    FEATURE_MEANING,
    FINAL_FEATURES,
    count_words,
    extract_pos_features,
    extract_stylometry,
)
from .text_pipeline import PreparedText, prepare_text

logger = logging.getLogger("inkprint.detector")

# Display bands agreed in the team's results-screen mockup
# (transformer/train_transformer.py, BANDS). Upper bound -> label.
BANDS = [(0.25, "Human"), (0.50, "Possibly AI"), (0.75, "Likely AI"), (1.00, "Very likely AI")]

# Confidence-range settings, also from train_transformer.py aggregate_document().
RANGE_Z = 1.96          # ~95% interval
RANGE_MIN_VARIANCE = 0.01


def _quiet(fn, *args):
    """Call a scikit-learn method without the 'X does not have valid feature
    names' warning: the model was fitted on a DataFrame, we pass the same
    columns in the same order as a NumPy array."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="X does not have valid feature names")
        return fn(*args)


class ModelConfigError(RuntimeError):
    """The frozen config does not describe the features this app computes."""


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def likelihood_band(probability: float) -> str:
    for upper, name in BANDS:
        if probability <= upper:
            return name
    return BANDS[-1][1]


def _read_metrics(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            row["metric"]: {
                "estimate": float(row["estimate"]),
                "ci_low": float(row["ci_low"]),
                "ci_high": float(row["ci_high"]),
            }
            for row in csv.DictReader(handle)
        }


@dataclass
class Detector:
    pipe: object
    config: dict
    split_nlp: object
    pos_nlp: object
    model_sha256: str
    model_dir: Path
    evasion_defence: bool = True

    # -- loading -----------------------------------------------------------
    @classmethod
    def load(
        cls,
        model_path: Path,
        config_path: Path,
        *,
        evasion_defence: bool = True,
    ) -> "Detector":
        fingerprint = sha256_of(model_path)  # shown as the model version only

        config = json.loads(config_path.read_text(encoding="utf-8"))
        if list(config["features"]) != FINAL_FEATURES:
            raise ModelConfigError("Frozen feature list differs from app/ml/features.py")

        pipe = joblib.load(model_path)

        # Two spaCy pipelines, exactly as in Assignment 2:
        #   sentence_splitter.py loads it without ner, lemmatizer, attribute_ruler;
        #   build_final_sentence_feature_table.py loads it without ner, lemmatizer
        #   (attribute_ruler is what fills token.pos_ for the POS features).
        split_nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer", "attribute_ruler"])
        pos_nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])

        logger.info("Model loaded (version %s)", fingerprint[:12])
        return cls(
            pipe=pipe,
            config=config,
            split_nlp=split_nlp,
            pos_nlp=pos_nlp,
            model_sha256=fingerprint,
            model_dir=model_path.parent,
            evasion_defence=evasion_defence,
        )

    @property
    def threshold(self) -> float:
        return float(self.config.get("threshold", 0.5))

    # -- scoring -----------------------------------------------------------
    def featurise(self, sentences: list[str]) -> np.ndarray:
        """Build the (n_sentences x 15) feature matrix in FINAL_FEATURES order."""
        word_counts = [count_words(s) for s in sentences]
        doc_avg = float(np.mean(word_counts)) if word_counts else 0.0
        rows = []
        for sentence, doc in zip(sentences, self.pos_nlp.pipe(sentences, batch_size=64)):
            row = extract_stylometry(sentence, doc_avg)
            row.update(extract_pos_features(doc))
            rows.append([row[name] for name in FINAL_FEATURES])
        return np.asarray(rows, dtype=float)

    def contributions(self, x: np.ndarray) -> np.ndarray:
        """Exact per-feature contributions to each sentence's logit.

        logit = intercept + sum(standardised_feature * coefficient).
        Same as sentence_logit_contributions() in Assignment 2 src/modeling.py.
        """
        z = _quiet(self.pipe[:-1].transform, x)
        return z * self.pipe.named_steps["clf"].coef_[0]

    @staticmethod
    def pool(probabilities: np.ndarray, n_words: np.ndarray) -> tuple[float, np.ndarray]:
        """sqrt-length weighted mean, as pool_documents(method='sqrt_length')."""
        weights = np.sqrt(np.maximum(n_words.astype(float), 1.0))
        return float(np.sum(weights * probabilities) / np.sum(weights)), weights

    @staticmethod
    def confidence_range(probabilities: np.ndarray, weights: np.ndarray, score: float) -> tuple[float, float]:
        """Sentence-agreement range: score +/- 1.96 * sqrt((variance + 0.01) / n).

        The team's formula (train_transformer.py aggregate_document) adapted to
        the weighted score: weighted variance, and Kish's effective sample size
        in place of n so a few very long sentences don't look like many.

        It widens when there are few sentences or they disagree. It is a
        HEURISTIC: sentences from one document are not independent, and we have
        not measured how often the true label falls inside it, so it must not be
        presented as a validated 95% confidence interval.
        """
        variance = float(np.sum(weights * (probabilities - score) ** 2) / np.sum(weights))
        n_effective = float(np.sum(weights) ** 2 / np.sum(weights**2))
        half_width = RANGE_Z * math.sqrt((variance + RANGE_MIN_VARIANCE) / n_effective)
        return max(0.0, score - half_width), min(1.0, score + half_width)

    def analyse(self, text: str, *, linear_regex: bool = True) -> dict:
        prepared: PreparedText = prepare_text(
            text, self.split_nlp, evasion_defence=self.evasion_defence, linear_regex=linear_regex
        )
        if not prepared.sentences:
            raise ValueError("no_sentences")

        x = self.featurise(prepared.sentences)
        probabilities = _quiet(self.pipe.predict_proba, x)[:, 1]
        contributions = self.contributions(x)
        n_words = x[:, FINAL_FEATURES.index("n_words")]

        score, weights = self.pool(probabilities, n_words)
        low, high = self.confidence_range(probabilities, weights, score)
        threshold = self.threshold

        sentences = []
        for i, sentence in enumerate(prepared.sentences):
            order = np.argsort(contributions[i])
            toward_ai = [self._factor(j, x[i, j], contributions[i, j]) for j in order[::-1][:3] if contributions[i, j] > 0]
            toward_human = [self._factor(j, x[i, j], contributions[i, j]) for j in order[:3] if contributions[i, j] < 0]
            sentences.append(
                {
                    "index": i,
                    "text": sentence,
                    "score": round(float(probabilities[i]), 4),
                    "band": likelihood_band(float(probabilities[i])),
                    "word_count": int(n_words[i]),
                    "toward_ai": toward_ai,
                    "toward_human": toward_human,
                }
            )

        # Document-level style summary: the weighted average contribution of each
        # feature across sentences (same weights as the score), largest first.
        doc_contrib = (weights[:, None] * contributions).sum(axis=0) / weights.sum()
        doc_values = (weights[:, None] * x).sum(axis=0) / weights.sum()
        style = sorted(
            (self._factor(j, doc_values[j], doc_contrib[j]) for j in range(len(FINAL_FEATURES))),
            key=lambda f: abs(f["contribution"]),
            reverse=True,
        )

        if low >= threshold:
            certainty = "clear"
        elif high < threshold:
            certainty = "clear"
        else:
            certainty = "uncertain"

        return {
            "label": "ai" if score >= threshold else "human",
            "band": likelihood_band(score),
            "score": round(score, 4),
            "range": {"low": round(low, 4), "high": round(high, 4)},
            "threshold": threshold,
            "certainty": certainty,
            "sentence_count": len(sentences),
            "word_count": int(n_words.sum()),
            "unscored_fragments": prepared.unscored_fragments,
            "sentences": sentences,
            "passage": [
                {"text": seg.text, "sentence_index": seg.sentence_index}
                for seg in prepared.segments
            ],
            "style_summary": style[:6],
            "normalisation": {
                "applied": self.evasion_defence,
                "invisible_removed": prepared.normalisation.invisible_removed,
                "homoglyphs_replaced": prepared.normalisation.homoglyphs_replaced,
                "marks_removed": prepared.normalisation.marks_removed,
            },
        }

    @staticmethod
    def _factor(j: int, value: float, contribution: float) -> dict:
        name = FINAL_FEATURES[j]
        return {
            "feature": name,
            "label": FEATURE_LABEL[name],
            "value": round(float(value), 4),
            "contribution": round(float(contribution), 4),
        }

    # -- model information ---------------------------------------------------
    def info(self) -> dict:
        coef = self.pipe.named_steps["clf"].coef_[0]
        return {
            "name": "Sentence-level Logistic Regression",
            "version": self.model_sha256[:12],
            "threshold": self.threshold,
            "pooling": self.config.get("pooling", "sqrt_length"),
            "bands": [{"upper": upper, "label": label} for upper, label in BANDS],
            "features": [
                {
                    "feature": name,
                    "label": FEATURE_LABEL[name],
                    "meaning": FEATURE_MEANING[name],
                    "coefficient": round(float(coef[j]), 4),
                    "direction": "ai" if coef[j] >= 0 else "human",
                }
                for j, name in enumerate(FINAL_FEATURES)
            ],
            "evaluation": {
                "validation": {
                    k: round(float(v), 4)
                    for k, v in self.config.get("validation_metrics", {}).items()
                    if k in {"accuracy", "precision", "recall", "f1", "roc_auc", "n_documents"}
                },
                "drcat_test": _read_metrics(self.model_dir / "test_metrics_drcat.csv"),
                "hc3_external": _read_metrics(self.model_dir / "test_metrics_hc3.csv"),
            },
        }
