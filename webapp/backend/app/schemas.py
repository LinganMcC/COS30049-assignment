"""Request and response models.

Constraints that describe the input live here, so FastAPI rejects bad input
with 422 before any model code runs, and /docs shows the limits.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .config import get_settings


def count_input_words(text: str) -> int:
    """Words for the input limits: whitespace-separated pieces with a letter or digit.

    "résumé", "work-life", "don't" and "2024" each count once, the way a person
    counts them. The front end uses the same rule (lib/text.js countWords).
    This is NOT the model's feature n_words, which keeps its Assignment 2 definition.
    """
    return sum(1 for token in text.split() if any(ch.isalnum() for ch in token))

_settings = get_settings()

EXAMPLE_TEXT = (
    "Remote work has fundamentally transformed the modern workplace, offering both "
    "significant opportunities and notable challenges. Furthermore, employees now enjoy "
    "greater flexibility, which can lead to improved work-life balance. However, the "
    "absence of in-person interaction may weaken team cohesion."
)


class DetectRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"examples": [{"text": EXAMPLE_TEXT}]},
    )

    text: str = Field(
        ...,
        min_length=1,
        max_length=_settings.max_chars,
        description=(
            f"English prose to analyse: {_settings.min_words} to {_settings.max_words} words, "
            f"at most {_settings.max_chars:,} characters."
        ),
    )

    @field_validator("text")
    @classmethod
    def check_word_count(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Text cannot be empty.")
        words = count_input_words(value)
        if words < _settings.min_words:
            raise ValueError(
                f"Text has {words} words; at least {_settings.min_words} are needed for a meaningful score."
            )
        if words > _settings.max_words:
            raise ValueError(f"Text has {words} words; the limit is {_settings.max_words}.")
        return value


class Factor(BaseModel):
    feature: str = Field(description="Feature id, e.g. long_word_ratio")
    label: str = Field(description="Short name for display")
    value: float = Field(description="The feature's value for this sentence (or document average)")
    contribution: float = Field(
        description="Push on the sentence's log-odds (before conversion to a probability): positive = toward AI, negative = toward human. Not percentage points."
    )


class SentenceResult(BaseModel):
    index: int
    text: str
    score: float = Field(ge=0, le=1, description="Probability this sentence is AI-generated")
    band: str
    word_count: int
    toward_ai: list[Factor]
    toward_human: list[Factor]


class PassageSegment(BaseModel):
    text: str
    sentence_index: int | None = Field(
        description="Index into sentences, or null for spacing and unscored fragments"
    )


class ScoreRange(BaseModel):
    low: float = Field(ge=0, le=1)
    high: float = Field(ge=0, le=1)


class NormalisationInfo(BaseModel):
    applied: bool
    invisible_removed: int
    homoglyphs_replaced: int
    marks_removed: int


class DetectResponse(BaseModel):
    label: Literal["ai", "human"] = Field(description="Verdict at the model's decision threshold")
    band: str = Field(description="Human / Possibly AI / Likely AI / Very likely AI")
    score: float = Field(ge=0, le=1, description="Estimated probability the passage is AI-generated")
    range: ScoreRange = Field(description="Sentence-agreement range: how far the score could move given how much the sentence scores disagree. A heuristic, not a validated confidence interval.")
    threshold: float
    certainty: Literal["clear", "uncertain"] = Field(
        description="'uncertain' when the range crosses the threshold"
    )
    sentence_count: int
    word_count: int
    unscored_fragments: int
    sentences: list[SentenceResult]
    passage: list[PassageSegment]
    style_summary: list[Factor] = Field(description="Features with the largest average push on the sentence scores (log-odds), weighted like the document score. Associations, not an exact breakdown of the percentage.")
    normalisation: NormalisationInfo


class FeatureInfo(BaseModel):
    feature: str
    label: str
    meaning: str
    coefficient: float
    direction: Literal["ai", "human"]


class Band(BaseModel):
    upper: float
    label: str


class MetricWithCI(BaseModel):
    estimate: float
    ci_low: float
    ci_high: float


class Evaluation(BaseModel):
    validation: dict[str, float]
    drcat_test: dict[str, MetricWithCI]
    hc3_external: dict[str, MetricWithCI]


class Limits(BaseModel):
    min_words: int
    max_words: int
    max_chars: int


class ModelInfoResponse(BaseModel):
    name: str
    version: str
    threshold: float
    pooling: str
    bands: list[Band]
    features: list[FeatureInfo]
    evaluation: Evaluation
    limits: Limits


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model_loaded: bool


class FieldError(BaseModel):
    field: str
    message: str


class ErrorBody(BaseModel):
    code: str = Field(examples=["validation_error"])
    message: str = Field(examples=["Text has 8 words; at least 20 are needed for a meaningful score."])
    fields: list[FieldError] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    """Every error from this API, of any status code, has this shape."""

    error: ErrorBody
