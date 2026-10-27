"""Pydantic request and response models.

The constraints here are the back end's real input validation; the front end
repeats them only so the user gets feedback before sending.
"""
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.config import get_settings

_settings = get_settings()
Band = Literal['Human', 'Possibly AI', 'Likely AI', 'Very likely AI']

_EXAMPLE_TEXT = ('Renewable energy has become a central topic in discussions about the '
                 'future of our planet. Solar and wind power offer clean alternatives to '
                 'fossil fuels. I reckon my uncle would still rather burn coal though.')


class AnalyseRequest(BaseModel):
    # extra='forbid': unknown fields are an error, not silently ignored.
    model_config = ConfigDict(extra='forbid', json_schema_extra={'examples': [{'text': _EXAMPLE_TEXT}]})

    text: str = Field(min_length=1, max_length=_settings.max_chars,
                      description=f'The text to check: at most {_settings.max_words} words '
                                  f'and {_settings.max_chars} characters.')

    @field_validator('text')
    @classmethod
    def within_word_limit(cls, value: str) -> str:
        if not value.strip():
            raise ValueError('Text must contain more than whitespace.')
        words = len(re.findall(r'\S+', value))
        if words > _settings.max_words:
            raise ValueError(f'Text has {words} words; the limit is {_settings.max_words}.')
        return value


class SentenceResult(BaseModel):
    index: int = Field(description='Position of the sentence in the text, from 0.')
    text: str
    scored: bool = Field(description='False for sentences under 5 words, which are not scored.')
    score: float | None = Field(ge=0, le=1, description='Probability the sentence is AI-generated.')
    band: Band | None


class DocumentResult(BaseModel):
    score: float = Field(ge=0, le=1, description='Mean of the sentence scores.')
    low: float = Field(ge=0, le=1, description='Lower end of the ~95% confidence range.')
    high: float = Field(ge=0, le=1, description='Upper end of the ~95% confidence range.')
    band: Band
    scored_sentences: int


class AnalyseResponse(BaseModel):
    document: DocumentResult
    sentences: list[SentenceResult]
    excluded_sentences: int = Field(description='Sentences under 5 words that were not scored.')
    characters_normalised: int = Field(
        description='Look-alike or invisible characters replaced before scoring.')

    model_config = ConfigDict(json_schema_extra={'examples': [{
        'document': {'score': 0.62, 'low': 0.31, 'high': 0.93, 'band': 'Likely AI',
                     'scored_sentences': 3},
        'sentences': [
            {'index': 0, 'text': 'Renewable energy has become a central topic in discussions '
                                 'about the future of our planet.',
             'scored': True, 'score': 0.91, 'band': 'Very likely AI'},
            {'index': 1, 'text': 'Solar and wind power offer clean alternatives to fossil fuels.',
             'scored': True, 'score': 0.78, 'band': 'Very likely AI'},
            {'index': 2, 'text': 'I reckon my uncle would still rather burn coal though.',
             'scored': True, 'score': 0.17, 'band': 'Human'},
        ],
        'excluded_sentences': 0,
        'characters_normalised': 0,
    }]})


class BandInfo(BaseModel):
    name: Band
    upper: float


class ConfigResponse(BaseModel):
    max_words: int
    max_chars: int
    min_sentence_words: int
    bands: list[BandInfo]


class HealthResponse(BaseModel):
    status: Literal['ok']
    model_loaded: bool


class ErrorBody(BaseModel):
    code: str = Field(examples=['invalid_input'])
    message: str = Field(examples=['The request is not valid.'])
    details: list[dict] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    """Shape of every error response, documented in /docs."""
    error: ErrorBody
