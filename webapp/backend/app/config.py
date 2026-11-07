"""Every setting comes from environment variables (or backend/.env).

Nothing secret lives in the source code. Copy .env.example to .env to run.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BACKEND_DIR / ".env", extra="ignore")

    # The single front-end origin allowed by CORS (scheme + host + port, exact).
    frontend_origin: str = "http://localhost:5173"

    # Model files (copied from logistic_regression/outputs/05_frozen_model).
    model_path: Path = BACKEND_DIR / "model" / "logistic_regression.joblib"
    model_config_path: Path = BACKEND_DIR / "model" / "frozen_config.json"

    # Input limits.
    #   max_words: the 1,000-word limit agreed in Assignment 1.
    #   min_words: below ~20 words there are only one or two sentences, so the
    #              range is meaningless and the score swings on a single sentence.
    #   max_chars: a safety net only. English averages ~6 characters per word
    #              including spaces, so 1,000 real words never come close;
    #              it stops one "word" of 50,000 letters reaching the model.
    max_words: int = 1000
    min_words: int = 20
    max_chars: int = 20_000
    max_body_bytes: int = 128 * 1024  # whole HTTP body (non-English letters take up to 4 bytes)

    # Defence A: denial of service (section 2.7, Advanced)
    dos_guard_enabled: bool = True
    max_concurrent_analyses: int = 4

    # Section 2.8: detection evasion (Unicode normalisation before the model)
    evasion_defence_enabled: bool = True

    show_docs: bool = True


@lru_cache
def get_settings() -> Settings:
    return Settings()
