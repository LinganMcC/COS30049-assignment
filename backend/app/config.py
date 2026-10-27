"""All settings in one place, read from environment variables or backend/.env.

Nothing secret or environment-specific is hard-coded elsewhere in the app.
"""
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BACKEND_DIR / '.env', extra='ignore')

    frontend_origin: str = 'http://localhost:5173'

    model_dir: Path = BACKEND_DIR / 'model'

    max_words: int = 1000
    max_chars: int = 8000
    max_body_bytes: int = 32 * 1024

    dos_protection: bool = True
    inference_workers: int = 2
    inference_queue: int = 4
    inference_timeout_seconds: float = 10.0
    torch_threads: int = 2

    normalise_input: bool = True

    security_headers: bool = True
    enable_docs: bool = True


@lru_cache
def get_settings() -> Settings:
    return Settings()
