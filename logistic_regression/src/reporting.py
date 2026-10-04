"""Small output helpers."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def markdown_table(df: pd.DataFrame, *, max_rows: int | None = None) -> str:
    shown = df if max_rows is None else df.head(max_rows)
    try:
        return shown.to_markdown(index=False)
    except ImportError:
        return shown.to_string(index=False)
