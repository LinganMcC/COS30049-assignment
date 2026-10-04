"""Minimal plotting helpers for tuning/evaluation outputs."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_threshold_sweep(sweep: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for metric in ["precision", "recall", "f1", "macro_f1", "accuracy"]:
        ax.plot(sweep["threshold"], sweep[metric], label=metric)
    ax.set_xlabel("Classification threshold")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.02)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_top_coefficients(coefficients: pd.DataFrame, path: Path, n: int = 15) -> None:
    shown = coefficients.head(n).sort_values("coefficient")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(shown["feature"], shown["coefficient"])
    ax.axvline(0, linewidth=1)
    ax.set_xlabel("Standardised Logistic Regression coefficient")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
