"""Configuration for the Logistic Regression baseline.

Expected project layout::

    project/
    ├── preprocess/
    ├── feature_extraction/
    │   └── feature_config.py
    └── logistic_regression/
        └── config.py

The Logistic Regression folder does not duplicate feature-extraction code.
Set LR_FEATURE_DIR if your feature_extraction folder lives somewhere else.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent

FEATURE_DIR = Path(
    os.environ.get("LR_FEATURE_DIR", PROJECT_ROOT / "feature_extraction")
).resolve()
if not (FEATURE_DIR / "feature_config.py").exists():
    raise ImportError(
        "Could not find feature_extraction/feature_config.py.\n"
        f"Expected: {FEATURE_DIR / 'feature_config.py'}\n"
        "Keep logistic_regression beside feature_extraction, or set LR_FEATURE_DIR."
    )

sys.path.insert(0, str(FEATURE_DIR))
from feature_config import FINAL_FEATURES, FEATURE_MEANING  # noqa: E402

DATA_DIR = Path(
    os.environ.get("LR_DATA_DIR", PROJECT_ROOT / "data" / "features")
).resolve()
OUTPUT_DIR = Path(
    os.environ.get("LR_OUTPUT_DIR", ROOT / "outputs")
).resolve()

DATASETS = {
    "train": DATA_DIR / "drcat_train_final_sentence_features.csv",
    "validation": DATA_DIR / "drcat_validation_final_sentence_features.csv",
    "test": DATA_DIR / "drcat_test_final_sentence_features.csv",
    "hc3": DATA_DIR / "hc3_external_test_final_sentence_features.csv",
}

FROZEN_DIR = OUTPUT_DIR / "05_frozen_model"
FROZEN_MODEL_PATH = FROZEN_DIR / "logistic_regression.joblib"
FROZEN_CONFIG_PATH = FROZEN_DIR / "frozen_config.json"

COARSE_RECOMMENDATION_PATH = OUTPUT_DIR / "02_coarse_tuning" / "recommended_config.json"
FINE_RECOMMENDATION_PATH = OUTPUT_DIR / "03_fine_tuning" / "recommended_config.json"


def stage_dir(name: str) -> Path:
    path = OUTPUT_DIR / name
    path.mkdir(parents=True, exist_ok=True)
    return path


RANDOM_STATE = 42
N_SPLITS = 5

# These are research candidates, not silently fixed choices.
SENTENCE_WEIGHTING_OPTIONS = ("uniform", "doc_equal", "sqrt_doc")
CLASS_BALANCE_OPTIONS = ("none", "doc_balanced")
POOLING_OPTIONS = ("mean", "sqrt_length")

# Logistic Regression solver/convergence settings are engineering settings, not
# hyperparameters we grid-search unless convergence actually fails.
LR_BASE_PARAMS = {
    "solver": "lbfgs",
    "max_iter": 3000,
    "random_state": RANDOM_STATE,
}

# Stage 1: broad regularisation search.
COARSE_C_GRID = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)

# Stage 2: focused search around the best coarse C.
FINE_C_MULTIPLIERS = (0.3, 0.5, 1.0, 2.0, 3.0)

# Validation is reserved for threshold selection after model-family choices are
# made using TRAIN grouped CV.
THRESHOLD_GRID = np.round(np.arange(0.05, 0.9501, 0.01), 2).tolist()
THRESHOLD_OBJECTIVE = "macro_f1"
DEFAULT_THRESHOLD = 0.5

# Final uncertainty uses a group/cluster bootstrap, not a sentence or ordinary
# document bootstrap, because documents sharing group_key are related.
N_BOOTSTRAP = int(os.environ.get("LR_N_BOOTSTRAP", 2000))
CI_LEVEL = 0.95
