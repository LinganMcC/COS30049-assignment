"""Fit, inspect, score and freeze Logistic Regression pipelines."""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config
from config import FINAL_FEATURES
from .pooling import pool_documents
from .weights import compute_training_weights


def build_pipeline(c_value: float) -> Pipeline:
    params = dict(config.LR_BASE_PARAMS)
    params["C"] = float(c_value)
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(**params)),
        ]
    )


def fit_pipeline(
    train_df: pd.DataFrame,
    *,
    weighting: str,
    class_balance: str,
    c_value: float,
) -> tuple[Pipeline, dict]:
    sample_weight = compute_training_weights(train_df, weighting, class_balance)
    pipe = build_pipeline(c_value)

    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        try:
            pipe.fit(
                train_df[FINAL_FEATURES],
                train_df["label"].to_numpy(dtype=int),
                clf__sample_weight=sample_weight,
            )
        except ConvergenceWarning as exc:
            raise RuntimeError(
                "Logistic Regression failed to converge. Do not hide this warning. "
                "Inspect scaling/data and only then consider changing solver/max_iter."
            ) from exc

    clf = pipe.named_steps["clf"]
    info = {
        "n_iter": int(clf.n_iter_[0]),
        "C": float(c_value),
        "weighting": weighting,
        "class_balance": class_balance,
        "sample_weight_min": float(sample_weight.min()),
        "sample_weight_max": float(sample_weight.max()),
        "sample_weight_mean": float(sample_weight.mean()),
        "missing_values_seen_at_fit": int(train_df[FINAL_FEATURES].isna().sum().sum()),
    }
    return pipe, info


def predict_sentence_proba(pipe: Pipeline, df: pd.DataFrame) -> np.ndarray:
    return pipe.predict_proba(df[FINAL_FEATURES])[:, 1]


def score_documents(pipe: Pipeline, df: pd.DataFrame, pooling: str):
    sentence_p = predict_sentence_proba(pipe, df)
    return sentence_p, pool_documents(df, sentence_p, pooling)


def standardized_features(pipe: Pipeline, df: pd.DataFrame) -> np.ndarray:
    return pipe[:-1].transform(df[FINAL_FEATURES])


def sentence_logit_contributions(pipe: Pipeline, df: pd.DataFrame) -> pd.DataFrame:
    """Exact per-feature contributions to each sentence's LR linear logit.

    For a sentence, logit = intercept + sum(standardized_feature * coefficient).
    These are NOT claimed to be exact contributions to the final pooled document
    probability, because sigmoid + pooling is nonlinear.
    """
    z = standardized_features(pipe, df)
    coef = pipe.named_steps["clf"].coef_[0]
    contrib = z * coef
    return pd.DataFrame(contrib, columns=FINAL_FEATURES, index=df.index)


def coefficient_table(pipe: Pipeline) -> pd.DataFrame:
    coef = pipe.named_steps["clf"].coef_[0]
    table = pd.DataFrame(
        {
            "feature": FINAL_FEATURES,
            "coefficient": coef,
            "abs_coefficient": np.abs(coef),
            "odds_ratio_per_1sd": np.exp(coef),
            "direction": np.where(coef >= 0, "toward AI", "toward Human"),
            "meaning": [config.FEATURE_MEANING.get(f, "") for f in FINAL_FEATURES],
        }
    )
    return table.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)


def save_frozen(pipe: Pipeline, frozen_config: dict) -> None:
    config.FROZEN_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, config.FROZEN_MODEL_PATH)
    config.FROZEN_CONFIG_PATH.write_text(
        json.dumps(frozen_config, indent=2), encoding="utf-8"
    )


def load_frozen() -> tuple[Pipeline, dict]:
    if not config.FROZEN_MODEL_PATH.exists() or not config.FROZEN_CONFIG_PATH.exists():
        raise SystemExit(
            "No frozen model found. Run scripts/04_validate_freeze.py after selecting "
            "the TRAIN-CV configuration."
        )
    cfg = json.loads(config.FROZEN_CONFIG_PATH.read_text(encoding="utf-8"))
    if list(cfg["features"]) != list(FINAL_FEATURES):
        raise SystemExit("Frozen feature list differs from feature_extraction/feature_config.py")
    return joblib.load(config.FROZEN_MODEL_PATH), cfg
