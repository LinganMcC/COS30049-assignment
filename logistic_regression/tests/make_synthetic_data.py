from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

FEATURES = [
    "n_words", "long_word_ratio", "func_word_ratio", "unique_word_ratio",
    "comma_rate", "exclaim", "apostrophe_rate", "starts_with_transition",
    "rel_len_deviation", "noun_ratio", "adjective_ratio", "adverb_ratio",
    "pronoun_ratio", "auxiliary_ratio", "conjunction_ratio",
]


def make_split(n_docs: int, seed: int, prefix: str, shift: float = 1.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    generators = ["gpt", "llama", "claude"]

    for d in range(n_docs):
        label = d % 2
        n_sentences = int(rng.integers(3, 10))
        generator = rng.choice(generators) if label else "human"
        # Two documents per group; some groups can include both labels.
        group_key = f"{prefix}_group_{d // 2}"
        lengths = rng.poisson(17 + 4 * label * shift, n_sentences).clip(2)
        mean_len = lengths.mean()

        for i, n_words in enumerate(lengths):
            row = {
                "sentence_id": f"{prefix}{d}::s{i}",
                "document_id": f"{prefix}{d}",
                "sentence_index": i,
                "text": f"Synthetic sentence {i} for document {d}.",
                "label": label,
                "label_origin": "document_inherited",
                "source_dataset": "synthetic",
                "group_key": group_key,
                "domain": "essay",
                "generator": generator,
                "n_words": float(n_words),
                "long_word_ratio": float(np.clip(rng.normal(0.20 + 0.05 * label * shift, 0.06), 0, 1)),
                "func_word_ratio": float(np.clip(rng.normal(0.42 - 0.04 * label * shift, 0.06), 0, 1)),
                "unique_word_ratio": float(np.clip(rng.normal(0.88 + 0.02 * label, 0.05), 0, 1)),
                "comma_rate": float(np.clip(rng.normal(0.04 + 0.02 * label, 0.025), 0, None)),
                "exclaim": float(rng.random() < (0.08 if label == 0 else 0.03)),
                "apostrophe_rate": float(np.clip(rng.normal(0.025 if label == 0 else 0.01, 0.015), 0, None)),
                "starts_with_transition": float(rng.random() < (0.05 + 0.10 * label * shift)),
                "rel_len_deviation": float(abs(n_words - mean_len) / max(mean_len, 1)),
                "noun_ratio": float(np.clip(rng.normal(0.27 + 0.04 * label, 0.05), 0, 1)),
                "adjective_ratio": float(np.clip(rng.normal(0.07 + 0.02 * label, 0.03), 0, 1)),
                "adverb_ratio": float(np.clip(rng.normal(0.05, 0.025), 0, 1)),
                "pronoun_ratio": float(np.clip(rng.normal(0.10 - 0.03 * label, 0.035), 0, 1)),
                "auxiliary_ratio": float(np.clip(rng.normal(0.07, 0.025), 0, 1)),
                "conjunction_ratio": float(np.clip(rng.normal(0.05, 0.02), 0, 1)),
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    # Exercise the imputer/warning path.
    if len(df) >= 5:
        df.loc[df.sample(3, random_state=seed).index, "noun_ratio"] = np.nan
    return df


def write_all(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    make_split(80, 1, "tr").to_csv(folder / "drcat_train_final_sentence_features.csv", index=False)
    make_split(30, 2, "va").to_csv(folder / "drcat_validation_final_sentence_features.csv", index=False)
    make_split(30, 3, "te").to_csv(folder / "drcat_test_final_sentence_features.csv", index=False)
    make_split(30, 4, "hc", shift=0.5).to_csv(folder / "hc3_external_test_final_sentence_features.csv", index=False)
