from __future__ import annotations

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

import config
from src.data_utils import load_feature_table
from src.modeling import load_frozen, score_documents, sentence_logit_contributions
from src.reporting import write_text


def analyse(name: str, pipe, cfg: dict, n_docs: int) -> None:
    out = config.stage_dir("07_error_analysis")
    df = load_feature_table(config.DATASETS[name], name.upper())
    sentence_p, docs = score_documents(pipe, df, cfg["pooling"])
    threshold = float(cfg["threshold"])

    docs = docs.copy()
    docs["prediction"] = (docs["doc_proba"] >= threshold).astype(int)
    docs["outcome"] = np.select(
        [
            (docs["label"] == 1) & (docs["prediction"] == 1),
            (docs["label"] == 0) & (docs["prediction"] == 0),
            (docs["label"] == 0) & (docs["prediction"] == 1),
            (docs["label"] == 1) & (docs["prediction"] == 0),
        ],
        ["TP", "TN", "FP", "FN"],
        default="?",
    )
    docs["confidence_from_threshold"] = (docs["doc_proba"] - threshold).abs()
    docs.reset_index().to_csv(out / f"{name}_document_outcomes.csv", index=False)

    contributions = sentence_logit_contributions(pipe, df)
    sentence_rows = df[["document_id", "n_words"]].copy()
    sentence_rows["sentence_proba"] = sentence_p
    if "text" in df.columns:
        sentence_rows["text"] = df["text"].astype(str)

    lines = [
        f"# Error analysis - {name.upper()}",
        "",
        "Feature contribution values below are exact contributions to a selected sentence's LR linear logit.",
        "They are not presented as an exact decomposition of the pooled document probability.",
        "",
    ]

    for outcome, title in (("FP", "False positives: Human predicted AI"), ("FN", "False negatives: AI predicted Human")):
        errors = docs[docs["outcome"] == outcome].sort_values("confidence_from_threshold", ascending=False).head(n_docs)
        lines += [f"## {title}", f"Total {outcome}: {int((docs['outcome'] == outcome).sum())}", ""]

        for doc_id, row in errors.iterrows():
            idx = df.index[df["document_id"] == doc_id]
            if outcome == "FP":
                chosen_idx = idx[np.argmax(sentence_p[idx])]
            else:
                chosen_idx = idx[np.argmin(sentence_p[idx])]

            contrib = contributions.loc[chosen_idx].sort_values()
            toward_human = contrib.head(4)
            toward_ai = contrib.tail(4).sort_values(ascending=False)
            sent_text = str(df.loc[chosen_idx, "text"]) if "text" in df.columns else "(text column unavailable)"

            lines += [
                f"### `{doc_id}` — document score {row['doc_proba']:.3f}",
                f"Selected sentence probability: {sentence_p[chosen_idx]:.3f}",
                f"Selected sentence: {sent_text}",
                "",
                "Strongest sentence-logit pushes toward AI: " + ", ".join(f"{k}={v:+.3f}" for k, v in toward_ai.items()),
                "Strongest sentence-logit pushes toward Human: " + ", ".join(f"{k}={v:+.3f}" for k, v in toward_human.items()),
                "",
                "**Your notes:** _What pattern do you see?_",
                "",
            ]

    write_text(out / f"{name}_error_analysis.md", "\n".join(lines))
    print(f"Saved {name} error analysis to {out}")


def main(args) -> None:
    pipe, cfg = load_frozen()
    for name in args.datasets:
        analyse(name, pipe, cfg, args.n_docs)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["test", "hc3"], choices=["validation", "test", "hc3"])
    ap.add_argument("--n-docs", type=int, default=5)
    main(ap.parse_args())
