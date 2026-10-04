"""Optional experiments. These do not belong to the core training pipeline."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

import config
from src.data_utils import load_feature_table
from src.metrics import compute_metrics
from src.modeling import fit_pipeline, load_frozen, score_documents
from src.reporting import markdown_table, write_text

SLICE_COLUMNS = ("generator", "domain", "source_dataset", "label_origin")


def recalc_rel_len_deviation(df: pd.DataFrame) -> pd.DataFrame:
    """Recalculate document-relative length after sentence truncation."""
    out = df.copy()
    if "rel_len_deviation" not in out.columns:
        return out
    mean_words = out.groupby("document_id")["n_words"].transform("mean")
    safe_mean = mean_words.where(mean_words > 0, 1.0)
    out["rel_len_deviation"] = (out["n_words"] - mean_words).abs() / safe_mean
    return out


def slice_analysis(name, df, docs, threshold):
    meta = df.groupby("document_id", sort=False).first()
    pred = (docs["doc_proba"] >= threshold).astype(int)
    rows = []
    for column in SLICE_COLUMNS:
        if column not in meta.columns:
            continue
        for value, doc_ids in meta.groupby(column, dropna=False).groups.items():
            sub = docs.loc[list(doc_ids)]
            sub_pred = pred.loc[list(doc_ids)]
            rows.append(
                {
                    "dataset": name,
                    "slice_by": column,
                    "value": value,
                    "n_documents": len(sub),
                    "actual_ai_share": float(sub["label"].mean()),
                    "predicted_ai_share": float(sub_pred.mean()),
                    "accuracy": float((sub_pred == sub["label"]).mean()),
                    "mean_document_score": float(sub["doc_proba"].mean()),
                }
            )
    return pd.DataFrame(rows)


def short_document_stress(name, df, pipe, cfg, ks):
    rank = df.groupby("document_id").cumcount()
    rows = []
    for k in ks:
        sub = df.copy() if k == 0 else df[rank < k].copy()
        sub = recalc_rel_len_deviation(sub)
        _, docs = score_documents(pipe, sub, cfg["pooling"])
        m = compute_metrics(docs["label"], docs["doc_proba"], cfg["threshold"])
        rows.append(
            {
                "dataset": name,
                "first_k_sentences": "all" if k == 0 else k,
                **{key: m[key] for key in ["n_documents", "accuracy", "precision", "recall", "f1", "roc_auc", "average_precision"]},
            }
        )
    return pd.DataFrame(rows)


def leave_one_generator_out(train, validation, cfg, min_ai_docs):
    if "generator" not in train.columns or "generator" not in validation.columns:
        return pd.DataFrame()

    generators = sorted(train.loc[train["label"] == 1, "generator"].dropna().unique())
    rows = []
    for generator in generators:
        val_ai_docs = validation.loc[
            (validation["label"] == 1) & (validation["generator"] == generator), "document_id"
        ].nunique()
        if val_ai_docs < min_ai_docs:
            continue

        reduced = train[~((train["label"] == 1) & (train["generator"] == generator))].copy()
        model, _ = fit_pipeline(
            reduced,
            weighting=cfg["weighting"],
            class_balance=cfg["class_balance"],
            c_value=cfg["C"],
        )
        vsub = validation[(validation["label"] == 0) | (validation["generator"] == generator)].copy()
        _, docs = score_documents(model, vsub, cfg["pooling"])
        m = compute_metrics(docs["label"], docs["doc_proba"], cfg["threshold"])
        rows.append(
            {
                "held_out_generator": generator,
                "n_ai_documents": val_ai_docs,
                "recall": m["recall"],
                "f1": m["f1"],
                "roc_auc": m["roc_auc"],
                "average_precision": m["average_precision"],
            }
        )
    return pd.DataFrame(rows)


def main(args) -> None:
    out = config.stage_dir("08_experiments")
    pipe, cfg = load_frozen()

    slice_frames = []
    trunc_frames = []
    for name in args.datasets:
        df = load_feature_table(config.DATASETS[name], name.upper())
        _, docs = score_documents(pipe, df, cfg["pooling"])
        slice_frames.append(slice_analysis(name, df, docs, cfg["threshold"]))
        trunc_frames.append(short_document_stress(name, df, pipe, cfg, args.ks))

    slices = pd.concat(slice_frames, ignore_index=True) if slice_frames else pd.DataFrame()
    trunc = pd.concat(trunc_frames, ignore_index=True) if trunc_frames else pd.DataFrame()
    slices.to_csv(out / "slice_analysis.csv", index=False)
    trunc.to_csv(out / "short_document_stress.csv", index=False)

    train = load_feature_table(config.DATASETS["train"], "TRAIN")
    validation = load_feature_table(config.DATASETS["validation"], "VALIDATION")
    logo = leave_one_generator_out(train, validation, cfg, args.min_ai_docs)
    logo.to_csv(out / "leave_one_generator_out.csv", index=False)

    report = "\n".join(
        [
            "# Optional stress experiments",
            "",
            "## Slice analysis",
            markdown_table(slices) if len(slices) else "No slice columns available.",
            "",
            "## Short-document stress",
            markdown_table(trunc),
            "",
            "## Leave-one-generator-out on validation",
            markdown_table(logo) if len(logo) else "No eligible generators.",
        ]
    )
    write_text(out / "stress_tests.md", report)
    print(f"Optional experiment outputs: {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["test", "hc3"], choices=["validation", "test", "hc3"])
    ap.add_argument("--ks", nargs="+", type=int, default=[1, 2, 3, 5, 10, 0], help="0 means all sentences")
    ap.add_argument("--min-ai-docs", type=int, default=5)
    main(ap.parse_args())
