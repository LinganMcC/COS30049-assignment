from __future__ import annotations

import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

import config
from src.data_utils import load_feature_table
from src.metrics import compute_metrics, group_bootstrap_ci
from src.modeling import load_frozen, score_documents
from src.reporting import markdown_table, write_text


def evaluate(name: str, pipe, cfg: dict) -> tuple[dict, pd.DataFrame]:
    out = config.stage_dir(f"06_final_eval/{name}")
    df = load_feature_table(config.DATASETS[name], name.upper())
    sentence_p, docs = score_documents(pipe, df, cfg["pooling"])

    metrics = compute_metrics(docs["label"], docs["doc_proba"], cfg["threshold"])
    ci = group_bootstrap_ci(
        docs,
        threshold=cfg["threshold"],
        n_bootstrap=config.N_BOOTSTRAP,
        ci_level=config.CI_LEVEL,
        random_state=config.RANDOM_STATE,
    )

    docs.reset_index().to_csv(out / "document_scores.csv", index=False)
    pd.DataFrame([metrics]).to_csv(out / "metrics.csv", index=False)
    ci.to_csv(out / "metrics_with_group_bootstrap_ci.csv", index=False)

    confusion = pd.DataFrame(
        [[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]],
        index=["actual_human", "actual_ai"],
        columns=["predicted_human", "predicted_ai"],
    )
    confusion.to_csv(out / "confusion_matrix.csv")

    text = "\n".join(
        [
            f"# Final evaluation - {name.upper()}",
            "",
            markdown_table(pd.DataFrame([metrics])),
            "",
            "## Group-bootstrap confidence intervals",
            "",
            markdown_table(ci),
            "",
            "## Confusion matrix",
            "",
            confusion.to_markdown(),
        ]
    )
    write_text(out / "evaluation_report.md", text)
    return metrics, ci


def main() -> None:
    pipe, cfg = load_frozen()
    test_metrics, _ = evaluate("test", pipe, cfg)
    hc3_metrics, _ = evaluate("hc3", pipe, cfg)

    comparison = pd.DataFrame(
        [
            {"dataset": "DRCAT test", **{k: test_metrics[k] for k in ["accuracy", "precision", "recall", "f1", "roc_auc", "average_precision"]}},
            {"dataset": "HC3 external", **{k: hc3_metrics[k] for k in ["accuracy", "precision", "recall", "f1", "roc_auc", "average_precision"]}},
        ]
    )
    out = config.stage_dir("06_final_eval/comparison")
    comparison.to_csv(out / "drcat_vs_hc3.csv", index=False)
    write_text(out / "drcat_vs_hc3.md", "# DRCAT test vs HC3 external\n\n" + markdown_table(comparison))

    print("Final evaluation complete. No tuning was performed on TEST or HC3.")
    print(f"Comparison: {out / 'drcat_vs_hc3.csv'}")


if __name__ == "__main__":
    main()
