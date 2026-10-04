from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from src.data_utils import inspect_table, load_feature_table
from src.reporting import markdown_table, write_json, write_text


def main() -> None:
    out = config.stage_dir("01_inspection")
    train = load_feature_table(config.DATASETS["train"], "TRAIN")
    summary, ranges = inspect_table(train)

    write_json(out / "train_summary.json", summary)
    ranges.to_csv(out / "feature_ranges.csv", index_label="feature")

    text = ["# TRAIN inspection", "", "## Summary", ""]
    for key, value in summary.items():
        text.append(f"- **{key}:** {value}")
    text += ["", "## Feature ranges", "", markdown_table(ranges.reset_index())]
    write_text(out / "train_inspection.md", "\n".join(text))

    print("TRAIN inspection complete")
    print(f"Documents: {summary['n_documents']:,}")
    print(f"Sentences: {summary['n_sentences']:,}")
    print(f"Groups: {summary['n_groups']:,}")
    print(f"Missing model-feature values: {summary['missing_feature_values']:,}")
    print(f"Saved to: {out}")


if __name__ == "__main__":
    main()
