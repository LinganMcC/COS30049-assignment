

import argparse
import hashlib
import re
import unicodedata

import pandas as pd


CANONICAL_COLUMNS = ["text", "label", "source_dataset", "domain", "generator", "group_key"]


# ----------------------------------------------------------------------
# Cleaning (applied identically to every source, after adaptation)
# ----------------------------------------------------------------------
def clean_text(text) -> str:
    """Minimal cleaning: fix encoding artifacts, normalize whitespace,
    strip stray HTML. Deliberately does NOT lowercase or strip
    punctuation -- those are style signal for this task, not noise."""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"<[^>]+>", "", text)
    return text.strip()


def _hash(s: str) -> str:
    return hashlib.md5(str(s).lower().strip().encode("utf-8")).hexdigest()


# ----------------------------------------------------------------------
# Per-source adapters -- each maps a different native shape into
# CANONICAL_COLUMNS
# ----------------------------------------------------------------------
def adapt_drcat(path: str) -> pd.DataFrame:
    """Base dataset: train_v2_drcat_02.csv. One row per document,
    columns: text, label, prompt_name, source, RDizzl3_seven."""
    df = pd.read_csv(path)
    return pd.DataFrame({
        "text": df["text"],
        "label": df["label"].astype(int),
        "source_dataset": "drcat_v2",
        "domain": "student_essay",
        "generator": df["source"],
        "group_key": df["prompt_name"],   # exact topic id available -- use it directly
    })


def adapt_hc3(path: str) -> pd.DataFrame:
    """HC3-style dataset: one row per QUESTION, with human_answers and
    chatgpt_answers as LIST columns (loaded from JSONL) -- a genuinely
    different shape from drcat's one-row-per-example format, and it has
    NO prompt_name equivalent, only a `question` and a `source` (domain)
    field. group_key here is a hash of the question text: every answer
    to the SAME question is grouped together so they can't split across
    train/val, even though there's no explicit topic id column."""
    df = pd.read_json(path, lines=True)
    rows = []
    for _, r in df.iterrows():
        domain = r.get("source", "unknown")
        q_key = _hash(r.get("question", ""))
        for ans in r.get("human_answers", []):
            rows.append({"text": ans, "label": 0, "source_dataset": "hc3",
                         "domain": domain, "generator": "human", "group_key": q_key})
        for ans in r.get("chatgpt_answers", []):
            rows.append({"text": ans, "label": 1, "source_dataset": "hc3",
                         "domain": domain, "generator": "chatgpt", "group_key": q_key})
    return pd.DataFrame(rows, columns=CANONICAL_COLUMNS)


def adapt_generic_binary_csv(path: str, text_col: str, label_col: str,
                              source_name: str, domain: str = "unknown",
                              group_col: str = None, positive_value=1) -> pd.DataFrame:
    """Generic fallback adapter for a CSV with SOME text column and SOME
    binary-ish label column under different names -- covers most
    Kaggle-style 'AI vs human text' datasets without writing a bespoke
    function per dataset. If group_col isn't available in that dataset,
    group_key falls back to a per-row unique id (== no grouping, a
    documented limitation, not silently ignored)."""
    df = pd.read_csv(path)
    label = (df[label_col] == positive_value).astype(int) if df[label_col].dtype == object \
        else df[label_col].astype(int)

    if group_col and group_col in df.columns:
        group_key = df[group_col].astype(str)
    else:
        print(f"WARNING: '{source_name}' has no group column -- falling back to "
              f"per-row unique ids (no topic-leakage protection for this source).")
        group_key = pd.Series([f"{source_name}_{i}" for i in range(len(df))])

    return pd.DataFrame({
        "text": df[text_col],
        "label": label,
        "source_dataset": source_name,
        "domain": domain,
        "generator": label.map({0: "human", 1: "unknown_ai"}),
        "group_key": group_key,
    })


# ----------------------------------------------------------------------
# Unification: clean + dedup ACROSS all sources
# ----------------------------------------------------------------------
def unify(frames: list, min_chars: int = 30, max_chars: int = 20000) -> pd.DataFrame:
    df = pd.concat(frames, ignore_index=True)[CANONICAL_COLUMNS]
    df["text"] = df["text"].apply(clean_text)

    lengths = df["text"].str.len()
    before = len(df)
    df = df[(lengths >= min_chars) & (lengths <= max_chars)].reset_index(drop=True)
    print(f"Dropped {before - len(df)} rows outside [{min_chars}, {max_chars}] chars")

    # dedup ACROSS sources (some public datasets reuse the same underlying
    # human/AI text, e.g. GPT-2 output or essay corpora appearing in
    # multiple places)
    df["_hash"] = df["text"].apply(_hash)
    before = len(df)
    df = df.drop_duplicates(subset="_hash").drop(columns="_hash").reset_index(drop=True)
    print(f"Dropped {before - len(df)} cross-source duplicate texts")

    print("\nRows per source:")
    print(df.groupby("source_dataset")["label"].agg(["count", "mean"]).rename(columns={"mean": "pct_ai"}))

    n_singleton_groups = (df["group_key"].value_counts() == 1).sum()
    print(f"\n{n_singleton_groups} / {df['group_key'].nunique()} group_keys are singleton "
          f"(no natural grouping available for those rows -- expected for sources "
          f"like the generic-CSV fallback).")

    return df


def main(drcat_path, hc3_path, out_path):
    frames = []
    if drcat_path:
        print(f"Adapting drcat source: {drcat_path}")
        frames.append(adapt_drcat(drcat_path))
    if hc3_path:
        print(f"Adapting HC3-style source: {hc3_path}")
        frames.append(adapt_hc3(hc3_path))

    if not frames:
        raise SystemExit("No input sources given -- pass at least --drcat or --hc3")

    unified = unify(frames)
    unified.to_csv(out_path, index=False)
    print(f"\nSaved unified dataset ({len(unified)} rows) to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--drcat", default=None, help="path to train_v2_drcat_02.csv")
    parser.add_argument("--hc3", default=None, help="path to an HC3-style JSONL file")
    parser.add_argument("--out", default="unified_dataset.csv")
    args = parser.parse_args()
    main(args.drcat, args.hc3, args.out)
