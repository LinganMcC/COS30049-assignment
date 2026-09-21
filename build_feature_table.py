"""Build one row of DOCUMENT features per text, without copying document labels to sentences.

Usage:
  python build_feature_table.py --input unified_dataset.csv --output document_features.csv
  python build_feature_table.py --input unified_dataset.csv --output quick_features.csv --sample-per-class 200

Input can also be the raw DRCAT CSV. Preserves provenance columns as evaluation metadata,
not predictive features. Requires features.py in the same directory.
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from features import (
    document_features, featurize_passage, SENTENCE_FEATURE_NAMES,
    DOCUMENT_FEATURE_NAMES,
)

# We keep the existing 14 MIL sentence features unchanged, but avoid introducing
# average sentence length as a separate document classifier feature. The other
# sentence averages let the document models use signals already in your project.
AGGREGATED_SENTENCE_FEATURES = [name for name in SENTENCE_FEATURE_NAMES if name != 'n_words']
MODEL_FEATURE_NAMES = DOCUMENT_FEATURE_NAMES + [f'sent_avg_{name}' for name in AGGREGATED_SENTENCE_FEATURES]
METADATA = ['label', 'source_dataset', 'domain', 'generator', 'group_key']


def main(args):
    df = pd.read_csv(args.input)
    if 'text' not in df or 'label' not in df:
        raise ValueError('Input must contain text and label columns.')
    df = df.dropna(subset=['text', 'label']).copy()
    df['text'] = df['text'].astype(str)
    df = df[df['text'].str.strip().str.len() >= 30].copy()
    df['label'] = pd.to_numeric(df['label'], errors='raise').astype(int)
    if not set(df['label'].unique()).issubset({0, 1}):
        raise ValueError('Expected 0=human and 1=AI labels only.')
    if args.sample_per_class:
        # Sample within each class without losing the original label column.
        parts = [part.sample(n=min(len(part), args.sample_per_class),
                             random_state=args.seed)
                 for _, part in df.groupby('label')]
        df = pd.concat(parts, ignore_index=True)
    df = df.reset_index(drop=True)
    df['source_dataset'] = df.get('source_dataset', pd.Series('drcat_v2', index=df.index)).fillna('unknown')
    df['domain'] = df.get('domain', pd.Series('student_essay', index=df.index)).fillna('unknown')
    df['generator'] = df.get('generator', df.get('source', pd.Series('unknown', index=df.index))).fillna('unknown')
    if 'group_key' not in df:
        if 'prompt_name' not in df:
            raise ValueError('Need group_key or prompt_name. Do not fall back to random row split.')
        df['group_key'] = df['prompt_name']
    # Prevent similarly named groups in independent datasets being merged accidentally.
    # HC3 question hashes are group keys; DRCAT prompt names are group keys.
    df['group_key'] = df['source_dataset'].astype(str) + '::' + df['group_key'].astype(str)
    if df['group_key'].isna().any():
        raise ValueError('Missing group keys: repair before training.')

    rows = []
    for i, text in enumerate(df['text']):
        doc_values = document_features(text)
        sentences, X_sent = featurize_passage(text)
        if X_sent.shape[0] == 0:
            continue
        sentence_means = dict(zip(SENTENCE_FEATURE_NAMES, np.mean(X_sent, axis=0)))
        row = {key: float(doc_values[key]) for key in DOCUMENT_FEATURE_NAMES}
        row.update({f'sent_avg_{k}': float(sentence_means[k]) for k in AGGREGATED_SENTENCE_FEATURES})
        row.update({key: df.at[i, key] for key in METADATA})
        row['text'] = text  # only for optional TF-IDF experiment and error analysis
        row['analysis_n_sentences'] = len(sentences)  # NOT a predictive feature
        row['analysis_n_words'] = sum(len(s.split()) for s in sentences)  # NOT a predictive feature
        rows.append(row)
        if (i + 1) % 5000 == 0:
            print(f'Featurized {i+1:,}/{len(df):,} documents', flush=True)
    if not rows:
        raise ValueError('No usable documents.')
    table = pd.DataFrame(rows)
    assert list(table[MODEL_FEATURE_NAMES].columns) == MODEL_FEATURE_NAMES
    if not np.isfinite(table[MODEL_FEATURE_NAMES].to_numpy(dtype=float)).all():
        raise ValueError('Non-finite features; inspect input text.')
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)
    print(f'Wrote {len(table):,} documents x {len(MODEL_FEATURE_NAMES)} model features: {output}')
    print('Labels:', table.label.value_counts().to_dict())
    print('Unique groups:', table.group_key.nunique())
    print('Predictive columns:', MODEL_FEATURE_NAMES)
    print('text/label/source_dataset/domain/generator/group_key are not in the classifier feature matrix.')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--output', default='document_features.csv')
    ap.add_argument('--sample-per-class', type=int, default=0)
    ap.add_argument('--seed', type=int, default=42)
    main(ap.parse_args())
