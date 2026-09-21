"""OPTIONAL: measure whether an actual reference-LM log-perplexity adds useful signal.

Run on your own Internet-connected computer only if you choose to test perplexity:
  python -m pip install torch transformers
  python test_perplexity_ablation.py --input document_features.csv --sample-per-class 100

First run downloads distilbert/distilgpt2 from Hugging Face. One forward pass per
text makes this an intentionally SMALL controlled experiment, not full-dataset
feature extraction. Never call a word-frequency heuristic 'perplexity'.

Evaluation: fixed grouped split; same sampled rows for style-only vs style+LM;
threshold selected on validation; AUC/F1 reported on an untouched test split.
"""
import argparse
import math
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, f1_score
from train_document_models import split_once, choose_threshold
from build_feature_table import MODEL_FEATURE_NAMES


def calculate_log_perplexity(texts, model_name, max_tokens):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    lm = AutoModelForCausalLM.from_pretrained(model_name)
    lm.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lm.to(device)
    result = []
    with torch.no_grad():
        for i, text in enumerate(texts):
            tokens = tokenizer(str(text), return_tensors='pt', truncation=True,
                               max_length=max_tokens)['input_ids'].to(device)
            if tokens.shape[1] < 2:
                result.append(float('nan'))
                continue
            loss = lm(input_ids=tokens, labels=tokens).loss.item()
            result.append(float(loss))  # log(perplexity); avoids overflow
            if (i + 1) % 25 == 0:
                print(f'LM-scored {i+1}/{len(texts)} passages', flush=True)
    return result


def report(name, tr, val, test, cols):
    model = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                          LogisticRegression(max_iter=2000, class_weight='balanced'))
    model.fit(tr[cols], tr.label)
    val_probs = model.predict_proba(val[cols])[:, 1]
    threshold = choose_threshold(val.label, val_probs)
    test_probs = model.predict_proba(test[cols])[:, 1]
    print(f'{name}: test ROC-AUC={roc_auc_score(test.label, test_probs):.3f}, '
          f'test F1={f1_score(test.label, test_probs >= threshold):.3f}; '
          f'validation-tuned threshold={threshold:.2f}')
    return test_probs


def main(args):
    df = pd.read_csv(args.input)
    if not set(MODEL_FEATURE_NAMES + ['text', 'label', 'group_key']).issubset(df):
        raise ValueError('Use the output of build_feature_table.py as input.')
    # Stratified sampling only chooses examples; it does NOT allocate groups
    # across train/test. The group split happens AFTER sampling.
    selected = [part.sample(n=min(args.sample_per_class, len(part)), random_state=args.seed)
                for _, part in df.groupby('label')]
    subset = pd.concat(selected, ignore_index=True)
    train, val, test = split_once(subset, args.seed)
    print('Same grouped train/validation/test documents are used for both models.')
    texts = pd.concat([train, val, test]).text.tolist()
    logppl = calculate_log_perplexity(texts, args.model_name, args.max_tokens)
    joined = pd.concat([train, val, test]).copy()
    joined['log_perplexity'] = logppl
    joined = joined.dropna(subset=['log_perplexity'])
    tr = joined.loc[joined.index.isin(train.index)]
    va = joined.loc[joined.index.isin(val.index)]
    te = joined.loc[joined.index.isin(test.index)]
    if any(x.label.nunique() != 2 for x in [tr, va, te]):
        raise ValueError('Subset split lost a class; increase sample size or change --seed.')
    val_auc = roc_auc_score(te.label, te.log_perplexity)
    print('Log-perplexity alone: test AUROC separation =', round(max(val_auc, 1-val_auc), 3))
    a = report('Style only', tr, va, te, MODEL_FEATURE_NAMES)
    b = report('Style + log-perplexity', tr, va, te, MODEL_FEATURE_NAMES + ['log_perplexity'])
    # Diagnostic results; avoid choosing your FINAL design using this test set.
    print('Difference in test AUROC (style+LM minus style only):',
          round(roc_auc_score(te.label, b) - roc_auc_score(te.label, a), 4))
    print('Interpret only as a small preliminary ablation. Repeat with multiple seeds '
          'and a separate HC3/M4 external test to judge cross-domain usefulness.')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input', default='document_features.csv')
    p.add_argument('--sample-per-class', type=int, default=100)
    p.add_argument('--max-tokens', type=int, default=256)
    p.add_argument('--model-name', default='distilbert/distilgpt2')
    p.add_argument('--seed', type=int, default=42)
    main(p.parse_args())
