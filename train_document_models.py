"""Grouped train/validation/test comparison of document-style models and optional TF-IDF.

Run after build_feature_table.py:
  python train_document_models.py --input document_features.csv --output-dir results --with-tfidf

The learner sees numeric text-derived features only. Source, generator, prompt and label
are never predictive features. Label is the target. Group is used for split only.
"""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score,
                             precision_score, recall_score, f1_score, confusion_matrix)
from sklearn.feature_extraction.text import TfidfVectorizer
from build_feature_table import MODEL_FEATURE_NAMES


def score(y, probabilities, threshold):
    predicted = (probabilities >= threshold).astype(int)
    result = {
        'accuracy': accuracy_score(y, predicted),
        'precision_ai': precision_score(y, predicted, zero_division=0),
        'recall_ai': recall_score(y, predicted, zero_division=0),
        'f1_ai': f1_score(y, predicted, zero_division=0),
        'roc_auc': roc_auc_score(y, probabilities) if len(np.unique(y)) == 2 else np.nan,
        'pr_auc': average_precision_score(y, probabilities) if len(np.unique(y)) == 2 else np.nan,
        'tn_fp_fn_tp': confusion_matrix(y, predicted, labels=[0, 1]).ravel().tolist(),
    }
    return result


def choose_threshold(y, val_probs):
    # Threshold is tuned ONLY on validation, NEVER on test.
    thresholds = np.linspace(0.10, 0.90, 81)
    f1_values = [f1_score(y, val_probs >= t, zero_division=0) for t in thresholds]
    return float(thresholds[int(np.argmax(f1_values))])


def split_once(df, seed):
    # Two-stage split: 70% train / ~15% validation / ~15% test BY GROUP.
    split1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    i_train, i_hold = next(split1.split(df, df.label, groups=df.group_key))
    train, hold = df.iloc[i_train].copy(), df.iloc[i_hold].copy()
    split2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed + 1)
    i_val, i_test = next(split2.split(hold, hold.label, groups=hold.group_key))
    val, test = hold.iloc[i_val].copy(), hold.iloc[i_test].copy()
    for name, sub in [('train', train), ('validation', val), ('test', test)]:
        if sub.label.nunique() != 2:
            raise ValueError(f'{name} has only one class. Try another --seed or add more groups.')
    return train, val, test


def make_models(n_train, seed):
    models = {
        'logistic_style': make_pipeline(
            SimpleImputer(strategy='median'), StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight='balanced', random_state=seed)),
        'random_forest_style': make_pipeline(
            SimpleImputer(strategy='median'),
            RandomForestClassifier(n_estimators=200, min_samples_leaf=3,
                                   class_weight='balanced_subsample', random_state=seed,
                                   n_jobs=-1)),
        'hist_gradient_boosting_style': make_pipeline(
            SimpleImputer(strategy='median'),
            HistGradientBoostingClassifier(max_iter=120, max_leaf_nodes=15,
                                           l2_regularization=1.0, random_state=seed)),
    }
    try:
        from xgboost import XGBClassifier
        models['xgboost_style'] = make_pipeline(
            SimpleImputer(strategy='median'),
            XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                          subsample=0.9, colsample_bytree=0.9, eval_metric='logloss',
                          tree_method='hist', random_state=seed, n_jobs=4))
    except ImportError:
        print('Optional xgboost not installed: skipping XGBoost (pip install xgboost).')
    return models


def main(args):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input).dropna(subset=['text', 'label', 'group_key'])
    required = set(MODEL_FEATURE_NAMES + ['text', 'label', 'group_key'])
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f'Missing feature-table columns: {missing}')
    df['label'] = df['label'].astype(int)
    train, val, test = split_once(df, args.seed)
    for name, split_df in [('train', train), ('validation', val), ('test', test)]:
        split_df.to_csv(out / f'{name}_documents.csv', index=False)
        print(f'{name}: {len(split_df):,} rows, {split_df.group_key.nunique()} groups; '
              f'labels {split_df.label.value_counts().to_dict()}')
    assert not set(train.group_key) & set(val.group_key)
    assert not set(train.group_key) & set(test.group_key)
    assert not set(val.group_key) & set(test.group_key)

    # EDA must use TRAIN ONLY. These single-feature AUCs are screening results,
    # not final performance claims; absolute AUROC ignores direction of association.
    feature_rows = []
    for col in MODEL_FEATURE_NAMES:
        vals = pd.to_numeric(train[col], errors='coerce')
        if vals.notna().sum() != len(vals) or vals.nunique() < 2:
            auc = np.nan
        else:
            auc_raw = roc_auc_score(train.label, vals)
            auc = max(auc_raw, 1 - auc_raw)
        feature_rows.append({
            'feature': col,
            'human_train_mean': vals[train.label == 0].mean(),
            'ai_train_mean': vals[train.label == 1].mean(),
            'single_feature_train_auc_separation': auc,
        })
    pd.DataFrame(feature_rows).sort_values(
        'single_feature_train_auc_separation', ascending=False
    ).to_csv(out / 'train_only_feature_screening.csv', index=False)
    train[MODEL_FEATURE_NAMES].corr(method='spearman').to_csv(out / 'train_only_feature_correlation.csv')
    print('Saved TRAIN-ONLY feature screening and correlation CSVs.')

    X_train, y_train = train[MODEL_FEATURE_NAMES], train.label.to_numpy()
    X_val, y_val = val[MODEL_FEATURE_NAMES], val.label.to_numpy()
    X_test, y_test = test[MODEL_FEATURE_NAMES], test.label.to_numpy()
    models = make_models(len(train), args.seed)
    if args.with_tfidf:
        # TF-IDF is an ALTERNATIVE feature representation, not another classifier.
        models['tfidf_logistic_baseline'] = make_pipeline(
            TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2,
                            sublinear_tf=True),
            LogisticRegression(max_iter=2000, class_weight='balanced'))

    metrics = []
    prediction_table = test[['text', 'label', 'group_key'] +
                            [c for c in ['source_dataset', 'domain', 'generator'] if c in test]].copy()
    for name, model in models.items():
        is_text = name == 'tfidf_logistic_baseline'
        tr = train.text.tolist() if is_text else X_train
        va = val.text.tolist() if is_text else X_val
        te = test.text.tolist() if is_text else X_test
        print(f'Fitting {name} ...', flush=True)
        model.fit(tr, y_train)
        val_p = model.predict_proba(va)[:, 1]
        threshold = choose_threshold(y_val, val_p)
        test_p = model.predict_proba(te)[:, 1]
        val_result = score(y_val, val_p, threshold)
        test_result = score(y_test, test_p, threshold)
        for split_name, result in [('validation', val_result), ('test', test_result)]:
            metrics.append({'model': name, 'split': split_name,
                            'threshold_tuned_on_validation': threshold, **result})
        prediction_table[name + '_prob_ai'] = test_p
        prediction_table[name + '_pred_ai'] = (test_p >= threshold).astype(int)
        joblib.dump({'model': model, 'threshold': threshold,
                     'feature_columns': None if is_text else MODEL_FEATURE_NAMES},
                    out / (name + '.joblib'))
        print(f'  validation ROC-AUC={val_result["roc_auc"]:.3f}; '
              f'test ROC-AUC={test_result["roc_auc"]:.3f}; '
              f'test F1={test_result["f1_ai"]:.3f}; threshold={threshold:.2f}')
    pd.DataFrame(metrics).to_csv(out / 'model_comparison.csv', index=False)
    prediction_table.to_csv(out / 'test_predictions_for_error_analysis.csv', index=False)
    print(f'Results saved to: {out.resolve()}')
    print('IMPORTANT: The test set was used for final reporting only. Do not pick '
          'features/hyperparameters using the test set; rerun a fresh external test for final claims.')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default='document_features.csv')
    ap.add_argument('--output-dir', default='results')
    ap.add_argument('--with-tfidf', action='store_true')
    ap.add_argument('--seed', type=int, default=42)
    main(ap.parse_args())
