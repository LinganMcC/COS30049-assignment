# Logistic Regression baseline

This folder is the cleaned Logistic Regression part of the AI-generated content detector.
It assumes your project already has separate `preprocess/` and `feature_extraction/` folders.
It **does not duplicate** `features.py`, `feature_config.py`, or the feature-table builder.

## Expected project layout

```text
project/
├── preprocess/
├── feature_extraction/
│   ├── build_final_sentence_feature_table.py
│   ├── feature_config.py
│   └── features.py
├── data/
│   └── features/
│       ├── drcat_train_final_sentence_features.csv
│       ├── drcat_validation_final_sentence_features.csv
│       ├── drcat_test_final_sentence_features.csv
│       └── hc3_external_test_final_sentence_features.csv
└── logistic_regression/
    ├── config.py
    ├── run_development.py
    ├── run_final_evaluation.py
    ├── scripts/
    ├── src/
    ├── experiments/
    └── tests/
```

If your directories are elsewhere, set:

```bash
export LR_FEATURE_DIR=/path/to/feature_extraction
export LR_DATA_DIR=/path/to/data/features
export LR_OUTPUT_DIR=/path/to/output
```

On Windows Git Bash, the same `export` syntax works for that terminal session.

---

## What changed from the original ZIP

1. **No duplicate feature-extraction package.** The model imports `FINAL_FEATURES` from the real sibling `feature_extraction/feature_config.py`.
2. **Grouped CV is constructed at document level.** `StratifiedGroupKFold` sees one row per document and groups by `group_key`; sentence rows are mapped back only after the document fold is decided.
3. **Sentence weighting is research, not assumption.** CV compares `uniform`, `doc_equal`, and `sqrt_doc`.
4. **Class balancing compares only `none` vs `doc_balanced`.** `sklearn class_weight='balanced'` was removed because it balances sentence counts and can double-count weighting logic.
5. **C uses two-stage tuning.** Coarse search first, then a focused fine search.
6. **Pooling is selected with TRAIN CV.** Mean and sqrt-length pooling are compared without an arbitrary `0.002` rule.
7. **Model ranking uses ROC-AUC + Average Precision.** Threshold-0.5 Accuracy/F1 are diagnostics only during model-family selection.
8. **Average Precision is called `average_precision`, not `pr_auc`.**
9. **Validation is reserved for threshold selection.** Threshold grid is 0.05–0.95 by 0.01; macro-F1 chooses the operating point, ties go closest to 0.5.
10. **TEST and HC3 require an explicit final command.** Development code cannot accidentally run them.
11. **95% CIs use group/cluster bootstrap by `group_key`.** This respects prompt/question clustering.
12. **Error-analysis contributions are sentence-logit contributions only.** The code no longer pretends an averaged z-score × coefficient exactly decomposes a nonlinear pooled document probability.
13. **Stress tests are optional experiments.** They are outside the core pipeline.
14. **Short-document stress recalculates `rel_len_deviation`** after truncating sentences.
15. **Missing values are reported, then median-imputed using TRAIN-fitted medians.**

---

# Recommended workflow

## 0. Install

```bash
cd logistic_regression
pip install -r requirements.txt
```

## 1. Inspect TRAIN + run the broad tuning experiment

```bash
python run_development.py
```

This creates:

```text
outputs/
├── 01_inspection/
└── 02_coarse_tuning/
    ├── cv_fold_structure.csv
    ├── all_fold_results.csv
    ├── configuration_summary.csv
    ├── recommended_config.json
    └── coarse_tuning_report.md
```

The full coarse experiment is:

```text
sentence weighting:
    uniform
    doc_equal
    sqrt_doc

class balance:
    none
    doc_balanced

C:
    0.001
    0.01
    0.1
    1
    10
    100

pooling:
    mean
    sqrt_length
```

There are 3 × 2 × 6 = 36 fitted LR configurations per fold. Pooling reuses the same sentence probabilities, so it does not refit the model.

**This is the first important decision point.** Look at `configuration_summary.csv` rather than blindly accepting rank 1.

Useful columns:

```text
mean_roc_auc
std_roc_auc
mean_average_precision
std_average_precision
mean_iterations
selection_score
```

`selection_score` is simply:

```text
(mean ROC-AUC + mean Average Precision) / 2
```

It is a transparent recommendation score, not a claim that it is the only valid scientific objective.

## 2. Fine-tune C around the selected coarse region

If you agree with the coarse recommendation:

```bash
python scripts/03_fine_tuning.py
```

For example, if coarse `C=0.1`, the automatic fine grid is approximately:

```text
0.03, 0.05, 0.1, 0.2, 0.3
```

You can override every part:

```bash
python scripts/03_fine_tuning.py \
    --weighting sqrt_doc \
    --class-balance doc_balanced \
    --pooling sqrt_length \
    --center-c 0.1 \
    --c-values 0.03 0.05 0.1 0.2 0.3
```

Inspect:

```text
outputs/03_fine_tuning/configuration_summary.csv
```

## 3. Validate the chosen model configuration WITHOUT freezing first

Example:

```bash
python scripts/04_validate_freeze.py \
    --weighting sqrt_doc \
    --class-balance doc_balanced \
    --c 0.1 \
    --pooling sqrt_length \
    --no-freeze
```

Or explicitly accept the CV recommendation:

```bash
python scripts/04_validate_freeze.py --use-cv-recommendation --no-freeze
```

This fits the selected configuration on all TRAIN data, scores VALIDATION, and produces:

```text
outputs/04_validation/
├── validation_document_scores.csv
├── threshold_sweep.csv
├── threshold_sweep.png
├── coefficients.csv
├── coefficients.png
├── candidate_config.json
└── validation_report.md
```

Threshold selection uses **macro-F1** across 0.05–0.95 in steps of 0.01. ROC-AUC and Average Precision do not change when the threshold changes.

## 4. Freeze only after you accept the configuration

Run the same command without `--no-freeze`:

```bash
python scripts/04_validate_freeze.py \
    --weighting sqrt_doc \
    --class-balance doc_balanced \
    --c 0.1 \
    --pooling sqrt_length
```

or:

```bash
python scripts/04_validate_freeze.py --use-cv-recommendation
```

This writes:

```text
outputs/05_frozen_model/
├── logistic_regression.joblib
└── frozen_config.json
```

## 5. Only now run TEST + HC3

```bash
python run_final_evaluation.py
```

This produces final DRCAT test and HC3 external results with:

```text
Accuracy
Precision
Recall
F1
ROC-AUC
Average Precision
confusion matrix
95% group-bootstrap confidence intervals
```

The command also produces FP/FN error analysis.

Do **not** return to tuning because you dislike TEST or HC3 results. Those sets are evaluation only.

---

# Optional experiments

```bash
python experiments/stress_tests.py
```

This is intentionally outside the main pipeline. It includes:

- generator/domain/source slice analysis;
- short-document / first-k-sentence robustness;
- leave-one-generator-out retraining on TRAIN with evaluation on VALIDATION.

These experiments should support the report, not determine the frozen TEST/HC3 model after final evaluation has been seen.

---

# How to read the tuning results

Do not look only at rank 1. Check:

1. Is the winner consistently better across folds, or only better by a tiny amount?
2. Do nearby C values perform almost identically? If yes, prefer the simpler/more regularised region rather than pretending a tiny difference is meaningful.
3. Does `sqrt_doc` genuinely beat `doc_equal`/`uniform`? If not, remove the extra complexity.
4. Does `doc_balanced` help? If results are essentially the same, `none` is simpler.
5. Does `sqrt_length` pooling consistently improve both ROC-AUC and Average Precision? If not, simple mean pooling is easier to justify.
6. Is `mean_iterations` comfortably below `max_iter=3000`? If yes, there is no reason to tune solver/max_iter.

This is the evidence we should use together to decide the final Logistic Regression configuration.
