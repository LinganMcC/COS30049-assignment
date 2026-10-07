# COS30049 AI-Generated Content Detection

This repository contains the data preparation, sentence segmentation, feature extraction, and Logistic Regression baseline for the COS30049 AI-Generated Content Detection project.

The current README documents the reproducible pipeline **from raw datasets through the completed Logistic Regression model**. The `K-Means/` and later Transformer/model work are separate team components and can be documented when those implementations are finalised.

---

## 1. Project structure

```text
COS30049-assignment/
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── sentences/
│   └── features/
│
├── Preprocess/
│   ├── data_adapters.py
│   ├── data_cleaning.py
│   ├── data_splitting.py
│   ├── download_hc3.py
│   ├── preprocess.py
│   ├── preprocess_report.py
│   └── sentence_splitter.py
│
├── Feature_extraction/
│   ├── build_final_sentence_feature_table.py
│   ├── feature_config.py
│   └── features.py
│
├── logistic_regression/
│   ├── config.py
│   ├── requirements.txt
│   ├── run_development.py
│   ├── run_final_evaluation.py
│   ├── scripts/
│   │   ├── 01_inspect_train.py
│   │   ├── 02_coarse_tuning.py
│   │   ├── 03_fine_tuning.py
│   │   ├── 04_validate_freeze.py
│   │   ├── 05_final_eval.py
│   │   └── 06_error_analysis.py
│   ├── src/
│   ├── experiments/
│   ├── tests/
│   └── outputs/
│
├── K-Means/                         # teammate component
├── results/
│   └── feature_analysis/
│
├── analyze_features_for_report.py
├── evaluate_full_train_15_features.py
├── requirements.txt
└── README.md
```

### Main pipeline

```text
Raw DRCAT + HC3
        ↓
Document adaptation and cleaning
        ↓
DRCAT TRAIN / VALIDATION / TEST
HC3 external test
        ↓
spaCy sentence segmentation
        ↓
15 sentence-level features
        ↓
DRCAT TRAIN grouped CV
        ↓
Logistic Regression tuning
        ↓
VALIDATION threshold selection
        ↓
Frozen Logistic Regression model
        ↓
DRCAT TEST + HC3 external evaluation
        ↓
False-positive / false-negative analysis
```

---

## 2. Environment setup

Run commands from the **project root** unless a later section explicitly says to change directory.

### 2.1 Optional virtual environment

```bash
python -m venv .venv
```

Activate it.

**Windows Git Bash**

```bash
source .venv/Scripts/activate
```

**Windows PowerShell**

```powershell
.venv\Scripts\Activate.ps1
```

**Linux/macOS**

```bash
source .venv/bin/activate
```

### 2.2 Install project dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r logistic_regression/requirements.txt
```

The preprocessing and final feature extractor use spaCy. Install spaCy and its English model if they are not already installed:

```bash
python -m pip install spacy
python -m spacy download en_core_web_sm
```

The Logistic Regression-specific requirements include packages such as `joblib` and `tabulate`, which are needed for saving the model and generating Markdown result tables.

---

## 3. Prepare the raw datasets

Create the raw-data folder if necessary:

```bash
mkdir -p data/raw
```

### 3.1 DRCAT

Obtain the **DAIGT V2 Train Dataset (DRCAT)** and place:

```text
train_v2_drcat_02.csv
```

at:

```text
data/raw/train_v2_drcat_02.csv
```

Dataset source:

```text
https://www.kaggle.com/datasets/thedrcat/daigt-v2-train-dataset
```

### 3.2 HC3

The project contains a downloader for the official English HC3 `all.jsonl` file.

Run:

```bash
python Preprocess/download_hc3.py --out data/raw/hc3_all.jsonl
```

Expected file:

```text
data/raw/hc3_all.jsonl
```

If the file already exists, the downloader keeps the existing copy. Use `--overwrite` only if a fresh copy is intentionally required.

---

## 4. Run document-level preprocessing

Run:

```bash
python Preprocess/preprocess.py \
    --drcat data/raw/train_v2_drcat_02.csv \
    --hc3 data/raw/hc3_all.jsonl
```

The frozen preprocessing configuration uses:

```text
HC3 minimum document length: 30 characters
HC3 maximum document length: 20,000 characters
HC3 external question groups: 1,000
random seed: 42
```

These are already the script defaults, so they do not need to be supplied unless intentionally changed.

### What preprocessing does

The pipeline:

1. adapts DRCAT and HC3 into the common schema;
2. performs conservative text cleaning;
3. removes same-label duplicates;
4. applies the 30–20,000 character filter to HC3 only;
5. creates leakage-safe DRCAT prompt-group TRAIN / VALIDATION / TEST partitions;
6. creates the balanced HC3 external test;
7. generates preprocessing audit files.

### Expected output

```text
data/processed/
├── unified_documents.csv
├── drcat_train.csv
├── drcat_validation.csv
├── drcat_test.csv
├── hc3_external_test.csv
├── transformation_examples.csv
└── preprocess_report.txt
```

Expected final document counts:

| Dataset | Documents |
|---|---:|
| DRCAT TRAIN | 31,119 |
| DRCAT VALIDATION | 7,133 |
| DRCAT TEST | 6,612 |
| HC3 external test | 2,000 |

`unified_documents.csv` is for audit, analysis, and visualisation. **Do not randomly split this file for model development.** DRCAT remains the development dataset and HC3 remains external evaluation data.

---

## 5. Split documents into sentences

The final sentence splitter uses spaCy `en_core_web_sm`.

Run all processed datasets:

```bash
python Preprocess/sentence_splitter.py \
    --processed-dir data/processed \
    --out-dir data/sentences
```

Expected sentence files:

```text
data/sentences/
├── drcat_train_sentences.csv
├── drcat_validation_sentences.csv
├── drcat_test_sentences.csv
├── hc3_external_test_sentences.csv
├── drcat_train_sentences_removed_fragments.csv
├── drcat_validation_sentences_removed_fragments.csv
├── drcat_test_sentences_removed_fragments.csv
└── hc3_external_test_sentences_removed_fragments.csv
```

The `*_removed_fragments.csv` files are audit files for fragments containing no alphabetic characters.

Each retained sentence preserves:

```text
sentence_id
document_id
sentence_index
text
label
label_origin
source_dataset
group_key
domain
generator
```

`label_origin` is recorded as `document_inherited` because the source datasets provide document-level labels rather than independently labelled sentences.

Expected usable sentence counts:

| Dataset | Sentences |
|---|---:|
| DRCAT TRAIN | 592,757 |
| DRCAT VALIDATION | 158,996 |
| DRCAT TEST | 151,878 |
| HC3 external test | 14,752 |

---

## 6. Build the final 15-feature sentence tables

The production feature extractor uses:

- 9 stylometric / lexical features;
- 6 spaCy POS-ratio features.

The frozen feature list is:

```text
n_words
long_word_ratio
func_word_ratio
unique_word_ratio
comma_rate
exclaim
apostrophe_rate
starts_with_transition
rel_len_deviation
noun_ratio
adjective_ratio
adverb_ratio
pronoun_ratio
auxiliary_ratio
conjunction_ratio
```

Create the feature output directory:

```bash
mkdir -p data/features
```

### 6.1 TRAIN

```bash
python Feature_extraction/build_final_sentence_feature_table.py \
    --input data/sentences/drcat_train_sentences.csv \
    --output data/features/drcat_train_final_sentence_features.csv
```

### 6.2 VALIDATION

```bash
python Feature_extraction/build_final_sentence_feature_table.py \
    --input data/sentences/drcat_validation_sentences.csv \
    --output data/features/drcat_validation_final_sentence_features.csv
```

### 6.3 DRCAT TEST

```bash
python Feature_extraction/build_final_sentence_feature_table.py \
    --input data/sentences/drcat_test_sentences.csv \
    --output data/features/drcat_test_final_sentence_features.csv
```

### 6.4 HC3 external test

```bash
python Feature_extraction/build_final_sentence_feature_table.py \
    --input data/sentences/hc3_external_test_sentences.csv \
    --output data/features/hc3_external_test_final_sentence_features.csv
```

Expected output:

```text
data/features/
├── drcat_train_final_sentence_features.csv
├── drcat_validation_final_sentence_features.csv
├── drcat_test_final_sentence_features.csv
└── hc3_external_test_final_sentence_features.csv
```

The feature builder validates that all 15 model features exist and stops if any NaN or infinite model-feature values are detected.

---

## 7. Optional TRAIN-only feature analysis for the report

This step is **analysis only**. It does not retrain or modify the model.

Run from the project root:

```bash
python analyze_features_for_report.py \
    --input data/features/drcat_train_final_sentence_features.csv \
    --output-dir results/feature_analysis/report
```

Expected outputs:

```text
results/feature_analysis/report/
├── feature_class_summary.csv
├── feature_effect_sizes.png
└── feature_correlation_heatmap.png
```

These files provide:

- Human and AI means / medians for each final feature;
- Cohen's d Human–AI effect sizes;
- Pearson correlations among the 15 final features.

Only **DRCAT TRAIN** is used for this analysis so VALIDATION, TEST, and HC3 do not influence the feature investigation.

---

# 8. Logistic Regression baseline

The Logistic Regression stage uses the four final feature tables generated above.

## 8.1 Folder-name compatibility

The current project folder is named:

```text
Feature_extraction/
```

while `logistic_regression/config.py` supports an environment variable for locating the feature configuration.

### Windows Git Bash / Linux / macOS

Change into the Logistic Regression folder and set the paths:

```bash
cd logistic_regression

export LR_FEATURE_DIR="../Feature_extraction"
export LR_DATA_DIR="../data/features"
export LR_OUTPUT_DIR="outputs"
```

### Windows PowerShell

```powershell
cd logistic_regression

$env:LR_FEATURE_DIR="../Feature_extraction"
$env:LR_DATA_DIR="../data/features"
$env:LR_OUTPUT_DIR="outputs"
```

The following Logistic Regression commands are run from inside:

```text
logistic_regression/
```

---

## 8.2 Step 1 — inspect TRAIN and run coarse grouped-CV tuning

Run:

```bash
python run_development.py
```

This runs:

```text
scripts/01_inspect_train.py
scripts/02_coarse_tuning.py
```

It uses DRCAT TRAIN only and **does not access DRCAT TEST or HC3**.

The coarse search compares:

```text
Sentence weighting:
    uniform
    doc_equal
    sqrt_doc

Class balancing:
    none
    doc_balanced

C:
    0.001
    0.01
    0.1
    1
    10
    100

Document pooling:
    mean
    sqrt_length
```

Expected outputs:

```text
outputs/
├── 01_inspection/
│   ├── feature_ranges.csv
│   ├── train_inspection.md
│   └── train_summary.json
│
└── 02_coarse_tuning/
    ├── all_fold_results.csv
    ├── coarse_tuning_report.md
    ├── configuration_summary.csv
    ├── cv_fold_structure.csv
    └── recommended_config.json
```

The main file to inspect is:

```text
outputs/02_coarse_tuning/configuration_summary.csv
```

Model-family selection uses:

```text
selection_score =
(mean ROC-AUC + mean Average Precision) / 2
```

as a transparent recommendation score.

---

## 8.3 Step 2 — reproduce the final fine C search

The final project fixed the coarse choices as:

```text
sentence weighting = uniform
class balance       = doc_balanced
pooling             = sqrt_length
```

and then compared the following fine C values:

```text
0.0001
0.0003
0.001
0.003
0.01
```

Run:

```bash
python scripts/03_fine_tuning.py \
    --weighting uniform \
    --class-balance doc_balanced \
    --pooling sqrt_length \
    --c-values 0.0001 0.0003 0.001 0.003 0.01
```

Expected outputs:

```text
outputs/03_fine_tuning/
├── all_fold_results.csv
├── configuration_summary.csv
├── fine_tuning_report.md
└── recommended_config.json
```

The selected value in the frozen project is:

```text
C = 0.0003
```

---

## 8.4 Step 3 — validation, threshold selection, and model freezing

Run:

```bash
python scripts/04_validate_freeze.py \
    --weighting uniform \
    --class-balance doc_balanced \
    --c 0.0003 \
    --pooling sqrt_length
```

This:

1. fits the chosen configuration using all DRCAT TRAIN data;
2. scores DRCAT VALIDATION;
3. sweeps classification thresholds from 0.05 to 0.95;
4. selects the threshold using macro-F1;
5. creates Logistic Regression coefficient evidence;
6. freezes the fitted model and final configuration.

Expected validation outputs:

```text
outputs/04_validation/
├── candidate_config.json
├── coefficients.csv
├── coefficients.png
├── threshold_sweep.csv
├── threshold_sweep.png
├── validation_document_scores.csv
└── validation_report.md
```

Expected frozen model:

```text
outputs/05_frozen_model/
├── frozen_config.json
└── logistic_regression.joblib
```

The final frozen configuration should contain:

```text
sentence weighting = uniform
class balance       = doc_balanced
regularisation      = L2
C                   = 0.0003
solver              = lbfgs
max_iter            = 3000
pooling             = sqrt_length
threshold           = 0.50
random_state        = 42
```

If a reviewer wants to inspect VALIDATION before writing the frozen model, add:

```text
--no-freeze
```

to the validation command, inspect the outputs, and then rerun the command without `--no-freeze`.

---

## 8.5 Step 4 — final DRCAT TEST and HC3 external evaluation

Only run this after the model has been frozen.

```bash
python run_final_evaluation.py
```

This runs:

```text
scripts/05_final_eval.py
scripts/06_error_analysis.py
```

The final evaluation computes:

```text
Accuracy
Precision
Recall
F1
ROC-AUC
Average Precision
Confusion matrix
95% group-bootstrap confidence intervals
```

for both DRCAT TEST and HC3 external evaluation.

Expected outputs:

```text
outputs/06_final_eval/
├── test/
│   ├── confusion_matrix.csv
│   ├── document_scores.csv
│   ├── evaluation_report.md
│   ├── metrics.csv
│   └── metrics_with_group_bootstrap_ci.csv
│
├── hc3/
│   ├── confusion_matrix.csv
│   ├── document_scores.csv
│   ├── evaluation_report.md
│   ├── metrics.csv
│   └── metrics_with_group_bootstrap_ci.csv
│
└── comparison/
    ├── drcat_vs_hc3.csv
    └── drcat_vs_hc3.md
```

The same command also generates:

```text
outputs/07_error_analysis/
├── test_document_outcomes.csv
├── test_error_analysis.md
├── hc3_document_outcomes.csv
└── hc3_error_analysis.md
```

These files contain the false-positive / false-negative examples and sentence-level Logistic Regression contributions used for manual error analysis.

---

## 8.6 Expected final Logistic Regression results

These values can be used as a reproducibility check.

| Metric | DRCAT TEST | HC3 external |
|---|---:|---:|
| Accuracy | 0.888 | 0.606 |
| Precision | 0.894 | 0.645 |
| Recall | 0.809 | 0.470 |
| F1 | 0.850 | 0.544 |
| ROC-AUC | 0.959 | 0.656 |
| Average Precision | 0.940 | 0.602 |

Small differences may occur only if the underlying raw data, package versions, or project configuration are changed. The frozen project uses `random_state = 42`.

---

## 9. Optional Logistic Regression stress experiments

These experiments are separate from the core frozen pipeline.

From inside `logistic_regression/`:

```bash
python experiments/stress_tests.py
```

They include additional robustness and subgroup investigations. They should not be used to retune the already-frozen model after TEST or HC3 results have been observed.

---

## 10. Quick reproduction checklist

From the project root:

```bash
# 1. Install
python -m pip install -r requirements.txt
python -m pip install -r logistic_regression/requirements.txt
python -m pip install spacy
python -m spacy download en_core_web_sm

# 2. Download HC3
python Preprocess/download_hc3.py --out data/raw/hc3_all.jsonl

# 3. Preprocess DRCAT + HC3
python Preprocess/preprocess.py \
    --drcat data/raw/train_v2_drcat_02.csv \
    --hc3 data/raw/hc3_all.jsonl

# 4. Sentence splitting
python Preprocess/sentence_splitter.py \
    --processed-dir data/processed \
    --out-dir data/sentences

# 5. Build TRAIN features
python Feature_extraction/build_final_sentence_feature_table.py \
    --input data/sentences/drcat_train_sentences.csv \
    --output data/features/drcat_train_final_sentence_features.csv

# 6. Build VALIDATION features
python Feature_extraction/build_final_sentence_feature_table.py \
    --input data/sentences/drcat_validation_sentences.csv \
    --output data/features/drcat_validation_final_sentence_features.csv

# 7. Build TEST features
python Feature_extraction/build_final_sentence_feature_table.py \
    --input data/sentences/drcat_test_sentences.csv \
    --output data/features/drcat_test_final_sentence_features.csv

# 8. Build HC3 features
python Feature_extraction/build_final_sentence_feature_table.py \
    --input data/sentences/hc3_external_test_sentences.csv \
    --output data/features/hc3_external_test_final_sentence_features.csv

# 9. Optional feature-analysis evidence
python analyze_features_for_report.py \
    --input data/features/drcat_train_final_sentence_features.csv \
    --output-dir results/feature_analysis/report
```

Then enter Logistic Regression:

```bash
cd logistic_regression

export LR_FEATURE_DIR="../Feature_extraction"
export LR_DATA_DIR="../data/features"
export LR_OUTPUT_DIR="outputs"
```

Run development and final modelling:

```bash
# 10. TRAIN inspection + coarse tuning
python run_development.py

# 11. Fine C tuning
python scripts/03_fine_tuning.py \
    --weighting uniform \
    --class-balance doc_balanced \
    --pooling sqrt_length \
    --c-values 0.0001 0.0003 0.001 0.003 0.01

# 12. VALIDATION + freeze
python scripts/04_validate_freeze.py \
    --weighting uniform \
    --class-balance doc_balanced \
    --c 0.0003 \
    --pooling sqrt_length

# 13. TEST + HC3 + error analysis
python run_final_evaluation.py
```

At this point the full pipeline through the **Logistic Regression baseline** has been reproduced.

---

## 11. Important data-use rules

To preserve the evaluation design:

- **DRCAT TRAIN** is used for model fitting and grouped cross-validation.
- **DRCAT VALIDATION** is used for threshold selection.
- **DRCAT TEST** is final internal evaluation only.
- **HC3** is external evaluation only.
- Do not tune the Logistic Regression model after viewing TEST or HC3 results.
- Metadata columns such as `document_id`, `group_key`, `source_dataset`, `domain`, and `generator` are not predictive Logistic Regression features.
- The classifier uses only the frozen 15 numerical sentence features defined in `Feature_extraction/feature_config.py`.

---

## 12. Current project scope

This README currently documents:

```text
✓ data collection setup
✓ preprocessing
✓ dataset transformation
✓ leakage-safe splitting
✓ sentence segmentation
✓ final 15-feature extraction
✓ TRAIN-only feature analysis
✓ Logistic Regression tuning
✓ validation and freezing
✓ DRCAT TEST evaluation
✓ HC3 external evaluation
✓ error analysis
```

The following team components are outside the current version of this README and should be documented after their implementations are finalised:

```text
K-Means clustering
Transformer / other beyond-unit model
```
