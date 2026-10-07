# COS30049 Assignment 2 — AI-Generated Content Detection

This repository contains the final machine-learning work for COS30049 Assignment 2:

- DRCAT + HC3 preprocessing and transformation
- leakage-safe DRCAT TRAIN / VALIDATION / TEST splitting
- HC3 external challenge set
- spaCy sentence segmentation
- 15 interpretable stylometric / POS features
- Logistic Regression baseline
- K-Means clustering within a single class
- Transformer encoder trained from scratch
- internal DRCAT evaluation and external HC3 evaluation

The final prediction model reported in the assignment is the Transformer. Logistic Regression is retained as an interpretable baseline, while K-Means is used for unsupervised within-class analysis.

---

## 1. Environment setup with Conda

Run commands from the project root unless stated otherwise.

```bash
conda create -n cos30049-ai-detection python=3.12 pip -y
conda activate cos30049-ai-detection

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

To leave the environment:

```bash
conda deactivate
```

A GPU is optional. PyTorch will use CUDA automatically when a compatible NVIDIA GPU is available; otherwise the Transformer can run on CPU.

---

## 2. Submitted processed datasets

The assignment ZIP is intended to include already-processed data so the marker does **not** need to redownload DRCAT or HC3 just to inspect or reproduce the final model work.

Recommended submitted data:

```text
data/
├── processed/
│   └── unified_documents.csv
│
├── sentences/
│   ├── drcat_train_sentences.csv
│   ├── drcat_validation_sentences.csv
│   ├── drcat_test_sentences.csv
│   └── hc3_external_test_sentences.csv
│
└── features/
    ├── drcat_train_final_sentence_features.csv
    ├── drcat_validation_final_sentence_features.csv
    ├── drcat_test_final_sentence_features.csv
    └── hc3_external_test_final_sentence_features.csv
```

### Fastest marking path

If the sentence CSVs and feature CSVs are already included:

- **Transformer:** use the submitted sentence CSVs directly.
- **Logistic Regression:** use the submitted final feature CSVs directly.
- **K-Means:** use `drcat_train_final_sentence_features.csv` directly.
- Preprocessing and feature extraction only need to be rerun if full reproduction from the raw sources is desired.

`unified_documents.csv` is included as transformation/audit evidence showing the common schema created from the different DRCAT and HC3 formats. It is not the file used directly to train the final models.

Expected counts used in the report:

| Split | Documents | Sentences |
|---|---:|---:|
| DRCAT TRAIN | 31,119 | 592,757 |
| DRCAT VALIDATION | 7,133 | 158,996 |
| DRCAT TEST | 6,612 | 151,878 |
| HC3 external test | 2,000 | 14,752 |

The sentence labels are inherited from the source document labels, recorded with:

```text
label_origin = document_inherited
```

---

## 3. Rebuild the processed datasets from raw data (optional)

This section is optional if the processed data above is already supplied.

### 3.1 Raw DRCAT

Place the DAIGT V2 / DRCAT CSV at:

```text
data/raw/train_v2_drcat_02.csv
```

### 3.2 Download HC3

```bash
python Preprocess/download_hc3.py --out data/raw/hc3_all.jsonl
```

### 3.3 Preprocess both sources

```bash
python Preprocess/preprocess.py \
  --drcat data/raw/train_v2_drcat_02.csv \
  --hc3 data/raw/hc3_all.jsonl \
  --out-dir data/processed
```

This creates the common schema, leakage-safe DRCAT document partitions, the balanced HC3 external test, transformation examples, and a preprocessing report.

### 3.4 Sentence splitting

```bash
python Preprocess/sentence_splitter.py \
  --processed-dir data/processed \
  --out-dir data/sentences
```

The final sentence splitter uses spaCy `en_core_web_sm`.

---

## 4. Build the final 15-feature tables

Skip this section if the four final feature CSVs are already included.

```bash
python Feature_extraction/build_final_sentence_feature_table.py \
  --input data/sentences/drcat_train_sentences.csv \
  --output data/features/drcat_train_final_sentence_features.csv

python Feature_extraction/build_final_sentence_feature_table.py \
  --input data/sentences/drcat_validation_sentences.csv \
  --output data/features/drcat_validation_final_sentence_features.csv

python Feature_extraction/build_final_sentence_feature_table.py \
  --input data/sentences/drcat_test_sentences.csv \
  --output data/features/drcat_test_final_sentence_features.csv

python Feature_extraction/build_final_sentence_feature_table.py \
  --input data/sentences/hc3_external_test_sentences.csv \
  --output data/features/hc3_external_test_final_sentence_features.csv
```

Final features:

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

Metadata columns such as `document_id`, `group_key`, `source_dataset`, `domain`, `generator`, and `label_origin` are retained for grouping/auditing but are not predictive model features.

---

## 5. Logistic Regression baseline

The Logistic Regression baseline uses the four final feature tables.

From the project root:

```bash
cd logistic_regression
```

### Windows Git Bash

```bash
export LR_FEATURE_DIR="../Feature_extraction"
export LR_DATA_DIR="../data/features"
export LR_OUTPUT_DIR="outputs"
```

### 5.1 Development / grouped cross-validation

```bash
python run_development.py
```

### 5.2 Fine-tuning around the final selected region

```bash
python scripts/03_fine_tuning.py \
  --weighting uniform \
  --class-balance doc_balanced \
  --pooling sqrt_length \
  --c-values 0.0001 0.0003 0.001 0.003 0.01
```

### 5.3 Freeze the final selected configuration

```bash
python scripts/04_validate_freeze.py \
  --weighting uniform \
  --class-balance doc_balanced \
  --c 0.0003 \
  --pooling sqrt_length
```

### 5.4 Final DRCAT TEST + HC3 evaluation

```bash
python run_final_evaluation.py
```

Then return to the project root:

```bash
cd ..
```

Frozen model files:

```text
logistic_regression/outputs/05_frozen_model/
├── logistic_regression.joblib
└── frozen_config.json
```

Reported document-level results:

| Metric | DRCAT TEST | HC3 external |
|---|---:|---:|
| Accuracy | 0.888 | 0.606 |
| Precision | 0.894 | 0.645 |
| Recall | 0.809 | 0.470 |
| F1 | 0.850 | 0.544 |
| ROC-AUC | 0.959 | 0.656 |
| Average Precision | 0.940 | 0.602 |

---

## 6. K-Means clustering

K-Means is applied **within one class at a time** and does not use the label as a clustering feature.

The final script is:

```text
K-Means/k-means_single_class.py
```

It reads:

```text
data/features/drcat_train_final_sentence_features.csv
```

The current script uses constants near the top:

```python
CLASS_LABEL = 0   # 0 = Human, 1 = AI
N_CLUSTERS = 4
```

### Reproduce the Human clustering

Set:

```python
CLASS_LABEL = 0
```

then run from the project root:

```bash
python "K-Means/k-means_single_class.py"
```

### Reproduce the AI clustering

Set:

```python
CLASS_LABEL = 1
```

and run the same command again:

```bash
python "K-Means/k-means_single_class.py"
```

The script generates the cluster summary and serialises the fitted scaler and K-Means model:

```text
cluster_summary.csv
scaler_single_class.joblib
kmeans_single_class.joblib
```

Because the filenames are reused, preserve/rename the Human outputs before rerunning the script for AI (or vice versa) if both fitted versions are required.

The report interprets clusters using:

- cluster sizes;
- feature means compared with the class-wide mean;
- the most distinctive features in standard-deviation units;
- features that are unusually often zero;
- representative sentences near each cluster centre.

K-Means is an unsupervised analysis model and is not used as the final end-user classifier.

---

## 7. Transformer encoder — final prediction model

The Transformer is trained from scratch and learns a byte-level BPE vocabulary using DRCAT TRAIN only.

Train/evaluate from the project root:

```bash
python transformer/train_transformer.py \
  --sentence-dir data/sentences \
  --output-dir results/transformer
```

Default final settings:

```text
BPE vocabulary     = 8,000
max tokens         = 64
encoder layers     = 4
attention heads    = 4
embedding size     = 256
feed-forward size  = 1,024
dropout            = 0.1
optimizer          = AdamW
learning rate      = 3e-4
batch size         = 128
epochs             = 5
random seed        = 42
minimum sentence   = 5 words
```

The training script:

1. trains the tokenizer on DRCAT TRAIN only;
2. trains the Transformer;
3. keeps the epoch with the best validation sentence ROC-AUC;
4. selects sentence/document thresholds using VALIDATION only;
5. evaluates the frozen model on DRCAT TEST and HC3;
6. saves final prediction/result files.

Main output files include:

```text
results/transformer/
├── best_model.pt
├── tokenizer.json
├── transformer_comparison.csv
├── test_sentence_predictions.csv
├── test_document_predictions.csv
├── hc3_external_sentence_predictions.csv
└── hc3_external_document_predictions.csv
```

Reported document-level results:

| Metric | LR DRCAT | Transformer DRCAT | LR HC3 | Transformer HC3 |
|---|---:|---:|---:|---:|
| Accuracy | 0.888 | 0.984 | 0.606 | 0.842 |
| Precision | 0.894 | 0.974 | 0.645 | 0.888 |
| Recall | 0.809 | 0.986 | 0.470 | 0.783 |
| F1 | 0.850 | 0.980 | 0.544 | 0.832 |
| ROC-AUC | 0.959 | 0.998 | 0.656 | 0.932 |

### Predict new text

The repository contains:

```text
transformer/check_my_text.py
```

Create a text file such as:

```text
sample.txt
```

Then run the prediction script according to the import paths in the submitted repository.

If the current repository version supports direct execution from the project root:

```bash
python transformer/check_my_text.py sample.txt
```

The script loads the submitted Transformer checkpoint and tokenizer, splits the input into sentences, excludes sentences under five words, prints sentence-level AI probabilities, and aggregates them into a document score and confidence range.

---

## 8. TRAIN-only feature analysis used in the report

```bash
python analyze_features_for_report.py \
  --input data/features/drcat_train_final_sentence_features.csv \
  --output-dir results/feature_analysis/report
```

This produces TRAIN-only feature summaries, Cohen's d effect sizes and Pearson-correlation analysis without using VALIDATION, TEST or HC3 to choose features.

---

## 9. Fast reproduction path for markers

If all processed datasets and final artifacts are supplied:

1. Create the Conda environment.
2. Install `requirements.txt`.
3. Install `en_core_web_sm`.
4. Use `data/features/*.csv` directly for Logistic Regression and K-Means.
5. Use `data/sentences/*.csv` directly for Transformer training/evaluation.
6. Inspect the supplied frozen Logistic Regression model and Transformer result/model files.
7. Raw DRCAT/HC3 downloading and preprocessing are optional.

This is the recommended marking path because the assignment submission already includes the processed data used by the final machine-learning work.

---

## 10. What should be included in the Canvas ZIP

Recommended:

```text
README.md
requirements.txt

Preprocess/
Feature_extraction/
logistic_regression/
K-Means/
transformer/
analyze_features_for_report.py

data/processed/unified_documents.csv

data/sentences/
├── drcat_train_sentences.csv
├── drcat_validation_sentences.csv
├── drcat_test_sentences.csv
└── hc3_external_test_sentences.csv

data/features/
├── drcat_train_final_sentence_features.csv
├── drcat_validation_final_sentence_features.csv
├── drcat_test_final_sentence_features.csv
└── hc3_external_test_final_sentence_features.csv

logistic_regression/outputs/05_frozen_model/
├── logistic_regression.joblib
└── frozen_config.json

results/transformer/
├── best_model.pt              # include if this is the checkpoint used for the reported results
├── tokenizer.json
└── transformer_comparison.csv

K-Means report/result files actually used in the report
```

For K-Means, include any saved scaler/K-Means `.joblib` files if they were generated and retained in the final project. If the final repository only retains the script and report outputs, do not invent additional artifacts; the clustering can be reproduced directly from the submitted TRAIN feature CSV.

Do **not** include unnecessary files such as:

```text
.venv/
venv/
__pycache__/
*.pyc
.git/
large obsolete checkpoints
duplicate experimental outputs
raw downloaded datasets unless intentionally required
```

---

## 11. GitHub versus Canvas

GitHub is used for version control. The Canvas ZIP is the actual assessment package.

Large processed datasets, generated outputs, and binary model files may remain ignored on GitHub if that is your repository policy. However, the **Canvas ZIP should contain the processed model inputs and the final artifacts actually needed to inspect/reproduce the submitted models**, even when those files are excluded by `.gitignore`.

Do not create the Canvas ZIP by downloading the GitHub repository if ignored model/data files are required. Build the ZIP from the final local project folder.
