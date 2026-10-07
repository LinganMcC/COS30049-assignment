# Logistic Regression coarse TRAIN-CV tuning

CV folds were created on one-row-per-document data and grouped by group_key.
The recommendation uses only threshold-independent ROC-AUC and Average Precision.
It is a recommendation, not a frozen decision.

## Fold structure

|   fold | split      |   n_documents |   n_groups |   human_documents |   ai_documents |   ai_share |
|-------:|:-----------|--------------:|-----------:|------------------:|---------------:|-----------:|
|      1 | train      |         24361 |          9 |             16149 |           8212 |   0.337096 |
|      1 | validation |          6758 |          2 |              2719 |           4039 |   0.597662 |
|      2 | train      |         23747 |          8 |             13680 |          10067 |   0.423927 |
|      2 | validation |          7372 |          3 |              5188 |           2184 |   0.296256 |
|      3 | train      |         26753 |          9 |             15356 |          11397 |   0.426008 |
|      3 | validation |          4366 |          2 |              3512 |            854 |   0.195602 |
|      4 | train      |         23672 |          9 |             15128 |           8544 |   0.360933 |
|      4 | validation |          7447 |          2 |              3740 |           3707 |   0.497784 |
|      5 | train      |         25943 |          9 |             15159 |          10784 |   0.415681 |
|      5 | validation |          5176 |          2 |              3709 |           1467 |   0.283423 |

## Top 20 configurations

| weighting   | class_balance   |       C | pooling     |   mean_roc_auc |   std_roc_auc |   mean_average_precision |   std_average_precision |   mean_accuracy_at_0_5 |   mean_f1_at_0_5 |   mean_iterations |   max_iterations |   selection_score |   rank |
|:------------|:----------------|--------:|:------------|---------------:|--------------:|-------------------------:|------------------------:|-----------------------:|-----------------:|------------------:|-----------------:|------------------:|-------:|
| uniform     | doc_balanced    |   0.001 | sqrt_length |       0.968392 |    0.00419865 |                 0.948749 |               0.0208921 |               0.911682 |         0.872077 |              10   |               10 |          0.958571 |      1 |
| uniform     | doc_balanced    |   0.01  | sqrt_length |       0.968382 |    0.00416529 |                 0.948668 |               0.0210283 |               0.911661 |         0.871886 |              10.2 |               11 |          0.958525 |      2 |
| uniform     | doc_balanced    |   0.1   | sqrt_length |       0.968381 |    0.00416025 |                 0.948658 |               0.0210432 |               0.911607 |         0.871824 |              10.4 |               11 |          0.95852  |      3 |
| uniform     | doc_balanced    |  10     | sqrt_length |       0.968381 |    0.00415958 |                 0.948656 |               0.0210451 |               0.911607 |         0.871824 |              10.4 |               11 |          0.958519 |      4 |
| uniform     | doc_balanced    | 100     | sqrt_length |       0.968381 |    0.0041596  |                 0.948656 |               0.0210451 |               0.911607 |         0.871824 |              10.4 |               11 |          0.958519 |      5 |
| uniform     | doc_balanced    |   1     | sqrt_length |       0.968381 |    0.00415978 |                 0.948657 |               0.0210449 |               0.911607 |         0.871824 |              10.4 |               11 |          0.958519 |      6 |
| uniform     | doc_balanced    |   0.001 | mean        |       0.967918 |    0.00431489 |                 0.947502 |               0.0214185 |               0.908992 |         0.87081  |              10   |               10 |          0.95771  |      7 |
| uniform     | doc_balanced    |   0.01  | mean        |       0.967919 |    0.00426677 |                 0.94744  |               0.0215503 |               0.909    |         0.870764 |              10.2 |               11 |          0.95768  |      8 |
| uniform     | doc_balanced    |   0.1   | mean        |       0.967918 |    0.00426153 |                 0.94743  |               0.0215669 |               0.908921 |         0.870644 |              10.4 |               11 |          0.957674 |      9 |
| uniform     | doc_balanced    |   1     | mean        |       0.967918 |    0.00426077 |                 0.947428 |               0.0215673 |               0.908921 |         0.870644 |              10.4 |               11 |          0.957673 |     10 |
| uniform     | doc_balanced    | 100     | mean        |       0.967918 |    0.00426048 |                 0.947428 |               0.0215679 |               0.908921 |         0.870644 |              10.4 |               11 |          0.957673 |     11 |
| uniform     | doc_balanced    |  10     | mean        |       0.967918 |    0.00426051 |                 0.947428 |               0.0215678 |               0.908921 |         0.870644 |              10.4 |               11 |          0.957673 |     12 |
| uniform     | none            |   0.001 | sqrt_length |       0.967487 |    0.00420978 |                 0.947521 |               0.0212722 |               0.7673   |         0.578478 |              10   |               10 |          0.957504 |     13 |
| uniform     | none            |   0.01  | sqrt_length |       0.967435 |    0.00416784 |                 0.947354 |               0.0214139 |               0.769552 |         0.583864 |              10.4 |               11 |          0.957395 |     14 |
| uniform     | none            |   0.1   | sqrt_length |       0.967438 |    0.00416383 |                 0.94735  |               0.0214142 |               0.769953 |         0.584848 |              10.2 |               11 |          0.957394 |     15 |
| uniform     | none            |   1     | sqrt_length |       0.967438 |    0.00416356 |                 0.947349 |               0.0214154 |               0.76998  |         0.584926 |              10.2 |               11 |          0.957393 |     16 |
| uniform     | none            | 100     | sqrt_length |       0.967437 |    0.00416363 |                 0.947349 |               0.0214155 |               0.76998  |         0.584926 |              10.2 |               11 |          0.957393 |     17 |
| uniform     | none            |  10     | sqrt_length |       0.967437 |    0.00416362 |                 0.947349 |               0.0214155 |               0.76998  |         0.584926 |              10.2 |               11 |          0.957393 |     18 |
| uniform     | none            |   0.001 | mean        |       0.966828 |    0.00426903 |                 0.94589  |               0.0220305 |               0.773459 |         0.595772 |              10   |               10 |          0.956359 |     19 |
| uniform     | none            |   0.1   | mean        |       0.96681  |    0.00419986 |                 0.945767 |               0.02215   |               0.776094 |         0.602547 |              10.2 |               11 |          0.956288 |     20 |

## Recommended starting point for fine C search

```json
{
  "weighting": "uniform",
  "class_balance": "doc_balanced",
  "C": 0.001,
  "pooling": "sqrt_length",
  "mean_roc_auc": 0.9683924366176736,
  "mean_average_precision": 0.9487494924806322,
  "selection_score": 0.9585709645491529,
  "note": "Recommendation only. Inspect the full CV tables before freezing."
}
```