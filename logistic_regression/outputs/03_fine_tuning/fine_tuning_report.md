# Fine C tuning

Fixed weighting: `uniform`
Fixed class balance: `doc_balanced`
Fixed pooling: `sqrt_length`
C values: `[0.0001, 0.0003, 0.001, 0.003, 0.01]`

| weighting   | class_balance   |      C | pooling     |   mean_roc_auc |   std_roc_auc |   mean_average_precision |   std_average_precision |   mean_accuracy_at_0_5 |   mean_f1_at_0_5 |   mean_iterations |   max_iterations |   selection_score |   rank |
|:------------|:----------------|-------:|:------------|---------------:|--------------:|-------------------------:|------------------------:|-----------------------:|-----------------:|------------------:|-----------------:|------------------:|-------:|
| uniform     | doc_balanced    | 0.0003 | sqrt_length |       0.96839  |    0.00431065 |                 0.948906 |               0.0205719 |               0.911456 |         0.872147 |              10   |               10 |          0.958648 |      1 |
| uniform     | doc_balanced    | 0.0001 | sqrt_length |       0.968202 |    0.00468025 |                 0.949008 |               0.0199012 |               0.911903 |         0.872952 |               9   |                9 |          0.958605 |      2 |
| uniform     | doc_balanced    | 0.001  | sqrt_length |       0.968392 |    0.00419865 |                 0.948749 |               0.0208921 |               0.911682 |         0.872077 |              10   |               10 |          0.958571 |      3 |
| uniform     | doc_balanced    | 0.003  | sqrt_length |       0.968384 |    0.00417362 |                 0.948691 |               0.0209886 |               0.911634 |         0.871814 |              10.2 |               11 |          0.958538 |      4 |
| uniform     | doc_balanced    | 0.01   | sqrt_length |       0.968382 |    0.00416529 |                 0.948668 |               0.0210283 |               0.911661 |         0.871886 |              10.2 |               11 |          0.958525 |      5 |

## Recommendation

```json
{
  "weighting": "uniform",
  "class_balance": "doc_balanced",
  "C": 0.0003,
  "pooling": "sqrt_length",
  "mean_roc_auc": 0.9683896909300611,
  "mean_average_precision": 0.9489057941918014,
  "selection_score": 0.9586477425609312,
  "note": "Recommendation only. Inspect the full CV tables before freezing."
}
```