# Validation and threshold selection

Model-family choices below came from TRAIN CV; VALIDATION is used only for threshold selection.

```json
{
  "weighting": "uniform",
  "class_balance": "doc_balanced",
  "C": 0.0003,
  "pooling": "sqrt_length"
}
```

Selected threshold by macro-F1: **0.50**

## Validation metrics

|   threshold |   n_documents |   accuracy |   precision |   recall |       f1 |   macro_f1 |   roc_auc |   average_precision |   tn |   fp |   fn |   tp |
|------------:|--------------:|-----------:|------------:|---------:|---------:|-----------:|----------:|--------------------:|-----:|-----:|-----:|-----:|
|         0.5 |          7133 |   0.833731 |    0.784592 |  0.76638 | 0.775379 |   0.821698 |  0.914377 |            0.871634 | 3900 |  562 |  624 | 2047 |

## Coefficients

| feature                |   coefficient |   abs_coefficient |   odds_ratio_per_1sd | direction    | meaning                                                                              |
|:-----------------------|--------------:|------------------:|---------------------:|:-------------|:-------------------------------------------------------------------------------------|
| long_word_ratio        |    0.611466   |        0.611466   |             1.84313  | toward AI    | Fraction of words with at least 7 characters.                                        |
| comma_rate             |    0.537595   |        0.537595   |             1.71189  | toward AI    | Number of commas divided by sentence word count.                                     |
| rel_len_deviation      |   -0.272767   |        0.272767   |             0.76127  | toward Human | Absolute sentence word-count deviation from its document mean, divided by that mean. |
| starts_with_transition |    0.272526   |        0.272526   |             1.31328  | toward AI    | 1 when the sentence starts with a configured transition phrase.                      |
| auxiliary_ratio        |   -0.256451   |        0.256451   |             0.773793 | toward Human | Fraction of alphabetic tokens tagged AUX by spaCy.                                   |
| adverb_ratio           |   -0.212831   |        0.212831   |             0.808292 | toward Human | Fraction of alphabetic tokens tagged ADV by spaCy.                                   |
| unique_word_ratio      |    0.205433   |        0.205433   |             1.22806  | toward AI    | Fraction of distinct lower-cased word tokens in the sentence.                        |
| exclaim                |    0.202004   |        0.202004   |             1.22385  | toward AI    | 1 when the sentence contains an exclamation mark, otherwise 0.                       |
| adjective_ratio        |    0.194462   |        0.194462   |             1.21466  | toward AI    | Fraction of alphabetic tokens tagged ADJ by spaCy.                                   |
| apostrophe_rate        |    0.191947   |        0.191947   |             1.21161  | toward AI    | Number of apostrophes divided by sentence word count.                                |
| conjunction_ratio      |    0.129799   |        0.129799   |             1.1386   | toward AI    | Fraction of alphabetic tokens tagged CCONJ or SCONJ by spaCy.                        |
| noun_ratio             |   -0.124922   |        0.124922   |             0.882566 | toward Human | Fraction of alphabetic tokens tagged NOUN or PROPN by spaCy.                         |
| func_word_ratio        |    0.0635726  |        0.0635726  |             1.06564  | toward AI    | Fraction of words in the configured function-word list.                              |
| pronoun_ratio          |    0.0577617  |        0.0577617  |             1.05946  | toward AI    | Fraction of alphabetic tokens tagged PRON by spaCy.                                  |
| n_words                |    0.00418873 |        0.00418873 |             1.0042   | toward AI    | Number of alphabetic/apostrophe word tokens in the sentence.                         |