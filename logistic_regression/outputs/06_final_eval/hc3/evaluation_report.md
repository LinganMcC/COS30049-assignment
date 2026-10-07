# Final evaluation - HC3

|   threshold |   n_documents |   accuracy |   precision |   recall |       f1 |   macro_f1 |   roc_auc |   average_precision |   tn |   fp |   fn |   tp |
|------------:|--------------:|-----------:|------------:|---------:|---------:|-----------:|----------:|--------------------:|-----:|-----:|-----:|-----:|
|         0.5 |          2000 |     0.6055 |    0.644719 |     0.47 | 0.543667 |   0.598121 |  0.656008 |            0.601711 |  741 |  259 |  530 |  470 |

## Group-bootstrap confidence intervals

| metric            |   estimate |   ci_low |   ci_high |   valid_bootstrap_draws |
|:------------------|-----------:|---------:|----------:|------------------------:|
| accuracy          |   0.6055   | 0.586    |  0.6245   |                    2000 |
| precision         |   0.644719 | 0.61812  |  0.670853 |                    2000 |
| recall            |   0.47     | 0.437    |  0.501    |                    2000 |
| f1                |   0.543667 | 0.516611 |  0.570154 |                    2000 |
| roc_auc           |   0.656008 | 0.633915 |  0.677141 |                    2000 |
| average_precision |   0.601711 | 0.580955 |  0.62512  |                    2000 |

## Confusion matrix

|              |   predicted_human |   predicted_ai |
|:-------------|------------------:|---------------:|
| actual_human |               741 |            259 |
| actual_ai    |               530 |            470 |