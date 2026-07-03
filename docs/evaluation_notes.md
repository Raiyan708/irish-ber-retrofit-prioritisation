# Evaluation Notes

This project uses several evaluation metrics because the retrofit-priority target is imbalanced.

Accuracy is reported, but it should not be interpreted alone. Since most dwellings are labelled as retrofit priority, a model can achieve high accuracy by mainly predicting the majority class.

Balanced accuracy is useful because it considers performance across both classes.

Macro F1-score is useful because it gives equal importance to both classes instead of being dominated by the majority class.

Recall for the retrofit-priority class is important because the project aims to identify dwellings that may need retrofit attention.

ROC-AUC is useful because it measures how well the model separates retrofit-priority and non-retrofit-priority dwellings across different classification thresholds.
