# Full Methodology & Reproducibility Notes

This document holds the detailed methodology, experiment justifications, and
reproducibility notes that support the [main README](../README.md). It exists
so the README can stay short and scannable while nothing rigorous gets lost —
every claim in the README is backed by something concrete here or in
`results/`.

## Contents

- [Mathematical Formulation](#mathematical-formulation)
- [Data Cleaning](#data-cleaning)
- [Leakage Handling](#leakage-handling)
- [Selected Modelling Features](#selected-modelling-features)
- [Preprocessing & Train/Test Split](#preprocessing-traintest-split)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Cross-Validation (5-Fold)](#cross-validation-5-fold)
- [Sample-Size Sensitivity](#sample-size-sensitivity)
- [SHAP Methodology](#shap-methodology)
- [SHAP vs. Permutation Importance](#shap-vs-permutation-importance)
- [Saved Model & Explainer](#saved-model-explainer)
- [Deployment Notes](#deployment-notes)
- [Full Output File Listing](#full-output-file-listing)

---

## Mathematical Formulation

This project is formulated as a supervised binary classification problem.

For each dwelling `i`, the model observes a feature vector `x_i` containing
physical, geographical, and heating-system characteristics (dwelling type,
county, year of construction, floor area, U-values, heating fuel, glazing
percentage, building-envelope areas). The response variable is:

```text
y_i = 1 if the dwelling is retrofit priority
y_i = 0 otherwise
```

The modelling objective is to learn a function `f(x_i) -> y_i`. For
probabilistic classifiers, `f` can also be read as estimating
`P(y_i = 1 | x_i)`, where larger values indicate higher predicted
retrofit-priority risk.

The problem is evaluated on a held-out test set using accuracy, balanced
accuracy, macro F1-score, ROC-AUC, precision, recall, and confusion
matrices. Because the target is imbalanced, balanced accuracy and macro
F1-score are treated as more informative than accuracy alone.

---

## Data Cleaning

Implemented in [`src/data_cleaning.py`](../src/data_cleaning.py). Steps:

1. Loads selected columns from the raw BER dataset.
2. Renames columns into consistent snake_case format.
3. Converts numerical fields into valid numeric types.
4. Converts placeholder missing values (blank strings, `NA`, `nan`, `None`,
   and similar) into proper missing values.
5. Applies data-quality rules for invalid construction years, U-values,
   physical area values, glazing percentages, volume values, and storey
   counts.
6. Creates the binary `retrofit_priority` target.
7. Removes target-leakage columns from the modelling dataset (see below).
8. Saves a model-ready dataset locally.
9. Saves audit tables for cleaning, missingness, leakage, and
   selected-feature rationale.

Generated audit files: `results/tables/cleaning_summary.csv`,
`data_quality_audit.csv`, `leakage_audit.csv`,
`selected_features_rationale.csv`, `missingness_after_cleaning.csv`.

---

## Leakage Handling

Because the target is derived from BER rating, the following columns were
excluded from the modelling features to prevent the model from learning
directly from variables that reveal the outcome:

| Column                   | Decision           | Reason                                                           |
| ------------------------ | ------------------- | ------------------------------------------------------------------ |
| `EnergyRating`            | Target source only | Used to define `retrofit_priority`, then removed from predictors |
| `BerRating`               | Dropped             | Numerical BER score closely determines the BER band              |
| `CO2Rating`               | Dropped             | BER-related assessment output                                    |
| `TotalCO2Emissions`       | Dropped             | Calculated energy-performance output                             |
| `TotalPrimaryEnergyFact`  | Dropped             | Calculated energy-performance output                             |
| `MPCDERValue`             | Dropped             | Derived assessment value                                         |
| `EPCDERValue`              | Dropped             | Derived assessment value                                         |

---

## Selected Modelling Features

County, dwelling type, year of construction, rating type, ground floor
area, wall/roof/floor/window/door U-values and areas, number of storeys,
main space- and water-heating fuel, thermal era, glazing percentage,
volume.

The derived feature `building_age` was not used because it is directly
calculated from `year_of_construction`; keeping only `year_of_construction`
avoids a redundant predictor and improves interpretability.

---

## Preprocessing & Train/Test Split

Implemented in [`src/preprocessing.py`](../src/preprocessing.py) and
[`src/train_models.py`](../src/train_models.py), using a scikit-learn
`ColumnTransformer`: numerical imputation + scaling, categorical imputation
+ one-hot encoding, stratified train-test splitting.

A stratified sample of 100,000 rows is used for the main pipeline (model
comparison, tuning, cross-validation, explainability), split 80,000
train / 20,000 test. See [Sample-Size Sensitivity](#sample-size-sensitivity)
for why this doesn't undermine the results.

---

## Hyperparameter Tuning

Implemented in `tune_hist_gradient_boosting()` in
[`src/train_models.py`](../src/train_models.py).

Once Hist Gradient Boosting was selected as the best-performing model
family on default hyperparameters, it was tuned with `RandomizedSearchCV`:

- 25 randomly sampled candidates × 5-fold stratified CV on the training
  split (80,000 rows), scored by macro F1.
- Search space: `learning_rate` (log-uniform, 0.01-0.3), `max_iter`
  (100-400), `max_leaf_nodes` (15-127), `min_samples_leaf` (10-99),
  `l2_regularization` (0-2).

Best hyperparameters found (best CV macro F1 = 0.8872):

| Hyperparameter      | Best Value |
| -------------------- | ---------: |
| `learning_rate`       |      0.138 |
| `max_iter`            |        360 |
| `max_leaf_nodes`      |        120 |
| `min_samples_leaf`    |         50 |
| `l2_regularization`   |      1.458 |

Before vs. after tuning, evaluated on the held-out test set:

| Metric                       | Default | Tuned  |
| ----------------------------- | ------: | -----: |
| Accuracy                     |  0.9586 | 0.9602 |
| Balanced accuracy            |  0.8544 | 0.8634 |
| Retrofit-priority precision  |  0.9657 | 0.9679 |
| Retrofit-priority recall     |  0.9885 | 0.9881 |
| Retrofit-priority F1-score   |  0.9769 | 0.9779 |
| Macro F1-score               |  0.8859 | 0.8918 |
| ROC-AUC                      |  0.9688 | 0.9704 |

Full results: `results/tables/hgb_tuning_before_after.csv`,
`results/tables/hgb_best_params.csv`.

Tuning improved every metric except retrofit-priority recall, which
dropped marginally (0.9885 → 0.9881, a 4-in-10,000 change). Balanced
accuracy showed the largest gain (+0.009). Because the tuned model is at
least as good as the default on every metric that matters here, it
replaces the default-hyperparameter model as the final saved pipeline
(`models/best_model.joblib`) used throughout the rest of the project.

---

## Cross-Validation (5-Fold)

Implemented in `run_cross_validation()` in
[`src/evaluate_models.py`](../src/evaluate_models.py).

The single 80/20 test-split result is how the final model is actually
evaluated for deployment, but it only gives one sample of performance. To
check it isn't a lucky (or unlucky) split, 5-fold stratified CV was also
run with the same tuned hyperparameters, on the training portion only
(the held-out test set stayed completely untouched). Each fold trains a
fresh copy of the full pipeline from scratch.

| Metric                       |   Mean | Std    |
| ----------------------------- | -----: | -----: |
| Accuracy                     | 0.9589 | 0.0015 |
| Balanced accuracy            | 0.8521 | 0.0073 |
| Retrofit-priority precision  | 0.9651 | 0.0018 |
| Retrofit-priority recall     | 0.9896 | 0.0017 |
| Retrofit-priority F1-score   | 0.9772 | 0.0008 |
| Macro F1-score               | 0.8862 | 0.0047 |
| ROC-AUC                      | 0.9673 | 0.0023 |

Full results: `results/tables/cv_5fold_summary.csv` (mean/std),
`results/tables/cv_5fold_raw_scores.csv` (per-fold).

The cross-validated means are close to the single-split test metrics
(within ~0.01-0.02 on every metric) with small standard deviations (all
under 0.01), so the single-split result is not an artefact of a
favourable split. Both are shown because they answer different questions:
the single split reflects the actual deployed model's performance on
genuinely unseen data, while CV reflects the stability of the model
configuration across different random partitions of the training data.

---

## Sample-Size Sensitivity

Implemented in
[`src/sample_size_experiment.py`](../src/sample_size_experiment.py) (a
standalone, optional script — not part of the main `run_pipeline.py`).

All main results use a stratified 100,000-row sample of the 790,389-row
dataset, mainly for faster iteration on the more expensive steps
(hyperparameter search, SHAP). To check this doesn't affect the
conclusions, the final tuned pipeline (same hyperparameters, fresh 80/20
split each time) was re-trained from scratch on 100k, 300k, and the full
dataset:

| Sample size    | Train rows | Fit time | Accuracy | Balanced Acc. | Macro F1 | ROC-AUC |
| --------------: | ---------: | -------: | -------: | -------------: | -------: | ------: |
| 100,000         |     80,000 |    2.2 s |   0.9603 |          0.8634 |   0.8918 |  0.9704 |
| 300,000         |    240,000 |    7.4 s |   0.9635 |          0.8697 |   0.9002 |  0.9729 |
| 790,389 (full)  |    632,311 |   38.8 s |   0.9651 |          0.8715 |   0.9042 |  0.9753 |

Full results: `results/tables/sample_size_comparison.csv`. Figure:
`results/figures/sample_size_comparison.png`.

Full-dataset training took under 40 seconds — Hist Gradient Boosting
scales well, so compute was not actually a constraint. Every metric
improves modestly and monotonically with more data, with diminishing
returns. This confirms the results reported throughout the project are
not an artefact of the smaller sample — they hold, and improve slightly,
at full scale.

---

## SHAP Methodology

Implemented in [`src/explainability.py`](../src/explainability.py). SHAP
values were computed with `TreeExplainer` on a stratified sample of 2,000
test-set rows.

Because the preprocessing pipeline one-hot encodes categorical columns,
SHAP values are computed on the encoded matrix and then mapped back to
original feature names: one-hot columns belonging to the same categorical
feature (e.g. `dwelling_type_Detached house`,
`dwelling_type_Semi-detached house`, ...) are summed to a single mean
absolute SHAP value per original feature. Numeric features are shown with
their original, unscaled values (real U-values and m² areas) rather than
the standardised values the model actually sees — this keeps SHAP outputs
directly comparable with the permutation importance table, which also
operates on original feature columns.

Full results: `results/tables/shap_feature_importance.csv`.

---

## SHAP vs. Permutation Importance

The two methods agree closely: 8 of the top 10 features are shared
(`ground_floor_area`, `u_value_window`, `year_of_construction`,
`u_value_wall`, `main_water_heating_fuel`, `u_value_roof`,
`u_value_floor`, `wall_area`). SHAP additionally surfaces `county_name`
and `volume` in its top 10, while permutation importance instead surfaces
`floor_area` and `main_space_heating_fuel`.

The two methods disagree mainly on *ranking order* rather than *which
features matter* — permutation importance ranks `ground_floor_area` first
(largest drop in macro F1 when shuffled), while SHAP ranks
`u_value_window` first (largest average contribution to individual
predictions). This is expected: permutation importance measures the
effect on overall model *performance*, while SHAP measures the average
magnitude of each feature's contribution to individual model *outputs* —
a feature can be highly influential on individual predictions without
being the single most damaging feature to shuffle, and vice versa. The
agreement between two independent methods increases confidence that these
dwelling-envelope and construction-year features are genuinely driving
the model, rather than being an artefact of one particular method.

Full comparison: `results/tables/shap_vs_permutation_importance.csv`.

---

## Saved Model & Explainer

The final trained pipeline (preprocessing + tuned Hist Gradient Boosting)
is saved to `models/best_model.joblib`, produced by
[`src/train_models.py`](../src/train_models.py) and committed to the repo
(the one exception carved out of the `models/*.joblib` rule in
`.gitignore`), so it can be loaded directly — no retraining — by anything
that needs predictions, including the dashboard. At ~1 MB it's small
enough to ship safely.

```python
import joblib

pipeline = joblib.load("models/best_model.joblib")
prediction = pipeline.predict(new_dwelling_df)          # 0 = not priority, 1 = retrofit priority
probability = pipeline.predict_proba(new_dwelling_df)[:, 1]
```

**Why the SHAP explainer isn't persisted separately.** Rather than also
saving a fitted SHAP explainer object, it's reconstructed on the fly:

```python
import shap

explainer = shap.TreeExplainer(pipeline.named_steps["model"])
```

`shap.TreeExplainer` reads the trained tree structure directly from the
model object, so building it involves no retraining or refitting on data —
it takes under 50ms regardless of dataset size. Saving a second serialized
explainer file would duplicate the same tree information for no runtime
benefit.

---

## Deployment Notes

The dashboard is deployed on
[Streamlit Community Cloud](https://ml-retrofit-ber.streamlit.app). Two
non-obvious things had to be handled to get there:

**Pinned library versions.** `models/best_model.joblib` is a pickled
object, and scikit-learn's internal representation of fitted objects
(e.g. `ColumnTransformer`) has changed between versions in ways that
break loading a pickle saved by an older version with a newer one.
`requirements.txt` pins `scikit-learn==1.6.1` and `joblib==1.4.2` to
match the exact versions the model was saved with. If you ever retrain
and re-save the model with different library versions, update these pins
to match.

**Explicit Python version.** Pinning the library versions alone isn't
enough: Streamlit Cloud's default Python version can be too new to have
a prebuilt wheel for an older pin like `scikit-learn==1.6.1`, forcing a
source build that's likely to fail. The Python version has to be set
explicitly to **3.11** (or 3.12) in the app's settings — under
**"Advanced settings"** at creation time, or **⋮ → Settings → General →
Python version** on an already-deployed app. A `runtime.txt` file at the
repo root was tried first but is **not** honoured by Streamlit Community
Cloud — the UI setting is the only mechanism that works. Changing it on
an already-deployed app requires a full **reboot** (⋮ → "Reboot app"),
not just a redeploy.

---

## Full Output File Listing

### Tables (`results/tables/`)

```text
cleaning_summary.csv
data_quality_audit.csv
leakage_audit.csv
selected_features_rationale.csv
model_metrics.csv
model_metrics_rounded.csv
best_model_evaluation_summary.csv
best_model_classification_report.csv
best_model_confusion_matrix.csv
hgb_tuning_before_after.csv
hgb_best_params.csv
cv_5fold_summary.csv
cv_5fold_raw_scores.csv
sample_size_comparison.csv
permutation_importance.csv
shap_feature_importance.csv
shap_vs_permutation_importance.csv
top_features_rounded.csv
key_results_summary.txt
```

### Figures (`results/figures/`)

```text
ber_rating_distribution.png
target_distribution.png
top_missingness_model_ready.png
ground_floor_area_by_target.png
year_of_construction_by_target.png
u_value_window_by_target.png
u_value_wall_by_target.png
u_value_roof_by_target.png
model_comparison_macro_f1.png
model_comparison_roc_auc.png
model_comparison_priority_recall.png
confusion_matrix_hist_gradient_boosting_tuned.png
roc_curve_hist_gradient_boosting_tuned.png
sample_size_comparison.png
permutation_importance_top_features.png
shap_summary_beeswarm.png
shap_waterfall_confident_priority.png
shap_waterfall_borderline.png
```
