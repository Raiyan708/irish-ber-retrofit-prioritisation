# Explainable Machine Learning for Irish Residential Energy Retrofit Prioritisation Using BER Data

## Project Overview

This project investigates whether Irish Building Energy Rating (BER) data can be used to identify residential dwellings that may require energy retrofit priority.

The project develops a reproducible and explainable machine-learning pipeline that classifies dwellings into:

* **Retrofit priority**
* **Not retrofit priority**

The target is based on BER rating:

* Homes rated **B2 or better** are labelled as **not retrofit priority**.
* Homes rated **worse than B2** are labelled as **retrofit priority**.

This project is framed as a decision-support analysis for residential energy retrofit planning. It is not intended to replace official BER assessment, engineering inspection, or policy decision-making.

---

## Research Question

Can Irish BER data be used to predict residential retrofit priority, and can explainable machine learning identify the main dwelling characteristics associated with poor energy performance?

---

## Mathematical Modelling Formulation

This project is formulated as a supervised binary classification problem.

For each dwelling `i`, the model observes a feature vector `x_i`, which contains physical, geographical, and heating-system characteristics such as dwelling type, county, year of construction, floor area, U-values, heating fuel, glazing percentage, and building-envelope areas.

The response variable is:

```text
y_i = 1 if the dwelling is retrofit priority
y_i = 0 otherwise
```

The modelling objective is to learn a function:

```text
f(x_i) -> y_i
```
For probabilistic classifiers, the model can also be interpreted as estimating the probability `P(y_i = 1 | x_i)`, where larger values indicate higher predicted retrofit-priority risk.

that predicts whether a dwelling should be classified as retrofit priority based on its observed characteristics.

The problem is evaluated on a held-out test set using accuracy, balanced accuracy, macro F1-score, ROC-AUC, precision, recall, and confusion matrices. Because the target variable is imbalanced, balanced accuracy and macro F1-score are treated as more informative than accuracy alone.

---

## Motivation

Residential energy efficiency is an important part of Ireland’s sustainability and climate-transition agenda. Many Irish homes differ in construction age, dwelling type, floor area, insulation quality, heating systems, glazing, and energy performance.

Because it is not practical to retrofit all homes immediately, a data-driven approach can help explore which dwelling characteristics are most associated with poor energy performance.

This project uses machine learning to predict retrofit-priority status and explain the main features influencing those predictions.

---

## Dataset

The project uses the cleaned Irish Building Energy Rating dataset:

**Ahern, C., Raushan, K., and Essien-Thompson, E. (2024). *Irish Building Energy Rating Database - (Cleaned 05/10/2023)*. Mendeley Data, Version 1. DOI: 10.17632/yhgdzfpnym.1**

The dataset contains a cleaned pull from the Irish BER database dated 5 October 2023. The file used locally in this project is:

```text
BER (filtered and cleaned 20231005).csv
```

For coding consistency, the dataset file should be renamed locally as:

```text
ber_cleaned_20231005.csv
```

The raw dataset is not included in this GitHub repository because it is large. To reproduce the project, download the dataset separately from Mendeley Data and place it at:

```text
data/raw/ber_cleaned_20231005.csv
```

The processed modelling dataset is generated automatically by the cleaning pipeline and saved locally to:

```text
data/processed/ber_model_ready.csv
```

Both raw and processed data files are excluded from version control using `.gitignore`.

After cleaning and target creation, the model-ready dataset contains:

```text
790,389 rows
22 columns
```
---

## Target Definition

The binary target variable is:

```text
retrofit_priority
```

The target is created from the `EnergyRating` column.

```text
retrofit_priority = 0
if BER rating is A1, A2, A3, B1, or B2

retrofit_priority = 1
if BER rating is B3, C1, C2, C3, D1, D2, E1, E2, F, or G
```

This means homes worse than BER B2 are treated as retrofit-priority dwellings.

Target distribution:

```text
Retrofit priority:     88.84%
Not retrofit priority: 11.16%
```

Because the target is imbalanced, the project evaluates models using balanced accuracy, macro F1-score, ROC-AUC, precision, recall, and confusion matrices rather than accuracy alone.

---

## Project Workflow

```text
Raw BER data
    ↓
Data cleaning and quality checks
    ↓
Target creation
    ↓
Leakage audit and feature selection
    ↓
Exploratory data analysis
    ↓
Preprocessing pipeline
    ↓
Model training and baseline comparison
    ↓
Final model evaluation
    ↓
Explainability analysis
    ↓
Results summary
```

---

## Repository Structure

```text
projects-qudsia-raiyan/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/
│   └── processed/
│
├── docs/
│   └── project_log.md
│
├── models/
│
├── notebooks/
│   └── 01_data_inspection.ipynb
│
├── reports/
│   └── Literature_Review.pdf
│
├── results/
│   ├── figures/
│   └── tables/
│
└── src/
    ├── config.py
    ├── data_cleaning.py
    ├── eda.py
    ├── preprocessing.py
    ├── train_models.py
    ├── evaluate_models.py
    ├── explainability.py
    ├── results_summary.py
    └── run_pipeline.py
```

---

## Tech Stack

* Python
* pandas
* NumPy
* scikit-learn
* matplotlib
* seaborn
* joblib
* Jupyter Notebook
* Git
* GitHub

No cloud tools such as AWS, Airflow, or SageMaker are required for this project. The project is published as a reproducible GitHub repository with an end-to-end local pipeline.

---

## Data Cleaning

The data-cleaning pipeline is implemented in:

```text
src/data_cleaning.py
```

The cleaning stage performs the following steps:

1. Loads selected columns from the raw BER dataset.
2. Renames columns into consistent snake_case format.
3. Converts numerical fields into valid numeric types.
4. Converts placeholder missing values such as blank strings, `NA`, `nan`, `None`, and similar values into proper missing values.
5. Applies data-quality rules for:

   * invalid construction years;
   * invalid U-values;
   * invalid physical area values;
   * invalid glazing percentages;
   * invalid volume values;
   * invalid storey counts.
6. Creates the binary `retrofit_priority` target.
7. Removes target-leakage columns from the modelling dataset.
8. Saves a model-ready dataset locally.
9. Saves audit tables for cleaning, missingness, leakage, and selected-feature rationale.

Important generated audit files:

```text
results/tables/cleaning_summary.csv
results/tables/data_quality_audit.csv
results/tables/leakage_audit.csv
results/tables/selected_features_rationale.csv
results/tables/missingness_after_cleaning.csv
```

---

## Leakage Handling

Because the target is derived from BER rating, care was taken to avoid target leakage.

The following columns were excluded from the modelling features:

| Column                   | Decision           | Reason                                                           |
| ------------------------ | ------------------ | ---------------------------------------------------------------- |
| `EnergyRating`           | Target source only | Used to define `retrofit_priority`, then removed from predictors |
| `BerRating`              | Dropped            | Numerical BER score closely determines the BER band              |
| `CO2Rating`              | Dropped            | BER-related assessment output                                    |
| `TotalCO2Emissions`      | Dropped            | Calculated energy-performance output                             |
| `TotalPrimaryEnergyFact` | Dropped            | Calculated energy-performance output                             |
| `MPCDERValue`            | Dropped            | Derived assessment value                                         |
| `EPCDERValue`            | Dropped            | Derived assessment value                                         |

This prevents the model from learning directly from variables that reveal the BER outcome.

---

## Selected Modelling Features

The model uses physical, dwelling-level, and heating-system variables such as:

* county;
* dwelling type;
* year of construction;
* rating type;
* ground floor area;
* wall, roof, floor, window, and door U-values;
* wall, roof, floor, window, and door areas;
* number of storeys;
* main space-heating fuel;
* main water-heating fuel;
* thermal era;
* glazing percentage;
* volume.

The derived feature `building_age` was not used because it is directly calculated from `year_of_construction`. To avoid redundant predictors and improve interpretability, the project keeps `year_of_construction` only.

---

## Exploratory Data Analysis

The EDA stage is implemented in:

```text
src/eda.py
```

It generates summary tables and figures to understand the dataset before modelling.

Key EDA outputs include:

* BER rating distribution;
* retrofit-priority class distribution;
* county distribution;
* dwelling type distribution;
* heating fuel distribution;
* missingness summary;
* construction year distribution;
* ground floor area distribution;
* priority rate by dwelling type;
* priority rate by heating fuel;
* numerical feature comparisons by target class.

Example figures:

### BER Rating Distribution

![BER Rating Distribution](results/figures/ber_rating_distribution.png)

### Retrofit Priority Class Distribution

![Retrofit Priority Class Distribution](results/figures/target_distribution.png)

### Top Missingness Rates

![Top Missingness Rates](results/figures/top_missingness_model_ready.png)

### Ground Floor Area by Retrofit-Priority Class

![Ground Floor Area by Target](results/figures/ground_floor_area_by_target.png)

### Year of Construction by Retrofit-Priority Class

![Year of Construction by Target](results/figures/year_of_construction_by_target.png)

---

## Machine-Learning Pipeline

The machine-learning pipeline uses `scikit-learn` tools for reproducible preprocessing and modelling.

Main files:

```text
src/preprocessing.py
src/train_models.py
src/evaluate_models.py
```

The preprocessing pipeline uses:

* `ColumnTransformer`;
* numerical imputation;
* numerical scaling;
* categorical imputation;
* one-hot encoding;
* stratified train-test splitting.

A stratified sample of 100,000 rows is used for modelling. This keeps training computationally efficient while preserving the class distribution.

The train-test split is:

```text
Training rows: 80,000
Test rows:     20,000
```

---

## Models Used

The project compares a dummy baseline with four machine-learning models:

1. Dummy majority-class baseline
2. Logistic Regression
3. Decision Tree
4. Random Forest
5. Hist Gradient Boosting

The dummy baseline is important because the target is imbalanced. Since most dwellings are retrofit priority, a naive model that always predicts the majority class can achieve high accuracy without learning meaningful patterns.

---

## Evaluation Metrics

The project uses:

* accuracy;
* balanced accuracy;
* precision for retrofit-priority class;
* recall for retrofit-priority class;
* F1-score for retrofit-priority class;
* macro F1-score;
* ROC-AUC;
* confusion matrix.

Accuracy alone is not sufficient because the dataset is imbalanced. Macro F1-score, balanced accuracy, and ROC-AUC give a more reliable view of model performance.

---

## Model Results

The best-performing model was:

```text
Hist Gradient Boosting
```

Hist Gradient Boosting was selected as the final model because it achieved the highest macro F1-score and ROC-AUC among the non-baseline models, while also maintaining very high recall for retrofit-priority dwellings. Although Random Forest achieved slightly higher balanced accuracy, Hist Gradient Boosting provided the strongest overall trade-off for this project.

Full model comparison:

| Model                  | Accuracy | Balanced Accuracy | Priority Recall | Macro F1-score | ROC-AUC |
| ---------------------- | -------: | ----------------: | --------------: | -------------: | ------: |
| Dummy Baseline         |   0.8884 |            0.5000 |          1.0000 |         0.4705 |  0.5000 |
| Logistic Regression    |   0.8896 |            0.8820 |          0.8917 |         0.7864 |  0.9564 |
| Decision Tree          |   0.9112 |            0.8718 |          0.9224 |         0.8110 |  0.9000 |
| Random Forest          |   0.9556 |            0.8748 |          0.9787 |         0.8849 |  0.9667 |
| Hist Gradient Boosting |   0.9586 |            0.8544 |          0.9885 |         0.8859 |  0.9688 |

Final best-model performance:

| Metric                      |  Value |
| --------------------------- | -----: |
| Accuracy                    | 0.9586 |
| Balanced accuracy           | 0.8544 |
| Retrofit-priority precision | 0.9657 |
| Retrofit-priority recall    | 0.9885 |
| Retrofit-priority F1-score  | 0.9769 |
| Macro F1-score              | 0.8859 |
| ROC-AUC                     | 0.9688 |

The dummy majority-class baseline achieved high accuracy because most dwellings in the dataset are labelled as retrofit priority. However, its balanced accuracy and ROC-AUC are both 0.5000, showing that it has no real discrimination ability. This demonstrates why accuracy alone is misleading for this imbalanced classification task.

The machine-learning models substantially improve balanced accuracy, macro F1-score, and ROC-AUC compared with the dummy baseline. This suggests that the models learn meaningful relationships between dwelling characteristics and retrofit-priority status.

### Model Comparison by Macro F1-Score

![Model Comparison Macro F1](results/figures/model_comparison_macro_f1.png)

### Model Comparison by ROC-AUC

![Model Comparison ROC-AUC](results/figures/model_comparison_roc_auc.png)

### Model Comparison by Retrofit-Priority Recall

![Model Comparison Priority Recall](results/figures/model_comparison_priority_recall.png)


---

## Explainability Analysis

Explainability is implemented in:

```text
src/explainability.py
```

Permutation importance was used to identify which original dwelling features most influenced model performance.

Permutation importance measures how much the model performance decreases when the values of one feature are randomly shuffled. If shuffling a feature causes a large drop in performance, the model is relying strongly on that feature.

This method was chosen because it is model-agnostic and can be applied to the final trained pipeline without needing to inspect model-specific internal parameters.

Top 10 features by permutation importance:

| Rank | Feature                   | Interpretation                                                                                  |
| ---: | ------------------------- | ----------------------------------------------------------------------------------------------- |
|    1 | `ground_floor_area`       | Larger or smaller dwelling size affects heating demand and energy performance.                  |
|    2 | `u_value_window`          | Window thermal transmittance affects heat loss through glazing.                                 |
|    3 | `year_of_construction`    | Construction period is linked to building standards and insulation quality.                     |
|    4 | `u_value_wall`            | Wall thermal performance affects heat loss through the building envelope.                       |
|    5 | `main_water_heating_fuel` | Water-heating fuel type is related to energy-system efficiency.                                 |
|    6 | `u_value_roof`            | Roof thermal performance affects heat loss through the roof or attic.                           |
|    7 | `wall_area`               | Larger wall area can increase total envelope heat loss.                                         |
|    8 | `u_value_floor`           | Floor thermal performance affects heat loss through floors.                                     |
|    9 | `dwelling_type`           | Detached, semi-detached, apartment, and other dwelling types have different heat-loss patterns. |
|   10 | `floor_area`              | Floor area represents building size and contributes to energy-demand differences.               |

These features are physically meaningful because they relate to dwelling size, construction age, heat loss through the building envelope, heating systems, and dwelling form.

This supports the interpretability of the model because the strongest predictors are consistent with building-energy principles rather than arbitrary or unexplained variables.

A limitation of permutation importance is that correlated features can share importance. Therefore, the results should be interpreted as evidence of model reliance, not as proof of direct causal effects.

### Permutation Importance

![Permutation Importance](results/figures/permutation_importance_top_features.png)

---

## Key Interpretation

The results suggest that BER data can be used to classify retrofit-priority dwellings with strong predictive performance.

The best model, Hist Gradient Boosting, achieved high recall for retrofit-priority dwellings. This is important because a retrofit-prioritisation model should identify as many poor-performing dwellings as possible.

The explainability analysis showed that the most influential features are related to building size, thermal transmittance, construction year, heating fuel, dwelling type, and envelope area. These variables are consistent with known building-energy principles.

---

## How to Run the Project
## Setup Notes

Before running the pipeline, make sure the required folder structure exists locally:

- data/raw/
- data/processed/
- results/figures/
- results/tables/
- models/

The raw BER dataset should be placed inside data/raw/ using the expected file name:

ber_cleaned_20231005.csv

The generated processed files, figures, tables, and trained models are created when the pipeline is run locally.

This project was developed using:

```text
Python 3.13.5
```

The Python dependencies are listed in:

```text
requirements.txt
```

The main packages used are `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `jupyter`, `joblib`, and `openpyxl`.

### 1. Clone the repository

```bash
git clone https://github.com/ACM40960/projects-qudsia-raiyan.git
cd projects-qudsia-raiyan
```

### 2. Install requirements

```bash
python -m pip install -r requirements.txt
```

### 3. Add the raw dataset

Download the cleaned BER dataset from Mendeley Data and rename the file to:

```text
ber_cleaned_20231005.csv
```

Place it in:

```text
data/raw/ber_cleaned_20231005.csv
```

The raw dataset is not committed to this repository because it is large.

### 4. Run the full pipeline

```bash
python src/run_pipeline.py
```

This command runs the complete project workflow:

```text
data cleaning
exploratory data analysis
model training
final model evaluation
explainability analysis
results summary
```

The final processed dataset, trained model, figures, and result tables are generated locally.

### Optional: Run each stage separately

```bash
python src/data_cleaning.py
python src/eda.py
python src/train_models.py
python src/evaluate_models.py
python src/explainability.py
python src/results_summary.py
```

---

## Main Output Files

### Tables

```text
results/tables/cleaning_summary.csv
results/tables/data_quality_audit.csv
results/tables/leakage_audit.csv
results/tables/selected_features_rationale.csv
results/tables/model_metrics.csv
results/tables/model_metrics_rounded.csv
results/tables/best_model_evaluation_summary.csv
results/tables/best_model_classification_report.csv
results/tables/best_model_confusion_matrix.csv
results/tables/permutation_importance.csv
results/tables/top_features_rounded.csv
results/tables/key_results_summary.txt
```

### Figures

```text
results/figures/ber_rating_distribution.png
results/figures/target_distribution.png
results/figures/top_missingness_model_ready.png
results/figures/ground_floor_area_by_target.png
results/figures/year_of_construction_by_target.png
results/figures/u_value_window_by_target.png
results/figures/u_value_wall_by_target.png
results/figures/u_value_roof_by_target.png
results/figures/model_comparison_macro_f1.png
results/figures/model_comparison_roc_auc.png
results/figures/model_comparison_priority_recall.png
results/figures/permutation_importance_top_features.png
```

---

## Limitations

This project should be interpreted as a data-driven decision-support analysis, not as a replacement for official BER assessment or professional retrofit inspection.

Main limitations:

* The retrofit-priority label is derived from BER rating and is a simplified modelling definition.
* BER data is based on standardised assessment assumptions, not actual household energy consumption.
* The model uses selected dwelling-level variables rather than every available BER field.
* Some BER fields were excluded because of missingness, interpretation difficulty, or leakage risk.
* The model was trained on a stratified sample of 100,000 rows for computational efficiency.
* The analysis identifies statistical associations, not direct causal effects.
* Further validation would be needed before using this type of model in operational retrofit planning.

---

## Future Work

Possible extensions include:

* train selected models on a larger sample or the full dataset;
* compare different retrofit-priority thresholds;
* include more carefully screened BER variables;
* perform county-level or dwelling-type-specific modelling;
* add SHAP explanations for local prediction-level interpretability;
* explore regression models for numerical energy-performance indicators;
* validate results on newer BER records if available.

---

## Authors

* Muhammad Raiyan
* Qudsia Samar Babu Khadar

---

## Module

**ACM 40960: Projects in Mathematical Modelling**
University College Dublin

---

## References

* Sustainable Energy Authority of Ireland. [Building Energy Rating information](https://www.seai.ie/ber) and [BER Research Tool](https://ndber.seai.ie/BERResearchTool/ber/search.aspx).

* Government of Ireland. [National Retrofit Plan](https://www.gov.ie/en/department-of-climate-energy-and-the-environment/publications/national-retrofit-plan/).

* Ahern, C., Raushan, K., and Essien-Thompson, E. (2024). [*Irish Building Energy Rating Database - (Cleaned 05/10/2023)*](https://data.mendeley.com/datasets/yhgdzfpnym/1). Mendeley Data, Version 1. DOI: [10.17632/yhgdzfpnym.1](https://doi.org/10.17632/yhgdzfpnym.1).

* Ribeiro, M. T., Singh, S., and Guestrin, C. (2016). “Why Should I Trust You?” Explaining the Predictions of Any Classifier. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining.(https://arxiv.org/abs/1602.04938).

* Lundberg, S. M., and Lee, S. I. (2017). A Unified Approach to Interpreting Model Predictions. Advances in Neural Information Processing Systems.(https://arxiv.org/abs/1705.07874).
