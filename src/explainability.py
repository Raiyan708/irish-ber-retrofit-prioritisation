"""
Explainability analysis for the Irish BER retrofit-prioritisation project.

This script:
1. Loads the best trained model.
2. Recreates the same modelling sample and train/test split.
3. Computes permutation importance on the test set.
4. Computes SHAP values for the trained Hist Gradient Boosting pipeline,
   mapped back to interpretable (pre-one-hot) feature names.
5. Saves feature-importance tables and plots for both methods.

Run from the project root:

    python src/explainability.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import shap

from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance

from config import (
    MODEL_READY_DATA_FILE,
    RESULTS_TABLES_DIR,
    RESULTS_FIGURES_DIR,
    RANDOM_STATE,
    MODEL_SAMPLE_SIZE,
    PROJECT_ROOT,
)

from preprocessing import TARGET_COLUMN
from train_models import stratified_sample


MODELS_DIR = PROJECT_ROOT / "models"
BEST_MODEL_FILE = MODELS_DIR / "best_model.joblib"

# Number of test-set rows used for SHAP value computation.
SHAP_SAMPLE_SIZE = 2000


def load_data():
    """Load model-ready data and reproduce the modelling sample."""
    df = pd.read_csv(MODEL_READY_DATA_FILE, low_memory=False)
    df = stratified_sample(df, MODEL_SAMPLE_SIZE)

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    return X_train, X_test, y_train, y_test


def compute_permutation_importance(model, X_test, y_test):
    """
    Compute permutation importance at original feature level.

    Since the full pipeline accepts the raw dataframe, permutation importance
    here measures how much model performance decreases when each original
    feature column is randomly shuffled.
    """
    print("Computing permutation importance...")

    # Use a smaller test sample for speed.
    if len(X_test) > 10000:
        X_perm, _, y_perm, _ = train_test_split(
            X_test,
            y_test,
            train_size=10000,
            stratify=y_test,
            random_state=RANDOM_STATE,
        )
    else:
        X_perm = X_test
        y_perm = y_test

    result = permutation_importance(
        model,
        X_perm,
        y_perm,
        n_repeats=5,
        random_state=RANDOM_STATE,
        scoring="f1_macro",
        n_jobs=1,
    )

    importance_df = pd.DataFrame(
        {
            "feature": X_perm.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values(by="importance_mean", ascending=False)

    return importance_df


def save_importance_plot(importance_df, top_n=15):
    """Save top feature importance plot."""
    top_features = importance_df.head(top_n).sort_values(
        by="importance_mean",
        ascending=True,
    )

    plt.figure(figsize=(10, 7))
    plt.barh(top_features["feature"], top_features["importance_mean"])
    plt.xlabel("Mean decrease in macro F1 after permutation")
    plt.ylabel("Feature")
    plt.title("Top Feature Importances by Permutation Importance")
    plt.tight_layout()

    output_path = RESULTS_FIGURES_DIR / "permutation_importance_top_features.png"
    plt.savefig(output_path, dpi=300)
    plt.close()


def select_shap_sample(X_test, y_test, sample_size=SHAP_SAMPLE_SIZE):
    """Take a stratified subsample of the test set for SHAP computation."""
    if len(X_test) <= sample_size:
        return X_test, y_test

    X_shap, _, y_shap, _ = train_test_split(
        X_test,
        y_test,
        train_size=sample_size,
        stratify=y_test,
        random_state=RANDOM_STATE,
    )
    return X_shap, y_shap


def get_clean_feature_names(preprocessor):
    """
    Return the ColumnTransformer's output feature names with the
    'numeric__' / 'categorical__' pipeline prefixes stripped, so SHAP
    plots show interpretable names (e.g. 'dwelling_type_Detached house')
    instead of raw encoded column identifiers.
    """
    raw_names = preprocessor.get_feature_names_out()
    return [name.split("__", 1)[1] for name in raw_names]


def compute_shap_values(pipeline, X_sample):
    """
    Compute SHAP values for the trained Hist Gradient Boosting model.

    The ColumnTransformer is applied first (as it is inside the pipeline),
    and the resulting encoded matrix is wrapped in a DataFrame with clean
    feature names so that SHAP's plotting functions display interpretable
    labels rather than positional array indices. Numeric columns are then
    given original, unscaled display values, so waterfall/force plots show
    real-world units (e.g. U-values, m^2) instead of standardised z-scores.
    """
    print(f"Computing SHAP values on {len(X_sample)} test-set rows...")

    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    X_transformed = preprocessor.transform(X_sample)
    feature_names = get_clean_feature_names(preprocessor)

    X_transformed_df = pd.DataFrame(
        X_transformed,
        columns=feature_names,
        index=X_sample.index,
    )

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_transformed_df)

    numeric_features = X_sample.select_dtypes(include=["int64", "float64"]).columns.tolist()
    display_df = X_transformed_df.copy()
    for column in numeric_features:
        display_df[column] = X_sample[column].values
    shap_values.display_data = display_df.values

    return shap_values, X_transformed_df, feature_names


def save_shap_beeswarm(shap_values, top_n=15):
    """Save the global SHAP summary (beeswarm) plot."""
    plt.figure()
    shap.plots.beeswarm(shap_values, max_display=top_n, show=False)
    plt.title("SHAP Summary: Impact of Each Feature on Retrofit-Priority Prediction")
    plt.tight_layout()

    output_path = RESULTS_FIGURES_DIR / "shap_summary_beeswarm.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def aggregate_shap_by_original_feature(shap_values, categorical_features):
    """
    Aggregate one-hot encoded SHAP values back to original feature level.

    For each categorical feature, the mean absolute SHAP value across all
    of its one-hot encoded columns is summed to give a single importance
    score per original dwelling characteristic, directly comparable with
    the permutation importance results (which operate on original columns).
    """
    values = np.abs(shap_values.values)
    feature_names = shap_values.feature_names

    mean_abs_shap = values.mean(axis=0)

    rows = []
    grouped_categorical = {feature: 0.0 for feature in categorical_features}

    for name, importance in zip(feature_names, mean_abs_shap):
        matched_categorical = next(
            (feature for feature in categorical_features if name.startswith(feature + "_")),
            None,
        )
        if matched_categorical is not None:
            grouped_categorical[matched_categorical] += importance
        else:
            rows.append({"feature": name, "mean_abs_shap": importance})

    for feature, importance in grouped_categorical.items():
        rows.append({"feature": feature, "mean_abs_shap": importance})

    importance_df = pd.DataFrame(rows).sort_values(
        by="mean_abs_shap", ascending=False
    ).reset_index(drop=True)

    return importance_df


def save_shap_vs_permutation_comparison(shap_importance_df, permutation_importance_df, top_n=10):
    """Save a side-by-side ranking comparison between SHAP and permutation importance."""
    shap_rank = shap_importance_df.head(top_n)[["feature", "mean_abs_shap"]].reset_index(drop=True)
    shap_rank.insert(0, "shap_rank", shap_rank.index + 1)

    perm_rank = permutation_importance_df.head(top_n)[["feature", "importance_mean"]].reset_index(drop=True)
    perm_rank.insert(0, "permutation_rank", perm_rank.index + 1)

    comparison_df = shap_rank.merge(
        perm_rank,
        on="feature",
        how="outer",
        suffixes=("_shap", "_permutation"),
    )

    output_path = RESULTS_TABLES_DIR / "shap_vs_permutation_importance.csv"
    comparison_df.to_csv(output_path, index=False)

    return comparison_df, output_path


def select_local_examples(pipeline, X_sample):
    """
    Pick one confidently retrofit-priority prediction (probability near 1)
    and one borderline prediction (probability near the 0.5 decision
    boundary) for local SHAP explanation plots.
    """
    y_prob = pipeline.predict_proba(X_sample)[:, 1]

    confident_idx = int(np.argmax(y_prob))
    borderline_idx = int(np.argmin(np.abs(y_prob - 0.5)))

    return confident_idx, borderline_idx, y_prob[confident_idx], y_prob[borderline_idx]


def save_shap_waterfall(shap_values, index, output_filename, title):
    """Save a local SHAP waterfall plot for a single prediction."""
    plt.figure()
    shap.plots.waterfall(shap_values[index], show=False)
    plt.title(title)
    plt.tight_layout()

    output_path = RESULTS_FIGURES_DIR / output_filename
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    return output_path


def run_shap_analysis(pipeline, X_test, y_test, categorical_features, permutation_importance_df):
    """Run the full SHAP explainability analysis and save all outputs."""
    print("Starting SHAP explainability analysis...")

    X_shap, _ = select_shap_sample(X_test, y_test)

    shap_values, X_transformed_df, _ = compute_shap_values(pipeline, X_shap)

    save_shap_beeswarm(shap_values)

    shap_importance_df = aggregate_shap_by_original_feature(shap_values, categorical_features)
    shap_importance_path = RESULTS_TABLES_DIR / "shap_feature_importance.csv"
    shap_importance_df.to_csv(shap_importance_path, index=False)

    comparison_df, comparison_path = save_shap_vs_permutation_comparison(
        shap_importance_df, permutation_importance_df
    )

    confident_idx, borderline_idx, confident_prob, borderline_prob = select_local_examples(
        pipeline, X_shap
    )

    save_shap_waterfall(
        shap_values,
        confident_idx,
        "shap_waterfall_confident_priority.png",
        f"SHAP Waterfall: Confident Retrofit-Priority Prediction (P = {confident_prob:.3f})",
    )
    save_shap_waterfall(
        shap_values,
        borderline_idx,
        "shap_waterfall_borderline.png",
        f"SHAP Waterfall: Borderline Prediction (P = {borderline_prob:.3f})",
    )

    print("SHAP explainability analysis complete.")
    print(f"SHAP feature importance saved to: {shap_importance_path}")
    print(f"SHAP vs permutation comparison saved to: {comparison_path}")
    print(f"Confident example: predicted P(priority) = {confident_prob:.4f}")
    print(f"Borderline example: predicted P(priority) = {borderline_prob:.4f}")
    print("Top 10 features by mean absolute SHAP value:")
    print(shap_importance_df.head(10))

    return shap_importance_df, comparison_df


def main():
    """Run explainability analysis."""
    print("Starting explainability analysis...")

    if not BEST_MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Best model not found: {BEST_MODEL_FILE}\n"
            "Run python src/train_models.py first."
        )

    RESULTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    model = joblib.load(BEST_MODEL_FILE)

    _, X_test, _, y_test = load_data()

    importance_df = compute_permutation_importance(model, X_test, y_test)

    importance_path = RESULTS_TABLES_DIR / "permutation_importance.csv"
    importance_df.to_csv(importance_path, index=False)

    save_importance_plot(importance_df, top_n=15)

    print("Permutation importance analysis complete.")
    print(f"Permutation importance saved to: {importance_path}")
    print("Top 10 features:")
    print(importance_df.head(10))

    categorical_features = X_test.select_dtypes(include=["object", "string"]).columns.tolist()

    run_shap_analysis(model, X_test, y_test, categorical_features, importance_df)

    print("\nExplainability analysis complete.")


if __name__ == "__main__":
    main()