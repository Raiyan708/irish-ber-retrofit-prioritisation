"""
Sample-size sensitivity check for the Irish BER retrofit-prioritisation project.

This script re-trains the final Hist Gradient Boosting pipeline (using the
same tuned hyperparameters already saved in models/best_model.joblib) on
progressively larger stratified samples of the model-ready dataset, to check
whether the results reported for the 100,000-row modelling sample hold as
more data is used.

This is a standalone, optional check (not part of the main pipeline in
run_pipeline.py), since the main project results are all based on the
100,000-row sample for computational consistency and reproducibility.

Run from the project root:

    python src/sample_size_experiment.py
"""

import time

import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from config import (
    MODEL_READY_DATA_FILE,
    RESULTS_TABLES_DIR,
    RESULTS_FIGURES_DIR,
    RANDOM_STATE,
    PROJECT_ROOT,
)

from preprocessing import TARGET_COLUMN, get_feature_columns, build_preprocessor
from train_models import stratified_sample


MODELS_DIR = PROJECT_ROOT / "models"
BEST_MODEL_FILE = MODELS_DIR / "best_model.joblib"

# Sample sizes to compare. None means the full model-ready dataset.
SAMPLE_SIZES = [100_000, 300_000, None]


def load_tuned_hyperparameters():
    """Load the tuned Hist Gradient Boosting hyperparameters from the saved final model."""
    if not BEST_MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Best model not found: {BEST_MODEL_FILE}. "
            "Run python src/train_models.py first."
        )

    best_pipeline = joblib.load(BEST_MODEL_FILE)
    hgb_params = best_pipeline.named_steps["model"].get_params()

    return {
        key: hgb_params[key]
        for key in [
            "learning_rate",
            "max_iter",
            "max_leaf_nodes",
            "min_samples_leaf",
            "l2_regularization",
        ]
    }


def evaluate_sample_size(df, sample_size, hyperparameters):
    """Train and evaluate the final model on one sample size, on a fresh 80/20 split."""
    sample_df = stratified_sample(df, sample_size) if sample_size is not None else df

    X = sample_df.drop(columns=[TARGET_COLUMN])
    y = sample_df[TARGET_COLUMN]

    numeric_features, categorical_features = get_feature_columns(sample_df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    preprocessor = build_preprocessor(numeric_features, categorical_features)
    model = HistGradientBoostingClassifier(random_state=RANDOM_STATE, **hyperparameters)
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

    print(
        f"\nFitting on {len(X_train):,} train rows "
        f"(sample size: {sample_size if sample_size is not None else 'full dataset'})..."
    )
    start_time = time.time()
    pipeline.fit(X_train, y_train)
    fit_seconds = time.time() - start_time

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    result = {
        "sample_size": sample_size if sample_size is not None else len(df),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "fit_seconds": round(fit_seconds, 2),
        "accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "precision_priority": precision_score(y_test, y_pred, pos_label=1, zero_division=0),
        "recall_priority": recall_score(y_test, y_pred, pos_label=1, zero_division=0),
        "f1_priority": f1_score(y_test, y_pred, pos_label=1, zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }

    print(f"Fit time: {fit_seconds:.1f}s | macro F1: {result['f1_macro']:.4f}")

    return result


def save_results(results):
    """Save the sample-size comparison table and a summary plot."""
    results_df = pd.DataFrame(results)

    results_path = RESULTS_TABLES_DIR / "sample_size_comparison.csv"
    results_df.to_csv(results_path, index=False)

    plt.figure(figsize=(8, 6))
    plt.plot(results_df["sample_size"], results_df["f1_macro"], marker="o", label="Macro F1-score")
    plt.plot(
        results_df["sample_size"],
        results_df["balanced_accuracy"],
        marker="o",
        label="Balanced accuracy",
    )
    plt.plot(results_df["sample_size"], results_df["roc_auc"], marker="o", label="ROC-AUC")
    plt.xlabel("Modelling sample size (rows)")
    plt.ylabel("Score")
    plt.title("Final Model Performance vs. Modelling Sample Size")
    plt.legend()
    plt.tight_layout()

    plot_path = RESULTS_FIGURES_DIR / "sample_size_comparison.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()

    return results_path, plot_path


def main():
    """Run the sample-size sensitivity check."""
    print("Starting sample-size sensitivity check...")

    if not MODEL_READY_DATA_FILE.exists():
        raise FileNotFoundError(
            f"Model-ready data not found: {MODEL_READY_DATA_FILE}. "
            "Run python src/data_cleaning.py first."
        )

    RESULTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    hyperparameters = load_tuned_hyperparameters()
    print(f"Using tuned hyperparameters: {hyperparameters}")

    df = pd.read_csv(MODEL_READY_DATA_FILE, low_memory=False)
    print(f"Full model-ready dataset: {df.shape}")

    results = [
        evaluate_sample_size(df, sample_size, hyperparameters)
        for sample_size in SAMPLE_SIZES
    ]

    results_path, plot_path = save_results(results)

    print("\nSample-size sensitivity check complete.")
    print(f"Results saved to: {results_path}")
    print(f"Plot saved to: {plot_path}")
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    main()
