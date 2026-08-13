"""
Final evaluation script for the Irish BER retrofit-prioritisation project.

This script evaluates the saved best model on the test set and saves
additional evaluation outputs for documentation.

Run from the project root:

    python src/evaluate_models.py
"""

import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

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

# Cross-validation settings for the final model.
CV_FOLDS = 5
CV_SCORING = {
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "precision_priority": "precision",
    "recall_priority": "recall",
    "f1_priority": "f1",
    "f1_macro": "f1_macro",
    "roc_auc": "roc_auc",
}


def load_train_test_data():
    """Load model-ready data and reproduce the train/test split."""
    if not MODEL_READY_DATA_FILE.exists():
        raise FileNotFoundError(
            f"Model-ready data not found: {MODEL_READY_DATA_FILE}. "
            "Run python src/data_cleaning.py first."
        )

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


def evaluate_best_model(model, X_test, y_test):
    """Evaluate saved best model and return summary metrics."""
    y_pred = model.predict(X_test)

    summary = {
        "accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "precision_priority": precision_score(
            y_test, y_pred, pos_label=1, zero_division=0
        ),
        "recall_priority": recall_score(
            y_test, y_pred, pos_label=1, zero_division=0
        ),
        "f1_priority": f1_score(
            y_test, y_pred, pos_label=1, zero_division=0
        ),
        "f1_macro": f1_score(
            y_test, y_pred, average="macro", zero_division=0
        ),
    }

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        summary["roc_auc"] = roc_auc_score(y_test, y_prob)
    else:
        summary["roc_auc"] = None

    return summary, y_pred


def save_outputs(summary, y_test, y_pred):
    """Save final evaluation tables."""
    RESULTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    summary_df = pd.DataFrame([summary]).round(4)
    summary_df.to_csv(
        RESULTS_TABLES_DIR / "best_model_evaluation_summary.csv",
        index=False,
    )

    report = classification_report(
        y_test,
        y_pred,
        target_names=["Not retrofit priority", "Retrofit priority"],
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(
        RESULTS_TABLES_DIR / "best_model_classification_report.csv"
    )

    labels = ["Not priority", "Priority"]

    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=["Actual not priority", "Actual priority"],
        columns=["Predicted not priority", "Predicted priority"],
    )
    cm_df.to_csv(
        RESULTS_TABLES_DIR / "best_model_confusion_matrix.csv"
    )

    # Row-normalized (each row of ground truth sums to 1), per evaluator
    # guidance: raw counts alone hide class-imbalance effects, so every
    # row/column should also be shown as a proportion.
    cm_normalized = confusion_matrix(y_test, y_pred, normalize="true")
    cm_normalized_df = pd.DataFrame(
        cm_normalized,
        index=["Actual not priority", "Actual priority"],
        columns=["Predicted not priority", "Predicted priority"],
    ).round(4)
    cm_normalized_df.to_csv(
        RESULTS_TABLES_DIR / "best_model_confusion_matrix_normalized.csv"
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    ConfusionMatrixDisplay(cm, display_labels=labels).plot(
        ax=axes[0], values_format="d", colorbar=False
    )
    axes[0].set_title("Counts")

    ConfusionMatrixDisplay(cm_normalized, display_labels=labels).plot(
        ax=axes[1], values_format=".1%", colorbar=False
    )
    axes[1].set_title("Normalized (row = % of actual class)")

    fig.suptitle("Final Model Confusion Matrix: Hist Gradient Boosting (Tuned)")
    plt.tight_layout()

    output_path = RESULTS_FIGURES_DIR / "best_model_confusion_matrix.png"
    plt.savefig(output_path, dpi=300)
    plt.close()


def run_cross_validation(model, X_train, y_train):
    """
    Run stratified k-fold cross-validation on the training split, using the
    same (already-tuned) hyperparameters as the saved final model.

    A fresh, unfitted clone of the pipeline is used so each fold trains from
    scratch. This runs on the training split only, keeping the held-out test
    set exclusively for the single-split evaluation above.
    """
    print(f"\nRunning {CV_FOLDS}-fold stratified cross-validation on the final model...")

    unfitted_pipeline = clone(model)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    cv_results = cross_validate(
        unfitted_pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=CV_SCORING,
        n_jobs=-1,
    )

    return cv_results


def save_cross_validation_results(cv_results):
    """Save per-fold scores and a mean/std summary table."""
    fold_scores = {
        metric: cv_results[f"test_{metric}"] for metric in CV_SCORING
    }
    fold_scores_df = pd.DataFrame(fold_scores)
    fold_scores_df.index.name = "fold"
    fold_scores_df.index = fold_scores_df.index + 1

    fold_scores_path = RESULTS_TABLES_DIR / "cv_5fold_raw_scores.csv"
    fold_scores_df.to_csv(fold_scores_path)

    summary_df = pd.DataFrame(
        {
            "metric": list(CV_SCORING.keys()),
            "mean": [fold_scores_df[m].mean() for m in CV_SCORING],
            "std": [fold_scores_df[m].std() for m in CV_SCORING],
        }
    ).round(4)

    summary_path = RESULTS_TABLES_DIR / "cv_5fold_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    return summary_df, fold_scores_path, summary_path


def main():
    """Run final best-model evaluation."""
    print("Starting final best-model evaluation...")

    if not BEST_MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Best model not found: {BEST_MODEL_FILE}. "
            "Run python src/train_models.py first."
        )

    model = joblib.load(BEST_MODEL_FILE)
    X_train, X_test, y_train, y_test = load_train_test_data()

    summary, y_pred = evaluate_best_model(model, X_test, y_test)
    save_outputs(summary, y_test, y_pred)

    print("Final best-model evaluation complete.")
    print("Single-split evaluation summary:")
    for metric, value in summary.items():
        print(f"{metric}: {value:.4f}" if value is not None else f"{metric}: None")

    cv_results = run_cross_validation(model, X_train, y_train)
    cv_summary_df, fold_scores_path, summary_path = save_cross_validation_results(cv_results)

    print(f"\n{CV_FOLDS}-fold cross-validation complete.")
    print(f"Per-fold scores saved to: {fold_scores_path}")
    print(f"Mean/std summary saved to: {summary_path}")
    print("Cross-validation summary (mean ± std):")
    print(cv_summary_df.to_string(index=False))


if __name__ == "__main__":
    main()