"""
Results summary script for the Irish BER retrofit-prioritisation project.

This script:
1. Loads model evaluation metrics.
2. Loads permutation-importance results.
3. Creates rounded result tables.
4. Generates poster-ready model comparison plots.
5. Saves a short key-results summary for README/poster writing.

Run from the project root:

    python src/results_summary.py
"""

import pandas as pd
import matplotlib.pyplot as plt

from config import RESULTS_TABLES_DIR, RESULTS_FIGURES_DIR


MODEL_NAME_MAP = {
    "logistic_regression": "Logistic Regression",
    "dummy_most_frequent": "Dummy Baseline",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "hist_gradient_boosting": "Hist Gradient Boosting",
}


def load_metrics():
    """Load model metrics table."""
    metrics_path = RESULTS_TABLES_DIR / "model_metrics.csv"

    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Model metrics file not found: {metrics_path}\n"
            "Run python src/train_models.py first."
        )

    metrics = pd.read_csv(metrics_path)
    metrics["model_display"] = metrics["model"].map(MODEL_NAME_MAP).fillna(metrics["model"])

    return metrics


def load_tuned_metrics():
    """Load the tuned Hist Gradient Boosting metrics, if available."""
    tuning_path = RESULTS_TABLES_DIR / "hgb_tuning_before_after.csv"

    if not tuning_path.exists():
        return None

    tuning = pd.read_csv(tuning_path)
    return tuning[tuning["model"] == "hist_gradient_boosting_tuned"].iloc[0]


def load_importance():
    """Load permutation importance table."""
    importance_path = RESULTS_TABLES_DIR / "permutation_importance.csv"

    if not importance_path.exists():
        raise FileNotFoundError(
            f"Permutation importance file not found: {importance_path}\n"
            "Run python src/explainability.py first."
        )

    importance = pd.read_csv(importance_path)

    return importance


def save_rounded_tables(metrics, importance):
    """Save rounded tables for README and poster use."""
    metrics_rounded = metrics.copy()

    numeric_cols = metrics_rounded.select_dtypes(include=["float64", "float32"]).columns
    metrics_rounded[numeric_cols] = metrics_rounded[numeric_cols].round(4)

    metrics_rounded = metrics_rounded[
        [
            "model",
            "model_display",
            "accuracy",
            "balanced_accuracy",
            "precision_priority",
            "recall_priority",
            "f1_priority",
            "f1_macro",
            "roc_auc",
        ]
    ]

    metrics_rounded.to_csv(
        RESULTS_TABLES_DIR / "model_metrics_rounded.csv",
        index=False,
    )

    top_features = importance.head(15).copy()
    top_features["importance_mean"] = top_features["importance_mean"].round(5)
    top_features["importance_std"] = top_features["importance_std"].round(5)

    top_features.to_csv(
        RESULTS_TABLES_DIR / "top_features_rounded.csv",
        index=False,
    )

    return metrics_rounded, top_features


def save_metric_bar_plot(metrics, metric, title, ylabel, output_name):
    """Save a model comparison bar chart for one metric."""
    plot_data = metrics.sort_values(metric, ascending=False)

    plt.figure(figsize=(9, 6))
    plt.bar(plot_data["model_display"], plot_data[metric])
    plt.title(title)
    plt.xlabel("Model")
    plt.ylabel(ylabel)
    plt.ylim(0, 1)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    output_path = RESULTS_FIGURES_DIR / output_name
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_all_model_comparison_plots(metrics):
    """Create poster-ready comparison plots."""
    save_metric_bar_plot(
        metrics=metrics,
        metric="f1_macro",
        title="Model Comparison by Macro F1-Score",
        ylabel="Macro F1-score",
        output_name="model_comparison_macro_f1.png",
    )

    save_metric_bar_plot(
        metrics=metrics,
        metric="roc_auc",
        title="Model Comparison by ROC-AUC",
        ylabel="ROC-AUC",
        output_name="model_comparison_roc_auc.png",
    )

    save_metric_bar_plot(
        metrics=metrics,
        metric="recall_priority",
        title="Model Comparison by Retrofit-Priority Recall",
        ylabel="Recall for retrofit-priority class",
        output_name="model_comparison_priority_recall.png",
    )


def save_key_results_summary(metrics_rounded, top_features, tuned_metrics=None):
    """Save a short text summary of the most important results."""
    best_by_macro_f1 = metrics_rounded.sort_values("f1_macro", ascending=False).iloc[0]
    best_by_auc = metrics_rounded.sort_values("roc_auc", ascending=False).iloc[0]
    best_by_recall = metrics_rounded.sort_values("recall_priority", ascending=False).iloc[0]

    top_feature_names = top_features["feature"].head(10).tolist()

    summary_lines = [
        "Key Results Summary",
        "===================",
        "",
        f"Best model family by macro F1-score (default hyperparameters): "
        f"{best_by_macro_f1['model_display']}",
        f"Macro F1-score: {best_by_macro_f1['f1_macro']}",
        f"Accuracy: {best_by_macro_f1['accuracy']}",
        f"Balanced accuracy: {best_by_macro_f1['balanced_accuracy']}",
        f"Priority-class recall: {best_by_macro_f1['recall_priority']}",
        f"ROC-AUC: {best_by_macro_f1['roc_auc']}",
        "",
        f"Best model by ROC-AUC: {best_by_auc['model_display']} "
        f"with ROC-AUC = {best_by_auc['roc_auc']}",
        "",
        f"Best model by priority-class recall: {best_by_recall['model_display']} "
        f"with recall = {best_by_recall['recall_priority']}",
        "",
    ]

    if tuned_metrics is not None:
        summary_lines.extend(
            [
                "Final model (Hist Gradient Boosting, tuned via RandomizedSearchCV):",
                f"Macro F1-score: {round(tuned_metrics['f1_macro'], 4)}",
                f"Accuracy: {round(tuned_metrics['accuracy'], 4)}",
                f"Balanced accuracy: {round(tuned_metrics['balanced_accuracy'], 4)}",
                f"Priority-class recall: {round(tuned_metrics['recall_priority'], 4)}",
                f"ROC-AUC: {round(tuned_metrics['roc_auc'], 4)}",
                "",
            ]
        )

    summary_lines.append("Top 10 features by permutation importance:")

    for idx, feature in enumerate(top_feature_names, start=1):
        summary_lines.append(f"{idx}. {feature}")

    summary_lines.extend(
        [
            "",
            "Interpretation:",
            "The strongest predictors are mainly related to building size, U-values, "
            "construction year, heating fuel, wall area and glazing. This is consistent "
            "with building-energy principles because heat loss, dwelling size, age and "
            "heating systems are directly linked to residential energy performance.",
        ]
    )

    output_path = RESULTS_TABLES_DIR / "key_results_summary.txt"

    with open(output_path, "w", encoding="utf-8") as file:
        file.write("\n".join(summary_lines))


def main():
    """Run results summary script."""
    print("Starting results summary...")

    RESULTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    metrics = load_metrics()
    importance = load_importance()
    tuned_metrics = load_tuned_metrics()

    metrics_rounded, top_features = save_rounded_tables(metrics, importance)

    save_all_model_comparison_plots(metrics)

    save_key_results_summary(metrics_rounded, top_features, tuned_metrics)

    print("Results summary complete.")
    print(f"Rounded metrics saved to: {RESULTS_TABLES_DIR / 'model_metrics_rounded.csv'}")
    print(f"Top features saved to: {RESULTS_TABLES_DIR / 'top_features_rounded.csv'}")
    print(f"Key summary saved to: {RESULTS_TABLES_DIR / 'key_results_summary.txt'}")
    print(f"Figures saved to: {RESULTS_FIGURES_DIR}")


if __name__ == "__main__":
    main()