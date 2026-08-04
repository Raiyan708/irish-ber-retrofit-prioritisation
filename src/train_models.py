"""
Model training and evaluation pipeline for the Irish BER retrofit-prioritisation project.

Run from the project root:

    python src/train_models.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import joblib

from scipy.stats import randint, uniform, loguniform

from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

from config import (
    MODEL_READY_DATA_FILE,
    RESULTS_TABLES_DIR,
    RESULTS_FIGURES_DIR,
    RANDOM_STATE,
    MODEL_SAMPLE_SIZE,
    PROJECT_ROOT,
)

from preprocessing import (
    TARGET_COLUMN,
    get_feature_columns,
    build_preprocessor,
)


MODELS_DIR = PROJECT_ROOT / "models"

# Hyperparameter search budget for the final Hist Gradient Boosting model.
HGB_SEARCH_N_ITER = 25
HGB_SEARCH_CV_FOLDS = 5

HGB_PARAM_DISTRIBUTIONS = {
    "model__learning_rate": loguniform(0.01, 0.3),
    "model__max_iter": randint(100, 400),
    "model__max_leaf_nodes": randint(15, 128),
    "model__min_samples_leaf": randint(10, 100),
    "model__l2_regularization": uniform(0, 2),
}


def load_model_data():
    """Load cleaned modelling dataset."""
    if not MODEL_READY_DATA_FILE.exists():
        raise FileNotFoundError(
            f"Model-ready data not found: {MODEL_READY_DATA_FILE}. "
            "Run python src/data_cleaning.py first."
        )

    df = pd.read_csv(MODEL_READY_DATA_FILE, low_memory=False)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found.")

    return df


def stratified_sample(df, sample_size):
    """Take a stratified sample for faster modelling."""
    if sample_size is None or len(df) <= sample_size:
        return df

    sampled_df, _ = train_test_split(
        df,
        train_size=sample_size,
        stratify=df[TARGET_COLUMN],
        random_state=RANDOM_STATE,
    )

    return sampled_df.reset_index(drop=True)


def get_models():
    """Define baseline and candidate machine-learning models."""
    return {
        "dummy_most_frequent": DummyClassifier(
            strategy="most_frequent",
            random_state=RANDOM_STATE,
        ),
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "decision_tree": DecisionTreeClassifier(
            max_depth=12,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=18,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=150,
            learning_rate=0.08,
            random_state=RANDOM_STATE,
        ),
    }


def tune_hist_gradient_boosting(preprocessor, X_train, y_train):
    """
    Search Hist Gradient Boosting hyperparameters with RandomizedSearchCV.

    Uses stratified k-fold cross-validation on the training split, scoring
    by macro F1 (consistent with how the final model was originally
    selected among model families).
    """
    print(
        f"\nTuning hist_gradient_boosting with RandomizedSearchCV "
        f"({HGB_SEARCH_N_ITER} candidates, {HGB_SEARCH_CV_FOLDS}-fold CV)..."
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", HistGradientBoostingClassifier(random_state=RANDOM_STATE)),
        ]
    )

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=HGB_PARAM_DISTRIBUTIONS,
        n_iter=HGB_SEARCH_N_ITER,
        cv=HGB_SEARCH_CV_FOLDS,
        scoring="f1_macro",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)

    print(f"Best CV macro F1: {search.best_score_:.4f}")
    print(f"Best params: {search.best_params_}")

    return search


def save_tuning_results(search, default_metrics, tuned_metrics):
    """Save the before/after tuning comparison table and best hyperparameters."""
    default_row = dict(default_metrics)
    default_row["model"] = "hist_gradient_boosting_default"

    tuned_row = dict(tuned_metrics)
    tuned_row["model"] = "hist_gradient_boosting_tuned"

    comparison_df = pd.DataFrame([default_row, tuned_row])
    comparison_path = RESULTS_TABLES_DIR / "hgb_tuning_before_after.csv"
    comparison_df.to_csv(comparison_path, index=False)

    best_params = {
        key.replace("model__", ""): value
        for key, value in search.best_params_.items()
    }
    best_params_df = pd.DataFrame(
        list(best_params.items()), columns=["hyperparameter", "best_value"]
    )
    best_params_df["cv_macro_f1"] = search.best_score_
    best_params_path = RESULTS_TABLES_DIR / "hgb_best_params.csv"
    best_params_df.to_csv(best_params_path, index=False)

    return comparison_path, best_params_path


def evaluate_model(model_name, pipeline, X_test, y_test):
    """Evaluate model and return metric dictionary."""
    y_pred = pipeline.predict(X_test)

    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "precision_priority": precision_score(y_test, y_pred, pos_label=1, zero_division=0),
        "recall_priority": recall_score(y_test, y_pred, pos_label=1, zero_division=0),
        "f1_priority": f1_score(y_test, y_pred, pos_label=1, zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
    }

    if hasattr(pipeline, "predict_proba"):
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        metrics["roc_auc"] = roc_auc_score(y_test, y_prob)
    else:
        metrics["roc_auc"] = None

    return metrics


def save_confusion_matrix(model_name, pipeline, X_test, y_test):
    """Save confusion matrix figure."""
    y_pred = pipeline.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Not priority", "Priority"],
    )

    fig, ax = plt.subplots(figsize=(7, 6))
    display.plot(ax=ax, values_format="d")
    ax.set_title(f"Confusion Matrix: {model_name}")
    plt.tight_layout()

    output_path = RESULTS_FIGURES_DIR / f"confusion_matrix_{model_name}.png"
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_roc_curve(model_name, pipeline, X_test, y_test):
    """Save ROC curve figure if probabilities are available."""
    if not hasattr(pipeline, "predict_proba"):
        return

    fig, ax = plt.subplots(figsize=(7, 6))
    RocCurveDisplay.from_estimator(pipeline, X_test, y_test, ax=ax)
    ax.set_title(f"ROC Curve: {model_name}")
    plt.tight_layout()

    output_path = RESULTS_FIGURES_DIR / f"roc_curve_{model_name}.png"
    plt.savefig(output_path, dpi=300)
    plt.close()


def main():
    """Run the full model training pipeline."""
    print("Starting model training pipeline...")

    RESULTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_model_data()
    print(f"Loaded model-ready dataset: {df.shape}")

    df = stratified_sample(df, MODEL_SAMPLE_SIZE)
    print(f"Using modelling dataset: {df.shape}")

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    numeric_features, categorical_features = get_feature_columns(df)

    print(f"Numerical features: {len(numeric_features)}")
    print(f"Categorical features: {len(categorical_features)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    print(f"Training rows: {X_train.shape[0]}")
    print(f"Test rows: {X_test.shape[0]}")

    preprocessor = build_preprocessor(numeric_features, categorical_features)
    models = get_models()

    all_metrics = []
    trained_pipelines = {}

    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        pipeline.fit(X_train, y_train)

        metrics = evaluate_model(model_name, pipeline, X_test, y_test)
        all_metrics.append(metrics)

        save_confusion_matrix(model_name, pipeline, X_test, y_test)
        save_roc_curve(model_name, pipeline, X_test, y_test)

        trained_pipelines[model_name] = pipeline

        print(metrics)

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df = metrics_df.sort_values(
        by=["f1_macro", "balanced_accuracy", "roc_auc"],
        ascending=False,
    )

    metrics_path = RESULTS_TABLES_DIR / "model_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    best_model_name = metrics_df.iloc[0]["model"]
    best_model_macro_f1 = metrics_df.iloc[0]["f1_macro"]
    best_pipeline = trained_pipelines[best_model_name]

    print("\nBaseline model comparison complete.")
    print(f"Best model with default hyperparameters: {best_model_name}")

    # Hyperparameter tuning for the project's chosen final model.
    default_hgb_metrics = next(
        m for m in all_metrics if m["model"] == "hist_gradient_boosting"
    )

    search = tune_hist_gradient_boosting(preprocessor, X_train, y_train)
    tuned_pipeline = search.best_estimator_
    tuned_metrics = evaluate_model(
        "hist_gradient_boosting_tuned", tuned_pipeline, X_test, y_test
    )

    save_confusion_matrix("hist_gradient_boosting_tuned", tuned_pipeline, X_test, y_test)
    save_roc_curve("hist_gradient_boosting_tuned", tuned_pipeline, X_test, y_test)

    comparison_path, best_params_path = save_tuning_results(
        search, default_hgb_metrics, tuned_metrics
    )

    print("\nHyperparameter tuning complete.")
    print(f"Before/after comparison saved to: {comparison_path}")
    print(f"Best hyperparameters saved to: {best_params_path}")
    print(f"Default HGB macro F1: {default_hgb_metrics['f1_macro']:.4f}")
    print(f"Tuned HGB macro F1:   {tuned_metrics['f1_macro']:.4f}")

    final_pipeline = best_pipeline
    final_model_name = best_model_name
    if tuned_metrics["f1_macro"] >= best_model_macro_f1:
        final_pipeline = tuned_pipeline
        final_model_name = "hist_gradient_boosting_tuned"

    best_model_path = MODELS_DIR / "best_model.joblib"
    joblib.dump(final_pipeline, best_model_path)

    print("\nModel training complete.")
    print(f"Metrics saved to: {metrics_path}")
    print(f"Final selected model: {final_model_name}")
    print(f"Final model saved locally to: {best_model_path}")


if __name__ == "__main__":
    main()