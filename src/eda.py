"""
Exploratory Data Analysis for the Irish BER retrofit-prioritisation project.

This script:
1. Loads the cleaned model-ready dataset.
2. Loads BER rating distribution from the raw dataset.
3. Saves summary tables.
4. Saves poster-ready figures into results/figures/.

Run from the project root with:

    python src/eda.py


This step will answer:

What does the BER dataset look like?
How many homes are retrofit priority?
Which dwelling types are common?
Which counties appear most?
How does construction year relate to retrofit priority?
Which heating fuels are common among priority homes?
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config import (
    RAW_DATA_FILE,
    MODEL_READY_DATA_FILE,
    RESULTS_FIGURES_DIR,
    RESULTS_TABLES_DIR,
    PRIORITY_RATINGS,
    RANDOM_STATE,
)


def save_bar_plot(series, title, xlabel, ylabel, output_path, rotation=45):
    """Save a simple bar plot from a pandas Series."""
    plt.figure(figsize=(10, 6))
    series.plot(kind="bar")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=rotation, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_target_distribution(df):
    """Plot retrofit-priority target distribution."""
    counts = df["retrofit_priority"].value_counts().sort_index()
    counts.index = ["Not retrofit priority", "Retrofit priority"]

    counts.to_csv(RESULTS_TABLES_DIR / "target_distribution.csv")

    save_bar_plot(
        series=counts,
        title="Retrofit Priority Class Distribution",
        xlabel="Class",
        ylabel="Number of dwellings",
        output_path=RESULTS_FIGURES_DIR / "target_distribution.png",
        rotation=0,
    )


def plot_ber_rating_distribution():
    """Plot BER rating distribution from raw EnergyRating column."""
    ratings = pd.read_csv(
        RAW_DATA_FILE,
        usecols=["EnergyRating"],
        low_memory=False,
    )

    rating_order = [
        "A1", "A2", "A3",
        "B1", "B2", "B3",
        "C1", "C2", "C3",
        "D1", "D2",
        "E1", "E2",
        "F", "G",
    ]

    ratings["EnergyRating"] = ratings["EnergyRating"].astype(str).str.strip()

    counts = ratings["EnergyRating"].value_counts()
    counts = counts.reindex(rating_order, fill_value=0)

    counts.to_csv(RESULTS_TABLES_DIR / "ber_rating_distribution.csv")

    save_bar_plot(
        series=counts,
        title="BER Rating Distribution",
        xlabel="BER rating",
        ylabel="Number of dwellings",
        output_path=RESULTS_FIGURES_DIR / "ber_rating_distribution.png",
        rotation=0,
    )


def plot_top_categories(df, column, title, output_name, top_n=10):
    """Plot top categories for a categorical column."""
    if column not in df.columns:
        print(f"Skipping {column}: column not found.")
        return

    counts = df[column].value_counts(dropna=False).head(top_n)
    counts.to_csv(RESULTS_TABLES_DIR / f"{output_name}.csv")

    save_bar_plot(
        series=counts,
        title=title,
        xlabel=column,
        ylabel="Number of dwellings",
        output_path=RESULTS_FIGURES_DIR / f"{output_name}.png",
        rotation=45,
    )


def plot_priority_rate_by_category(df, column, title, output_name, top_n=10):
    """
    Plot retrofit-priority rate for the most common categories.
    This is useful for interpretation.
    """
    if column not in df.columns:
        print(f"Skipping {column}: column not found.")
        return

    top_categories = df[column].value_counts().head(top_n).index
    subset = df[df[column].isin(top_categories)].copy()

    priority_rate = (
        subset.groupby(column)["retrofit_priority"]
        .mean()
        .sort_values(ascending=False)
    )

    priority_rate.to_csv(RESULTS_TABLES_DIR / f"{output_name}.csv")

    plt.figure(figsize=(10, 6))
    priority_rate.plot(kind="bar")
    plt.title(title)
    plt.xlabel(column)
    plt.ylabel("Proportion classified as retrofit priority")
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(RESULTS_FIGURES_DIR / f"{output_name}.png", dpi=300)
    plt.close()


def plot_numeric_distribution(df, column, title, output_name):
    """Plot distribution of a numerical column."""
    if column not in df.columns:
        print(f"Skipping {column}: column not found.")
        return

    values = pd.to_numeric(df[column], errors="coerce").dropna()

    plt.figure(figsize=(10, 6))
    sns.histplot(values, bins=50)
    plt.title(title)
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(RESULTS_FIGURES_DIR / f"{output_name}.png", dpi=300)
    plt.close()


def save_eda_summary(df):
    """Save simple EDA summary statistics."""
    summary = pd.DataFrame(
        {
            "metric": [
                "rows",
                "columns",
                "retrofit_priority_count_0",
                "retrofit_priority_count_1",
                "retrofit_priority_proportion_0",
                "retrofit_priority_proportion_1",
            ],
            "value": [
                df.shape[0],
                df.shape[1],
                df["retrofit_priority"].value_counts().get(0, 0),
                df["retrofit_priority"].value_counts().get(1, 0),
                df["retrofit_priority"].value_counts(normalize=True).get(0, 0),
                df["retrofit_priority"].value_counts(normalize=True).get(1, 0),
            ],
        }
    )

    summary.to_csv(RESULTS_TABLES_DIR / "eda_summary.csv", index=False)

    missingness = df.isna().mean().sort_values(ascending=False)
    missingness.to_csv(RESULTS_TABLES_DIR / "model_ready_missingness.csv")

def plot_missingness_summary(df, top_n=15):
    """Plot top missingness rates in the model-ready dataset."""
    missingness = df.isna().mean().sort_values(ascending=False).head(top_n)
    missingness.to_csv(RESULTS_TABLES_DIR / "top_missingness_model_ready.csv")

    plt.figure(figsize=(10, 6))
    missingness.sort_values(ascending=True).plot(kind="barh")
    plt.title("Top Missingness Rates in Model-Ready Dataset")
    plt.xlabel("Missing-value proportion")
    plt.ylabel("Feature")
    plt.xlim(0, 1)
    plt.tight_layout()
    plt.savefig(RESULTS_FIGURES_DIR / "top_missingness_model_ready.png", dpi=300)
    plt.close()


def save_numeric_summary_by_target(df, numeric_columns):
    """Save mean, median and standard deviation of selected numeric features by target class."""
    available_columns = [col for col in numeric_columns if col in df.columns]

    summary = (
        df.groupby("retrofit_priority")[available_columns]
        .agg(["mean", "median", "std"])
        .round(3)
    )

    summary.to_csv(RESULTS_TABLES_DIR / "numeric_summary_by_target.csv")


def plot_numeric_by_target_boxplot(df, column, title, output_name):
    """Save a boxplot of a numeric feature by retrofit-priority class."""
    if column not in df.columns:
        print(f"Skipping {column}: column not found.")
        return

    plot_df = df[[column, "retrofit_priority"]].dropna().copy()

    # Sample for speed and readable plotting.
    if len(plot_df) > 50_000:
        plot_df = plot_df.sample(n=50_000, random_state=RANDOM_STATE)

    plot_df["retrofit_priority_label"] = plot_df["retrofit_priority"].map(
        {
            0: "Not priority",
            1: "Priority",
        }
    )

    plt.figure(figsize=(8, 6))
    sns.boxplot(
        data=plot_df,
        x="retrofit_priority_label",
        y=column,
        showfliers=False,
    )
    plt.title(title)
    plt.xlabel("Retrofit-priority class")
    plt.ylabel(column)
    plt.tight_layout()
    plt.savefig(RESULTS_FIGURES_DIR / f"{output_name}.png", dpi=300)
    plt.close()


def main():
    """Run full EDA script."""
    print("Starting EDA...")

    RESULTS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)

    if not MODEL_READY_DATA_FILE.exists():
        raise FileNotFoundError(
            f"Model-ready data not found: {MODEL_READY_DATA_FILE}\n"
            "Run python src/data_cleaning.py first."
        )

    df = pd.read_csv(MODEL_READY_DATA_FILE, low_memory=False)

    save_eda_summary(df)

    plot_ber_rating_distribution()
    plot_target_distribution(df)

    plot_missingness_summary(df, top_n=15)

    key_numeric_columns = [
        "ground_floor_area",
        "year_of_construction",
        "u_value_wall",
        "u_value_roof",
        "u_value_floor",
        "u_value_window",
        "u_value_door",
        "glazing_percent",
        "volume",
    ]

    save_numeric_summary_by_target(df, key_numeric_columns)

    plot_numeric_by_target_boxplot(
        df,
        column="ground_floor_area",
        title="Ground Floor Area by Retrofit-Priority Class",
        output_name="ground_floor_area_by_target",
    )

    plot_numeric_by_target_boxplot(
        df,
        column="year_of_construction",
        title="Year of Construction by Retrofit-Priority Class",
        output_name="year_of_construction_by_target",
    )

    plot_numeric_by_target_boxplot(
        df,
        column="u_value_window",
        title="Window U-Value by Retrofit-Priority Class",
        output_name="u_value_window_by_target",
    )

    plot_numeric_by_target_boxplot(
        df,
        column="u_value_wall",
        title="Wall U-Value by Retrofit-Priority Class",
        output_name="u_value_wall_by_target",
    )

    plot_numeric_by_target_boxplot(
        df,
        column="u_value_roof",
        title="Roof U-Value by Retrofit-Priority Class",
        output_name="u_value_roof_by_target",
    )

    plot_top_categories(
        df,
        column="county_name",
        title="Top 10 Counties by Number of BER Records",
        output_name="top_counties",
        top_n=10,
    )

    plot_top_categories(
        df,
        column="dwelling_type",
        title="Top Dwelling Types",
        output_name="dwelling_type_distribution",
        top_n=10,
    )

    plot_top_categories(
        df,
        column="main_space_heating_fuel",
        title="Top Main Space Heating Fuels",
        output_name="main_space_heating_fuel_distribution",
        top_n=10,
    )

    plot_priority_rate_by_category(
        df,
        column="dwelling_type",
        title="Retrofit-Priority Rate by Dwelling Type",
        output_name="priority_rate_by_dwelling_type",
        top_n=10,
    )

    plot_priority_rate_by_category(
        df,
        column="main_space_heating_fuel",
        title="Retrofit-Priority Rate by Main Space Heating Fuel",
        output_name="priority_rate_by_heating_fuel",
        top_n=10,
    )

    plot_numeric_distribution(
        df,
        column="year_of_construction",
        title="Distribution of Year of Construction",
        output_name="year_of_construction_distribution",
    )

    plot_numeric_distribution(
        df,
        column="ground_floor_area",
        title="Distribution of Ground Floor Area",
        output_name="ground_floor_area_distribution",
    )

    print("EDA complete.")
    print(f"Figures saved to: {RESULTS_FIGURES_DIR}")
    print(f"Tables saved to: {RESULTS_TABLES_DIR}")


if __name__ == "__main__":
    main()
