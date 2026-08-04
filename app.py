"""
Streamlit dashboard for the Irish BER retrofit-prioritisation project.

Loads the saved final pipeline (models/best_model.joblib) and reconstructs a
SHAP TreeExplainer from it at startup -- nothing is retrained. Run with:

    streamlit run app.py
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"

REPO_URL = (
    "https://github.com/ACM40960/"
    "Explainable-Machine-Learning-for-Irish-Residential-Energy-Retrofit-Prioritisation-Using-BER-Data"
)

# Feature ranges/defaults below were computed from the training dataset
# (1st/99th percentile, median) so sliders stay in a realistic range without
# shipping the ~130 MB processed dataset with the app.
NUMERIC_FEATURES = {
    "year_of_construction": {
        "label": "Year of construction",
        "min": 1700,
        "max": 2023,
        "default": 1997,
        "step": 1,
    },
    "no_storeys": {
        "label": "Number of storeys",
        "min": 1,
        "max": 6,
        "default": 2,
        "step": 1,
    },
    "ground_floor_area": {
        "label": "Ground floor area (m²)",
        "min": 10.0,
        "max": 300.0,
        "default": 100.0,
        "step": 1.0,
    },
    "wall_area": {
        "label": "Wall area (m²)",
        "min": 0.0,
        "max": 250.0,
        "default": 89.0,
        "step": 1.0,
    },
    "roof_area": {
        "label": "Roof area (m²)",
        "min": 0.0,
        "max": 220.0,
        "default": 59.0,
        "step": 1.0,
    },
    "floor_area": {
        "label": "Total floor area (m²)",
        "min": 0.0,
        "max": 200.0,
        "default": 59.0,
        "step": 1.0,
    },
    "window_area": {
        "label": "Window area (m²)",
        "min": 1.0,
        "max": 60.0,
        "default": 16.5,
        "step": 0.5,
    },
    "door_area": {
        "label": "Door area (m²)",
        "min": 0.0,
        "max": 8.0,
        "default": 2.3,
        "step": 0.1,
    },
    "volume": {
        "label": "Dwelling volume (m³)",
        "min": 20.0,
        "max": 1600.0,
        "default": 455.0,
        "step": 5.0,
    },
    "glazing_percent": {
        "label": "Glazing ratio (glazed area / total area)",
        "min": 0.0,
        "max": 1.0,
        "default": 0.20,
        "step": 0.01,
    },
    "u_value_wall": {
        "label": "Wall U-value (W/m²K)",
        "min": 0.10,
        "max": 3.0,
        "default": 0.40,
        "step": 0.01,
    },
    "u_value_roof": {
        "label": "Roof U-value (W/m²K)",
        "min": 0.10,
        "max": 2.0,
        "default": 0.25,
        "step": 0.01,
    },
    "u_value_floor": {
        "label": "Floor U-value (W/m²K)",
        "min": 0.10,
        "max": 1.2,
        "default": 0.44,
        "step": 0.01,
    },
    "u_value_window": {
        "label": "Window U-value (W/m²K)",
        "min": 0.7,
        "max": 5.8,
        "default": 2.70,
        "step": 0.01,
    },
    "u_value_door": {
        "label": "Door U-value (W/m²K)",
        "min": 0.8,
        "max": 4.0,
        "default": 3.00,
        "step": 0.01,
    },
}

CATEGORICAL_FEATURES = {
    "county_name": {
        "label": "County",
        "options": sorted(
            [
                "Co. Cork", "Co. Dublin", "Co. Kildare", "Co. Galway", "Co. Meath",
                "Co. Wexford", "Co. Tipperary", "Co. Kerry", "Co. Donegal", "Co. Louth",
                "Co. Mayo", "Co. Wicklow", "Co. Clare", "Co. Limerick", "Dublin 15",
                "Limerick City", "Co. Westmeath", "Cork City", "Dublin 24", "Co. Laois",
                "Galway City", "Co. Sligo", "Co. Kilkenny", "Co. Cavan", "Co. Waterford",
                "Co. Offaly", "Co. Carlow", "Dublin 8", "Dublin 11", "Dublin 18",
                "Co. Roscommon", "Dublin 9", "Dublin 12", "Waterford City", "Dublin 7",
                "Dublin 16", "Dublin 5", "Dublin 14", "Co. Monaghan", "Dublin 4",
                "Dublin 13", "Co. Longford", "Dublin 22", "Dublin 1", "Dublin 3",
                "Co. Leitrim", "Dublin 6", "Dublin 17", "Dublin 6W", "Dublin 2",
                "Dublin 10", "Dublin 20", "Dublin 21", "Dublin 19", "Dublin 23",
            ]
        ),
        "default": "Co. Cork",
    },
    "dwelling_type": {
        "label": "Dwelling type",
        "options": sorted(
            [
                "Detached house", "Semi-detached house", "Mid-terrace house",
                "End of terrace house", "Mid-floor apartment", "Ground-floor apartment",
                "Top-floor apartment", "House", "Maisonette", "Apartment",
            ]
        ),
        "default": "Detached house",
    },
    "thermal_era": {
        "label": "Thermal era",
        "options": ["Pre", "Post"],
        "default": "Post",
    },
    "main_space_heating_fuel": {
        "label": "Main space-heating fuel",
        "options": sorted(
            [
                "Mains Gas", "Heating Oil", "Electricity", "Solid Multi-Fuel",
                "Bulk LPG (propane or butane)", "Manufactured Smokeless Fuel",
                "House Coal", "Wood Pellets (bulk supply for", "Sod Peat",
                "Bottled LPG", "Wood Logs", "Wood Pellets (in bags for seco",
                "Peat Briquettes", "Anthracite", "Wood Chips",
                "Electricity - Standard Domesti", "Electricity - On-peak Night-Ra",
                "Electricity - Off-peak Night-R", "Biodiesel from renewable sourc",
            ]
        ),
        "default": "Mains Gas",
    },
    "main_water_heating_fuel": {
        "label": "Main water-heating fuel",
        "options": sorted(
            [
                "Mains Gas", "Heating Oil", "Electricity", "Solid Multi-Fuel",
                "Bulk LPG (propane or butane)", "Manufactured Smokeless Fuel",
                "House Coal", "Wood Pellets (bulk supply for", "Sod Peat",
                "Bottled LPG", "Wood Logs", "Wood Pellets (in bags for seco",
                "Peat Briquettes", "Anthracite", "Wood Chips",
                "Electricity - Standard Domesti", "Biodiesel from renewable sourc",
                "Bioethanol from renewable sour", "Electricity - On-peak Night-Ra",
                "Electricity - Off-peak Night-R",
            ]
        ),
        "default": "Mains Gas",
    },
}

# Not shown in the main form (technical BER-assessment field, not a physical
# dwelling characteristic) but still required by the pipeline.
TYPE_OF_RATING_OPTIONS = ["Existing", "Final"]
TYPE_OF_RATING_DEFAULT = "Existing"


@st.cache_resource
def load_pipeline():
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_explainer(_pipeline):
    return shap.TreeExplainer(_pipeline.named_steps["model"])


@st.cache_data
def load_table(filename):
    return pd.read_csv(TABLES_DIR / filename)


def build_input_row(values):
    return pd.DataFrame([values])


def explain_prediction(pipeline, explainer, input_row):
    """Compute a SHAP waterfall figure for a single prediction, with real-unit display values."""
    preprocessor = pipeline.named_steps["preprocessor"]

    transformed = preprocessor.transform(input_row)
    feature_names = [name.split("__", 1)[1] for name in preprocessor.get_feature_names_out()]
    transformed_df = pd.DataFrame(transformed, columns=feature_names, index=input_row.index)

    shap_values = explainer(transformed_df)

    display_df = transformed_df.copy()
    for column in NUMERIC_FEATURES:
        display_df[column] = input_row[column].values
    shap_values.display_data = display_df.values

    fig = plt.figure()
    shap.plots.waterfall(shap_values[0], show=False)
    plt.tight_layout()
    return fig


def render_predict_tab(pipeline, explainer):
    st.subheader("Enter dwelling characteristics")
    st.caption(
        "Fields match what the trained model expects. Defaults are set to typical "
        "(median) values from the training data."
    )

    with st.form("predict_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Location & dwelling**")
            county_name = st.selectbox(
                CATEGORICAL_FEATURES["county_name"]["label"],
                CATEGORICAL_FEATURES["county_name"]["options"],
                index=CATEGORICAL_FEATURES["county_name"]["options"].index(
                    CATEGORICAL_FEATURES["county_name"]["default"]
                ),
            )
            dwelling_type = st.selectbox(
                CATEGORICAL_FEATURES["dwelling_type"]["label"],
                CATEGORICAL_FEATURES["dwelling_type"]["options"],
                index=CATEGORICAL_FEATURES["dwelling_type"]["options"].index(
                    CATEGORICAL_FEATURES["dwelling_type"]["default"]
                ),
            )
            year_of_construction = st.slider(
                NUMERIC_FEATURES["year_of_construction"]["label"],
                NUMERIC_FEATURES["year_of_construction"]["min"],
                NUMERIC_FEATURES["year_of_construction"]["max"],
                NUMERIC_FEATURES["year_of_construction"]["default"],
                NUMERIC_FEATURES["year_of_construction"]["step"],
            )
            no_storeys = st.slider(
                NUMERIC_FEATURES["no_storeys"]["label"],
                NUMERIC_FEATURES["no_storeys"]["min"],
                NUMERIC_FEATURES["no_storeys"]["max"],
                NUMERIC_FEATURES["no_storeys"]["default"],
                NUMERIC_FEATURES["no_storeys"]["step"],
            )
            thermal_era = st.selectbox(
                CATEGORICAL_FEATURES["thermal_era"]["label"],
                CATEGORICAL_FEATURES["thermal_era"]["options"],
                index=CATEGORICAL_FEATURES["thermal_era"]["options"].index(
                    CATEGORICAL_FEATURES["thermal_era"]["default"]
                ),
            )

        with col2:
            st.markdown("**Size & areas**")
            area_values = {}
            for feature in [
                "ground_floor_area", "wall_area", "roof_area", "floor_area",
                "window_area", "door_area", "volume", "glazing_percent",
            ]:
                spec = NUMERIC_FEATURES[feature]
                area_values[feature] = st.slider(
                    spec["label"], spec["min"], spec["max"], spec["default"], spec["step"]
                )

        with col3:
            st.markdown("**Thermal performance (U-values)**")
            u_values = {}
            for feature in [
                "u_value_wall", "u_value_roof", "u_value_floor",
                "u_value_window", "u_value_door",
            ]:
                spec = NUMERIC_FEATURES[feature]
                u_values[feature] = st.slider(
                    spec["label"], spec["min"], spec["max"], spec["default"], spec["step"]
                )

            st.markdown("**Heating**")
            main_space_heating_fuel = st.selectbox(
                CATEGORICAL_FEATURES["main_space_heating_fuel"]["label"],
                CATEGORICAL_FEATURES["main_space_heating_fuel"]["options"],
                index=CATEGORICAL_FEATURES["main_space_heating_fuel"]["options"].index(
                    CATEGORICAL_FEATURES["main_space_heating_fuel"]["default"]
                ),
            )
            main_water_heating_fuel = st.selectbox(
                CATEGORICAL_FEATURES["main_water_heating_fuel"]["label"],
                CATEGORICAL_FEATURES["main_water_heating_fuel"]["options"],
                index=CATEGORICAL_FEATURES["main_water_heating_fuel"]["options"].index(
                    CATEGORICAL_FEATURES["main_water_heating_fuel"]["default"]
                ),
            )

        with st.expander("Advanced (BER assessment field)"):
            type_of_rating = st.selectbox(
                "Type of rating",
                TYPE_OF_RATING_OPTIONS,
                index=TYPE_OF_RATING_OPTIONS.index(TYPE_OF_RATING_DEFAULT),
            )

        submitted = st.form_submit_button("Predict retrofit priority", type="primary")

    if not submitted:
        return

    input_values = {
        "county_name": county_name,
        "dwelling_type": dwelling_type,
        "year_of_construction": year_of_construction,
        "type_of_rating": type_of_rating,
        **area_values,
        **u_values,
        "no_storeys": no_storeys,
        "main_space_heating_fuel": main_space_heating_fuel,
        "main_water_heating_fuel": main_water_heating_fuel,
        "thermal_era": thermal_era,
    }
    input_row = build_input_row(input_values)

    prediction = int(pipeline.predict(input_row)[0])
    probability = float(pipeline.predict_proba(input_row)[:, 1][0])

    st.divider()
    result_col, gauge_col = st.columns([2, 1])
    with result_col:
        if prediction == 1:
            st.error(f"⚠️ **Retrofit priority** — predicted probability {probability:.1%}")
        else:
            st.success(f"✅ **Not retrofit priority** — predicted probability {probability:.1%}")
        st.caption(
            "\"Retrofit priority\" means the model predicts this dwelling is worse than "
            "BER B2 (SEAI's minimum recommended retrofit standard)."
        )
    with gauge_col:
        st.metric("P(retrofit priority)", f"{probability:.1%}")
        st.progress(probability)

    st.subheader("Why this prediction? (SHAP)")
    st.caption(
        "Red bars push the prediction toward retrofit priority; blue bars push toward "
        "not-priority. Values shown are the actual entered values."
    )
    fig = explain_prediction(pipeline, load_explainer(pipeline), input_row)
    st.pyplot(fig)
    plt.close(fig)


def render_model_tab():
    st.subheader("Model Comparison")
    st.caption("Default-hyperparameter comparison used to select the model family.")
    st.dataframe(load_table("model_metrics_rounded.csv"), use_container_width=True)

    st.subheader("Hyperparameter Tuning: Before vs. After")
    st.dataframe(load_table("hgb_tuning_before_after.csv"), use_container_width=True)

    st.subheader("5-Fold Cross-Validation (Final Model)")
    st.dataframe(load_table("cv_5fold_summary.csv"), use_container_width=True)

    st.subheader("Exploratory Data Analysis")
    col1, col2 = st.columns(2)
    col1.image(str(FIGURES_DIR / "ber_rating_distribution.png"), caption="BER Rating Distribution")
    col2.image(
        str(FIGURES_DIR / "target_distribution.png"),
        caption="Retrofit Priority Class Distribution",
    )

    st.subheader("Explainability")
    col1, col2 = st.columns(2)
    col1.image(
        str(FIGURES_DIR / "permutation_importance_top_features.png"),
        caption="Permutation Importance",
    )
    col2.image(str(FIGURES_DIR / "shap_summary_beeswarm.png"), caption="SHAP Summary")


def render_about_tab():
    st.subheader("About This Project")
    st.markdown(
        """
This dashboard predicts whether an Irish dwelling should be considered a
**retrofit priority**, using its Building Energy Rating (BER) characteristics
— things like construction year, floor area, wall/roof/window U-values,
and heating fuel.

**Target definition.** A dwelling is labelled *not retrofit priority* if its
BER is B2 or better, and *retrofit priority* otherwise. B2 was chosen
because it's SEAI's own recommended minimum retrofit standard, and the
specific target of Ireland's Climate Action Plan / National Retrofit Plan
(500,000 homes upgraded to B2 by 2030).

**Data source.** Ahern, C., Raushan, K., and Essien-Thompson, E. (2024).
*Irish Building Energy Rating Database (Cleaned 05/10/2023)*, Mendeley Data.
[DOI: 10.17632/yhgdzfpnym.1](https://doi.org/10.17632/yhgdzfpnym.1)

**Model.** Hist Gradient Boosting, tuned with `RandomizedSearchCV` and
validated with 5-fold cross-validation, trained on the columns shown in the
prediction form. Explainability uses both permutation importance and SHAP.

**Limitations.**
- This is a decision-support tool, not a substitute for an official BER
  assessment or a professional retrofit inspection.
- The retrofit-priority label is a simplified binary cutoff of a continuous
  energy-performance measure — a dwelling just below B2 and a dwelling far
  below it (e.g. rated G) get the same label here.
- BER figures reflect standardised assessment assumptions, not measured
  household energy use.
- Predictions reflect statistical association learned from historical BER
  records, not a certified engineering assessment or causal claim.

Full methodology, results, and code are in the project's GitHub repository.
"""
    )
    st.link_button("View full project on GitHub", REPO_URL)


def main():
    st.set_page_config(
        page_title="Irish BER Retrofit Priority Predictor",
        page_icon="🏠",
        layout="wide",
    )

    st.title("🏠 Irish BER Retrofit Priority Predictor")
    st.caption(
        "Explainable ML pipeline predicting residential retrofit priority from "
        "Building Energy Rating (BER) data — an MSc project for ACM40960, UCD."
    )

    pipeline = load_pipeline()

    predict_tab, model_tab, about_tab = st.tabs(
        ["🔮 Predict", "📊 Model & Data", "ℹ️ About"]
    )

    with predict_tab:
        render_predict_tab(pipeline, load_explainer(pipeline))

    with model_tab:
        render_model_tab()

    with about_tab:
        render_about_tab()


if __name__ == "__main__":
    main()
