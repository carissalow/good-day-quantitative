import pandas as pd
import numpy as np
import pingouin
import statsmodels as sm
import statsmodels.formula.api as smf

from statsmodels.graphics.gofplots import ProbPlot
from utils import load_config


def select_activities_based_on_endorsement(between_person_descriptives, threshold_min, threshold_max):
    """Select activities to analyze based on frequency of endorsement across all surveys
    Parameters:
        between_person_descriptives (pandas.DataFrame): Output of src/compute_between_person_descriptives.py
        threshold_min (float): Number between 0 and 1, inclusive, representing the minimum proportion of surveys
            on which the activity must have been endorsed
        threshold_max (float): Number between 0 and 1, inclusive, representing the maximum proportion of surveys
            on which the activity must have been endorsed
    Returns: list
    """
    selected_activities = (
        between_person_descriptives
        .pipe(lambda x: x[x["variable"].str.contains("what_did_you_spend_time_doing_today")])
        .pipe(lambda x: x[x["mean"].astype(float).between(threshold_min, threshold_max)]["variable"])
        .tolist()
    )
    return selected_activities

def create_univariable_model_formula(target_column, predictor_column):
    """Create a model formula from a target and single predictor to be passed to `smf.mixedlm()`"""
    return target_column + " ~ " + predictor_column

def fit_univariable_random_intercept_model(model_formula, data, group):
    """Fit a univariable linear mixed model with random intercepts for the specified grouping variable"""
    model = smf.mixedlm(model_formula, data, groups=data[group])
    model_fit = model.fit()
    return model_fit

def clean_model_fit_parameter_names(parameter_column):
    """Clean model fit parameter strings that appear in a model summary"""
    return parameter_column.str.replace("\\.|\\:|\\-", "", regex=True).str.replace(" ", "_").str.replace("^No", "n", regex=True).str.lower()

def extract_model_fit_parameters(fitted_model):
    """Extract model fit parameters from a fitted model"""
    model_summary = fitted_model.summary().tables[0]
    model_fit_params = pd.DataFrame(
        np.vstack([model_summary[[0, 1]].values, model_summary[[2, 3]].values]), 
        columns=["model_parameter", "model_parameter_value"]
    )
    model_fit_params["model_parameter"] = clean_model_fit_parameter_names(model_fit_params["model_parameter"])
    model_fit_params = model_fit_params[model_fit_params.fillna("")["model_parameter"] != ""]
    model_fit_params_wide = pd.DataFrame(columns=model_fit_params["model_parameter"].tolist())
    model_fit_params_wide.loc[0] = model_fit_params["model_parameter_value"].tolist()
    return model_fit_params_wide

def extract_model_fixed_effects(fitted_model):
    """Extract fixed effects estimates from a fitted model"""
    model_summary = fitted_model.summary().tables[1]
    model_fixed_effects = (
        model_summary
        .reset_index()
        .rename(columns={"index":"variable", "Coef.":"beta", "Std.Err.":"stderr", "P>|z|":"p_value", "[0.025":"ci_low", "0.975]":"ci_high"})
    )
    model_fixed_effects = model_fixed_effects[model_fixed_effects["variable"] != "Group Var"]
    return model_fixed_effects

def summarize_model_results(fitted_model, model_formula):
    """Summarize model fit and fixed effects estimates for a fitted model"""
    model_fit_params = extract_model_fit_parameters(fitted_model)
    model_fixed_effects = extract_model_fixed_effects(fitted_model)
    model_results = pd.concat([model_fit_params, model_fixed_effects], axis=1).ffill()
    model_results["model_formula"] = model_formula
    model_results[["model_formula"] + [col for col in model_results.columns if not "model_formula" in col]]
    return model_results

def adjust_model_p_values(model_results, p_adjust_method):
    """Correct for multiple comparisons by adjusting p-values for non-intercept model parameters"""
    if not p_adjust_method in ["bonf", "sidak", "holm", "fdr_bh", "fdr_by", "none"]:
        raise ValueError("`p_adjust_method` must be one of: bonf, sidak, holm, fdr_bh, fdr_by, none")
    predictor_p_values = model_results[model_results["variable"] != "Intercept"][["variable", "p_value"]]
    predictor_p_values["q_value"] = pingouin.multicomp(pvals=predictor_p_values["p_value"].astype(float), method=p_adjust_method)[1]
    predictor_p_values.drop("p_value", axis=1, inplace=True)
    model_results = model_results.merge(predictor_p_values, how="left")
    return model_results

def extract_model_diagnostics(fitted_model, model_formula):
    """Extract model fitted values and residuals for diagnostic plots"""
    fitted = fitted_model.fittedvalues
    resid = fitted_model.resid
    sample_quanitles = ProbPlot(resid).sample_quantiles
    theoretical_quantiles = ProbPlot(resid).theoretical_quantiles
    model_diagnostics = pd.DataFrame({
        "model_formula":model_formula, 
        "fitted":fitted,
        "resid":resid, 
        "sample_quantiles":sample_quanitles, 
        "theoretical_quantiles":theoretical_quantiles
    })
    return model_diagnostics

def main():
    config = load_config()
    model_params = config["ANALYSIS_PARAMETERS"]["UNIVARIABLE_MIXED_MODELS"]

    TARGET_COLUMN = model_params["TARGET_COLUMN"]
    ENDORSEMENT_THRESHOLD_MIN = model_params["FEATURE_COLUMNS"]["ENDORSEMENT_THRESHOLD_MIN"]
    ENDORSEMENT_THRESHOLD_MAX = model_params["FEATURE_COLUMNS"]["ENDORSEMENT_THRESHOLD_MAX"]
    ADJUST_P_VALUES = model_params["MULTIPLE_COMPARISONS"]["ADJUST_P_VALUES"]
    ADJUST_P_VALUES_METHOD = model_params["MULTIPLE_COMPARISONS"]["METHOD"]

    diary_data_clean = pd.read_csv("data/processed/good_day_diary_data_cleaned.csv", dtype={"id":str}, parse_dates=["date"])
    between_person_descriptives = pd.read_csv("output/descriptives/between_person_descriptive_statistics.csv")
    selected_activities = select_activities_based_on_endorsement(between_person_descriptives=between_person_descriptives, threshold_min=ENDORSEMENT_THRESHOLD_MIN, threshold_max=ENDORSEMENT_THRESHOLD_MAX)

    all_model_results = pd.DataFrame()
    all_model_diagnostics = pd.DataFrame()
    for activity in selected_activities:
        model_formula = create_univariable_model_formula(target_column=TARGET_COLUMN, predictor_column=activity)
        model_fit = fit_univariable_random_intercept_model(model_formula=model_formula, data=diary_data_clean, group="id")
        model_results = summarize_model_results(fitted_model=model_fit, model_formula=model_formula)
        all_model_results = pd.concat([all_model_results, model_results], axis=0)

        model_diagnostics = extract_model_diagnostics(fitted_model=model_fit, model_formula=model_formula)
        all_model_diagnostics = pd.concat([all_model_diagnostics, model_diagnostics], axis=0)

    if ADJUST_P_VALUES:
        all_model_results = adjust_model_p_values(model_results=all_model_results, p_adjust_method=ADJUST_P_VALUES_METHOD)

    all_model_results.reset_index(drop=True, inplace=True)
    all_model_results.to_csv("output/results/univariable_linear_mixed_models.csv", index=False)

    all_model_diagnostics.reset_index(drop=True, inplace=True)
    all_model_diagnostics.to_csv("output/results/univariable_linear_mixed_models_diagnostics.csv", index=False)


if __name__ == "__main__":
    main()