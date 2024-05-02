import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf


def intraclass_correlation_coefficient(between_group_variance, within_group_variance):
    """Calculate intraclass correlation coefficient from between-group (random effects) and within-group (residual) variance values"""
    icc = between_group_variance / (between_group_variance + within_group_variance)
    return icc

def compute_icc(data, target_variable, group_variable):
    """Fit an intercept-only linear mixed effects model and compute ICC from model variance components"""
    icc_model_formula = f"{target_variable} ~ 1"
    icc_model = smf.mixedlm(icc_model_formula, data, groups=data[group_variable])
    icc_model_fit = icc_model.fit()

    between_group_variance = float(icc_model_fit.summary().tables[1].iloc[1,0])  # i.e., random effects variance ("[group variable] Var" Coef.)
    within_group_variance = float(icc_model_fit.summary().tables[0].iloc[2,3])   # i.e., residual variance ("Scale")
    icc = intraclass_correlation_coefficient(between_group_variance=between_group_variance, within_group_variance=within_group_variance)

    icc_results = {
        "model_formula": [icc_model_formula],
        "target_variable": [target_variable],
        "group_variable": [group_variable],
        "n_obs": [int(icc_model_fit.summary().tables[0].iloc[1,1])],
        "n_groups": [int(icc_model_fit.summary().tables[0].iloc[2,1])],
        "min_group_size": [int(icc_model_fit.summary().tables[0].iloc[3,1])],
        "max_group_size": [int(icc_model_fit.summary().tables[0].iloc[4,1])],
        "mean_group_size": [float(icc_model_fit.summary().tables[0].iloc[5,1])],
        "between_group_variance": [between_group_variance],
        "within_group_variance": [within_group_variance],
        "icc": [icc]
    }
    return pd.DataFrame.from_dict(icc_results)

def main():
    daily_diary_data_clean = pd.read_csv(
        "data/processed/good_day_diary_data_cleaned.csv", 
        dtype={"id":"string"}, 
        parse_dates=["date"], 
        date_format = "%m/%d/%y"
    )
    
    icc_results = compute_icc(
        data=daily_diary_data_clean, 
        target_variable="how_would_you_describe_today", 
        group_variable="id"
    )
    icc_results.to_csv("output/descriptives/intraclass_correlation_coefficient.csv", index=False)

if __name__ == "__main__":
    main()