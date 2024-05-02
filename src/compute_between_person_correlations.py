import pandas as pd
import numpy as np
import pingouin

from utils import load_config 


def perform_pairwise_correlations(data, target_column, numeric_columns, method="pearson"):
    """
    Compute correlations between a target variable and a series of other variables
    Parameters:
        data: A pandas DataFrame containing at a minimum all variables for analysis
        target_column (str): The name of the variable to be correlated with all other specified variables.
            This variable must be numeric.
        numeric_columns (list): List of strings corresponding to the names of the numeric variables to be 
            correlated with the target column variable. These columns must be numeric.
        method (str): The type of correlation to perform. Can be one of: "pearson", "spearman", "kendall", 
            "bicor", "percbend", "shepherd", "skipped". Default is "pearson"
    Returns:
        pd.DataFrame with one row per variable in the numeric_columns list and columns for statistics
    """
    robust_methods = ["bicor", "percbend", "shepherd", "skipped"]
    methods = ["pearson", "spearman", "kendall"] + robust_methods
    if not method in methods:
        raise ValueError(f"`method` must be one of: {methods}")

    correlation_results = pd.DataFrame()

    for col in numeric_columns:
        fit = pingouin.corr(data[target_column], data[col], method=method).reset_index().rename(columns={"index":"method"})
        fit["target"] = target_column
        fit["variable"] = col
        correlation_results = pd.concat([correlation_results, fit], axis=0)

    correlation_results["r_ci_low"] = correlation_results["CI95%"].apply(lambda x: np.NaN if np.isnan(x).any() else x[0])
    correlation_results["r_ci_high"] = correlation_results["CI95%"].apply(lambda x: np.NaN if np.isnan(x).any() else x[1])
    correlation_results["r_ci_level"] = "95%"

    correlation_results.drop("CI95%", axis=1, inplace=True)
    correlation_results.reset_index(drop=True, inplace=True)
    correlation_results.rename(columns={"p-val":"p_value"}, inplace=True)
    if method == "pearson":
        correlation_results.rename(columns={"BF10":"bayes_factor"}, inplace=True)
    if method in robust_methods:
        correlation_results.rename(columns={"outliers":"n_outliers"}, inplace=True)

    return correlation_results


def perform_independent_samples_ttests(data, target_column, categorical_columns): 
    """
    Perform independent samples t-tests for a target column by a series of binary grouping columns
    Parameters:
        data: A pandas DataFrame containing at a minimum all variables for analysis
        target_column (str): The name of the variable to be correlated with all other specified variables.
            This variable must be numeric.
        categorical_columns (list): List of strings corresponding to the names of the categorical variables,
            each having exactly two groups (e.g., Female, Male), the means of which will be compared in the 
            test. These columns must be string or object columns
    Returns:
        pd.DataFrame with one row per variable in the categorical_columns list and columns for statistics
    """
    ttest_results = pd.DataFrame()

    for col in categorical_columns:
        groups = data[col].value_counts().keys().tolist()

        fit = pingouin.ttest(data[data[col] == groups[0]][target_column], data[data[col] == groups[1]][target_column], paired=False).reset_index(drop=True)
        fit["method"] = "Independent samples t-test"
        fit["target"] = target_column
        fit["variable"] = col

        for group in groups:
            data_for_group = data.loc[data[col] == group][target_column]
            stats_for_group = data_for_group.describe().to_frame().T[["count", "mean", "std", "min", "max"]].reset_index(drop=True)
            stats_for_group["count"] = stats_for_group["count"].astype(int)
            stats_for_group.columns = [f"group{str(groups.index(group))}_{col}" for col in stats_for_group.columns]
            stats_for_group[f"group{str(groups.index(group))}"] = group
            fit = pd.concat([fit, stats_for_group], axis=1)
       
        ttest_results = pd.concat([ttest_results, fit], axis=0)

    ttest_results["n"] = ttest_results["group0_count"] + ttest_results["group1_count"]
    ttest_results["t_ci_low"] = ttest_results["CI95%"].apply(lambda x: x[0])
    ttest_results["t_ci_high"] = ttest_results["CI95%"].apply(lambda x: x[1])
    ttest_results["t_ci_level"] = "95%"

    ttest_results.drop("CI95%", axis=1, inplace=True)
    ttest_results.reset_index(drop=True, inplace=True)
    ttest_results.rename(columns={"T":"t", "dof":"degrees_of_freedom", "p-val":"p_value", "cohen-d":"cohen_d_effect_size", "BF10":"bayes_factor"}, inplace=True)

    return ttest_results


def main():
    config = load_config()
    between_person_correlation_params = config["ANALYSIS_PARAMETERS"]["BETWEEN_PERSON_CORRELATIONS"]
    TARGET_COLUMN = between_person_correlation_params["TARGET_COLUMN"]
    CORRELATION_METHOD = between_person_correlation_params["METHOD"]
    ADJUST_P_VALUES = between_person_correlation_params["MULTIPLE_COMPARISONS"]["ADJUST_P_VALUES"]
    ADJUST_P_VALUES_METHOD = between_person_correlation_params["MULTIPLE_COMPARISONS"]["METHOD"]

    diary_data_descriptives = pd.read_csv("output/descriptives/within_person_descriptive_statistics.csv", dtype={"id":str})
    demographics_data = pd.read_csv("data/processed/good_day_demographics_data_cleaned.csv", dtype={"record_id":str})
    promis_data = pd.read_csv("data/processed/good_day_baseline_promis_data_cleaned.csv", dtype={"record_id":str})

    data_for_tests = (
        diary_data_descriptives[["id", TARGET_COLUMN]]
            .merge(demographics_data, left_on="id", right_on="record_id", how="left")
            .merge(promis_data, on="record_id", how="left")
            .drop(["record_id"], axis=1)
            .set_index("id")
    )

    numeric_columns = [col for col in data_for_tests.select_dtypes(include=[np.number]).columns.values.tolist() if not col == TARGET_COLUMN]
    categorical_columns = data_for_tests.select_dtypes(include=["object"]).columns.values.tolist()

    correlation_results = perform_pairwise_correlations(data=data_for_tests, target_column=TARGET_COLUMN, numeric_columns=numeric_columns, method=CORRELATION_METHOD)
    ttest_results = perform_independent_samples_ttests(data=data_for_tests, target_column=TARGET_COLUMN, categorical_columns=categorical_columns)

    if ADJUST_P_VALUES:
        p_values = pd.concat([correlation_results["p_value"], ttest_results["p_value"]], axis=0).reset_index(drop=True)
        q_values = pingouin.multicomp(pvals=p_values, method=ADJUST_P_VALUES_METHOD)[1]
        correlation_results["q_value"] = q_values[:correlation_results.shape[0]]
        ttest_results["q_value"] = q_values[correlation_results.shape[0]:]

    correlation_results.to_csv(f"output/results/between_person_{CORRELATION_METHOD}_correlations.csv", index=False)
    ttest_results.to_csv("output/results/between_person_independent_samples_ttests.csv", index=False)


if __name__ == "__main__":
    main()