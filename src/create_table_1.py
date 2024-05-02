import pandas as pd 
import numpy as np
from utils import format_activity_names

def format_point_estimate(estimate):
    return estimate.round(3).astype(str)

def format_confidence_interval(low, high):
    return low.round(3).astype(str) + ", " + high.round(3).astype(str)

def format_p_value(p_value):
    p_value = p_value.astype(float)
    formatted_p_value = np.where(p_value.round(3) < 0.001, "<0.001", p_value.round(3).astype(str))
    formatted_p_value = np.where(p_value.round(3) > 0.999, ">0.999", formatted_p_value)
    return formatted_p_value

def format_proportion_as_percent(proportion):
    return (proportion*100).round(2).astype(str) 

def format_n_and_percent(n, percent):
    return n.astype(int).astype(str) + " (" + percent + "%)"

def main():
    between_person_descriptives = pd.read_csv("output/descriptives/between_person_descriptive_statistics.csv")
    mixed_model_results = pd.read_csv("output/results/univariable_linear_mixed_models.csv")
    mixed_model_results = mixed_model_results.merge(between_person_descriptives, how="left")

    n_obs = mixed_model_results["n_observations"].unique()[0]

    selected_columns = ["activity", "n", "percent", "beta", "stderr", "confint", "p_value", "q_value"]
    selected_columns_new_names = ["Activity", "n", "%", "Beta", "SE", "95% CI", "P-value", "Q-value"]
    spanner_column_names = ["", f"Endorsement (N={n_obs})", "", "Association with daily goodness rating"] + np.repeat("", 4).tolist()

    results_table = (
        mixed_model_results
            .pipe(lambda x: x[x["variable"] != "Intercept"])
            .assign(activity = lambda x: format_activity_names(x["variable"]))
            .assign(beta = lambda x: format_point_estimate(x["beta"]))
            .assign(stderr = lambda x: format_point_estimate(x["stderr"]))
            .assign(confint = lambda x: format_confidence_interval(x["ci_low"], x["ci_high"]))
            .assign(p_value = lambda x: format_p_value(x["p_value"]))
            .assign(q_value = lambda x: format_p_value(x["q_value"]))
            .assign(percent = lambda x: format_proportion_as_percent(x["mean"]))
            .assign(n = lambda x: x["sum"].astype(int))
            .filter(selected_columns, axis=1)
            .sort_values("activity")
            .reset_index(drop=True)
            .rename(columns=dict(zip(selected_columns, selected_columns_new_names)))
    )

    results_table.columns = [spanner_column_names, results_table.columns.get_level_values(0)]
    results_table.to_csv("output/tables/table_1.csv", index=False)

if __name__ == "__main__":
    main()