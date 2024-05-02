import pandas as pd
import numpy as np


def compute_within_person_descriptives(data, goodness_columns, activity_columns):
    """Compute descriptive statistics within participants (i.e., separately for each unique ID)"""
    numeric_columns_agg_functions = dict.fromkeys(goodness_columns, ["count", "mean", "std", "median", "min", "max"])
    categorical_columns_agg_functions = dict.fromkeys(activity_columns, ["count", "sum", "mean"]) # the mean of a binary variable is the proportion of 1's

    within_person_descriptives = (
        data
            .filter(["id"] + goodness_columns + activity_columns)
            .groupby("id", observed=True)
            .agg({**numeric_columns_agg_functions, **categorical_columns_agg_functions})
            .round(10)
            .reset_index()
        )

    within_person_descriptives_columns = [f"{top}_{bottom}" if bottom != "" else f"{top}" for top, bottom in within_person_descriptives.columns.to_flat_index()]
    within_person_descriptives.droplevel(axis=1, level=0)
    within_person_descriptives.columns = within_person_descriptives_columns
    return within_person_descriptives


def compute_between_person_descriptives_longitudinal(data, goodness_columns, activity_columns):
    """Compute descriptive statistics across participants from longitudinal data with one or more values per participant"""
    numeric_columns_agg_functions = dict.fromkeys(goodness_columns, ["count", "mean", "std", "median", "min", "max"])
    categorical_columns_agg_functions = dict.fromkeys(activity_columns, ["count", "sum", "mean"])

    between_person_descriptives_longitudinal = (
        data
            .agg({**numeric_columns_agg_functions, **categorical_columns_agg_functions})
            .round(10)
            .transpose()
            .rename_axis("variable")
            .reset_index()
        )
    return between_person_descriptives_longitudinal


def compute_between_person_descriptives_cross_sectional(data, demographics_columns, promis_columns, survey_compliance_columns):
    """Compute descriptive statistics across participants from cross-sectional data with one value per participant"""
    numeric_demographics_columns = data[demographics_columns].select_dtypes(include=[np.number]).columns.values.tolist()
    categorical_demographics_columns = pd.get_dummies(data[demographics_columns]).columns.difference(numeric_demographics_columns)
    
    numeric_columns_agg_functions = dict.fromkeys(numeric_demographics_columns + promis_columns + survey_compliance_columns, ["count", "mean", "std", "median", "min", "max"])
    categorical_columns_agg_functions = dict.fromkeys(categorical_demographics_columns, ["count", "sum", "mean"])

    between_person_descriptives_cross_sectional = (
        data
            .filter(["id"] + demographics_columns + promis_columns + survey_compliance_columns)
            .drop_duplicates()
            .set_index("id")
            .pipe(pd.get_dummies, dtype="int")
            .reset_index()
            .agg({**numeric_columns_agg_functions, **categorical_columns_agg_functions})
            .round(10)
            .transpose()
            .rename_axis("variable")
            .reset_index()
    )
    return between_person_descriptives_cross_sectional


def compute_between_person_descritptives(data, goodness_columns, activity_columns, demographics_columns, promis_columns, survey_compliance_columns):
    """Compute descriptive statistics across participants"""
    between_person_descriptives_longitudinal = compute_between_person_descriptives_longitudinal(data, goodness_columns, activity_columns)
    between_person_descriptives_cross_sectional = compute_between_person_descriptives_cross_sectional(data, demographics_columns, promis_columns, survey_compliance_columns)
    between_person_descriptives = (
        pd.concat([between_person_descriptives_longitudinal, between_person_descriptives_cross_sectional], axis=0)
            .reset_index(drop=True)
            .assign(count = lambda x: x["count"].astype("Int64"))
            .assign(sum = lambda x: x["sum"].astype("Int64"))
    )
    return between_person_descriptives


def main():
    diary_data = pd.read_csv("data/processed/good_day_diary_data_cleaned.csv", dtype={"id":str})
    demographics_data = pd.read_csv("data/processed/good_day_demographics_data_cleaned.csv", dtype={"record_id":str})
    promis_data = pd.read_csv("data/processed/good_day_baseline_promis_data_cleaned.csv", dtype={"record_id":str})

    goodness_columns = ["how_would_you_describe_today"]
    activity_columns = [col for col in diary_data.columns if "what_did_you_spend_time_doing_today" in col]
    demographics_columns = ["age_years", "gender", "race"]
    promis_columns = [col for col in promis_data.columns if "baseline_promis" in col]

    data_for_descriptives = (
        diary_data
            .merge(demographics_data, left_on="id", right_on="record_id", how="left")
            .merge(promis_data, left_on="id", right_on="record_id", how="left")
            .filter(["id"] + goodness_columns + activity_columns + demographics_columns + promis_columns)
            .assign(n_responses = lambda x: x.groupby("id")["id"].transform("count"))
    )

    within_person_descriptives = compute_within_person_descriptives(
        data=data_for_descriptives, 
        goodness_columns=goodness_columns, 
        activity_columns=activity_columns
    )

    between_person_descriptives = compute_between_person_descritptives(
        data=data_for_descriptives, 
        goodness_columns=goodness_columns, 
        activity_columns=activity_columns, 
        demographics_columns=demographics_columns, 
        promis_columns=promis_columns,
        survey_compliance_columns=["n_responses"]
    )

    within_person_descriptives.to_csv("output/descriptives/within_person_descriptive_statistics.csv", index=False)
    between_person_descriptives.to_csv("output/descriptives/between_person_descriptive_statistics.csv", index=False)


if __name__ == "__main__":
    main()