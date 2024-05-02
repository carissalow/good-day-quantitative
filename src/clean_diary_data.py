import pandas as pd
import numpy as np
import re

from functools import wraps
from utils import load_config, rename_columns, remove_preceding_zeros, clean_study_date_data, assign_responses_to_date, initialize_logging

logger = initialize_logging(log_name="clean_diary_data")


def log_pipeline_step(func):
    """Write to a log file the number of rows, participants, and rows per participant in the dataset following each cleaning step"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        result_rows = result.shape[0]
        result_pids = [str(pid) for pid in result["id"].unique().tolist()]
        rows_per_pid = (
            result.copy()
                .assign(id = lambda x: x.id.astype("Int64")).sort_values("id")
                .groupby("id", dropna=False).size().to_frame()
                .reset_index().rename(columns={0:"n_rows"})
        )
        log_message = (
            f"Following pipeline step `{func.__name__}`:\n"
            f"The dataset has {result_rows} rows from {len(result_pids)} participants.\n"
            f"The dataset has the following number of rows per participant:\n{rows_per_pid}\n"
        )
        logger.info(log_message)
        return result
    return wrapper


@log_pipeline_step
def start_pipeline(data):
    """Helper function to capture info about the raw data file before cleaning steps are applied"""
    return data


@log_pipeline_step
def drop_invalid_responses(data, progress_threshold=50, participant_list=[]):
    """Filter out responses that are incomplete or associated with invalid participant IDs
    Parameters:
        data: A pandas data frame
        progress_threshold (int): Integer corresponding to percent progress at or above which a survey
            is considered sufficiently complete and retained in the output. The default is 50.
        participant_list (list): List of strings corresponding to valid participant IDs. The default
            is an empty list, in which case all participant IDs present in the data will be included
            in the output.
    Returns:
        pd.DataFrame with only rows meeting specified criteria retained.
    """
    if progress_threshold < 0 or progress_threshold > 100:
        raise ValueError("`progress_threshold` must be an integer between 0 and 100, inclusive")
        
    if len(participant_list) > 0:
        data = data[data["id"].isin(participant_list)]
    data = data[data["progress"] >= progress_threshold]
    return data


@log_pipeline_step
def drop_responses_outside_study_period(data, before_start_date=True, after_end_date=True):
    """Drop any responses collected prior to the participant's EMA start date or after the participant's study end date"""
    if before_start_date:
        data = data[data["date"] >= pd.to_datetime(data["ema_start_date"]).dt.date]
    if after_end_date:
        data = data[data["date"] <= pd.to_datetime(data["ema_end_date"]).dt.date]
    return data


@log_pipeline_step
def deduplicate_responses(data, deduplication_method="none", prioritize_nonmissing=True):
    """Deduplicate survey responses using the specified method
    Parameters:
        data: A pandas data frame
        deduplication_method (str): One of 'none', 'min', or 'max'.
            If the method is 'none', all responses will be retained. Thus, there may be two or more responses per 
            date per participant (as assigned based on the specified time threshold) in the resulting dataset. 
            If the method is 'min', the first response per date per participant will be retained. 
            If the method is 'max', the last response per date per participant will be retained.
        prioritize_nonmissing (bool): If True, responses with a non-missing value for the daily goodness 
            rating will be retained even if they do not meet the deduplication method's criteria (e.g., a survey
            completed later in the day will be retained if the method was 'min' but the earliest completed survey 
            is missing a value for the goodness rating)
    Returns:
        pd.DataFrame with only rows meeting specified criteria retained.
    """
    if deduplication_method not in ["none", "min", "max"]:
        raise ValueError("`deduplication_method` must be one of: none, min, max")

    if deduplication_method != "none":
        if prioritize_nonmissing:
            data["selected_timestamp"] = (
                data
                    .assign(selected_timestamp = data["time_minutes_since_midnight"].where(data["how_would_you_describe_today"].notna()))
                    .groupby(["id", "date"])["selected_timestamp"]
                    .transform(deduplication_method)
                    .fillna(data["time_minutes_since_midnight"])
                )   
        else:
            data["selected_timestamp"] = data.groupby(["id", "date"])["time_minutes_since_midnight"].transform(deduplication_method)
        data = data[data["time_minutes_since_midnight"] == data["selected_timestamp"]].drop(columns=["selected_timestamp"])
        
    return data


def clean_activity_endorsements(data):
    """For activity questions, replace missing response values with 0
    Due to the way Qualtrics records responses, for sufficiently complete surveys, we assume a missing value
    reflects that the partcipant did not endorse the activity, rather than that the participant skipped the question.
    """
    activity_columns = [col for col in data.columns if "what_did_you_spend_time_doing" in col]
    data[activity_columns] = data[activity_columns].fillna(value = 0)
    return data


def person_center_goodness_ratings(data):
    """Person-center daily goodness ratings two ways for between-subjects comparisons
    We perform the following calculations: (1) subtract participant-specific mean from each rating, and (2) compute 
    participant-specific z-score (subtract participant-specific mean and divide by the articipant-specific standard 
    deviation). Note that z-score is missing when SD is 0 (i.e., if there was a single response for the participant, 
    or there were multiple responses with no variance).
    """
    data["how_would_you_describe_today_mean_centered"] = data["how_would_you_describe_today"] - data.groupby(["id"])["how_would_you_describe_today"].transform("mean")
    data["how_would_you_describe_today_z_scored"] = data["how_would_you_describe_today_mean_centered"]/data.groupby(["id"])["how_would_you_describe_today"].transform("std")
    return data


def subset_columns(data, data_type="diary"):
    """Select subset of columns for analysis"""
    if data_type not in ["study_dates", "diary"]:
        raise ValueError("`data_type` must be one of: study_dates, diary")

    if data_type == "study_dates":
        data = data[["study_id", "ema_start_date", "ema_end_date"]]
    elif data_type == "diary":
        data["id"] = data["id"].astype("int")
        id_columns = ["response_id", "id", "date", "time_minutes_since_midnight", "progress"]
        outcome_columns = [col for col in data.columns if "how_would_you_describe_today" in col]
        activity_columns = [col for col in data.columns if "what_did_you_spend_time_doing" in col]
        data = data[id_columns + outcome_columns + activity_columns]

    return data


def main():
    config = load_config()

    PARTICIPANTS = config["PARTICIPANTS"]
    STUDY_DATE_DATA_FILE = config["RAW_DATA_FILES"]["STUDY_DATE_DATA"]
    DIARY_DATA_FILE = config["RAW_DATA_FILES"]["DIARY_DATA"]

    # data cleaning params
    data_cleaning_params = config["DATA_CLEANING_PARAMETERS"]["DIARY"]
    PROGRESS_THRESHOLD = data_cleaning_params["PROGRESS_THRESHOLD"]
    DATE_TIME_THRESHOLD = data_cleaning_params["DATE_TIME_THRESHOLD"]
    DROP_RESPONSES_BEFORE_START = data_cleaning_params["DROP_RESPONSES"]["BEFORE_STUDY_START_DATE"]
    DROP_RESPONSES_AFTER_END = data_cleaning_params["DROP_RESPONSES"]["AFTER_STUDY_END_DATE"]
    DEDUPLICATION_METHOD = data_cleaning_params["DEDUPLICATE_REPONSES"]["METHOD"]
    PRIORITIZE_NONMISSING = data_cleaning_params["DEDUPLICATE_REPONSES"]["PRIORITIZE_NONMISSING"]

    participant_ids = [str(pid) for pid in PARTICIPANTS]

    study_date_data = pd.read_csv(
        "data/raw/" + STUDY_DATE_DATA_FILE,
        dtype={"Study ID":"string"}, 
        parse_dates=["Interview Date", "EMA Start Date", "End Date"], 
        date_format = "%m/%d/%y"
    )

    study_date_data = (
        study_date_data
            .pipe(rename_columns)
            .pipe(remove_preceding_zeros, column="study_id")
            .pipe(clean_study_date_data)
            .pipe(subset_columns, data_type="study_dates")
    )

    daily_diary_data = pd.read_csv(
        "data/raw/" + DIARY_DATA_FILE, 
        skiprows=[0,2], 
        dtype={"id":"string"}, 
        parse_dates=["Start Date", "End Date", "Recorded Date"], 
        date_format = "%m/%d/%y %H:%M"
    )

    daily_diary_data = (
        daily_diary_data
            .pipe(start_pipeline)
            .pipe(rename_columns)
            .pipe(pd.merge, study_date_data, left_on=["id"], right_on=["study_id"], how="left")
            .pipe(assign_responses_to_date, date_time_threshold=DATE_TIME_THRESHOLD)
            .pipe(drop_invalid_responses, progress_threshold=PROGRESS_THRESHOLD, participant_list=participant_ids)
            .pipe(drop_responses_outside_study_period, before_start_date=DROP_RESPONSES_BEFORE_START, after_end_date=DROP_RESPONSES_AFTER_END)
            .pipe(deduplicate_responses, deduplication_method=DEDUPLICATION_METHOD, prioritize_nonmissing=PRIORITIZE_NONMISSING)
            .pipe(clean_activity_endorsements)
            .pipe(person_center_goodness_ratings)
            .pipe(subset_columns, data_type="diary")
            .sort_values(["id", "date"])
            .reset_index(drop=True)
        )

    daily_diary_data.to_csv("data/processed/good_day_diary_data_cleaned.csv", index=False)


if __name__ == "__main__":
    main()