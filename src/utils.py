import pandas as pd
import numpy as np
import re
import yaml
import os
import logging

def initialize_logging(log_name):
    logger = logging.getLogger(__name__)
    log_directory = os.getcwd() + "/" + "logs"
    log_file = f"{log_directory}/{log_name}.log"
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)
    if os.path.isfile(log_file):
        os.remove(log_file)
    fileHandler = logging.FileHandler(log_file)
    logger.addHandler(fileHandler)
    logger.setLevel(logging.INFO)
    return logger

def load_config():
    with open("config.yaml") as file:
        config = yaml.safe_load(file)
    return config

def generate_participant_list(min_id, max_id):
    """Create a list of participant IDs (strings) between `min_id` and `max_id`, inclusive"""
    return [str(x) for x in range(min_id, max_id+1)]

def rename_columns(data):
    """Clean column names in data exported from Qualtrics"""
    replacements = [
        ("\\(s\\)", "s"),
        (" \\(Select all that apply\\)", ""),
        (" \\(.*\\)", ""),
        ("\\?$|\\? ", ""),
        (" ____", "_"),
        ("-| -", ""),
        ("\\:", ""),
        (" ", "_"),
        (",", ""),
        ("/", "_")
    ]

    clean_column_names = data.columns
    for old, new in replacements:
        clean_column_names = [re.sub(old, new, string) for string in clean_column_names]
    clean_column_names = [col.lower() for col in clean_column_names]
    data.columns = clean_column_names
    return data


def remove_preceding_zeros(data, column):
    """Remove preceding zeros from a string column"""
    data[column] = data[column].replace("^0", "", regex=True)
    return data


def clean_study_date_data(data):
    """Clean up study start and end date data"""
    data["ema_start_date"] = data["ema_start_date"].fillna(data["interview_date"])
    data = data[data["study_id"].notna()].reset_index(drop=True) # & (data["study_id"].str.contains("Withdrawn") == False)
    data.rename(columns={"end_date":"ema_end_date"}, inplace=True)
    return data


def assign_responses_to_date(data, date_time_threshold=0):
    """Assign response to a date based on the survey start date-time
    Parameters:
        data (pandas.DataFrame): Dataset containing, at a minimum, a "start_date" column
        threshold_minutes_since_midnight (int): The time at which the day "begins" in minutes since midnight.
            If the survey start time is >= this threshold, `date` will be equal to the survey start date. If
            the survey start time is < this threshold, `date` will be equal to the survey start day minus 1 day
            (i.e., we assume such a response corresponds to the previous day). The default is 0.
    Returns:
        pandas.DataFrame with added `time_minutes_since_midnight` and `date` columns.
    """
    if date_time_threshold < 0 or date_time_threshold > 1439:
        raise ValueError("`date_time_threshold` must be an integer between 0 and 1439, inclusive")

    data["time_minutes_since_midnight"] = data["start_date"].apply(lambda x: int(x.strftime("%H"))*60 + int(x.strftime("%M")))
    data["date"] = np.where(
        data["time_minutes_since_midnight"] >= date_time_threshold, 
        pd.to_datetime(data["start_date"]).dt.date, 
        pd.to_datetime(data["start_date"]).dt.date - pd.Timedelta(1, unit = "D")
    )
    return data


def format_activity_names(activity_column):
    """Format activity names for presentation in a figure or table"""
    return (
        activity_column
            .str.replace("^what_did_you_spend_time_doing_today_", "", regex=True)
            .str.replace("_", " ")
            .str.replace(" call", "")
            .str.replace(" email.*", " or email", regex=True)
            .str.capitalize()
            .str.replace(" tv$", " TV", regex=True)
    )