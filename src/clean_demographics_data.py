import pandas as pd
import numpy as np

from utils import load_config, rename_columns, remove_preceding_zeros

def main():
    config = load_config()
    PARTICIPANTS = config["PARTICIPANTS"]
    DEMOGRAPHICS_DATA_FILE = config["RAW_DATA_FILES"]["DEMOGRAPHICS_DATA"]

    participant_ids = [str(pid) for pid in PARTICIPANTS]
    demographics_data = pd.read_csv("data/raw/" + DEMOGRAPHICS_DATA_FILE)

    demographics_data = (
        demographics_data
        .pipe(rename_columns)
        .pipe(remove_preceding_zeros, "record_id")
        .pipe(lambda x: x[x["record_id"].isin(participant_ids)])
        .rename(columns={"age":"age_years", "1._what_is_your_gender":"gender", "6._what_is_your_race_ethnicity":"race"})
        .filter(["record_id", "age_years", "gender", "race"])
        .reset_index(drop=True)
    )

    demographics_data.to_csv("data/processed/good_day_demographics_data_cleaned.csv", index=False)

if __name__ == "__main__":
    main()