import pandas as pd
import numpy as np

from utils import load_config, rename_columns, remove_preceding_zeros

def main():
    config = load_config()
    PARTICIPANTS = config["PARTICIPANTS"]
    PROMIS_DATA_FILE = config["RAW_DATA_FILES"]["PROMIS_DATA"]
    PROMIS_SCORE_TYPE = config["DATA_CLEANING_PARAMETERS"]["PROMIS"]["SCORE_TYPE"]
    if not PROMIS_SCORE_TYPE in ["raw_score", "t_score", "standard_error", "theta"]:
        raise ValueError("`PROMIS_SCORE_TYPE` must be one of: raw_score, t_score, standard_error, theta")

    participant_ids = participant_ids = [str(pid) for pid in PARTICIPANTS]
    promis_data = pd.read_csv("data/raw/" + PROMIS_DATA_FILE, dtype={"Record ID":"string"})

    promis_data = (
        promis_data
            .pipe(rename_columns)
            .pipe(remove_preceding_zeros, "record_id")
            .pipe(lambda x: x[["record_id"] + [col for col in x.columns if PROMIS_SCORE_TYPE in col]])
            .pipe(lambda x: x[x["record_id"].isin(participant_ids)])
    )

    promis_data.to_csv("data/processed/good_day_baseline_promis_data_cleaned.csv", index=False)

if __name__ == "__main__":
    main()