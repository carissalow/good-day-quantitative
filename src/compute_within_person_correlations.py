import pandas as pd
import numpy as np
import pingouin
from scipy import stats
from utils import load_config 


def compute_within_person_correlations(data, pids, activity_columns, target_column, p_adjust, p_adjust_method, p_adjust_level):
    if p_adjust:
        if not p_adjust_method in ["bonf", "sidak", "holm", "fdr_bh", "fdr_by", "none"]:
            raise ValueError("`p_adjust_method` must be one of: bonf, sidak, holm, fdr_bh, fdr_by, none")
        if not p_adjust_level in ["participant", "overall"]:
            raise ValueError("`p_adjust_level` must be one of: participant, overall")

    within_person_correlations = pd.DataFrame()
    for pid in pids:
        data_for_pid = data[data["id"] == pid].set_index("date")
        # if a participant has fewer than 2 observations or zero variance in the target column (e.g., goodness rating), we skip them
        if (data_for_pid.shape[0] < 2) or (data_for_pid[target_column].var() == 0):
            continue
        correlations_for_pid = pd.DataFrame()
        for activity in activity_columns:
            # if an activity has zero variance (i.e., it was always endorsed or never endorsed by the participant), we skip it
            if data_for_pid[activity].var() == 0:
                continue
            corr = stats.pointbiserialr(x=data_for_pid[activity], y=data_for_pid[target_column])
            correlations_for_activity = pd.DataFrame({"pid":[pid], "activity":[activity], "method":["Point biserial correlation"], "r":[corr[0]], "p_value":[corr[1]]})
            correlations_for_pid = pd.concat([correlations_for_pid, correlations_for_activity], axis=0)
        if p_adjust & (p_adjust_level == "participant"):
            correlations_for_pid["q_value"] = pingouin.multicomp(pvals=correlations_for_pid["p_value"], method=p_adjust_method)[1]
        within_person_correlations = pd.concat([within_person_correlations, correlations_for_pid], axis=0)
    
    if p_adjust & (p_adjust_level == "overall"):
        within_person_correlations["q_value"] = pingouin.multicomp(pvals=within_person_correlations["p_value"], method=p_adjust_method)[1]
    
    within_person_correlations.reset_index(drop=True, inplace=True) 
    return within_person_correlations


def main():
    config = load_config()
    within_person_correlation_params = config["ANALYSIS_PARAMETERS"]["WITHIN_PERSON_CORRELATIONS"]

    TARGET_COLUMN = within_person_correlation_params["TARGET_COLUMN"]
    ADJUST_P_VALUES = within_person_correlation_params["MULTIPLE_COMPARISONS"]["ADJUST_P_VALUES"]
    ADJUST_P_VALUES_METHOD = within_person_correlation_params["MULTIPLE_COMPARISONS"]["METHOD"]
    ADJUST_P_VALUES_LEVEL = within_person_correlation_params["MULTIPLE_COMPARISONS"]["LEVEL"]

    diary_data_clean = pd.read_csv("data/processed/good_day_diary_data_cleaned.csv", dtype={"id":str}, parse_dates=["date"])
    pids = diary_data_clean["id"].unique()
    activity_columns = [col for col in diary_data_clean if "what_did_you_spend_time_doing_today" in col] 

    within_person_correlations = compute_within_person_correlations(
        data=diary_data_clean,
        pids=pids,
        activity_columns=activity_columns,
        target_column=TARGET_COLUMN,
        p_adjust=ADJUST_P_VALUES,
        p_adjust_method=ADJUST_P_VALUES_METHOD,
        p_adjust_level=ADJUST_P_VALUES_LEVEL
    )

    within_person_correlations.to_csv("output/results/within_person_point_biserial_correlations.csv", index=False)

if __name__ == "__main__":
    main()