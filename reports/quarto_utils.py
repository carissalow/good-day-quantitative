import pandas as pd
import numpy as np
import plotnine as p9
import itertools
import yaml
import re

#### Load and parse analysis metadata ----

def read_config():
    with open("../config.yaml") as file:
        config = yaml.safe_load(file)
    return config

def parse_data_cleaning_log():
    with open("../logs/clean_diary_data.log", "r") as log:
        lines = log.readlines()

        step_name_regex = r"Following pipeline step `(.*)`\:"
        step_rows_regex = r"The dataset has (.*) rows from (.*) participants."
        step_rows_per_pid_regex = r"^(\d*)\s*(<NA>|\d*)\s*(\d*)"

        step_names = [re.search(step_name_regex, line).group(1) for line in lines if re.search(step_name_regex, line) is not None]
        step_n_rows = [re.search(step_rows_regex, line).group(1) for line in lines if re.search(step_rows_regex, line) is not None]
        step_n_pids = [re.search(step_rows_regex, line).group(2) for line in lines if re.search(step_rows_regex, line) is not None]
        step_pids = []
        step_rows_per_pid = []
    
        step_start_positions = [i for i, line in enumerate(lines) if re.search(step_name_regex, line) is not None]
        for i, _ in enumerate(step_start_positions):
            if i+1 < len(step_start_positions):
                slice = lines[step_start_positions[i]:step_start_positions[i+1]]
            else:
                slice = lines[step_start_positions[i]:]

            slice_pids = [re.search(step_rows_per_pid_regex, line).group(2) for line in slice if re.search(step_rows_per_pid_regex, line) is not None]
            slice_pids = [pid for pid in slice_pids if pid != ""]
            step_pids.append(slice_pids)

            slice_rows_per_pid = [re.search(step_rows_per_pid_regex, line).group(3) for line in slice if re.search(step_rows_per_pid_regex, line) is not None]
            slice_rows_per_pid = [row for row in slice_rows_per_pid if row != ""]
            step_rows_per_pid.append(slice_rows_per_pid)

    log_data = pd.DataFrame({'step':step_names, 'n_rows_after_step':step_n_rows, 'n_pids_after_step':step_n_pids, 'pids':step_pids, 'rows_per_pid':step_rows_per_pid})

    log_data = (
        pd.DataFrame({'step':step_names, 'n_rows_after_step':step_n_rows, 'n_pids_after_step':step_n_pids, 'pids':step_pids, 'rows_per_pid':step_rows_per_pid})
            .assign(n_rows_before_step = lambda x: np.where(x["n_rows_after_step"].shift().isnull(), x["n_rows_after_step"], x["n_rows_after_step"].shift()))
            .assign(n_rows_dropped = lambda x: x["n_rows_before_step"].astype(float) - x["n_rows_after_step"].astype(float))
    )
    return log_data

def create_readable_data_cleaning_report():
    def _extract_step_value(data, step_name, column):
        return data.query(f"step == '{step_name}'")[column].iloc[0]

    def _format_step_values(data, step_name):
        numerator = int(_extract_step_value(data, step_name, "n_rows_dropped"))
        denominator = int(_extract_step_value(data, step_name, "n_rows_before_step"))
        percent = round(_extract_step_value(data, step_name, "prop_rows_dropped")*100, 2)
        return f"n={numerator}/{denominator}, {percent}%"

    config = read_config()
    participants = [str(pid) for pid in config["PARTICIPANTS"]]

    log_data = parse_data_cleaning_log()

    log_data_per_pid = (
        log_data
            .filter(["step", "pids", "rows_per_pid"])
            .explode(["pids", "rows_per_pid"])
            .assign(rows_per_pid = lambda x: x["rows_per_pid"].astype(int))
            .assign(invalid_pid = lambda x: np.where(x["pids"].isin(participants), 0, 1))
    )

    pids_in_raw_data = log_data.query("step == 'start_pipeline'")["pids"].iloc[0]
    pids_in_clean_data = log_data.query("step == 'deduplicate_responses'")["pids"].iloc[0]
    invalid_pids = log_data_per_pid.query("invalid_pid == 1")["pids"].tolist()
    missing_pids = list(set(participants).difference(set(pids_in_raw_data)))
    dropped_pids = list(set([pid for pid in pids_in_raw_data if pid not in invalid_pids]).difference(set(pids_in_clean_data)))

    n_dropped_invalid_pid = log_data_per_pid.query("(step == 'start_pipeline') & (invalid_pid == 1)").agg({"rows_per_pid":"sum"}).iloc[0]
    n_before_dropped_incomplete = _extract_step_value(log_data, "drop_invalid_responses", "n_rows_before_step")
    n_dropped_incomplete =  n_dropped_invalid_pid - _extract_step_value(log_data, "drop_invalid_responses", "n_rows_dropped") 
    n_before_dropped_invalid_pid = int(float(_extract_step_value(log_data, "drop_invalid_responses", "n_rows_before_step")) - n_dropped_incomplete)

    drop_invalid_responses_details_rows = pd.DataFrame({ 
        "step":["drop_invalid_responses_incomplete", "drop_invalid_responses_invalid_pid"],
        "n_rows_before_step":[n_before_dropped_incomplete, n_before_dropped_invalid_pid],
        "n_rows_dropped":[n_dropped_incomplete, n_dropped_invalid_pid]
    })
        
    log_data = (
        log_data    
            .filter(["step", "n_rows_before_step", "n_rows_dropped"])
            .pipe(lambda x: pd.concat([x, drop_invalid_responses_details_rows], axis=0))
            .reset_index(drop=True)
            .assign(prop_rows_dropped = lambda x: x["n_rows_dropped"].astype(int)/x["n_rows_before_step"].astype(int))
    )

    return {
        "invalid_pids":invalid_pids,
        "missing_pids":missing_pids,
        "dropped_pids":dropped_pids,
        "start_pids":pids_in_raw_data,
        "start_rows":_extract_step_value(log_data, "start_pipeline", "n_rows_before_step"),
        "step_dropped_row_stats":{
            "drop_incomplete":_format_step_values(log_data, "drop_invalid_responses_incomplete"),
            "drop_invalid_pid":_format_step_values(log_data, "drop_invalid_responses_invalid_pid"),
            "drop_date":_format_step_values(log_data, "drop_responses_outside_study_period"),
            "drop_duplicates":_format_step_values(log_data, "deduplicate_responses")
        }
    }

def interpret_readable_data_cleaning_report(report, type):
    interpretation = ""
    if type == "invalid_pids":
        pid_list = [pid.replace("<", "").replace(">", "") for pid in report["invalid_pids"]]
        pid_list.sort()
        if len(pid_list) > 0:
            interpretation = f" (including responses for the following {len(report['invalid_pids'])} missing/invalid ID{'s' if len(pid_list) > 1 else ''}: {', '.join(pid_list)})"
    
    if type == "missing_pids":
        pid_list = report["missing_pids"]
        pid_list.sort()
        if len(pid_list) > 0:
            if len(pid_list) == 1:
                interpretation = f"There were no responses for expected participant ID {', '.join(pid_list)}."
            else:
                interpretation = f"There were no responses for the following {len(pid_list)} expected participant IDs: {', '.join(pid_list)}."
    
    if type == "dropped_pids":
        pid_list = report["dropped_pids"]
        pid_list.sort()
        if len(pid_list) > 0:
            if len(pid_list) == 1:
                interpretation = f"All responses for participant ID {''.join(pid_list)} were dropped."
            else:
                interpretation = f"All responses for the following {len(pid_list)} participant IDs were dropped: {', '.join(pid_list)}."
    return interpretation

def interpret_minutes_after_midnight(data_cleaning_params):
    minutes_after_midnight = data_cleaning_params["DATE_TIME_THRESHOLD"]
    if minutes_after_midnight == 0:
        return "12:00 AM and 11:59 PM on a given date"
    else:
        start_hours = minutes_after_midnight//60
        start_mins = minutes_after_midnight%60
        end_hours = (minutes_after_midnight-1)//60
        end_mins = (minutes_after_midnight-1)%60

        if start_hours < 12:
            start_time = f"{'12' if start_hours == 0 else start_hours}:{'0' + str(start_mins) if len(str(start_mins)) < 2 else start_mins} AM"
            end_time = f"{'12' if end_hours == 0 else end_hours}:{'0' + str(end_mins) if len(str(end_mins)) < 2 else end_mins} AM"
        else:
            start_time = f"{start_hours if start_hours == 12 else start_hours-12}:{'0' + str(start_mins) if len(str(start_mins)) < 2 else start_mins} PM"
            end_time = f"{end_hours if end_hours == 12 else end_hours-12}:{'0' + str(end_mins) if len(str(end_mins)) < 2 else end_mins} PM"
        return f"{start_time} on a given date and {end_time} on the following day"
    
def intepret_drop_responses(data_cleaning_params):
    before_start = data_cleaning_params["DROP_RESPONSES"]["BEFORE_STUDY_START_DATE"]
    after_end = data_cleaning_params["DROP_RESPONSES"]["AFTER_STUDY_END_DATE"]
    if before_start & after_end:
        return "before the participant's study start date and after the participant's study end date"
    elif before_start:
        return "before the participant's study start date" 
    else:
        return "after the participant's study end date"

def interpret_deduplicate_responses(data_cleaning_params):
    method = data_cleaning_params["DEDUPLICATE_REPONSES"]["METHOD"]
    keep_nonmiss = data_cleaning_params["DEDUPLICATE_REPONSES"]["PRIORITIZE_NONMISSING"] 
    if keep_nonmiss:
        keep_nonmiss_detail = " with a non-missing value for the day's overall goodness rating"
    else:
        keep_nonmiss_detail = ""
    return f"{'latest' if method == 'max' else 'earliest'} completed survey (i.e., the survey associated with the {method} timestamp){keep_nonmiss_detail}"

def interpret_padjust_method(multi_comp_setting):
    method_to_name = {
        "bonf":"using Bonferroni adjustment", 
        "sidak":"using Sidak adjustment", 
        "holm":"using Holm adjustment", 
        "fdr_bh":"by controlling the false discovery rate (Benjamini-Hochberg procedure)", 
        "fdr_by":"by controlling the false discovery rate (Benjamini-Yekutieli procedure)", 
        "none":"We did not correct for multiple comparisons"
    }
    if multi_comp_setting["ADJUST_P_VALUES"]:
        return "We corrected for multiple comparisons " + ("for each participant " if multi_comp_setting.get("LEVEL", "") == "participant" else "") + method_to_name[multi_comp_setting["METHOD"]]
    else:
        return "We did not correct for multiple comparisons"

def interpret_promis_score_type(promis_score_type_setting):
    score_type_to_name = {
        "raw_score":"raw score", 
        "t_score":"t-score", 
        "standard_error":"standard error", 
        "theta":"theta value"
    }
    return score_type_to_name[promis_score_type_setting]

#### Format values ----

def format_proportion_as_percent(proportion):
    return (proportion*100).round(2).astype(str) 

def format_n_and_percent(n, percent):
    return n.astype(int).astype(str) + " (" + percent + "%)"

def format_point_estimate(estimate, decimals=3):
    return estimate.round(decimals).astype(str)

def format_ttest_means(group, group_mean, group_std):
    return "*M (SD)*<sub>" + group + "</sub> = " + format_point_estimate(group_mean, 2) + " (" + format_point_estimate(group_std, 2) + ")"

def format_ttest_statistic(t, df):
    return "*t*(" + format_point_estimate(df, 2) + ") = " + format_point_estimate(t, 2)

def format_correlation_coefficient(r):
    return "*r* = " + format_point_estimate(r, 2)

def format_confidence_interval(low, high):
    return low.round(3).astype(str) + ", " + high.round(3).astype(str)

def format_p_value(p_value):
    p_value = p_value.astype(float)
    formatted_p_value = np.where(p_value.round(3) < 0.001, "<0.001", p_value.round(3).astype(str))
    formatted_p_value = np.where(p_value.round(3) > 0.999, ">0.999", formatted_p_value)
    return formatted_p_value

def format_activity_names(activity_column):
    """Convert names of activities contained in a specified column (pd.Series) to a readable format"""
    return (
        activity_column
            .str.replace("^what_did_you_spend_time_doing_today_", "", regex=True)
            .str.replace("_", " ")
            .str.replace(" call", "")
            .str.replace(" email.*", " or email", regex=True)
            .str.capitalize()
            .str.replace(" tv$", " TV", regex=True)
    )

def format_covariate_names(covariate_column):
    """Convert names of covariates contained in a specified column (pd.Series) to a readable format"""
    repl = lambda m: "&nbsp;&nbsp;&nbsp;" + m.group('two').capitalize()
    return (
        covariate_column
            .str.replace("^gender_|^race_", "&nbsp;&nbsp;&nbsp;", regex = True)
            .str.replace("_", " ")
            .str.capitalize()
            .str.replace("Age years", "Age (years)")
            .str.replace(r"(?P<one>&nbsp;&nbsp;&nbsp;)(?P<two>.*)", repl, regex=True)
            .str.replace("african american", "African American")
            .str.replace("caucasian", "Caucasian")
    )

def format_promis_names(covariate_column, promis_score_type_setting=["t_score", "raw_score", "standard_error", "theta"]):
    """Convert names of PROMIS domains contained in a specified column (pd.Series) to a readable format"""
    repl = lambda m: "&nbsp;&nbsp;&nbsp;" + m.group('two').capitalize()
    return (
        covariate_column
            .str.replace(" " + promis_score_type_setting.replace("_", " ") + "$", "", regex=True)
            .str.replace("^Baseline promis ", "&nbsp;&nbsp;&nbsp;", regex=True)
            .str.replace("anxiety ", "anxiety/")
            .str.replace("depression ", "depression/")
            .str.replace("roles ", "roles/")
            .str.replace(r"(?P<one>&nbsp;&nbsp;&nbsp;)(?P<two>.*)", repl, regex=True)
    )

def bold_labels(column):
    """Add Markdown syntax for bold text to any strings in a column (pd.Series) if the string is not indented (contains &nbsp;)"""
    return ["**" + x + "**" if "&nbsp;" not in x else x for x in column.tolist()]

#### Format dataframes to be passed to GT() to create presentation-worthy tables ----

def format_demographics(between_person_descriptives, promis_score_type_setting):
    def _format_single_characteristic(data, match, type=["numeric", "categorical"], sort_column="sum", sort_ascending=False, add_header_row=False, header_row_title=""):
        if type == "numeric":
            stat_function = lambda x: format_point_estimate(x["mean"], 2) + " (" + format_point_estimate(x["std"], 2) + ") " + "[" + format_point_estimate(x["min"], 2) + "-" + format_point_estimate(x["max"], 2) + "]"
        else:
            stat_function = lambda x: format_n_and_percent(n = x["sum"], percent = format_proportion_as_percent(x["mean"]))
        characteristic_rows = (
            data
                .query(f"variable.str.contains('{match}')")
                .assign(variable = lambda x: format_covariate_names(x["variable"]))
                .assign(statistic = stat_function)
                .sort_values(sort_column, ascending=sort_ascending)
                .filter(["variable", "count", "statistic"], axis=1)
        )
        if "promis" in match:
            characteristic_rows = characteristic_rows.assign(variable = lambda x: format_promis_names(x["variable"], promis_score_type_setting))
        if add_header_row:
            new_row = pd.DataFrame({"variable":header_row_title}, index=[0])
            return pd.concat([new_row, characteristic_rows]).reset_index(drop=True)
        else:
            return characteristic_rows

    age = _format_single_characteristic(between_person_descriptives, "age_years", "numeric")
    gender = _format_single_characteristic(between_person_descriptives, "gender_", "categorical", "sum", False, True, "Gender")
    race = _format_single_characteristic(between_person_descriptives, "race_", "categorical", "sum", False, True, "Race")
    promis = _format_single_characteristic(between_person_descriptives, "baseline_promis_", "numeric", "variable", True, True, "Baseline PROMIS " + interpret_promis_score_type(promis_score_type_setting))

    demographics = pd.concat([age, gender, race, promis], axis=0).reset_index(drop=True)
    max_n = int(demographics["count"].max())

    return (
        demographics
            .assign(variable = lambda x: bold_labels(x["variable"]))
            .filter(["variable", "statistic"])
            .fillna("")
            .rename(columns={"variable":"Characteristic", "statistic":f"N={str(max_n)}"})
    )

def format_between_person_goodness(between_person_descriptives):
    return (
        between_person_descriptives
            .query("variable == 'how_would_you_describe_today'")
            .drop(["variable", "sum"], axis=1)
            .rename(lambda x: x.capitalize(), axis=1)
            .rename(columns={"Count":"N observations", "Std":"SD"})
            .round(2)
    )

def format_within_person_goodness(within_person_descriptives):
    return (
        within_person_descriptives
            .filter(regex="^id$|^how_would_you_describe_today", axis=1)
            .rename(lambda x: x.replace("how_would_you_describe_today_", "").capitalize(), axis=1)
            .rename({"Id":"Participant ID", "Count":"N observations", "Std":"SD"}, axis=1)
            .sort_values(by="Participant ID", key=lambda x: x.astype(int))
            .reset_index(drop=True)
            .round(2)
    )

def format_icc_results(icc_results):
    return (
        icc_results
            .assign(var_bw = lambda x: x["between_group_variance"].round(4))
            .assign(var_wi = lambda x: x["within_group_variance"].round(4))
            .assign(icc = lambda x: x["icc"].round(4))
            .filter(["n_obs", "n_groups", "var_bw", "var_wi", "icc"])
            .rename(columns={"n_obs":"N observations", "n_groups":"N groups", "var_bw":"Between-group variance", "var_wi":"Within-group variance", "icc":"ICC"})
            .reset_index(drop=True)
    )

def format_between_person_activities(between_person_descriptives):
    return (
        between_person_descriptives
            .pipe(lambda x: x[x["variable"].str.contains("what_did_you_spend_time_doing_today_")])
            .assign(Activity = lambda x: format_activity_names(x["variable"]))
            .assign(n = lambda x: x["sum"].astype(int))
            .assign(mean = lambda x: (x["mean"]*100).round(2))
            .rename(columns={"mean":"%"})
            .filter(["Activity", "n", "%"])
            .sort_values("%", ascending=False)
            .reset_index(drop=True)
    )

def format_within_person_activities(within_person_descriptives):
    return (
        within_person_descriptives
            .melt(id_vars="id")
            .query("(variable.str.contains('what_did_you_spend_time_doing_today_')) & (variable.str.contains('_mean'))")
            .assign(variable = lambda x: format_activity_names(x["variable"].str.replace("_mean", "")))
            .assign(id = lambda x: x["id"].astype("category"))
            .sort_values(["id", "variable"])
            .reset_index()
    )

def format_between_person_association_results(correlations, ttests, promis_score_type_setting):
    def _get_location_to_insert_row(data, column, match):
        return data[data[column].str.contains(match)].iloc[0].name

    def _add_header_row(data, column, match, title):
        row_index = _get_location_to_insert_row(data, column, match)
        new_row = pd.DataFrame({column:title}, index=[0])
        data = pd.concat([data.iloc[:row_index], new_row, data.iloc[row_index:]]).reset_index(drop=True)
        return data

    formatted_correlations = (
        correlations
            .sort_values("variable")
            .reset_index(drop=True)
            .assign(statistic = lambda x: format_correlation_coefficient(x["r"]))
            .assign(p = lambda x: format_p_value(x["p_value"]))
            .assign(q = lambda x: format_p_value(x["q_value"]))
            .assign(variable = lambda x: format_covariate_names(x["variable"]))
            .pipe(_add_header_row, column="variable", match="Baseline promis", title="Baseline PROMIS " + interpret_promis_score_type(promis_score_type_setting))
            .assign(variable = lambda x: format_promis_names(x["variable"], promis_score_type_setting))
            .assign(variable = lambda x: bold_labels(x["variable"]))
            .fillna("")
            .filter(["variable", "statistic", "p", "q"])
    )

    formatted_ttests = (
        ttests
            .assign(means = lambda x: format_ttest_means(x["group0"], x["group0_mean"], x["group0_std"]) + "<br>" + format_ttest_means(x["group1"], x["group1_mean"], x["group1_std"]))
            .assign(statistic = lambda x: format_ttest_statistic(x["t"], x["degrees_of_freedom"]))
            .assign(statistic = lambda x: x["means"] + "<br>" + x["statistic"])
            .assign(p = lambda x: format_p_value(x["p_value"]))
            .assign(q = lambda x: format_p_value(x["q_value"]))
            .assign(variable = lambda x: x["variable"].str.capitalize())
            .assign(variable = lambda x: bold_labels(x["variable"]))
            .filter(["variable", "statistic", "p", "q"])
    )

    insert_row_index = _get_location_to_insert_row(data=formatted_correlations, column="variable", match="PROMIS")
    results = pd.concat([formatted_correlations.iloc[:insert_row_index], formatted_ttests, formatted_correlations.iloc[insert_row_index:]]).reset_index(drop=True)
    results.columns = ["Characteristic", "Test statistic", "p-value", "q-value"]
    return results

def format_mixed_model_results(mixed_model_results, between_person_descriptives):
    return (
        mixed_model_results
            .pipe(lambda x: x[x["variable"] != "Intercept"])
            .merge(between_person_descriptives, on="variable", how="left")
            .assign(n_percent = lambda x: format_n_and_percent(x["sum"], format_proportion_as_percent(x["mean"])))
            .assign(Activity = lambda x: format_activity_names(x["variable"]))
            .assign(Beta = lambda x: format_point_estimate(x["beta"]))
            .assign(SE = lambda x: format_point_estimate(x["stderr"]))
            .assign(conf_int = lambda x: format_confidence_interval(x["ci_low"], x["ci_high"]))
            .assign(conf_int = lambda x: x["conf_int"].str.replace(" ", ""))
            .assign(p_value = lambda x: format_p_value(x["p_value"]))
            .assign(q_value = lambda x: format_p_value(x["q_value"]))
            .sort_values("Activity")
            .filter(["Activity", "n_percent", "n_observations", "n_groups", "Beta", "SE", "conf_int", "p_value", "q_value"])
            .rename(columns={"n_percent":"n (%)", "n_observations":"N obs", "n_groups":"N groups", "conf_int":"95% CI", "p_value":"p-value", "q_value":"q-value"})
            .reset_index(drop=True)
    )

def format_within_person_correlation_results(results):
    pids = results["pid"].unique()
    activities = results["activity"].unique()
    pid_activity_combinations = pd.DataFrame(columns=["pid", "activity"], data=list(itertools.product(*[pids, activities])))
    return (
        results
            .merge(pid_activity_combinations, how="right")
            .assign(pid = lambda x: x["pid"].astype("category"))
            .assign(activity = lambda x: format_activity_names(x["activity"]))
    )

#### Create data visualizations ----

def create_heatmap(
        data, 
        x="", 
        y="", 
        fill="", 
        fill_cmap="RdBu",
        fill_breaks=(-0.99, -0.5, 0, 0.5, 0.99), 
        x_label="", 
        y_label="", 
        fill_label="", 
        title="", 
        fig_size=(7.5,6)
    ):
    return (
        p9.ggplot(data) 
            + p9.geom_tile(p9.aes(y=y, x=x, fill=fill), color="black", size=0.05)
            + p9.scale_x_discrete(expand=(0, 0))
            + p9.scale_y_discrete(expand=(0, 0), limits=data.sort_values(y)[y].unique()[::-1])
            + p9.scale_fill_cmap(cmap_name=fill_cmap, breaks=fill_breaks, na_value="grey")
            + p9.coord_equal() 
            + p9.guides(fill=p9.guide_colorbar(draw_ulim=False, draw_llim=False))
            + p9.labs(x=x_label, y=y_label, fill=fill_label, title=title)
            + p9.theme_light()
            + p9.theme(
                axis_text_x=p9.element_text(size=8),
                axis_text_y=p9.element_text(size=10),
                axis_title=p9.element_text(size=12),
                legend_title=p9.element_text(size=10),
                legend_ticks_length=0,
                legend_position="right",
                figure_size=fig_size
            )
    )
