import io
import math
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import pingouin as pg
import csv
import matplotlib.pyplot as plt
from scipy.stats import rankdata
import statsmodels.api as sm
import copy


"""
Expected columns in CSV:

    If not marked with "missing data is possible with empty string", the data is always present.

Went to bed; yyyy-mm-dd hh:mm:ss -> seconds since 00:00 of day before get up day
Woke up; yyyy-mm-dd hh:mm:ss -> seconds since 00:00 of get up day
Sleep Quality; int%
Time in bed (seconds); int
Time asleep (seconds); int
Asleep after (seconds); int
Regularity; float (from 0.0 to 1.0)
Did snore; false/true
Snore time (seconds); int
Coughing (per hour); float
Steps; int
Weather temperature (°C); int (missing data is possible with empty string)
Weather type; string (missing date possible with empty string; get all unique values in code)
Air Pressure (Pa); float (missing data is possible with empty string)
City; string (missing date possible with empty string; get all unique values in code)
Breathing disruptions (per hour); float
Ambient noise (dB);Ambient light (lux); float
Alertness score; int% (missing data is possible with empty string)
Alertness reaction time (seconds); float (missing data is possible with empty string; always present if alertness_score is present)
Alertness accuracy; int% (missing data is possible with empty string; always present if alertness_score is present)
Movements per hour; float
Wake up window start; yy-mm-dd hh:mm:ss (missing data is possible with empty string -> no wake up time set) -> seconds since 00:00 of get up day
Wake up window stop; yy-mm-dd hh:mm:ss (missing data is possible with empty string; present if wake_up_window_start is present) -> seconds since 00:00 of get up day
Notes; list of strings seperated by ":" (missing data is possible with empty string; get all unique values in code)
Mood; string (missing data is possible with empty string; unique values are "Bad", "OK", "Good") -> 0/1/2
"""


def yyyy_time_to_datetime(string):
    if string == "":
        return None
    return datetime.strptime(string, "%Y-%m-%d %H:%M:%S")


def yy_time_to_datetime(string):
    if string == "":
        return None
    return datetime.strptime(string, "%y-%m-%d %H:%M:%S")


def seconds_since_midnight(datetime):
    if datetime is None:
        return None
    return datetime.hour * 3600 + datetime.minute * 60 + datetime.second


def parse_float(s):
    if s == "" or s == "—":
        return None
    return float(s.replace(",", ".").rstrip("%"))


def find_first_appearance_of_factor(rows, factor):
    for i, row in enumerate(rows):
        if factor in row["Notes"]:
            return i

    for i, row in enumerate(rows):
        if row[factor] not in (None, ""):
            return i

    return None


def effective_sample_size(weights):
    w = np.asarray(weights, dtype=float)
    return (w.sum() ** 2) / (np.sum(w**2))


def shrink_correlation(corr, n_eff, k=1):
    return corr * (n_eff / (n_eff + k))


def convert_column_to_ranks(rows, column):
    values = [row[column] for row in rows]

    mask = [v is not None for v in values]

    ranked = rankdata([v for v, m in zip(values, mask) if m], method="average")

    it = iter(ranked)

    for i, m in enumerate(mask):
        if m:
            value = float(next(it))
            rows[i][column] = value
        else:
            rows[i][column] = None


def weighted_mean(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    return np.sum(values * weights) / np.sum(weights)


def weighted_partial_correlation(rows, factor, result, control_columns):

    # drop rows like how it was done in code previously
    first_appearance_index = find_first_appearance_of_factor(rows, factor)
    if first_appearance_index is None:
        print("Error: factor never appears!")
        return  # factor never appears

    rows_relevant = rows[first_appearance_index:]

    # if factor is a note, keep only rows with at least one note
    if factor in unique_notes:
        rows_relevant = [row for row in rows_relevant if len(row["Notes"]) > 0]

    control_means = {}

    for c in control_columns:
        vals = []
        ws = []
        for r in rows_relevant:
            v = r[c]
            w = r["Weight"]
            if v is not None and w is not None:
                vals.append(v)
                ws.append(w)

        if len(vals) == 0:
            control_means[c] = None
        else:
            control_means[c] = weighted_mean(vals, ws)

    data = []

    for r in rows_relevant:
        x = r[factor]
        y = r[result]
        w = r["Weight"]

        if x is None or y is None or w is None:
            continue

        controls = []
        for c in control_columns:
            v = r[c]
            if v is None:
                v = control_means[c]
            if v is None:
                break  # control has no mean at all → skip row
            controls.append(v)

        if len(controls) != len(control_columns):
            continue

        data.append((x, y, controls, w))

    if len(data) < 3:
        print(f"Error: too little data for factor {factor}")
        return None

    X_var = np.array([d[0] for d in data], dtype=float)
    Y_var = np.array([d[1] for d in data], dtype=float)
    Controls = np.array([d[2] for d in data], dtype=float)
    w = np.array([d[3] for d in data], dtype=float)

    Controls_const = sm.add_constant(Controls)

    # absorb weights via sqrt(w)
    sw = np.sqrt(w)

    Xw = Controls_const * sw[:, None]
    X_var_w = X_var * sw
    Y_var_w = Y_var * sw

    n_obs = len(X_var)
    n_params = Controls_const.shape[1]  # includes intercept

    if n_obs <= n_params:
        print(
            f"Error: not enough degrees of freedom for robust regression "
            f"(n_obs={n_obs}, n_params={n_params}) for factor {factor}"
        )
        return None

    # robust (Huber) regression
    res_x = sm.RLM(X_var_w, Xw, M=sm.robust.norms.HuberT()).fit().resid

    res_y = sm.RLM(Y_var_w, Xw, M=sm.robust.norms.HuberT()).fit().resid

    wsum = np.sum(w)
    mx = np.sum(w * res_x) / wsum
    my = np.sum(w * res_y) / wsum

    num = np.sum(w * (res_x - mx) * (res_y - my))
    den = np.sqrt(np.sum(w * (res_x - mx) ** 2) * np.sum(w * (res_y - my) ** 2))

    if den == 0:
        print(f"Error: denominator is 0 for Factor {factor}")
        return None

    corr = num / den

    n_eff = effective_sample_size(w)
    return float(shrink_correlation(corr, n_eff, k=1))


if __name__ == "__main__":
    # Import CSV data from file
    csv_file_name = "sleepdata.csv"

    column_names = []
    with open(csv_file_name, "r", encoding="utf-8-sig") as csvfile:
        csvreader = csv.reader(csvfile, delimiter=";")
        column_names = next(csvreader)  # get header row

    rows = []

    with open(csv_file_name, "r", encoding="utf-8-sig") as csvfile:
        csvreader = csv.DictReader(csvfile, delimiter=";")

        for row in csvreader:
            rows.append(row)  # each row is a list of strings

    print(f"Total rows (excluding header): {len(rows)}")

    # convert data formats from string to correct types

    for row in rows:

        # all yyyy-mm-dd hh:mm:ss to datetime
        for column in ("Went to bed", "Woke up"):
            row[column] = yyyy_time_to_datetime(row[column])

        # all yy-mm-dd hh:mm:ss to datetime
        for column in ("Wake up window start", "Wake up window stop"):
            row[column] = yy_time_to_datetime(row[column])

    # stores the latest date
    latest_date = rows[-1]["Woke up"]

    for row in rows:
        row["Age (days)"] = (latest_date - row["Woke up"]).days

    HALF_LIFE_DAYS = 365
    LAMBDA = math.log(2) / HALF_LIFE_DAYS

    for row in rows:
        row["Weight"] = math.exp(-LAMBDA * row["Age (days)"])

    for row in rows:

        # add week day column (day before woke up)
        row["Weekday"] = (row["Woke up"] - timedelta(days=1)).strftime("%A")

        # convert all datetime to seconds since midnight of specific day
        for column in ("Went to bed",):
            dt = row[column]
            if dt.date() == (row["Woke up"] - timedelta(days=1)).date():
                row[column] = seconds_since_midnight(dt)
            else:
                row[column] = (
                    seconds_since_midnight(dt) + 86400
                )  # add 24 hours when past midnight (this way no jump at 00:00)

        for column in ("Woke up", "Wake up window start", "Wake up window stop"):
            dt = row[column]
            row[column] = seconds_since_midnight(dt)

        # all int columns
        for column in (
            "Sleep Quality",
            "Time in bed (seconds)",
            "Time asleep (seconds)",
            "Asleep after (seconds)",
            "Snore time (seconds)",
            "Steps",
            "Weather temperature (°C)",
            "Alertness score",
            "Alertness accuracy",
        ):
            if row[column] == "":
                row[column] = None
            else:
                row[column] = int(row[column].rstrip("%"))

        # all float columns
        for column in (
            "Regularity",
            "Coughing (per hour)",
            "Air Pressure (Pa)",
            "Breathing disruptions (per hour)",
            "Ambient noise (dB)",
            "Ambient light (lux)",
            "Alertness reaction time (seconds)",
            "Movements per hour",
        ):
            row[column] = parse_float(row[column])

        # all bool columns
        for column in ("Did snore",):
            if row[column] == "":
                row[column] = None
            else:
                row[column] = 1 if row[column] == "true" else 0

        # mood column
        for column in ("Mood",):
            if row[column] == "":
                row[column] = None
            else:
                mood_mapping = {"Bad": 0, "OK": 1, "Good": 2}
                row[column] = mood_mapping.get(row[column], None)

        # make notes a list of strings
        if row["Notes"] == "":
            row["Notes"] = []
        else:
            row["Notes"] = row["Notes"].split(":")

        # weather and city "" -> None
        for column in ("Weather type", "City"):
            if row[column] == "":
                row[column] = None

        """ maiby because 0.0 could just be for bad sensor?
        if row["Ambient light (lux)"] == 0.0:
            row["Ambient light (lux)"] = None
        """

    print("After basic conversion:")
    print(rows[-1])

    print("\n---------------------------------\n")

    # average sleep quality
    average_sleep_quality = sum(row["Sleep Quality"] for row in rows) / len(rows)

    print(f"Average sleep quality: {average_sleep_quality}%")

    print("\n---------------------------------\n")

    # find all unique weather types
    unique_weather_types = sorted(
        {row["Weather type"] for row in rows if row["Weather type"] != None}
    )
    print("Unique weather types: ", unique_weather_types)

    # find all unique cities
    unique_cities = sorted({row["City"] for row in rows if row["City"] != None})
    print("Unique cities: ", unique_cities)

    # find all unique notes
    unique_notes = sorted({note for row in rows for note in row["Notes"]})

    print("Unique notes: ", unique_notes)

    # caluclate the average sleep quality difference in slitly different ways

    print("\n---------------------------------\n")

    note_quality_differences_nosep_all = {}
    for note in unique_notes:
        with_note = [row["Sleep Quality"] for row in rows if note in row["Notes"]]

        if not with_note:
            continue  # skip if not enough data

        avg_with = sum(with_note) / len(with_note)
        diff = avg_with - average_sleep_quality

        note_quality_differences_nosep_all[note] = diff

    # sort notes by difference
    sorted_notes_naive = sorted(
        note_quality_differences_nosep_all.items(), key=lambda x: x[1], reverse=True
    )
    print("\nSleep quality differences for notes (non separate diff, all data):")
    for note, diff in sorted_notes_naive:
        print(f"Note: {note}, Sleep Quality Difference: {diff}")

    print("\n---------------------------------\n")

    # same as above with the difference that only days are considered aftere the first appearance of the note
    note_quality_differences_nosep_nonall = {}
    for note in unique_notes:

        first_appearance_index = find_first_appearance_of_factor(rows, note)
        if first_appearance_index is None:
            continue  # note never appears

        rows_after_first_appearance = rows[first_appearance_index:]

        with_note = [
            row["Sleep Quality"]
            for row in rows_after_first_appearance
            if note in row["Notes"]
        ]

        if not with_note:
            continue  # skip if not enough data

        avg_with = sum(with_note) / len(with_note)
        diff = avg_with - average_sleep_quality

        note_quality_differences_nosep_nonall[note] = diff

    # sort notes by difference
    sorted_notes_naive = sorted(
        note_quality_differences_nosep_nonall.items(), key=lambda x: x[1], reverse=True
    )
    print(
        "\nSleep quality differences for notes (non separate diff, after first appearance):"
    )
    for note, diff in sorted_notes_naive:
        print(f"Note: {note}, Sleep Quality Difference: {diff}")

    print("\n---------------------------------\n")

    note_quality_differences_sep_all = {}
    for note in unique_notes:
        with_note = [row["Sleep Quality"] for row in rows if note in row["Notes"]]

        without_note = [
            row["Sleep Quality"] for row in rows if note not in row["Notes"]
        ]

        if not with_note or not without_note:
            continue  # skip if not enough data

        avg_with = sum(with_note) / len(with_note)
        avg_without = sum(without_note) / len(without_note)
        diff = avg_with - avg_without

        note_quality_differences_sep_all[note] = diff

    # sort notes by difference
    sorted_notes_naive = sorted(
        note_quality_differences_sep_all.items(), key=lambda x: x[1], reverse=True
    )
    print("\nSleep quality differences for notes (seperate diff, all data):")
    for note, diff in sorted_notes_naive:
        print(f"Note: {note}, Sleep Quality Difference: {diff}")

    print("\n---------------------------------\n")

    # same as above with the difference that only days are considered aftere the first appearance of the note
    note_quality_differences_sep_noall = {}
    for note in unique_notes:

        first_appearance_index = find_first_appearance_of_factor(rows, note)
        if first_appearance_index is None:
            continue  # note never appears

        rows_after_first_appearance = rows[first_appearance_index:]

        with_note = [
            row["Sleep Quality"]
            for row in rows_after_first_appearance
            if note in row["Notes"]
        ]

        without_note = [
            row["Sleep Quality"]
            for row in rows_after_first_appearance
            if note not in row["Notes"]
        ]

        if not with_note or not without_note:
            continue  # skip if not enough data

        avg_with = sum(with_note) / len(with_note)
        avg_without = sum(without_note) / len(without_note)
        diff = avg_with - avg_without

        note_quality_differences_sep_noall[note] = diff

    # sort notes by difference
    sorted_notes_naive = sorted(
        note_quality_differences_sep_noall.items(), key=lambda x: x[1], reverse=True
    )
    print(
        "\nSleep quality differences for notes (separate diff, after first appearance):"
    )
    for note, diff in sorted_notes_naive:
        print(f"Note: {note}, Sleep Quality Difference: {diff}")

    print("\n---------------------------------\n")

    # same as above with the difference than only days are considered that have at least one note
    note_quality_differences_sep_noall_atleastone = {}
    for note in unique_notes:

        first_appearance_index = find_first_appearance_of_factor(rows, note)
        if first_appearance_index is None:
            continue  # note never appears

        rows_after_first_appearance = rows[first_appearance_index:]

        with_note = [
            row["Sleep Quality"]
            for row in rows_after_first_appearance
            if note in row["Notes"]
        ]

        without_note = [
            row["Sleep Quality"]
            for row in rows_after_first_appearance
            if (
                note not in row["Notes"] and len(row["Notes"]) > 0
            )  # if not row["Notes"] == [] is probably faster
        ]

        if not with_note or not without_note:
            continue  # skip if not enough data

        avg_with = sum(with_note) / len(with_note)
        avg_without = sum(without_note) / len(without_note)
        diff = avg_with - avg_without

        note_quality_differences_sep_noall_atleastone[note] = diff

    # sort notes by difference
    sorted_notes_naive = sorted(
        note_quality_differences_sep_noall_atleastone.items(),
        key=lambda x: x[1],
        reverse=True,
    )
    print(
        "\nSleep quality differences for notes (separate diff, after first appearance, at least one note):"
    )
    for note, diff in sorted_notes_naive:
        print(f"Note: {note}, Sleep Quality Difference: {diff}")

    """
    Thoughts about these 4 methods of calculating sleep quality difference for notes:
    - first 4 methods give similar results for most notes
    - seperating is defenetly desirable although it needs more calculations
    - considering only days after first appearance seems like a good idea
    - all of these methods dont give the exact results one can see in the app
        the first three come close, but not exact even with different rounding methods
    - all of theme dont consider that a note could appear more often in specific conditions
        eg. one might use melatonin only when already knowing that sleep quality will be bad
    - if a note appears very rarely, the results can be very misleading
        eg. if only one day has the note and sleep quality was very bad that day, the note will be marked as very bad for sleep quality

    - with at least one note only days are considered where the user at least made the effort to even put in notes. (or also there might be times with premium abo and without).
      this is more robust
    
    In the following code I will use variations of the last code idea"
    """

    print("\n---------------------------------\n")

    # add notes indicator to rows (1 if note is present 0 if not)
    for row in rows:
        for note in unique_notes:
            if note in row["Notes"]:
                row[f"Note {note}"] = 1
            else:
                row[f"Note {note}"] = 0

    # add weather type indicator to rows
    for row in rows:
        for weather in unique_weather_types:
            if row["Weather type"] == weather:
                row[f"Weather type {weather}"] = 1
            else:
                row[f"Weather type {weather}"] = 0

    # add city indicator to rows
    for row in rows:
        for city in unique_cities:
            if row["City"] == city:
                row[f"City {city}"] = 1
            else:
                row[f"City {city}"] = 0

    factor_list = (
        [
            "Went to bed",
            "Time in bed (seconds)",
            "Regularity",
            "Steps",
            "Weather temperature (°C)",
            "Air Pressure (Pa)",
            "Ambient noise (dB)",
            "Ambient light (lux)",
            "Wake up window stop",
        ]
        + [f"Note {note}" for note in unique_notes]
        + [f"Weather type {weather}" for weather in unique_weather_types]
        + [f"City {city}" for city in unique_cities]
    )

    factor_list_app = (
        [f"Note {note}" for note in unique_notes]
        + [f"Weather type {weather}" for weather in unique_weather_types]
        + [f"City {city}" for city in unique_cities]
    )

    result_list = [
        "Sleep Quality",
        "Asleep after (seconds)",
        "Did snore",
        "Snore time (seconds)",
        "Coughing (per hour)",
        "Breathing disruptions (per hour)",
        "Alertness score",
        "Alertness reaction time (seconds)",
        "Alertness accuracy",
        "Movements per hour",
        "Mood",
    ]

    print("\n---------------------------------\n")

    rows_for_spearman = copy.deepcopy(rows)

    for factor in factor_list:
        convert_column_to_ranks(rows_for_spearman, factor)

    for result in result_list:
        convert_column_to_ranks(rows_for_spearman, result)

    correlation_results = {}

    for factor in factor_list_app:
        control_columns = [
            "Went to bed",
            "Time in bed (seconds)",
            "Regularity",
            "Steps",
            "Weather temperature (°C)",
            "Air Pressure (Pa)",
            "Ambient noise (dB)",
            "Ambient light (lux)",
            "Wake up window stop",
        ]  # [f for f in factor_list if f != factor]

        rho = weighted_partial_correlation(
            rows=rows_for_spearman,
            factor=factor,
            result="Sleep Quality",
            control_columns=control_columns,
        )

        correlation_results[factor] = rho

    correlation_results_filtered = {
        k: v for k, v in correlation_results.items() if v is not None
    }

    correlation_results = sorted(
        correlation_results_filtered.items(), key=lambda item: item[1], reverse=True
    )

    print("\nCorrelation Results with Spearman (ranks):")
    for key, value in correlation_results:
        print(f"{key} -> {value}")

    print("\n---------------------------------\n")

    correlation_results = {}

    print("\n\n")

    for factor in factor_list_app:
        control_columns = [
            "Went to bed",
            "Time in bed (seconds)",
            "Regularity",
            "Steps",
            "Weather temperature (°C)",
            "Air Pressure (Pa)",
            "Ambient noise (dB)",
            "Ambient light (lux)",
            "Wake up window stop",
        ]  # [f for f in factor_list if f != factor]

        rho = weighted_partial_correlation(
            rows=rows,
            factor=factor,
            result="Sleep Quality",
            control_columns=control_columns,
        )

        correlation_results[factor] = rho

    print("\n\n")

    correlation_results_filtered = {
        k: v for k, v in correlation_results.items() if v is not None
    }

    correlation_results = sorted(
        correlation_results_filtered.items(), key=lambda item: item[1], reverse=True
    )

    print("\nCorrelation Results with Pearson:")
    for key, value in correlation_results:
        print(f"{key} -> {value}")

    sleep_quality = np.array(
        [row["Sleep Quality"] for row in rows if row["Sleep Quality"] is not None]
    )

    sigma_Y = np.std(sleep_quality, ddof=1)  # sample std deviation

    expected_effects = {}

    for (
        factor,
        r,
    ) in correlation_results:  # correlation_results = list of (factor, Pearson r)
        factor_values = np.array(
            [row[factor] for row in rows if row[factor] is not None]
        )
        p = np.mean(factor_values)
        if p in (0, 1):  # avoid division by zero
            expected_effects[factor] = 0
        else:
            delta = r * sigma_Y / np.sqrt(p * (1 - p))
            expected_effects[factor] = delta

    expected_effects = sorted(
        expected_effects.items(), key=lambda item: item[1], reverse=True
    )

    print("\n---------------------------------\n")

    for key, value in expected_effects:
        print(f"{key} -> {value}")
