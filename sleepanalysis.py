import math
from datetime import datetime, timedelta
import numpy as np
import csv
import statsmodels.api as sm


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

    """
    In actual code for sleepcycle use known unique types
    """
    # find all unique weather types
    unique_weather_types = sorted(
        {row["Weather type"] for row in rows if row["Weather type"] != None}
    )

    # find all unique cities
    unique_cities = sorted({row["City"] for row in rows if row["City"] != None})

    # find all unique notes
    unique_notes = sorted({note for row in rows for note in row["Notes"]})

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

    # =========================
    # ALARM TIME NONPARAMETRIC MODEL
    # =========================

    DAY_SECONDS = 86400
    BANDWIDTH = 45 * 60  # minutes smoothing window (edit from around 30 to 90 minutes)

    # helper: circular distance in seconds
    def circ_dist(a, b):
        d = abs(a - b)
        return min(d, DAY_SECONDS - d)

    # collect data for rows WITH alarm and known sleep quality
    alarm_times = []
    alarm_quality = []
    alarm_weights = []

    for r in rows:
        t = r["Wake up window stop"]
        q = r["Sleep Quality"]
        w = r["Weight"]
        if t is not None and q is not None:
            alarm_times.append(t)
            alarm_quality.append(q)
            alarm_weights.append(w)

    alarm_times = np.array(alarm_times, dtype=float)
    alarm_quality = np.array(alarm_quality, dtype=float)
    alarm_weights = np.array(alarm_weights, dtype=float)

    # kernel prediction function
    def predict_alarm_quality(t_query, exclude_index=None):
        dists = np.array([circ_dist(t_query, t) for t in alarm_times])
        kernel = np.exp(-0.5 * (dists / BANDWIDTH) ** 2)

        w = kernel * alarm_weights

        if exclude_index is not None:
            w[exclude_index] = 0

        if w.sum() == 0:
            return None

        return np.sum(w * alarm_quality) / np.sum(w)

    # mean quality when NO alarm
    no_alarm_vals = [
        r["Sleep Quality"] * r["Weight"]
        for r in rows
        if r["Wake up window stop"] is None and r["Sleep Quality"] is not None
    ]
    no_alarm_w = [
        r["Weight"]
        for r in rows
        if r["Wake up window stop"] is None and r["Sleep Quality"] is not None
    ]

    if len(no_alarm_vals) > 0:
        no_alarm_mean = sum(no_alarm_vals) / sum(no_alarm_w)
    else:
        no_alarm_mean = np.mean(alarm_quality)  # fallback

    # assign columns
    for i, r in enumerate(rows):
        t = r["Wake up window stop"]

        if t is None:
            r["Alarm set"] = 0
            r["Alarm_quality_prediction"] = no_alarm_mean
        else:
            r["Alarm set"] = 1
            r["Alarm_quality_prediction"] = predict_alarm_quality(t)

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

    factor_list_notes = [f"Note {note}" for note in unique_notes]

    factor_list_weather = [
        f"Weather type {weather}" for weather in unique_weather_types
    ]

    factor_list_cities = [f"City {city}" for city in unique_cities]

    # correlation calculation

    control_columns = [
        "Alarm_quality_prediction",
        "Alarm set",
    ]

    correlation_results_notes = {}

    for factor in factor_list_notes:

        rho = weighted_partial_correlation(
            rows=rows,
            factor=factor,
            result="Sleep Quality",
            control_columns=control_columns,
        )

        correlation_results_notes[factor] = rho

    correlation_results_notes = {
        k: v for k, v in correlation_results_notes.items() if v is not None
    }

    correlation_results_weather = {}

    for factor in factor_list_weather:

        rho = weighted_partial_correlation(
            rows=rows,
            factor=factor,
            result="Sleep Quality",
            control_columns=control_columns,
        )

        correlation_results_weather[factor] = rho

    correlation_results_weather = {
        k: v for k, v in correlation_results_weather.items() if v is not None
    }

    correlation_results_cities = {}

    for factor in factor_list_cities:

        rho = weighted_partial_correlation(
            rows=rows,
            factor=factor,
            result="Sleep Quality",
            control_columns=control_columns,
        )

        correlation_results_cities[factor] = rho

    correlation_results_cities = {
        k: v for k, v in correlation_results_cities.items() if v is not None
    }

    sleep_quality = np.array(
        [row["Sleep Quality"] for row in rows if row["Sleep Quality"] is not None]
    )

    sigma_Y = np.std(sleep_quality, ddof=1)  # sample std deviation

    # calculate the expected effects

    expected_effects_notes = {}

    for (
        factor,
        r,
    ) in (
        correlation_results_notes.items()
    ):  # correlation_results = list of (factor, Pearson r)
        factor_values = np.array(
            [row[factor] for row in rows if row[factor] is not None]
        )
        p = np.mean(factor_values)
        if p in (0, 1):  # avoid division by zero
            expected_effects_notes[factor] = 0
        else:
            delta = r * sigma_Y / np.sqrt(p * (1 - p))
            expected_effects_notes[factor] = delta

    expected_effects_notes = sorted(
        expected_effects_notes.items(), key=lambda item: item[1], reverse=True
    )

    print("\nExpected effects of Notes:")

    for key, value in expected_effects_notes:
        print(f"{key} -> {int(value.round()):+} %")

    expected_effects_weather = {}

    for (
        factor,
        r,
    ) in (
        correlation_results_weather.items()
    ):  # correlation_results = list of (factor, Pearson r)
        factor_values = np.array(
            [row[factor] for row in rows if row[factor] is not None]
        )
        p = np.mean(factor_values)
        if p in (0, 1):  # avoid division by zero
            expected_effects_weather[factor] = 0
        else:
            delta = r * sigma_Y / np.sqrt(p * (1 - p))
            expected_effects_weather[factor] = delta

    expected_effects_weather = sorted(
        expected_effects_weather.items(), key=lambda item: item[1], reverse=True
    )

    print("\nExpected effects of Weather types:")

    for key, value in expected_effects_weather:
        print(f"{key} -> {int(value.round()):+} %")

    expected_effects_cities = {}

    for (
        factor,
        r,
    ) in (
        correlation_results_cities.items()
    ):  # correlation_results = list of (factor, Pearson r)
        factor_values = np.array(
            [row[factor] for row in rows if row[factor] is not None]
        )
        p = np.mean(factor_values)
        if p in (0, 1):  # avoid division by zero
            expected_effects_cities[factor] = 0
        else:
            delta = r * sigma_Y / np.sqrt(p * (1 - p))
            expected_effects_cities[factor] = delta

    expected_effects_cities = sorted(
        expected_effects_cities.items(), key=lambda item: item[1], reverse=True
    )

    print("\nExpected effects of Cities:")

    for key, value in expected_effects_cities:
        print(f"{key} -> {int(value.round()):+} %")

    import matplotlib.pyplot as plt

    # grid over 24h
    grid = np.linspace(0, DAY_SECONDS, 300)
    preds = [predict_alarm_quality(t) for t in grid]

    # convert seconds → hours for nicer axis
    grid_hours = grid / 3600

    plt.figure()
    plt.plot(grid_hours, preds)

    # scatter actual weighted points (optional but useful)
    actual_hours = alarm_times / 3600
    plt.scatter(actual_hours, alarm_quality)

    plt.xlabel("Alarm time (hours)")
    plt.ylabel("Expected sleep quality")
    plt.title("Weighted alarm-time effect")
    plt.show()
