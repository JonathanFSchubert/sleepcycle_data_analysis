"""
Parts of this code were written with help from LLMs.
"""

import csv
import math
import os
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import statsmodels.api as sm


# Optional fixed sleep goal parameters (set to None to optimize, or provide a specific value/range)
# For bedtime/alarm: time string like "22:30" (fixed) or "22:00 - 23:00" (range) or seconds (0-86400)
# For time_in_bed: float hours like 8.0 (fixed) or "7.5 - 9.0" (range as string with hours)
FIXED_BEDTIME = None           # e.g., "22:30" or "22:00 - 23:30" or 81000 (seconds) -> None to optimize
FIXED_ALARM_TIME = None        # e.g., "07:00" or "07:00 - 09:00" or 25200 (seconds) -> None to optimize
FIXED_TIME_IN_BED_HOURS = None # e.g., 8.0 or "7.5 - 9.0" -> None to optimize


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(SCRIPT_DIR, "sleepdata.csv")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
FIG_SIZE = 20


BANDWIDTH_DEFAULT = 0.2
BANDWIDTH_PRESSURE = 0.35
BANDWIDTH_ALARM = 45 * 60


# Tunable parameters for reliability and shrinkage
RELIABILITY_DENOM = 1.8  # denominator added to n_eff when computing reliability -> Increase to make reliabilities smaller (reduces trust in local data)
GLOBAL_BIAS_DEFAULT = 0.15  # baseline weight when combining predictions -> Increase to make predictions move closer to the baseline when reliability is low
BASELINE_PERCENTILE = 25    # percentile used as conservative baseline


def yyyy_time_to_datetime(string):
    if string == "":
        return None
    return datetime.strptime(string, "%Y-%m-%d %H:%M:%S")


def yy_time_to_datetime(string):
    if string == "":
        return None
    return datetime.strptime(string, "%y-%m-%d %H:%M:%S")


def seconds_since_midnight(dt):
    if dt is None:
        return None
    return dt.hour * 3600 + dt.minute * 60 + dt.second


def parse_time_to_seconds(time_input):
    """Convert time input (string like '22:30' or int seconds) to seconds since midnight."""
    if time_input is None:
        return None
    if isinstance(time_input, (int, float)):
        return int(time_input)
    if isinstance(time_input, str):
        parts = time_input.split(":")
        if len(parts) == 2:
            hours, minutes = int(parts[0]), int(parts[1])
            return hours * 3600 + minutes * 60
    return None


def parse_time_range(time_input):
    """
    Parse time range input.
    Returns: (min_seconds, max_seconds) if range, or (seconds, seconds) if single value, or (None, None) if None.
    Examples:
    - "09:00" -> (32400, 32400)
    - "09:00 - 10:00" -> (32400, 36000)
    - None -> (None, None)
    """
    if time_input is None:
        return None, None
    
    if isinstance(time_input, (int, float)):
        return int(time_input), int(time_input)
    
    if isinstance(time_input, str):
        # Check if it's a range (contains " - ")
        if " - " in time_input:
            parts = time_input.split(" - ")
            if len(parts) == 2:
                start_sec = parse_time_to_seconds(parts[0].strip())
                end_sec = parse_time_to_seconds(parts[1].strip())
                if start_sec is not None and end_sec is not None:
                    return start_sec, end_sec
        else:
            # Single value
            sec = parse_time_to_seconds(time_input.strip())
            if sec is not None:
                return sec, sec
    
    return None, None


def parse_tib_range(tib_input):
    """
    Parse time-in-bed range input.
    Returns: (min_hours, max_hours) if range, or (hours, hours) if single value, or (None, None) if None.
    Examples:
    - 8.0 -> (8.0, 8.0)
    - "7.5 - 9.0" -> (7.5, 9.0)
    - None -> (None, None)
    """
    if tib_input is None:
        return None, None
    
    if isinstance(tib_input, (int, float)):
        return float(tib_input), float(tib_input)
    
    if isinstance(tib_input, str):
        # Check if it's a range (contains " - ")
        if " - " in tib_input:
            parts = tib_input.split(" - ")
            if len(parts) == 2:
                try:
                    start = float(parts[0].strip())
                    end = float(parts[1].strip())
                    return start, end
                except ValueError:
                    pass
        else:
            # Single value
            try:
                val = float(tib_input.strip())
                return val, val
            except ValueError:
                pass
    
    return None, None


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
    first_appearance_index = find_first_appearance_of_factor(rows, factor)
    if first_appearance_index is None:
        print("Error: factor never appears!")
        return


    rows_relevant = rows
    """
    only uses rows after the user started using this note

    rows_relevant = rows[first_appearance_index:]
    """

    """
    only uses rows where at leat one notes was set. i though maiby it is good to remove rows where the user just didnt bother to use notes...
    not shure if this good.

    if factor.startswith("Note "):
        rows_relevant = [row for row in rows_relevant if len(row["Notes"]) > 0]
    """

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
                break
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
    sw = np.sqrt(w)
    Xw = Controls_const * sw[:, None]
    X_var_w = X_var * sw
    Y_var_w = Y_var * sw

    n_obs = len(X_var)
    n_params = Controls_const.shape[1]
    if n_obs <= n_params:
        print(
            f"Error: not enough degrees of freedom for robust regression "
            f"(n_obs={n_obs}, n_params={n_params}) for factor {factor}"
        )
        return None

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


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_text(filename, content):
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path


def sanitize_filename(name):
    safe = name.strip().lower().replace(" ", "_").replace("/", "_")
    safe = safe.replace("\\", "_").replace(":", "_")
    return safe


def weighted_quantile(values, weights, quantiles):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(mask):
        return np.full(len(quantiles), np.nan)

    vals = values[mask]
    ws = weights[mask]
    order = np.argsort(vals)
    vals = vals[order]
    ws = ws[order]

    cdf = np.cumsum(ws)
    if cdf[-1] == 0:
        return np.full(len(quantiles), np.nan)

    cdf = cdf / cdf[-1]
    return np.interp(quantiles, cdf, vals)


def kernel_predict(x_query, x_vals, y_vals, weights, bandwidth):
    x_vals = np.asarray(x_vals, dtype=float)
    y_vals = np.asarray(y_vals, dtype=float)
    weights = np.asarray(weights, dtype=float)

    mask = np.isfinite(x_vals) & np.isfinite(y_vals) & np.isfinite(weights)
    if not np.any(mask):
        return np.nan

    x_vals = x_vals[mask]
    y_vals = y_vals[mask]
    weights = weights[mask]

    dists = np.abs(x_vals - x_query)
    kernel = np.exp(-0.5 * (dists / bandwidth) ** 2)
    w = kernel * weights

    if w.sum() == 0:
        return np.nan

    return np.sum(w * y_vals) / np.sum(w)


def kernel_smooth_curve(x_grid, x_vals, y_vals, weights, bandwidth):
    return np.array([kernel_predict(x, x_vals, y_vals, weights, bandwidth) for x in x_grid])


def circular_time_distance(a, b, period=86400):
    d = np.abs(a - b)
    return np.minimum(d, period - d)


def kernel_predict_circular(x_query, x_vals, y_vals, weights, bandwidth, period=86400):
    dists = circular_time_distance(x_vals, x_query, period)
    kernel = np.exp(-0.5 * (dists / bandwidth) ** 2)
    w = kernel * weights
    if w.sum() == 0:
        return np.nan
    return np.sum(w * y_vals) / np.sum(w)


def kernel_reliability(x_query, x_vals, weights, bandwidth, circular=False, period=86400):
    x_vals = np.asarray(x_vals, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(x_vals) & np.isfinite(weights) & (weights > 0)
    if not np.any(mask):
        return 0.0

    x_vals = x_vals[mask]
    weights = weights[mask]
    if circular:
        dists = circular_time_distance(x_vals, x_query, period)
    else:
        dists = np.abs(x_vals - x_query)

    kernel = np.exp(-0.5 * (dists / bandwidth) ** 2)
    w = kernel * weights
    if w.sum() == 0:
        return 0.0

    return float(effective_sample_size(w) / (effective_sample_size(w) + RELIABILITY_DENOM))


def kernel_smooth_curve_circular(x_grid, x_vals, y_vals, weights, bandwidth, period=86400):
    return np.array([
        kernel_predict_circular(x, x_vals, y_vals, weights, bandwidth, period)
        for x in x_grid
    ])


def plot_reliability_time_of_day(x, y, weights, xlabel, ylabel, title, bandwidth=None):
    x_arr = np.asarray([xi for xi, yi, w in zip(x, y, weights) if xi is not None and yi is not None and w is not None], dtype=float)
    y_arr = np.asarray([yi for xi, yi, w in zip(x, y, weights) if xi is not None and yi is not None and w is not None], dtype=float)
    w_arr = np.asarray([w for xi, yi, w in zip(x, y, weights) if xi is not None and yi is not None and w is not None], dtype=float)
    if len(x_arr) == 0:
        return None

    x_arr = np.mod(x_arr, 86400)
    if bandwidth is None:
        bandwidth = max(circular_time_bandwidth(x_arr) * 0.25, 1e-6)

    grid = np.linspace(0, 86400, 300, endpoint=False)
    reliabilities = [kernel_reliability(x, x_arr, w_arr, bandwidth, circular=True) for x in grid]
    quality_preds = kernel_smooth_curve_circular(grid, x_arr, y_arr, w_arr, bandwidth)
    # use a conservative baseline (lower quartile) so low-reliability predictions are pulled down
    try:
        pct25 = float(np.nanpercentile(y_arr, BASELINE_PERCENTILE)) if len(y_arr) > 0 else float(np.nanmean(y_arr))
    except Exception:
        pct25 = float(np.nanmean(y_arr))
    try:
        local_min = float(np.nanmin(y_arr)) if len(y_arr) > 0 else pct25
    except Exception:
        local_min = pct25
    baseline = max(0.0, min(pct25, local_min - 1.0))
    # effective prediction pulls each prediction toward the conservative baseline by (1 - reliability)
    # use GLOBAL_BIAS_DEFAULT to control shrinkage strength
    quality_effective = [
        (pred * rel + baseline * GLOBAL_BIAS_DEFAULT) / (rel + GLOBAL_BIAS_DEFAULT)
        for pred, rel in zip(quality_preds, reliabilities)
    ]

    plt.figure(figsize=(FIG_SIZE + 2, FIG_SIZE + 1))
    ax = plt.gca()
    ax.plot(grid, reliabilities, color="tab:green", linewidth=2, label="Local reliability")
    point_sizes = 40 + 120 * (w_arr / max(w_arr.max(), 1e-9))
    ax.scatter(x_arr, np.full_like(x_arr, 0.05), s=point_sizes, alpha=0.4, color="gray", edgecolors="none", label="Data points")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Reliability")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    configure_time_of_day_axis(ax)

    ax2 = ax.twinx()
    ax2.plot(grid, quality_effective, color="tab:orange", linewidth=2, label="Effective quality")
    ax2.set_ylabel(ylabel)
    ax2.set_ylim(min(0, np.nanmin(quality_effective) - 5), max(100, np.nanmax(quality_effective) + 5))

    handles, labels = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles + handles2, labels + labels2, loc="upper right")
    return save_plot(title)


def plot_reliability_line(x, y, weights, xlabel, ylabel, title, bandwidth=None):
    x_arr = np.asarray([xi for xi, yi, w in zip(x, y, weights) if xi is not None and yi is not None and w is not None], dtype=float)
    y_arr = np.asarray([yi for xi, yi, w in zip(x, y, weights) if xi is not None and yi is not None and w is not None], dtype=float)
    w_arr = np.asarray([w for xi, yi, w in zip(x, y, weights) if xi is not None and yi is not None and w is not None], dtype=float)
    if len(x_arr) == 0:
        return None

    if bandwidth is None:
        bandwidth = max(np.std(x_arr) * 0.25, 1e-6)

    grid = np.linspace(x_arr.min(), x_arr.max(), 300)
    reliabilities = [kernel_reliability(x, x_arr, w_arr, bandwidth) for x in grid]
    quality_preds = kernel_smooth_curve(grid, x_arr, y_arr, w_arr, bandwidth)
    # use GLOBAL_BIAS_DEFAULT to control shrinkage strength
    try:
        pct25 = float(np.nanpercentile(y_arr, BASELINE_PERCENTILE)) if len(y_arr) > 0 else float(np.nanmean(y_arr))
    except Exception:
        pct25 = float(np.nanmean(y_arr))
    try:
        local_min = float(np.nanmin(y_arr)) if len(y_arr) > 0 else pct25
    except Exception:
        local_min = pct25
    baseline = max(0.0, min(pct25, local_min - 1.0))
    quality_effective = [
        (pred * rel + baseline * GLOBAL_BIAS_DEFAULT) / (rel + GLOBAL_BIAS_DEFAULT)
        for pred, rel in zip(quality_preds, reliabilities)
    ]

    plt.figure(figsize=(FIG_SIZE + 1, FIG_SIZE))
    ax = plt.gca()
    ax.plot(grid, reliabilities, color="tab:green", linewidth=2, label="Local reliability")
    point_sizes = 40 + 120 * (w_arr / max(w_arr.max(), 1e-9))
    ax.scatter(x_arr, np.full_like(x_arr, 0.05), s=point_sizes, alpha=0.4, color="gray", edgecolors="none", label="Data points")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Reliability")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if "hour" in xlabel.lower():
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.25))

    ax2 = ax.twinx()
    ax2.plot(grid, quality_effective, color="tab:orange", linewidth=2, label="Effective quality")
    ax2.set_ylabel(ylabel)
    ax2.set_ylim(min(0, np.nanmin(quality_effective) - 5), max(100, np.nanmax(quality_effective) + 5))

    handles, labels = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles + handles2, labels + labels2, loc="upper right")
    return save_plot(title)


def circular_time_bandwidth(x_arr, period=86400):
    angles = x_arr / period * 2 * np.pi
    mean_sin = np.mean(np.sin(angles))
    mean_cos = np.mean(np.cos(angles))
    R = np.hypot(mean_sin, mean_cos)
    if R <= 1e-12:
        return period / 4
    circ_std = np.sqrt(-2 * np.log(R)) * period / (2 * np.pi)
    return circ_std


def save_plot(title):
    filename = sanitize_filename(title)
    path = os.path.join(OUTPUT_DIR, f"{filename}.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return path


def format_seconds_to_24h_label(seconds):
    seconds = int(seconds) % 86400
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    return f"{hours:02d}:{minutes:02d}"


def seconds_to_hours(values):
    return [v / 3600.0 if v is not None else None for v in values]


def seconds_to_minutes(values):
    return [v / 60.0 if v is not None else None for v in values]


def configure_time_of_day_axis(ax):
    ax.xaxis.set_major_locator(ticker.MultipleLocator(3600))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1800))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda val, pos: format_seconds_to_24h_label(val)))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")


def plot_weighted_scatter_time_of_day(x, y, weights, xlabel, ylabel, title, invert_x=False, bandwidth=None):
    x_arr = np.asarray([xi for xi, yi, w in zip(x, y, weights) if xi is not None and yi is not None and w is not None], dtype=float)
    y_arr = np.asarray([yi for xi, yi, w in zip(x, y, weights) if xi is not None and yi is not None and w is not None], dtype=float)
    w_arr = np.asarray([w for xi, yi, w in zip(x, y, weights) if xi is not None and yi is not None and w is not None], dtype=float)

    if len(x_arr) == 0:
        return None

    x_arr = np.mod(x_arr, 86400)

    if bandwidth is None:
        bandwidth = max(circular_time_bandwidth(x_arr) * 0.25, 1e-6)

    grid = np.linspace(0, 86400, 300, endpoint=False)
    preds = kernel_smooth_curve_circular(grid, x_arr, y_arr, w_arr, bandwidth)

    plt.figure(figsize=(FIG_SIZE + 2, FIG_SIZE + 1))
    sizes = 70 * (w_arr / max(w_arr.max(), 1e-9))
    plt.scatter(x_arr, y_arr, s=sizes, alpha=0.6, color="tab:blue", edgecolors="none")
    plt.plot(grid, preds, color="red", linewidth=2, label="Weighted kernel smooth")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    ax = plt.gca()
    if invert_x:
        ax.invert_xaxis()
    configure_time_of_day_axis(ax)
    plt.legend()
    return save_plot(title)


def plot_weighted_scatter(x, y, weights, xlabel, ylabel, title, invert_x=False, bandwidth=None):
    x_arr = np.asarray([xi for xi, yi, w in zip(x, y, weights) if xi is not None and yi is not None and w is not None], dtype=float)
    y_arr = np.asarray([yi for xi, yi, w in zip(x, y, weights) if xi is not None and yi is not None and w is not None], dtype=float)
    w_arr = np.asarray([w for xi, yi, w in zip(x, y, weights) if xi is not None and yi is not None and w is not None], dtype=float)

    if len(x_arr) == 0:
        return None

    if bandwidth is None:
        bandwidth = max(np.std(x_arr) * 0.25, 1e-6)

    grid = np.linspace(x_arr.min(), x_arr.max(), 300)
    preds = kernel_smooth_curve(grid, x_arr, y_arr, w_arr, bandwidth)

    plt.figure(figsize=(FIG_SIZE + 1, FIG_SIZE))
    sizes = 70 * (w_arr / max(w_arr.max(), 1e-9))
    plt.scatter(x_arr, y_arr, s=sizes, alpha=0.6, color="tab:blue", edgecolors="none")
    plt.plot(grid, preds, color="red", linewidth=2, label="Weighted kernel smooth")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    ax = plt.gca()
    if "hour" in xlabel.lower():
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.25))
    if invert_x:
        ax.invert_xaxis()
    plt.legend()
    return save_plot(title)


def plot_weighted_boxplot(group_values, group_weights, labels, ylabel, title, invert_x=False):
    fig, ax = plt.subplots(figsize=(FIG_SIZE, FIG_SIZE))
    stats = []
    positions = list(range(1, len(labels) + 1))

    for values, weights in zip(group_values, group_weights):
        values = np.asarray(values, dtype=float)
        weights = np.asarray(weights, dtype=float)
        mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)

        if not np.any(mask):
            stats.append({
                "med": np.nan,
                "q1": np.nan,
                "q3": np.nan,
                "whislo": np.nan,
                "whishi": np.nan,
                "fliers": [],
            })
            continue

        vals = values[mask]
        ws = weights[mask]
        q1, med, q3 = weighted_quantile(vals, ws, [0.25, 0.5, 0.75])
        iqr = q3 - q1
        low_limit = q1 - 1.5 * iqr
        high_limit = q3 + 1.5 * iqr

        within_mask = (vals >= low_limit) & (vals <= high_limit)
        valid_vals = vals[within_mask]
        valid_ws = ws[within_mask]

        if len(valid_vals) == 0:
            whislo = np.min(vals)
            whishi = np.max(vals)
        else:
            # Whisker endpoints remain the most extreme non-outlier values,
            # but the fence itself is defined by weighted quartiles and IQR.
            whislo = np.min(valid_vals)
            whishi = np.max(valid_vals)

        fliers = vals[(vals < low_limit) | (vals > high_limit)].tolist()

        stats.append({
            "med": med,
            "q1": q1,
            "q3": q3,
            "whislo": whislo,
            "whishi": whishi,
            "fliers": fliers,
        })

    ax.bxp(stats, positions=positions, showfliers=True, widths=0.6, patch_artist=True)

    for i, (values, weights) in enumerate(zip(group_values, group_weights), start=1):
        values = np.asarray(values, dtype=float)
        weights = np.asarray(weights, dtype=float)
        mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
        if not np.any(mask):
            continue

        x_jitter = np.random.normal(i, 0.08, size=mask.sum())
        sizes = 20 + 120 * (weights[mask] / max(weights[mask].max(), 1e-9))
        ax.scatter(x_jitter, values[mask], s=sizes, alpha=0.5, color="gray", edgecolors="none")

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if invert_x:
        ax.invert_xaxis()

    return save_plot(title)


def plot_alarm_time(rows, bandwidth=BANDWIDTH_ALARM):
    alarm_times = [
        r["Wake up window stop"]
        for r in rows
        if r["Wake up window stop"] is not None
        and r["Sleep Quality"] is not None
        and r["Weight"] is not None
    ]
    alarm_quality = [
        r["Sleep Quality"]
        for r in rows
        if r["Wake up window stop"] is not None
        and r["Sleep Quality"] is not None
        and r["Weight"] is not None
    ]
    alarm_weights = [
        r["Weight"]
        for r in rows
        if r["Wake up window stop"] is not None
        and r["Sleep Quality"] is not None
        and r["Weight"] is not None
    ]

    if len(alarm_times) == 0:
        return None

    return plot_weighted_scatter_time_of_day(
        alarm_times,
        alarm_quality,
        alarm_weights,
        "Alarm time",
        "Expected sleep quality",
        "Alarm time vs Sleep quality",
        bandwidth=bandwidth,
    )


def compute_best_sleep_goal(rows, time_in_bed_min_hours=5.0, time_in_bed_max_hours=12.0, step_minutes=5, fixed_bedtime=None, fixed_alarm_time=None, fixed_time_in_bed_hours=None):
    """
    Compute best sleep goal by optimizing over bedtime, alarm time, and time in bed.
    
    Parameters:
    - rows: list of sleep data rows
    - time_in_bed_min_hours: minimum time in bed to consider (hours)
    - time_in_bed_max_hours: maximum time in bed to consider (hours)
    - step_minutes: step size for search grid (minutes)
    - fixed_bedtime: if provided (in seconds), use this bedtime and optimize the other two parameters
    - fixed_alarm_time: if provided (in seconds), use this alarm time and optimize the other two parameters
    - fixed_time_in_bed_hours: if provided (in hours), use this time in bed and optimize the other two parameters
    
    Returns: dict with alarm_time, bedtime, time_in_bed_hours, predicted_quality
    """
    # Prepare predictors (x values, y quality, weights)
    bed_x = np.array([r["Went to bed"] for r in rows if r["Went to bed"] is not None and r["Sleep Quality"] is not None and r["Weight"] is not None], dtype=float)
    bed_y = np.array([r["Sleep Quality"] for r in rows if r["Went to bed"] is not None and r["Sleep Quality"] is not None and r["Weight"] is not None], dtype=float)
    bed_w = np.array([r["Weight"] for r in rows if r["Went to bed"] is not None and r["Sleep Quality"] is not None and r["Weight"] is not None], dtype=float)

    alarm_x = np.array([r["Wake up window stop"] for r in rows if r["Wake up window stop"] is not None and r["Sleep Quality"] is not None and r["Weight"] is not None], dtype=float)
    alarm_y = np.array([r["Sleep Quality"] for r in rows if r["Wake up window stop"] is not None and r["Sleep Quality"] is not None and r["Weight"] is not None], dtype=float)
    alarm_w = np.array([r["Weight"] for r in rows if r["Wake up window stop"] is not None and r["Sleep Quality"] is not None and r["Weight"] is not None], dtype=float)

    tib_x = np.array([r["Time in bed (seconds)"] / 3600.0 for r in rows if r["Time in bed (seconds)"] is not None and r["Sleep Quality"] is not None and r["Weight"] is not None], dtype=float)
    tib_y = np.array([r["Sleep Quality"] for r in rows if r["Time in bed (seconds)"] is not None and r["Sleep Quality"] is not None and r["Weight"] is not None], dtype=float)
    tib_w = np.array([r["Weight"] for r in rows if r["Time in bed (seconds)"] is not None and r["Sleep Quality"] is not None and r["Weight"] is not None], dtype=float)

    # Fallback means
    global_mean = np.mean([r["Sleep Quality"] for r in rows if r["Sleep Quality"] is not None]) if any(r["Sleep Quality"] is not None for r in rows) else 0
    try:
        global_baseline = float(np.nanpercentile([r["Sleep Quality"] for r in rows if r["Sleep Quality"] is not None], BASELINE_PERCENTILE))
    except Exception:
        global_baseline = global_mean

    # Bandwidths
    if len(bed_x) > 0:
        bed_bw = max(circular_time_bandwidth(bed_x) * 0.25, 1e-6)
    else:
        bed_bw = BANDWIDTH_ALARM
    if len(alarm_x) > 0:
        alarm_bw = BANDWIDTH_ALARM
    else:
        alarm_bw = BANDWIDTH_ALARM
    if len(tib_x) > 0:
        tib_bw = max(np.std(tib_x) * 0.25, 1e-6)
    else:
        tib_bw = 0.5

    def local_effective_sample_size(weights):
        mask = np.isfinite(weights) & (weights > 0)
        if not np.any(mask):
            return 0.0
        w = np.asarray(weights, dtype=float)[mask]
        return float((w.sum() ** 2) / np.sum(w ** 2))

    def predict_with_reliability(query, x_vals, y_vals, weights, bandwidth, circular=False):
        if len(x_vals) == 0:
            return np.nan, 0.0

        x_arr = np.asarray(x_vals, dtype=float)
        y_arr = np.asarray(y_vals, dtype=float)
        w_arr = np.asarray(weights, dtype=float)

        if circular:
            dists = circular_time_distance(x_arr, query)
        else:
            dists = np.abs(x_arr - query)

        kernel = np.exp(-0.5 * (dists / bandwidth) ** 2)
        w = kernel * w_arr
        if w.sum() == 0:
            return np.nan, 0.0

        pred = np.sum(w * y_arr) / np.sum(w)
        n_eff = local_effective_sample_size(w)
        reliability = n_eff / (n_eff + RELIABILITY_DENOM)
        return float(pred), float(reliability)

    step_sec = int(step_minutes * 60)
    alarm_candidates = np.arange(0, 86400, step_sec, dtype=int)
    tib_step = step_minutes / 60.0
    tib_candidates = np.arange(time_in_bed_min_hours, time_in_bed_max_hours + tib_step / 2.0, tib_step)

    # Parse parameter ranges
    bedtime_min, bedtime_max = parse_time_range(fixed_bedtime)
    alarm_min, alarm_max = parse_time_range(fixed_alarm_time)
    tib_min_fixed, tib_max_fixed = parse_tib_range(fixed_time_in_bed_hours)

    # Apply range constraints to candidates
    if alarm_min is not None and alarm_max is not None:
        alarm_candidates = alarm_candidates[(alarm_candidates >= alarm_min) & (alarm_candidates <= alarm_max)]
    
    if tib_min_fixed is not None and tib_max_fixed is not None:
        tib_candidates = tib_candidates[(tib_candidates >= tib_min_fixed) & (tib_candidates <= tib_max_fixed)]

    best = None
    best_score = -1e9

    # Optimize over free parameters with constraints applied to candidates
    for tib in tib_candidates:
        for at in alarm_candidates:
            bedtime = (at - int(tib * 3600)) % 86400
            
            # Check bedtime constraint if specified
            if bedtime_min is not None and bedtime_max is not None:
                if bedtime_min <= bedtime_max:
                    # Normal range (e.g., 22:00 - 23:00)
                    if not (bedtime_min <= bedtime <= bedtime_max):
                        continue
                else:
                    # Wraparound range (e.g., 23:00 - 02:00 crosses midnight)
                    if not (bedtime >= bedtime_min or bedtime <= bedtime_max):
                        continue

            q_bed, rel_bed = predict_with_reliability(bedtime, bed_x, bed_y, bed_w, bed_bw, circular=True)
            q_alarm, rel_alarm = predict_with_reliability(at, alarm_x, alarm_y, alarm_w, alarm_bw, circular=True)
            q_tib, rel_tib = predict_with_reliability(tib, tib_x, tib_y, tib_w, tib_bw)

            eff_bed = (q_bed * rel_bed + global_baseline * GLOBAL_BIAS_DEFAULT) / (rel_bed + GLOBAL_BIAS_DEFAULT) if not np.isnan(q_bed) else global_baseline
            eff_alarm = (q_alarm * rel_alarm + global_baseline * GLOBAL_BIAS_DEFAULT) / (rel_alarm + GLOBAL_BIAS_DEFAULT) if not np.isnan(q_alarm) else global_baseline
            eff_tib = (q_tib * rel_tib + global_baseline * GLOBAL_BIAS_DEFAULT) / (rel_tib + GLOBAL_BIAS_DEFAULT) if not np.isnan(q_tib) else global_baseline

            total_reliability = rel_bed + rel_alarm + rel_tib
            total_weight = total_reliability + GLOBAL_BIAS_DEFAULT
            score = (eff_bed * rel_bed + eff_alarm * rel_alarm + eff_tib * rel_tib + global_baseline * GLOBAL_BIAS_DEFAULT) / total_weight

            if score > best_score:
                best_score = score
                best = {"alarm_time": int(at), "bedtime": int(bedtime), "time_in_bed_hours": float(tib), "predicted_quality": float(score)}

    return best


def load_rows():
    rows = []
    with open(CSV_FILE, "r", encoding="utf-8-sig") as csvfile:
        reader = csv.DictReader(csvfile, delimiter=";")
        for row in reader:
            rows.append(row)

    for row in rows:
        row["Went to bed"] = yyyy_time_to_datetime(row.get("Went to bed", ""))
        row["Woke up"] = yyyy_time_to_datetime(row.get("Woke up", ""))
        row["Wake up window start"] = yy_time_to_datetime(row.get("Wake up window start", ""))
        row["Wake up window stop"] = yy_time_to_datetime(row.get("Wake up window stop", ""))

        row["Sleep Quality"] = int(row["Sleep Quality"].rstrip("%")) if row.get("Sleep Quality") else None
        row["Time in bed (seconds)"] = int(row["Time in bed (seconds)"]) if row.get("Time in bed (seconds)") else None
        row["Time asleep (seconds)"] = int(row["Time asleep (seconds)"]) if row.get("Time asleep (seconds)") else None
        row["Asleep after (seconds)"] = int(row["Asleep after (seconds)"]) if row.get("Asleep after (seconds)") else None
        row["Snore time (seconds)"] = int(row["Snore time (seconds)"]) if row.get("Snore time (seconds)") else None
        row["Steps"] = int(row["Steps"]) if row.get("Steps") else None
        row["Weather temperature (°C)"] = int(row["Weather temperature (°C)"]) if row.get("Weather temperature (°C)") else None
        row["Alertness score"] = int(row["Alertness score"].rstrip("%")) if row.get("Alertness score") else None
        row["Alertness accuracy"] = int(row["Alertness accuracy"].rstrip("%")) if row.get("Alertness accuracy") else None

        row["Regularity"] = parse_float(row.get("Regularity", ""))
        row["Coughing (per hour)"] = parse_float(row.get("Coughing (per hour)", ""))
        row["Air Pressure (Pa)"] = parse_float(row.get("Air Pressure (Pa)", ""))
        row["Breathing disruptions (per hour)"] = parse_float(row.get("Breathing disruptions (per hour)", ""))
        row["Ambient noise (dB)"] = parse_float(row.get("Ambient noise (dB)", ""))
        row["Ambient light (lux)"] = parse_float(row.get("Ambient light (lux)", ""))
        row["Alertness reaction time (seconds)"] = parse_float(row.get("Alertness reaction time (seconds)", ""))
        row["Movements per hour"] = parse_float(row.get("Movements per hour", ""))

        did_snore = row.get("Did snore", "")
        row["Did snore"] = None if did_snore == "" else (1 if did_snore == "true" else 0)

        notes_value = row.get("Notes", "")
        row["Notes"] = [] if notes_value == "" else notes_value.split(":")

        for field in ("Weather type", "City"):
            if row.get(field, "") == "":
                row[field] = None

        if "Mood" in row:
            mood_text = row.get("Mood", "")
            if mood_text == "":
                row["Mood"] = None
            else:
                mood_mapping = {"Bad": 0, "OK": 1, "Good": 2}
                row["Mood"] = mood_mapping.get(mood_text, None)

    for index, row in enumerate(rows):
        row["Prev Sleep Quality"] = rows[index - 1]["Sleep Quality"] if index > 0 else None

    valid_woke = [row["Woke up"] for row in rows if row["Woke up"] is not None]
    latest_date = max(valid_woke) if valid_woke else None

    for row in rows:
        if latest_date is not None and row["Woke up"] is not None:
            row["Age (days)"] = (latest_date - row["Woke up"]).days
        else:
            row["Age (days)"] = None

    HALF_LIFE_DAYS = 365
    LAMBDA = math.log(2) / HALF_LIFE_DAYS

    for row in rows:
        if row["Age (days)"] is None:
            row["Weight"] = None
        else:
            row["Weight"] = math.exp(-LAMBDA * row["Age (days)"])

    for row in rows:
        if row["Woke up"] is not None:
            row["Weekday"] = (row["Woke up"] - timedelta(days=1)).strftime("%A")
        else:
            row["Weekday"] = None

        if row["Went to bed"] is not None and row["Woke up"] is not None:
            if row["Went to bed"].date() == (row["Woke up"] - timedelta(days=1)).date():
                row["Went to bed"] = seconds_since_midnight(row["Went to bed"])
            else:
                row["Went to bed"] = seconds_since_midnight(row["Went to bed"]) + 86400
        else:
            row["Went to bed"] = None

        for field in ("Woke up", "Wake up window start", "Wake up window stop"):
            row[field] = seconds_since_midnight(row[field])

        row["Sleep drug"] = 1 if "Sleep drug" in row["Notes"] else 0
        row["Coffee"] = 1 if "Coffee" in row["Notes"] else 0
        row["Tea"] = 1 if "Tea" in row["Notes"] else 0

    global unique_weather_types, unique_cities, unique_notes
    unique_weather_types = sorted({row["Weather type"] for row in rows if row["Weather type"] is not None})
    unique_cities = sorted({row["City"] for row in rows if row["City"] is not None})
    unique_notes = sorted({note for row in rows for note in row["Notes"]})

    for row in rows:
        for note in unique_notes:
            row[f"Note {note}"] = 1 if note in row["Notes"] else 0
        for weather in unique_weather_types:
            row[f"Weather type {weather}"] = 1 if row["Weather type"] == weather else 0
        for city in unique_cities:
            row[f"City {city}"] = 1 if row["City"] == city else 0

    return rows


def plot_all(rows):
    plots = []

    drug_yes = [row["Asleep after (seconds)"] / 60 for row in rows if row["Sleep drug"] == 1 and row["Asleep after (seconds)"] is not None]
    drug_yes_weights = [row["Weight"] for row in rows if row["Sleep drug"] == 1 and row["Asleep after (seconds)"] is not None and row["Weight"] is not None]
    drug_no = [row["Asleep after (seconds)"] / 60 for row in rows if row["Sleep drug"] == 0 and row["Asleep after (seconds)"] is not None]
    drug_no_weights = [row["Weight"] for row in rows if row["Sleep drug"] == 0 and row["Asleep after (seconds)"] is not None and row["Weight"] is not None]
    plots.append(plot_weighted_boxplot([drug_no, drug_yes], [drug_no_weights, drug_yes_weights], ["No sleep drug", "Sleep drug"], "Asleep after (minutes)", "Sleep drug vs time to fall asleep"))

    plots.append(plot_weighted_scatter([row["Sleep Quality"] for row in rows], [row["Alertness score"] for row in rows], [row["Weight"] for row in rows], "Sleep quality", "Alertness score", "Alertness vs Sleep quality"))

    plots.append(plot_weighted_scatter(seconds_to_hours([row["Time in bed (seconds)"] for row in rows]), [row["Alertness score"] for row in rows], [row["Weight"] for row in rows], "Time in bed (hours)", "Alertness score", "Alertness vs Time in bed"))

    plots.append(plot_weighted_scatter(seconds_to_hours([row["Time asleep (seconds)"] for row in rows]), [row["Alertness score"] for row in rows], [row["Weight"] for row in rows], "Time asleep (hours)", "Alertness score", "Alertness vs Time asleep"))

    coffee_yes = [row["Asleep after (seconds)"] / 60 for row in rows if row["Coffee"] == 1 and row["Asleep after (seconds)"] is not None]
    coffee_yes_weights = [row["Weight"] for row in rows if row["Coffee"] == 1 and row["Asleep after (seconds)"] is not None and row["Weight"] is not None]
    coffee_no = [row["Asleep after (seconds)"] / 60 for row in rows if row["Coffee"] == 0 and row["Asleep after (seconds)"] is not None]
    coffee_no_weights = [row["Weight"] for row in rows if row["Coffee"] == 0 and row["Asleep after (seconds)"] is not None and row["Weight"] is not None]
    plots.append(plot_weighted_boxplot([coffee_no, coffee_yes], [coffee_no_weights, coffee_yes_weights], ["No coffee", "Coffee"], "Asleep after (minutes)", "Coffee vs time to fall asleep"))

    tea_yes = [row["Asleep after (seconds)"] / 60 for row in rows if row["Tea"] == 1 and row["Asleep after (seconds)"] is not None]
    tea_yes_weights = [row["Weight"] for row in rows if row["Tea"] == 1 and row["Asleep after (seconds)"] is not None and row["Weight"] is not None]
    tea_no = [row["Asleep after (seconds)"] / 60 for row in rows if row["Tea"] == 0 and row["Asleep after (seconds)"] is not None]
    tea_no_weights = [row["Weight"] for row in rows if row["Tea"] == 0 and row["Asleep after (seconds)"] is not None and row["Weight"] is not None]
    plots.append(plot_weighted_boxplot([tea_no, tea_yes], [tea_no_weights, tea_yes_weights], ["No tea", "Tea"], "Asleep after (minutes)", "Tea vs time to fall asleep"))

    plots.append(plot_weighted_scatter([row["Prev Sleep Quality"] for row in rows], [row["Sleep Quality"] for row in rows], [row["Weight"] for row in rows], "Sleep quality previous day", "Sleep quality", "Sleep inertia / carryover effect", invert_x=True))

    plots.append(plot_weighted_scatter([row["Regularity"] for row in rows], [row["Sleep Quality"] for row in rows], [row["Weight"] for row in rows], "Regularity", "Sleep quality", "Sleep regularity vs Sleep quality"))

    plots.append(plot_weighted_scatter(seconds_to_hours([row["Time in bed (seconds)"] for row in rows]), [row["Sleep Quality"] for row in rows], [row["Weight"] for row in rows], "Time in bed (hours)", "Sleep quality", "Time in bed vs Sleep quality"))
    plots.append(plot_reliability_line(seconds_to_hours([row["Time in bed (seconds)"] for row in rows]), [row["Sleep Quality"] for row in rows], [row["Weight"] for row in rows], "Time in bed (hours)", "Sleep quality", "Time in bed vs Reliability"))

    plots.append(plot_weighted_scatter(seconds_to_hours([row["Time asleep (seconds)"] for row in rows]), [row["Sleep Quality"] for row in rows], [row["Weight"] for row in rows], "Time asleep (hours)", "Sleep quality", "Time asleep vs Sleep quality"))

    plots.append(plot_weighted_scatter_time_of_day([row["Went to bed"] for row in rows], [row["Sleep Quality"] for row in rows], [row["Weight"] for row in rows], "Bedtime", "Sleep quality", "Bedtime vs Sleep quality"))
    plots.append(plot_reliability_time_of_day([row["Went to bed"] for row in rows], [row["Sleep Quality"] for row in rows], [row["Weight"] for row in rows], "Bedtime", "Sleep quality", "Bedtime vs Reliability"))

    plots.append(plot_weighted_scatter_time_of_day([row["Went to bed"] for row in rows], seconds_to_minutes([row["Asleep after (seconds)" ] for row in rows]), [row["Weight"] for row in rows], "Bedtime", "Time to fall asleep (minutes)", "Bedtime vs Time to fall asleep"))

    plots.append(plot_alarm_time(rows))
    plots.append(plot_reliability_time_of_day([row["Wake up window stop"] for row in rows], [row["Sleep Quality"] for row in rows], [row["Weight"] for row in rows], "Alarm time", "Sleep quality", "Alarm time vs Reliability"))

    pressure = [row["Air Pressure (Pa)"] for row in rows if row["Air Pressure (Pa)"] is not None and row["Sleep Quality"] is not None and row["Weight"] is not None]
    quality = [row["Sleep Quality"] for row in rows if row["Air Pressure (Pa)"] is not None and row["Sleep Quality"] is not None and row["Weight"] is not None]
    pressure_weights = [row["Weight"] for row in rows if row["Air Pressure (Pa)"] is not None and row["Sleep Quality"] is not None and row["Weight"] is not None]
    plots.append(plot_weighted_scatter(pressure, quality, pressure_weights, "Air pressure (Pa)", "Sleep quality", "Air pressure vs Sleep quality", bandwidth=BANDWIDTH_PRESSURE))

    mood_bad = [row["Sleep Quality"] for row in rows if row.get("Mood") == 0 and row["Sleep Quality"] is not None]
    mood_bad_weights = [row["Weight"] for row in rows if row.get("Mood") == 0 and row["Sleep Quality"] is not None and row["Weight"] is not None]
    mood_ok = [row["Sleep Quality"] for row in rows if row.get("Mood") == 1 and row["Sleep Quality"] is not None]
    mood_ok_weights = [row["Weight"] for row in rows if row.get("Mood") == 1 and row["Sleep Quality"] is not None and row["Weight"] is not None]
    mood_good = [row["Sleep Quality"] for row in rows if row.get("Mood") == 2 and row["Sleep Quality"] is not None]
    mood_good_weights = [row["Weight"] for row in rows if row.get("Mood") == 2 and row["Sleep Quality"] is not None and row["Weight"] is not None]
    plots.append(plot_weighted_boxplot([mood_bad, mood_ok, mood_good], [mood_bad_weights, mood_ok_weights, mood_good_weights], ["Bad", "OK", "Good"], "Sleep quality", "Mood vs Sleep quality"))

    return plots


def run_analysis_prints(rows):
    DAY_SECONDS = 86400
    BANDWIDTH = 45 * 60

    # Build alarm prediction columns for use in partial correlation controls
    def circ_dist(a, b):
        d = abs(a - b)
        return min(d, DAY_SECONDS - d)

    alarm_times = []
    alarm_quality = []
    alarm_weights = []
    for r in rows:
        t = r["Wake up window stop"]
        q = r["Sleep Quality"]
        w = r["Weight"]
        if t is not None and q is not None and w is not None:
            alarm_times.append(t)
            alarm_quality.append(q)
            alarm_weights.append(w)

    alarm_times = np.array(alarm_times, dtype=float)
    alarm_quality = np.array(alarm_quality, dtype=float)
    alarm_weights = np.array(alarm_weights, dtype=float)

    def predict_alarm_quality(t_query, exclude_index=None):
        dists = np.array([circ_dist(t_query, t) for t in alarm_times])
        kernel = np.exp(-0.5 * (dists / BANDWIDTH) ** 2)
        w = kernel * alarm_weights
        if exclude_index is not None:
            w[exclude_index] = 0
        if w.sum() == 0:
            return None
        return np.sum(w * alarm_quality) / np.sum(w)

    no_alarm_vals = [
        r["Sleep Quality"] * r["Weight"]
        for r in rows
        if r["Wake up window stop"] is None and r["Sleep Quality"] is not None and r["Weight"] is not None
    ]
    no_alarm_w = [
        r["Weight"]
        for r in rows
        if r["Wake up window stop"] is None and r["Sleep Quality"] is not None and r["Weight"] is not None
    ]
    if len(no_alarm_vals) > 0:
        no_alarm_mean = sum(no_alarm_vals) / sum(no_alarm_w)
    else:
        no_alarm_mean = np.mean(alarm_quality) if len(alarm_quality) > 0 else 0

    for i, r in enumerate(rows):
        t = r["Wake up window stop"]
        if t is None:
            r["Alarm set"] = 0
            r["Alarm_quality_prediction"] = no_alarm_mean
        else:
            r["Alarm set"] = 1
            r["Alarm_quality_prediction"] = predict_alarm_quality(t)

    factor_list = [
        "Went to bed",
        "Time in bed (seconds)",
        "Regularity",
        "Regularity",
        "Steps",
        "Weather temperature (°C)",
        "Air Pressure (Pa)",
        "Ambient noise (dB)",
        "Ambient light (lux)",
        "Wake up window stop",
    ]
    factor_list += [f"Note {note}" for note in unique_notes]
    factor_list += [f"Weather type {weather}" for weather in unique_weather_types]
    factor_list += [f"City {city}" for city in unique_cities]

    factor_list_notes = [f"Note {note}" for note in unique_notes]
    factor_list_weather = [f"Weather type {weather}" for weather in unique_weather_types]
    factor_list_cities = [f"City {city}" for city in unique_cities]

    control_columns = ["Alarm_quality_prediction", "Alarm set"]

    correlation_results_notes = {}
    for factor in factor_list_notes:
        rho = weighted_partial_correlation(rows=rows, factor=factor, result="Sleep Quality", control_columns=control_columns)
        correlation_results_notes[factor] = rho
    correlation_results_notes = {k: v for k, v in correlation_results_notes.items() if v is not None}

    correlation_results_weather = {}
    for factor in factor_list_weather:
        rho = weighted_partial_correlation(rows=rows, factor=factor, result="Sleep Quality", control_columns=control_columns)
        correlation_results_weather[factor] = rho
    correlation_results_weather = {k: v for k, v in correlation_results_weather.items() if v is not None}

    correlation_results_cities = {}
    for factor in factor_list_cities:
        rho = weighted_partial_correlation(rows=rows, factor=factor, result="Sleep Quality", control_columns=control_columns)
        correlation_results_cities[factor] = rho
    correlation_results_cities = {k: v for k, v in correlation_results_cities.items() if v is not None}

    sleep_quality = np.array([row["Sleep Quality"] for row in rows if row["Sleep Quality"] is not None])
    sigma_Y = np.std(sleep_quality, ddof=1) if len(sleep_quality) > 1 else 0

    expected_effects_notes = {}
    for factor, r in correlation_results_notes.items():
        factor_values = np.array([row[factor] for row in rows if row[factor] is not None])
        p = np.mean(factor_values) if len(factor_values) > 0 else 0
        if p in (0, 1):
            expected_effects_notes[factor] = 0
        else:
            delta = r * sigma_Y / np.sqrt(p * (1 - p))
            expected_effects_notes[factor] = delta
    expected_effects_notes = sorted(expected_effects_notes.items(), key=lambda item: item[1], reverse=True)

    lines = []
    lines.append("Expected effects of Notes:")
    for key, value in expected_effects_notes:
        lines.append(f"{key} -> {int(value.round()):+} %")

    expected_effects_weather = {}
    for factor, r in correlation_results_weather.items():
        factor_values = np.array([row[factor] for row in rows if row[factor] is not None])
        p = np.mean(factor_values) if len(factor_values) > 0 else 0
        if p in (0, 1):
            expected_effects_weather[factor] = 0
        else:
            delta = r * sigma_Y / np.sqrt(p * (1 - p))
            expected_effects_weather[factor] = delta
    expected_effects_weather = sorted(expected_effects_weather.items(), key=lambda item: item[1], reverse=True)

    lines.append("\nExpected effects of Weather types:")
    for key, value in expected_effects_weather:
        lines.append(f"{key} -> {int(value.round()):+} %")

    expected_effects_cities = {}
    for factor, r in correlation_results_cities.items():
        factor_values = np.array([row[factor] for row in rows if row[factor] is not None])
        p = np.mean(factor_values) if len(factor_values) > 0 else 0
        if p in (0, 1):
            expected_effects_cities[factor] = 0
        else:
            delta = r * sigma_Y / np.sqrt(p * (1 - p))
            expected_effects_cities[factor] = delta
    expected_effects_cities = sorted(expected_effects_cities.items(), key=lambda item: item[1], reverse=True)

    lines.append("\nExpected effects of Cities:")
    for key, value in expected_effects_cities:
        lines.append(f"{key} -> {int(value.round()):+} %")

    # Compute best sleep goal (bedtime, alarm, time in bed)
    best = compute_best_sleep_goal(rows)
    if best is not None:
        lines.append("\nBest sleep goal (maximize expected sleep quality):")
        lines.append(f"Bedtime: {format_seconds_to_24h_label(best['bedtime'])}")
        lines.append(f"Alarm: {format_seconds_to_24h_label(best['alarm_time'])}")
        tib_h = int(best['time_in_bed_hours'])
        tib_min = int(round((best['time_in_bed_hours'] - tib_h) * 60))
        # Avoid showing 60 minutes (e.g., 11h 60m) — carry into hours
        if tib_min >= 60:
            tib_h += tib_min // 60
            tib_min = tib_min % 60
        lines.append(f"Time in bed: {tib_h}h {tib_min}m")
        lines.append(f"Predicted sleep quality: {best['predicted_quality']:.1f} %")

    # Compute constrained sleep goals if any fixed parameters are set
    if FIXED_BEDTIME is not None or FIXED_ALARM_TIME is not None or FIXED_TIME_IN_BED_HOURS is not None:
        best_constrained = compute_best_sleep_goal(rows, fixed_bedtime=FIXED_BEDTIME, fixed_alarm_time=FIXED_ALARM_TIME, fixed_time_in_bed_hours=FIXED_TIME_IN_BED_HOURS)
        if best_constrained is not None:
            lines.append("\nBest sleep goal (with constraints):")
            
            # Format bedtime with constraint info
            bedtime_min, bedtime_max = parse_time_range(FIXED_BEDTIME)
            if FIXED_BEDTIME is not None:
                if bedtime_min == bedtime_max:
                    lines.append(f"Bedtime: {format_seconds_to_24h_label(best_constrained['bedtime'])} (fixed to {format_seconds_to_24h_label(bedtime_min)})")
                else:
                    lines.append(f"Bedtime: {format_seconds_to_24h_label(best_constrained['bedtime'])} (range {format_seconds_to_24h_label(bedtime_min)} - {format_seconds_to_24h_label(bedtime_max)})")
            else:
                lines.append(f"Bedtime: {format_seconds_to_24h_label(best_constrained['bedtime'])}")
            
            # Format alarm with constraint info
            alarm_min, alarm_max = parse_time_range(FIXED_ALARM_TIME)
            if FIXED_ALARM_TIME is not None:
                if alarm_min == alarm_max:
                    lines.append(f"Alarm: {format_seconds_to_24h_label(best_constrained['alarm_time'])} (fixed to {format_seconds_to_24h_label(alarm_min)})")
                else:
                    lines.append(f"Alarm: {format_seconds_to_24h_label(best_constrained['alarm_time'])} (range {format_seconds_to_24h_label(alarm_min)} - {format_seconds_to_24h_label(alarm_max)})")
            else:
                lines.append(f"Alarm: {format_seconds_to_24h_label(best_constrained['alarm_time'])}")
            
            # Format time in bed with constraint info
            tib_h = int(best_constrained['time_in_bed_hours'])
            tib_min = int(round((best_constrained['time_in_bed_hours'] - tib_h) * 60))
            if tib_min >= 60:
                tib_h += tib_min // 60
                tib_min = tib_min % 60
            
            tib_min_fixed, tib_max_fixed = parse_tib_range(FIXED_TIME_IN_BED_HOURS)
            if FIXED_TIME_IN_BED_HOURS is not None:
                if tib_min_fixed == tib_max_fixed:
                    lines.append(f"Time in bed: {tib_h}h {tib_min}m (fixed to {tib_min_fixed:.1f}h)")
                else:
                    lines.append(f"Time in bed: {tib_h}h {tib_min}m (range {tib_min_fixed:.1f}h - {tib_max_fixed:.1f}h)")
            else:
                lines.append(f"Time in bed: {tib_h}h {tib_min}m")
            
            lines.append(f"Predicted sleep quality: {best_constrained['predicted_quality']:.1f} %")

    text_path = save_text("output.txt", "\n".join(lines) + "\n")
    print(f"Saved output to {text_path}")

    plot_alarm_time(rows)

    data = [
        [r["Sleep Quality"] for r in rows if r["Alarm set"] == 0 and r["Sleep Quality"] is not None],
        [r["Sleep Quality"] for r in rows if r["Alarm set"] == 1 and r["Sleep Quality"] is not None],
    ]
    weights = [
        [r["Weight"] for r in rows if r["Alarm set"] == 0 and r["Sleep Quality"] is not None and r["Weight"] is not None],
        [r["Weight"] for r in rows if r["Alarm set"] == 1 and r["Sleep Quality"] is not None and r["Weight"] is not None],
    ]
    plot_weighted_boxplot(data, weights, ["No alarm", "Alarm set"], "Sleep quality", "Sleep quality vs alarm set")


if __name__ == "__main__":
    ensure_output_dir()
    rows = load_rows()
    created = plot_all(rows)
    run_analysis_prints(rows)
    print(f"Saved {len([p for p in created if p])} plot files to {OUTPUT_DIR}")
