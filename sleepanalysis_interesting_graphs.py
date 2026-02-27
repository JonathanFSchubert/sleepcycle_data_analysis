"""
Parts of this code were written with help from LLMs.
"""

import csv
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt

CSV_FILE = "sleepdata.csv"

# =========================
# Helpers
# =========================


def yyyy_time_to_datetime(string):
    if string == "":
        return None
    return datetime.strptime(string, "%Y-%m-%d %H:%M:%S")


def parse_float(s):
    if s == "" or s == "—":
        return None
    return float(s.replace(",", ".").rstrip("%"))


# =========================
# Load data
# =========================

rows = []
with open(CSV_FILE, "r", encoding="utf-8-sig") as csvfile:
    reader = csv.DictReader(csvfile, delimiter=";")
    for row in reader:
        rows.append(row)

# convert types
for r in rows:
    r["Woke up"] = yyyy_time_to_datetime(r["Woke up"])

    r["Sleep Quality"] = (
        int(r["Sleep Quality"].rstrip("%")) if r["Sleep Quality"] else None
    )
    r["Time in bed (seconds)"] = (
        int(r["Time in bed (seconds)"]) if r["Time in bed (seconds)"] else None
    )
    r["Time asleep (seconds)"] = (
        int(r["Time asleep (seconds)"]) if r["Time asleep (seconds)"] else None
    )
    r["Asleep after (seconds)"] = (
        int(r["Asleep after (seconds)"]) if r["Asleep after (seconds)"] else None
    )
    r["Alertness score"] = (
        int(r["Alertness score"].rstrip("%")) if r["Alertness score"] else None
    )

    r["Notes"] = [] if r["Notes"] == "" else r["Notes"].split(":")

    r["Regularity"] = parse_float(r["Regularity"])

# =========================
# Derived columns
# =========================

# Sleep quality previous day
for i in range(len(rows)):
    if i == 0:
        rows[i]["Prev Sleep Quality"] = None
    else:
        rows[i]["Prev Sleep Quality"] = rows[i - 1]["Sleep Quality"]

# Note indicators
for r in rows:
    r["Sleep drug"] = 1 if "Sleep drug" in r["Notes"] else 0
    r["Coffee"] = 1 if "Coffee" in r["Notes"] else 0
    r["Tea"] = 1 if "Tea" in r["Notes"] else 0

# =========================
# Plot helper
# =========================

FIG_SIZE = 10


def scatter(x, y, xlabel, ylabel, title, invert_x=False):
    x_arr = np.array([xi for xi, yi in zip(x, y) if xi is not None and yi is not None])
    y_arr = np.array([yi for xi, yi in zip(x, y) if xi is not None and yi is not None])

    plt.figure(figsize=(FIG_SIZE, FIG_SIZE))
    plt.scatter(x_arr, y_arr, alpha=0.6)

    # linear fit
    if len(x_arr) > 1:
        m, b = np.polyfit(x_arr, y_arr, 1)
        plt.plot(x_arr, m * x_arr + b, color="red", linewidth=2)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if invert_x:
        plt.gca().invert_xaxis()
    plt.show()


def boxplot(groups, labels, ylabel, title, invert_x=False):
    data = []
    for g in groups:
        data.append([v for v in g if v is not None])

    plt.figure(figsize=(FIG_SIZE, FIG_SIZE))
    plt.boxplot(data, tick_labels=labels)
    plt.ylabel(ylabel)
    plt.title(title)
    if invert_x:
        plt.gca().invert_xaxis()
    plt.show()


# =========================
# Sleep drug note vs Asleep after
# =========================

drug_yes = [r["Asleep after (seconds)"] for r in rows if r["Sleep drug"] == 1]
drug_no = [r["Asleep after (seconds)"] for r in rows if r["Sleep drug"] == 0]

boxplot(
    [drug_no, drug_yes],
    ["No sleep drug", "Sleep drug"],
    "Asleep after (s)",
    "Sleep drug vs time to fall asleep",
)

# =========================
# Alertness score vs Sleep quality
# =========================

scatter(
    [r["Sleep Quality"] for r in rows],
    [r["Alertness score"] for r in rows],
    "Sleep quality",
    "Alertness score",
    "Alertness vs Sleep quality",
)

# =========================
# Alertness score vs Time in bed
# =========================

scatter(
    [r["Time in bed (seconds)"] for r in rows],
    [r["Alertness score"] for r in rows],
    "Time in bed (s)",
    "Alertness score",
    "Alertness vs Time in bed",
)

# =========================
# Alertness score vs Time asleep
# =========================

scatter(
    [r["Time asleep (seconds)"] for r in rows],
    [r["Alertness score"] for r in rows],
    "Time asleep (s)",
    "Alertness score",
    "Alertness vs Time asleep",
)

# =========================
# Coffee note vs Asleep after
# =========================

coffee_yes = [r["Asleep after (seconds)"] for r in rows if r["Coffee"] == 1]
coffee_no = [r["Asleep after (seconds)"] for r in rows if r["Coffee"] == 0]

boxplot(
    [coffee_no, coffee_yes],
    ["No coffee", "Coffee"],
    "Asleep after (s)",
    "Coffee vs time to fall asleep",
)

# =========================
# Tea note vs Asleep after
# =========================

tea_yes = [r["Asleep after (seconds)"] for r in rows if r["Tea"] == 1]
tea_no = [r["Asleep after (seconds)"] for r in rows if r["Tea"] == 0]
boxplot(
    [tea_no, tea_yes],
    ["No tea", "Tea"],
    "Asleep after (s)",
    "Tea vs time to fall asleep",
)

# =========================
# Quality day before vs Sleep quality
# =========================

scatter(
    [r["Prev Sleep Quality"] for r in rows],
    [r["Sleep Quality"] for r in rows],
    "Sleep quality previous day",
    "Sleep quality",
    "Sleep inertia / carryover effect",
    invert_x=True,
)

# =========================
# Sleep regularity vs Sleep quality
# =========================

scatter(
    [r["Regularity"] for r in rows],
    [r["Sleep Quality"] for r in rows],
    "Regularity",
    "Sleep quality",
    "Sleep regularity vs Sleep quality",
    invert_x=False,
)
