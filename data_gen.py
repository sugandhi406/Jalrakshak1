"""
JalRakshak AI - Simulated Data Generator
Generates realistic water-consumption data for a college/hostel campus in
Parbhani with 4 zones (blocks). Produces:
  - hourly_consumption.csv  (last 30 days, hourly, litres) -> for anomaly detection
  - daily_consumption.csv   (last 120 days, daily, litres) -> for forecasting

Includes deliberately injected anomalies (a slow leak, a sudden burst,
and continuous night-flow) so the anomaly detector has something real to catch.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)

ZONES = ["Block A (Boys Hostel)", "Block B (Girls Hostel)", "Block C (Day Scholars/Admin)", "Block D (Staff Quarters)"]

BASE_HOURLY_PROFILE = {
    # typical fraction of daily usage occurring in each hour (peaks: morning + evening)
    0: 0.005, 1: 0.003, 2: 0.002, 3: 0.002, 4: 0.01, 5: 0.04,
    6: 0.09, 7: 0.11, 8: 0.08, 9: 0.05, 10: 0.03, 11: 0.03,
    12: 0.05, 13: 0.05, 14: 0.03, 15: 0.03, 16: 0.04, 17: 0.05,
    18: 0.08, 19: 0.10, 20: 0.08, 21: 0.05, 22: 0.02, 23: 0.01,
}

ZONE_DAILY_BASE_LITRES = {
    "Block A (Boys Hostel)": 18000,
    "Block B (Girls Hostel)": 15000,
    "Block C (Day Scholars/Admin)": 6000,
    "Block D (Staff Quarters)": 4000,
}


def generate_daily(days=120):
    end = datetime(2026, 8, 30)
    start = end - timedelta(days=days - 1)
    dates = pd.date_range(start, end, freq="D")
    rows = []
    for zone, base in ZONE_DAILY_BASE_LITRES.items():
        for i, d in enumerate(dates):
            weekday_factor = 0.85 if d.weekday() >= 5 else 1.0  # lower usage on weekends
            seasonal = 1.0 + 0.15 * np.sin(2 * np.pi * i / 120)  # mild seasonal drift
            noise = np.random.normal(1.0, 0.06)
            occupancy_dip = 0.5 if (d.month == 5 and zone.startswith("Block A")) else 1.0  # summer vacation dip example
            litres = base * weekday_factor * seasonal * noise * occupancy_dip
            rows.append({"date": d.strftime("%Y-%m-%d"), "zone": zone, "litres": round(litres, 1)})

    df = pd.DataFrame(rows)

    # Inject a slow creeping leak in Block C over the last 10 days (gradual +40%)
    mask = (df["zone"] == "Block C (Day Scholars/Admin)") & (df["date"] >= (end - timedelta(days=9)).strftime("%Y-%m-%d"))
    idx = df[mask].index
    ramp = np.linspace(1.05, 1.45, len(idx))
    df.loc[idx, "litres"] = (df.loc[idx, "litres"] * ramp).round(1)

    return df


def generate_hourly(days=30):
    end = datetime(2026, 8, 30, 23, 0, 0)
    start = end - timedelta(days=days - 1)
    start = start.replace(hour=0, minute=0, second=0)
    timestamps = pd.date_range(start, end, freq="h")

    rows = []
    for zone, daily_base in ZONE_DAILY_BASE_LITRES.items():
        for ts in timestamps:
            hour = ts.hour
            frac = BASE_HOURLY_PROFILE[hour]
            noise = np.random.normal(1.0, 0.12)
            litres = max(daily_base * frac * noise, 0)
            rows.append({"timestamp": ts.strftime("%Y-%m-%d %H:%M"), "zone": zone, "litres": round(litres, 1)})

    df = pd.DataFrame(rows)

    # --- Inject anomaly 1: continuous night-flow leak in Block A, last 3 nights (1am-4am) ---
    leak_days = [end - timedelta(days=d) for d in range(3)]
    for d in leak_days:
        for h in [1, 2, 3, 4]:
            ts_str = d.replace(hour=h, minute=0, second=0).strftime("%Y-%m-%d %H:%M")
            mask = (df["zone"] == "Block A (Boys Hostel)") & (df["timestamp"] == ts_str)
            df.loc[mask, "litres"] = np.random.uniform(180, 260)  # should be near-zero normally

    # --- Inject anomaly 2: sudden burst spike in Block D, 2 days ago at 3pm ---
    burst_ts = (end - timedelta(days=2)).replace(hour=15, minute=0, second=0).strftime("%Y-%m-%d %H:%M")
    mask = (df["zone"] == "Block D (Staff Quarters)") & (df["timestamp"] == burst_ts)
    df.loc[mask, "litres"] = df.loc[mask, "litres"] * 6

    return df


if __name__ == "__main__":
    daily = generate_daily()
    hourly = generate_hourly()
    daily.to_csv("/home/claude/jalrakshak-ai/data/daily_consumption.csv", index=False)
    hourly.to_csv("/home/claude/jalrakshak-ai/data/hourly_consumption.csv", index=False)
    print("Generated daily_consumption.csv:", daily.shape)
    print("Generated hourly_consumption.csv:", hourly.shape)
