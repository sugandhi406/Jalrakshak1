"""
JalRakshak AI - Leakage Risk / Anomaly Detection
Combines a rule engine (continuous night-flow, sudden spikes) with a
statistical baseline model (rolling mean + std per zone/hour-of-day) and an
Isolation Forest for multivariate outlier confirmation.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest


def _load_hourly(path):
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    df["date"] = df["timestamp"].dt.date
    return df


def build_baseline(df):
    """Per-zone, per-hour-of-day baseline mean & std (using the full history)."""
    baseline = df.groupby(["zone", "hour"])["litres"].agg(["mean", "std"]).reset_index()
    baseline["std"] = baseline["std"].replace(0, np.nan).fillna(baseline["mean"] * 0.15 + 1)
    return baseline


def detect_anomalies(hourly_csv_path, z_threshold=3.0, night_hours=(1, 2, 3, 4), night_flow_threshold=50):
    """
    Returns a list of anomaly alert dicts:
      { zone, timestamp, litres, expected, z_score, type, severity, note }
    """
    df = _load_hourly(hourly_csv_path)
    baseline = build_baseline(df)
    merged = df.merge(baseline, on=["zone", "hour"], how="left")
    merged["z_score"] = (merged["litres"] - merged["mean"]) / merged["std"]

    alerts = []

    # --- Rule 1: statistical spike/dip anomalies (z-score) ---
    spikes = merged[merged["z_score"].abs() >= z_threshold].copy()
    for _, row in spikes.iterrows():
        severity = "High" if row["z_score"] >= z_threshold + 1.5 else "Medium"
        alerts.append({
            "zone": row["zone"],
            "timestamp": row["timestamp"].strftime("%Y-%m-%d %H:%M"),
            "litres": round(row["litres"], 1),
            "expected": round(row["mean"], 1),
            "z_score": round(row["z_score"], 2),
            "type": "sudden_spike" if row["z_score"] > 0 else "unexpected_dip",
            "severity": severity,
            "note": f"Usage was {round(row['litres'],0)}L vs expected ~{round(row['mean'],0)}L for this hour — "
                    f"{'far above' if row['z_score']>0 else 'far below'} normal baseline."
        })

    # --- Rule 2: continuous night-flow (classic leak signature) ---
    for zone in df["zone"].unique():
        zdf = df[df["zone"] == zone]
        for date, group in zdf[zdf["hour"].isin(night_hours)].groupby("date"):
            flowing_hours = group[group["litres"] > night_flow_threshold]
            if len(flowing_hours) >= 3:  # 3+ consecutive-ish night hours with real flow = likely leak
                total = flowing_hours["litres"].sum()
                alerts.append({
                    "zone": zone,
                    "timestamp": f"{date} (night window {night_hours[0]}:00-{night_hours[-1]}:00)",
                    "litres": round(total, 1),
                    "expected": "~0 (no expected activity at night)",
                    "z_score": None,
                    "type": "continuous_night_flow",
                    "severity": "High",
                    "note": f"Continuous water flow detected across {len(flowing_hours)} night hours "
                            f"({round(total,0)}L total) with no expected occupancy activity — classic leak signature."
                })

    # --- Cross-check with Isolation Forest for confidence ---
    if len(merged) > 20:
        features = merged[["litres", "hour"]].fillna(0)
        iso = IsolationForest(contamination=0.02, random_state=42)
        merged["iso_flag"] = iso.fit_predict(features)
        iso_conf = set(
            zip(
                merged.loc[merged["iso_flag"] == -1, "zone"],
                merged.loc[merged["iso_flag"] == -1, "timestamp"].dt.strftime("%Y-%m-%d %H:%M")
            )
        )
        for a in alerts:
            if (a["zone"], a["timestamp"]) in iso_conf:
                a["confidence"] = "Confirmed by multivariate model"
            else:
                a["confidence"] = "Rule/statistical detection"
    else:
        for a in alerts:
            a["confidence"] = "Rule/statistical detection"

    # Deduplicate & sort by severity/recency
    alerts = sorted(alerts, key=lambda x: (x["severity"] != "High", x["timestamp"]), reverse=False)
    return alerts


def summarize_alerts(alerts):
    by_zone = {}
    for a in alerts:
        by_zone.setdefault(a["zone"], []).append(a)
    return {zone: len(items) for zone, items in by_zone.items()}


if __name__ == "__main__":
    alerts = detect_anomalies("/home/claude/jalrakshak-ai/data/hourly_consumption.csv")
    print(f"Found {len(alerts)} anomalies")
    for a in alerts[:10]:
        print(a)
