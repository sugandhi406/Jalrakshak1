"""
JalRakshak AI - Demand Forecasting
Uses Holt-Winters (triple exponential smoothing) per zone to produce a
7-day rolling forecast, with weekly seasonality (weekday/weekend usage
pattern) captured automatically.
"""
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def _load_daily(path):
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def forecast_zone(df, zone, horizon=7):
    series = df[df["zone"] == zone].sort_values("date").set_index("date")["litres"]
    series = series.asfreq("D").interpolate()

    try:
        model = ExponentialSmoothing(
            series, trend="add", seasonal="add", seasonal_periods=7, damped_trend=True
        ).fit(optimized=True)
        forecast = model.forecast(horizon)
    except Exception:
        # Fallback: simple weekday-aware moving average if smoothing fails (e.g. too little data)
        last_28 = series.iloc[-28:]
        by_weekday = last_28.groupby(last_28.index.weekday).mean()
        future_dates = pd.date_range(series.index[-1] + pd.Timedelta(days=1), periods=horizon)
        forecast = pd.Series([by_weekday.get(d.weekday(), series.mean()) for d in future_dates], index=future_dates)

    hist_std = series.iloc[-28:].std()
    return {
        "zone": zone,
        "history": [{"date": d.strftime("%Y-%m-%d"), "litres": round(v, 1)} for d, v in series.iloc[-30:].items()],
        "forecast": [
            {
                "date": d.strftime("%Y-%m-%d"),
                "litres": round(max(v, 0), 1),
                "lower": round(max(v - 1.28 * hist_std, 0), 1),
                "upper": round(v + 1.28 * hist_std, 1),
            }
            for d, v in forecast.items()
        ],
    }


def forecast_all(daily_csv_path, horizon=7):
    df = _load_daily(daily_csv_path)
    results = {}
    for zone in df["zone"].unique():
        results[zone] = forecast_zone(df, zone, horizon)
    return results


if __name__ == "__main__":
    results = forecast_all("/home/claude/jalrakshak-ai/data/daily_consumption.csv")
    for zone, r in results.items():
        print(zone)
        for f in r["forecast"]:
            print("  ", f)
