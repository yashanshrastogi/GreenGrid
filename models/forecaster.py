import os
import sys
import warnings
import logging
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)
warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def run_prophet_synthetic(input_path: str, output_path: str) -> dict:
    from prophet import Prophet
    
    df = pd.read_csv(input_path, parse_dates=["timestamp"])
    hourly = (
        df.groupby(["room", pd.Grouper(key="timestamp", freq="1h")])["power_kw"]
        .mean()
        .reset_index()
    )
    hourly.columns = ["room", "ds", "y"]
    hourly = hourly.dropna().reset_index(drop=True)

    cutoff = hourly["ds"].max() - pd.Timedelta(days=7)
    results = []
    all_forecasts = []

    for room, g in hourly.groupby("room"):
        train = g[g["ds"] <= cutoff].reset_index(drop=True)
        test = g[g["ds"] > cutoff].reset_index(drop=True)
        if len(test) == 0:
            continue

        model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=False,
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10,
        )
        model.fit(train)

        future = model.make_future_dataframe(periods=len(test), freq="1h")
        forecast = model.predict(future)

        cols_to_keep = ["ds", "yhat", "yhat_lower", "yhat_upper", "trend"]
        if "daily" in forecast.columns: cols_to_keep.append("daily")
        if "weekly" in forecast.columns: cols_to_keep.append("weekly")

        eval_df = test.merge(
            forecast[cols_to_keep], on="ds", how="left"
        )
        mae = (eval_df["y"] - eval_df["yhat"]).abs().mean()
        rmse = ((eval_df["y"] - eval_df["yhat"]) ** 2).mean() ** 0.5

        results.append({
            "room": room,
            "train_hours": len(train),
            "test_hours": len(test),
            "mae_kw": round(mae, 4),
            "rmse_kw": round(rmse, 4),
            "mean_actual_kw": round(eval_df["y"].mean(), 4),
        })

        eval_df["room"] = room
        all_forecasts.append(eval_df)

    results_df = pd.DataFrame(results)
    overall_mae = results_df["mae_kw"].mean()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    forecast_df = pd.concat(all_forecasts, ignore_index=True)
    forecast_df.to_csv(output_path, index=False)

    return {"overall_mae": overall_mae, "per_room": results_df.to_dict("records")}


def run_prophet_real(long_df: pd.DataFrame, output_path: str, plot_path: str) -> dict:
    from prophet import Prophet

    circuit = "water_heater_ac"

    g = (
        long_df[long_df["circuit"] == circuit]
        .set_index("timestamp")
        .resample("1h")["power_kw"]
        .mean()
        .reset_index()
    )
    g.columns = ["ds", "y"]
    g = g.dropna().reset_index(drop=True)

    cutoff = g["ds"].max() - pd.Timedelta(days=7)
    train = g[g["ds"] <= cutoff]
    test = g[g["ds"] > cutoff].reset_index(drop=True)

    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=0.05,
    )
    model.fit(train)

    future = model.make_future_dataframe(periods=len(test), freq="1h")
    forecast = model.predict(future)

    cols_to_keep = ["ds", "yhat", "yhat_lower", "yhat_upper", "trend"]
    if "daily" in forecast.columns: cols_to_keep.append("daily")
    if "weekly" in forecast.columns: cols_to_keep.append("weekly")

    eval_df = test.merge(
        forecast[cols_to_keep], on="ds", how="left"
    )
    mae = (eval_df["y"] - eval_df["yhat"]).abs().mean()
    rmse = ((eval_df["y"] - eval_df["yhat"]) ** 2).mean() ** 0.5

    eval_df["circuit"] = circuit
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    eval_df.to_csv(output_path, index=False)

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(
        train["ds"].tail(7 * 24), train["y"].tail(7 * 24),
        color="gray", alpha=0.5, label="Train (prior week shown)"
    )
    ax.plot(eval_df["ds"], eval_df["y"], color="#2b6e4f", linewidth=1.2, label="Actual (held-out week)")
    ax.plot(eval_df["ds"], eval_df["yhat"], color="#e07a1f", linewidth=1.2,
            linestyle="--", label="Prophet forecast")
    ax.fill_between(eval_df["ds"], eval_df["yhat_lower"], eval_df["yhat_upper"],
                    color="#e07a1f", alpha=0.15)
    ax.set_title(f"GreenGrid — Prophet Forecast vs Actual ({circuit})")
    ax.set_ylabel("Power (kW, hourly mean)")
    ax.legend(loc="upper right")
    plt.tight_layout()
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=150)
    plt.close()

    return {"mae": mae, "rmse": rmse, "circuit": circuit}
