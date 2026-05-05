# generate presentation plots from saved simulation CSVs
# run lstm_sim.py and rf_sim.py (both modes) before running this
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import argparse
import os

DRIFT_THRESHOLD = 2000
RESULTS = "results"

KNOWN_EVENTS = {
    "2020-03-01": "COVID lockdowns",
    "2022-11-01": "ChatGPT launch",
}


def load(name):
    path = os.path.join(RESULTS, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df


def plot_drift_vs_static(model_name, color, out_file):
    drift = load(f"{model_name}_sim_drift.csv")
    static = load(f"{model_name}_sim_static.csv")

    drift_daily = drift.set_index("datetime")["error"].resample("D").mean()
    static_daily = static.set_index("datetime")["error"].resample("D").mean()

    retrain_dates = drift[drift["status"] == "RETRAINED"]["datetime"]

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

    for ax, daily, label, alpha in [
        (axes[0], static_daily, "no retraining", 0.4),
        (axes[1], drift_daily, "drift-aware", 0.4),
    ]:
        ax.bar(daily.index, daily.values, color=color, alpha=alpha, width=1)
        ax.plot(daily.index, daily.rolling(5, center=True).mean(),
                color=color, linewidth=2)
        ax.axhline(DRIFT_THRESHOLD, color="red", linewidth=1, linestyle="--")
        ax.set_ylabel("Absolute Error (MW)")
        mae = daily.mean()
        ax.set_title(f"{model_name.upper()} - {label}  |  MAE={mae:.0f} MW")
        ax.grid(True, alpha=0.2)

    for d in retrain_dates:
        axes[1].axvline(d, color="lime", alpha=0.7, linewidth=1, linestyle="--")

    legend_elements = [
        Line2D([0], [0], color=color,  linewidth=2,      label="5-day rolling avg"),
        Line2D([0], [0], color="red",  linestyle="--",   label=f"threshold ({DRIFT_THRESHOLD} MW)"),
        Line2D([0], [0], color="lime", linestyle="--",   label="retrain triggered"),
    ]
    axes[1].legend(handles=legend_elements, loc="upper right", fontsize=9)
    axes[1].set_xlabel("Date")

    mae_static = static_daily.mean()
    mae_drift  = drift_daily.mean()
    pct = (mae_static - mae_drift) / mae_static * 100
    plt.suptitle(
        f"{model_name.upper()}: Static vs Drift-Aware  |  "
        f"MAE improvement: {mae_static:.0f} to {mae_drift:.0f} MW  ({pct:+.1f}%)",
        fontsize=13
    )
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, out_file), dpi=150)
    print(f"saved {out_file}  (MAE {pct:+.1f}% improvement from retraining)")
    plt.show()


def plot_model_comparison():
    lstm = load("lstm_sim_drift.csv")
    rf   = load("rf_sim_drift.csv")

    lstm_daily = lstm.set_index("datetime")["error"].resample("D").mean()
    rf_daily   = rf.set_index("datetime")["error"].resample("D").mean()

    lstm_retrains = lstm[lstm["status"] == "RETRAINED"]["datetime"]
    rf_retrains = rf[rf["status"] == "RETRAINED"]["datetime"]

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

    for ax, daily, retrains, color, label in [
        (axes[0], lstm_daily, lstm_retrains, "steelblue", "LSTM"),
        (axes[1], rf_daily, rf_retrains, "darkorange", "Random Forest"),
    ]:
        ax.plot(daily.index, daily.values,                   color=color, linewidth=0.4, alpha=0.3)
        ax.plot(daily.index, daily.rolling(5, center=True).mean(), color=color, linewidth=2)
        ax.axhline(DRIFT_THRESHOLD, color="red", linewidth=1, linestyle="--")
        for d in retrains:
            ax.axvline(d, color="lime", alpha=0.5, linewidth=1, linestyle="--")
        mae = daily.mean()
        n = len(retrains)
        ax.set_title(f"{label} (drift-aware)  |  MAE={mae:.0f} MW  |  retrains={n}")
        ax.set_ylabel("Absolute Error (MW)")
        ax.grid(True, alpha=0.2)

    for ds, label in KNOWN_EVENTS.items():
        dt = pd.to_datetime(ds)
        for ax in axes:
            ax.axvline(dt, color="purple", alpha=0.5, linewidth=1.5)
        axes[0].text(dt, axes[0].get_ylim()[1] * 0.88, label,
                     rotation=90, fontsize=8, color="purple", va="top")

    legend_elements = [
        Line2D([0], [0], color="steelblue",  linewidth=2,    label="LSTM 5-day avg"),
        Line2D([0], [0], color="darkorange", linewidth=2,    label="RF 5-day avg"),
        Line2D([0], [0], color="red",        linestyle="--", label=f"threshold ({DRIFT_THRESHOLD} MW)"),
        Line2D([0], [0], color="lime",        linestyle="--", label="retrain triggered"),
        Line2D([0], [0], color="purple",      alpha=0.6,      label="known event"),
    ]
    axes[1].legend(handles=legend_elements, loc="upper right", fontsize=9)
    axes[1].set_xlabel("Date")
    plt.suptitle("LSTM vs Random Forest - Drift-Aware Forecasting", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "model_comparison.png"), dpi=150)
    print("saved model_comparison.png")
    plt.show()


def plot_actual_vs_pred(year=2021):
    lstm = load("lstm_sim_drift.csv")
    rf   = load("rf_sim_drift.csv")

    lstm_y = lstm[lstm["datetime"].dt.year == year].set_index("datetime")
    rf_y   = rf[rf["datetime"].dt.year == year].set_index("datetime")

    # daily averages so the chart isn't a wall of noise
    actual = lstm_y["actual"].resample("D").mean()
    lstm_pred = lstm_y["predicted"].resample("D").mean()
    rf_pred = rf_y["predicted"].resample("D").mean()

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(actual.index, actual.values, color="white", linewidth=2, label="Actual")
    ax.plot(lstm_pred.index, lstm_pred.values, color="steelblue", linewidth=1.5,
            linestyle="--", label="LSTM predicted")
    ax.plot(rf_pred.index, rf_pred.values, color="darkorange", linewidth=1.5,
            linestyle="--", label="RF predicted")

    ax.set_title(f"Actual vs Predicted Load (MW) - Daily Avg {year}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Load (MW)")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    out = f"actual_vs_pred_{year}.png"
    plt.savefig(os.path.join(RESULTS, out), dpi=150)
    print(f"saved {out}")
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2021,
                        help="year for actual vs predicted plot (default 2021)")
    parser.add_argument("--plots", nargs="+",
                        choices=["lstm", "rf", "compare", "actual", "all"],
                        default=["all"],
                        help="which plots to generate")
    args = parser.parse_args()

    run_all = "all" in args.plots

    if run_all or "lstm" in args.plots:
        plot_drift_vs_static("lstm", "steelblue", "lstm_comparison.png")

    if run_all or "rf" in args.plots:
        plot_drift_vs_static("rf", "darkorange", "rf_comparison.png")

    if run_all or "compare" in args.plots:
        plot_model_comparison()

    if run_all or "actual" in args.plots:
        plot_actual_vs_pred(year=args.year)


if __name__ == "__main__":
    main()
