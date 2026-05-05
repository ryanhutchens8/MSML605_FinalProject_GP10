import os
import numpy as np
import pandas as pd

LSTM_LOG = "results/lstm_sim_drift.csv"
RF_LOG = "results/rf_sim_drift.csv"
DRIFT_THRESHOLD = 2000
OUT = "results/system_metrics_report.txt"


def detection_delay_hours(log_df):
    log_df = log_df.copy()
    log_df["datetime"] = pd.to_datetime(log_df["datetime"])
    log_df = log_df.sort_values("datetime").reset_index(drop=True)

    retrain_idx = log_df.index[log_df["status"] == "RETRAINED"].tolist()
    delays = []

    for ri in retrain_idx:
        # walk backward until rolling avg drops below threshold
        j = ri
        while j >= 0 and log_df.loc[j, "rolling_avg_error"] > DRIFT_THRESHOLD:
            j -= 1
        hours_above = ri - j  # each row = 1 hour
        delays.append(hours_above)

    return delays


def simulated_coverage(log_df):
    log_df = log_df.copy()
    log_df["datetime"] = pd.to_datetime(log_df["datetime"])
    span_hours = (log_df["datetime"].max() - log_df["datetime"].min()).total_seconds() / 3600
    return len(log_df) / span_hours * 100 if span_hours > 0 else None


def summarize(label, log_path, lines):
    if not os.path.exists(log_path):
        lines.append(f"{label}: log not found at {log_path}\n")
        return

    df = pd.read_csv(log_path)
    retrain_rows = df[df["status"] == "RETRAINED"]

    lines.append(f"=== {label} ===")

    delays = detection_delay_hours(df)
    if delays:
        lines.append(f"retrain events:              {len(delays)}")
        lines.append(f"detection delay (h):")
        lines.append(f"  mean:                      {np.mean(delays):.1f}")
        lines.append(f"  median:                    {np.median(delays):.1f}")
        lines.append(f"  min:                       {np.min(delays):.1f}")
        lines.append(f"  max:                       {np.max(delays):.1f}")
    else:
        lines.append("no retrain events found")

    cov = simulated_coverage(df)
    if cov is not None:
        lines.append(f"simulation coverage:         {cov:.2f}%")

    mae = df["error"].mean()
    rmse = ((df["actual"] - df["predicted"]) ** 2).mean() ** 0.5
    mape = ((df["actual"] - df["predicted"]).abs() / df["actual"] * 100).mean()
    lines.append(f"MAE:                         {mae:.1f} MW")
    lines.append(f"RMSE:                        {rmse:.1f} MW")
    lines.append(f"MAPE:                        {mape:.2f}%")
    lines.append("")


def main():
    lines = []

    summarize("LSTM", LSTM_LOG, lines)
    summarize("Random Forest", RF_LOG, lines)

    report = "\n".join(lines)
    print(report)

    os.makedirs("results", exist_ok=True)
    with open(OUT, "w") as f:
        f.write(report)
    print(f"saved to {OUT}")


if __name__ == "__main__":
    main()
