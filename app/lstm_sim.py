import pandas as pd
import numpy as np
import os
import argparse
from collections import deque
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import joblib
import warnings
warnings.filterwarnings("ignore")

INPUT_FILE = "load_weather_full.csv"
MODEL_DIR = "models"

TRAIN_START = "2018-01-01"
TRAIN_END = "2018-12-31"
DEPLOY_START = "2019-01-01"

DRIFT_THRESHOLD = 2000
DRIFT_WINDOW = 120
COOLDOWN_HOURS = 14 * 24
RETRAIN_YEARS = 1
COMFORT_TEMP = 65
HORIZON = 6
SEQ_LEN = 48
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001
PATIENCE = 8

FEATURES = [
    "hdd", "cdd", "temp_f", "feels_like_f", "humidity_pct",
    "hour_sin", "hour_cos",
    "dayofweek_sin", "dayofweek_cos",
    "month_sin", "month_cos",
    "is_weekend",
    "lag_1h", "lag_24h", "lag_168h", "rolling_24h_mean"
]
TARGET = "target"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def prepare_hourly(df):
    df = df.copy()
    df = df.sort_values("datetime").reset_index(drop=True)

    hour = df["datetime"].dt.hour
    dow = df["datetime"].dt.dayofweek
    month = df["datetime"].dt.month

    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["dayofweek_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dayofweek_cos"] = np.cos(2 * np.pi * dow / 7)
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)
    df["is_weekend"] = (dow >= 5).astype(int)
    df["hdd"] = (COMFORT_TEMP - df["temp_f"]).clip(lower=0)
    df["cdd"] = (df["temp_f"] - COMFORT_TEMP).clip(lower=0)

    df["lag_1h"] = df["mw"].shift(1)
    df["lag_24h"] = df["mw"].shift(24)
    df["lag_168h"] = df["mw"].shift(168)
    df["rolling_24h_mean"] = df["mw"].shift(1).rolling(24).mean()

    df["target"] = df["mw"].shift(-HORIZON)

    df.dropna(inplace=True)
    df = df.reset_index(drop=True)
    return df


class LoadDataset(Dataset):
    def __init__(self, X, y, seq_len):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.y) - self.seq_len

    def __getitem__(self, idx):
        x_seq = self.X[idx: idx + self.seq_len]
        y_val = self.y[idx + self.seq_len - 1]
        return x_seq, y_val


class LoadLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(1)


def train_model(hourly, start, end, model_name="lstm"):
    subset = hourly[(hourly["datetime"] >= start) & (hourly["datetime"] <= end)].copy()

    if len(subset) < SEQ_LEN + 10:
        print(f"  not enough data ({len(subset)} rows), skipping.")
        return None, None

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X = scaler_X.fit_transform(subset[FEATURES].values)
    y = scaler_y.fit_transform(subset[[TARGET]].values).flatten()

    split = int(len(X) * 0.8)
    use_val = (len(X) - split) > SEQ_LEN

    train_loader = DataLoader(LoadDataset(X[:split], y[:split], SEQ_LEN),
                              batch_size=BATCH_SIZE, shuffle=True)
    if use_val:
        val_loader = DataLoader(LoadDataset(X[split:], y[split:], SEQ_LEN),
                                batch_size=BATCH_SIZE)

    model = LoadLSTM(input_size=len(FEATURES)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    best_weights = None
    wait = 0

    for epoch in range(EPOCHS):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(X_batch), y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        if not use_val:
            continue

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                val_loss += loss_fn(model(X_batch), y_batch).item()
        val_loss /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"  early stop at epoch {epoch+1}")
                break

    if best_weights:
        model.load_state_dict(best_weights)

    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, f"{model_name}.pt"))
    joblib.dump(scaler_X, os.path.join(MODEL_DIR, f"{model_name}_scaler_X.pkl"))
    joblib.dump(scaler_y, os.path.join(MODEL_DIR, f"{model_name}_scaler_y.pkl"))

    print(f"  trained on {start} to {end} ({len(subset)} hours)")
    return model, (scaler_X, scaler_y)


def predict(model, scalers, history):
    scaler_X, scaler_y = scalers
    if len(history) < SEQ_LEN:
        return None

    X_raw = history[FEATURES].values[-SEQ_LEN:]
    X_scaled = scaler_X.transform(X_raw)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        pred_scaled = model(X_tensor).item()

    return scaler_y.inverse_transform([[pred_scaled]])[0][0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=None, help="days to simulate from deploy start")
    parser.add_argument("--no-retrain", action="store_true", help="disable retraining (static model baseline)")
    args = parser.parse_args()

    mode = "static" if args.no_retrain else "drift"
    log_file = f"results/lstm_sim_{mode}.csv"
    plot_file = f"results/lstm_sim_{mode}.png"

    os.makedirs("results", exist_ok=True)

    df = pd.read_csv(INPUT_FILE)
    df["datetime"] = pd.to_datetime(df["datetime"])
    hourly = prepare_hourly(df)

    print(f"{len(hourly)} hours loaded ({hourly['datetime'].min()} to {hourly['datetime'].max()})")
    print(f"mode: {'static (no retraining)' if args.no_retrain else 'drift-aware (retraining enabled)'}")
    print(f"threshold: {DRIFT_THRESHOLD} MW  |  window: {DRIFT_WINDOW}h  |  cooldown: {COOLDOWN_HOURS}h")

    print(f"\ntraining on {TRAIN_START} to {TRAIN_END}...")
    model, scalers = train_model(hourly, TRAIN_START, TRAIN_END, model_name="lstm_2018")

    if model is None:
        print("training failed")
        return

    deploy = hourly[hourly["datetime"] >= DEPLOY_START]
    if args.days:
        end = pd.to_datetime(DEPLOY_START) + pd.Timedelta(days=args.days)
        deploy = deploy[deploy["datetime"] < end]
    deploy = deploy.reset_index(drop=True)
    print(f"\nsimulating {len(deploy)} hours (~{len(deploy)//24} days from {DEPLOY_START})\n")

    log = []
    retrain_events = []
    retrain_count = 0
    recent_errors = deque(maxlen=DRIFT_WINDOW)
    hours_since_retrain = COOLDOWN_HOURS

    for i, row in deploy.iterrows():
        dt = row["datetime"]
        history = hourly[hourly["datetime"] < dt]
        pred = predict(model, scalers, history)

        if pred is None:
            continue

        actual = row[TARGET]
        error = abs(actual - pred)
        recent_errors.append(error)
        hours_since_retrain += 1

        rolling_avg = sum(recent_errors) / len(recent_errors)
        drift_detected = (
            len(recent_errors) == DRIFT_WINDOW
            and rolling_avg > DRIFT_THRESHOLD
            and hours_since_retrain >= COOLDOWN_HOURS
        )

        status = "OK"

        if drift_detected:
            if args.no_retrain:
                status = "DRIFT_DETECTED"
                if i % 24 == 0:
                    print(f"  [{dt.strftime('%Y-%m-%d')}] drift {rolling_avg:.0f} MW (no retrain)")
            else:
                retrain_start = dt - pd.DateOffset(years=RETRAIN_YEARS)
                print(f"  [{dt.strftime('%Y-%m-%d %H:%M')}] DRIFT: {rolling_avg:.0f} MW, retraining...")
                new_model, new_scalers = train_model(
                    hourly, str(retrain_start), str(dt),
                    model_name=f"lstm_retrained_{dt.strftime('%Y%m%d_%H')}"
                )
                if new_model is not None:
                    model = new_model
                    scalers = new_scalers
                    retrain_events.append(dt)
                    retrain_count += 1
                    status = "RETRAINED"
                    recent_errors.clear()
                    hours_since_retrain = 0
        else:
            if i % 24 == 0:
                print(f"  [{dt.strftime('%Y-%m-%d')}] rolling avg {rolling_avg:.0f} MW")

        log.append({
            "datetime": dt.strftime("%Y-%m-%d %H:%M"),
            "actual": round(actual, 1),
            "predicted": round(pred, 1),
            "error": round(error, 1),
            "rolling_avg_error": round(rolling_avg, 1),
            "status": status,
        })

    log_df = pd.DataFrame(log)
    log_df.to_csv(log_file, index=False)

    mae = log_df["error"].mean()
    rmse = ((log_df["actual"] - log_df["predicted"]) ** 2).mean() ** 0.5
    mape = ((log_df["actual"] - log_df["predicted"]).abs() / log_df["actual"] * 100).mean()
    print(f"\nLSTM ({'static' if args.no_retrain else 'drift-aware'}): {retrain_count} retrains, MAE={mae:.1f} MW, RMSE={rmse:.1f} MW, MAPE={mape:.2f}%")
    print(f"saved to {log_file}")

    # --- plot ---
    log_df["datetime"] = pd.to_datetime(log_df["datetime"])
    daily_avg = log_df.set_index("datetime")["error"].resample("D").mean()

    season_colors = {
        12: "steelblue", 1: "steelblue", 2: "steelblue",
        3: "green",      4: "green",      5: "green",
        6: "red",        7: "red",        8: "red",
        9: "black",      10: "black",     11: "black",
    }

    fig, ax = plt.subplots(figsize=(16, 6))
    for month, group in daily_avg.groupby(daily_avg.index.month):
        ax.bar(group.index, group.values, color=season_colors[month], alpha=0.5, width=1)

    ax.plot(daily_avg.index, daily_avg.rolling(5, center=True).mean(),
            color="black", linewidth=2, label="5-day rolling avg")
    ax.axhline(DRIFT_THRESHOLD, color="red", linewidth=1, linestyle="--",
               label=f"threshold ({DRIFT_THRESHOLD} MW)")

    for event in retrain_events:
        ax.axvline(event, color="lime", alpha=0.8, linewidth=1, linestyle="--")

    known_events = {"2020-03-01": "COVID lockdowns", "2022-11-01": "ChatGPT launch"}
    for ds, label in known_events.items():
        dt_e = pd.to_datetime(ds)
        if daily_avg.index.min() <= dt_e <= daily_avg.index.max():
            ax.axvline(dt_e, color="purple", alpha=0.5, linewidth=1.5)
            ax.text(dt_e, ax.get_ylim()[1] * 0.92, label,
                    rotation=90, fontsize=8, color="purple", va="top")

    legend_elements = [
        Line2D([0], [0], color="steelblue", linewidth=6, alpha=0.5, label="Winter"),
        Line2D([0], [0], color="green", linewidth=6, alpha=0.5, label="Spring"),
        Line2D([0], [0], color="red", linewidth=6, alpha=0.5, label="Summer"),
        Line2D([0], [0], color="black", linewidth=6, alpha=0.5, label="Fall"),
        Line2D([0], [0], color="black", linewidth=2, label="5-day rolling avg"),
        Line2D([0], [0], color="red", linestyle="--", label=f"threshold ({DRIFT_THRESHOLD} MW)"),
        Line2D([0], [0], color="lime", linestyle="--", label="retrain triggered"),
        Line2D([0], [0], color="purple", alpha=0.6, label="known event"),
    ]
    mode_label = "static" if args.no_retrain else "drift-aware"
    ax.legend(handles=legend_elements, loc="upper left", fontsize=9)
    ax.set_title(f"LSTM load forecasting - {mode_label} | threshold {DRIFT_THRESHOLD} MW")
    ax.set_xlabel("Date")
    ax.set_ylabel("Absolute Error (MW)")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(plot_file, dpi=150)
    print(f"plot saved to {plot_file}")
    plt.show()


if __name__ == "__main__":
    main()
