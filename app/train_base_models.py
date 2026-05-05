# trains lstm_2018 and rf_2018 on 2018 data with HORIZON=6
# run this once before starting the stack, or any time HORIZON changes
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_DIR = os.getenv("MODEL_DIR", "models")
DATA_FILE = "load_weather_full.csv"
TRAIN_START = "2018-01-01"
TRAIN_END   = "2018-12-31"
COMFORT_TEMP = 65
HORIZON = 6
SEQ_LEN = 48
EPOCHS = 50
BATCH_SIZE = 32
LR = 0.001
PATIENCE = 8

FEATURES = [
    "hdd", "cdd", "temp_f", "feels_like_f", "humidity_pct",
    "hour_sin", "hour_cos",
    "dayofweek_sin", "dayofweek_cos",
    "month_sin", "month_cos",
    "is_weekend",
    "lag_1h", "lag_24h", "lag_168h", "rolling_24h_mean"
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Reproducibility -- match RF random_state=42 for the LSTM as well
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def prepare(df):
    df = df.copy().sort_values("datetime").reset_index(drop=True)
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
    return df.reset_index(drop=True)


class DS(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.y) - SEQ_LEN
    def __getitem__(self, i): return self.X[i:i+SEQ_LEN], self.y[i+SEQ_LEN-1]


class LoadLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(nn.Linear(hidden_size, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(1)


def save(name, scaler_X, scaler_y, model=None, rf=None):
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(scaler_X, f"{MODEL_DIR}/{name}_scaler_X.pkl")
    joblib.dump(scaler_y, f"{MODEL_DIR}/{name}_scaler_y.pkl")
    if model is not None:
        torch.save(model.state_dict(), f"{MODEL_DIR}/{name}.pt")
    if rf is not None:
        joblib.dump(rf, f"{MODEL_DIR}/{name}.pkl")
    logger.info(f"saved {name}")


def train_lstm(subset):
    sx, sy = StandardScaler(), StandardScaler()
    X = sx.fit_transform(subset[FEATURES].values)
    y = sy.fit_transform(subset[["target"]].values).flatten()
    split = int(len(X) * 0.8)
    tl = DataLoader(DS(X[:split], y[:split]), batch_size=BATCH_SIZE, shuffle=True)
    vl = DataLoader(DS(X[split:], y[split:]), batch_size=BATCH_SIZE) if (len(X)-split) > SEQ_LEN else None
    m = LoadLSTM(len(FEATURES)).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    loss_fn = nn.MSELoss()
    best, best_w, wait = float("inf"), None, 0
    for epoch in range(EPOCHS):
        m.train()
        for xb, yb in tl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss_fn(m(xb), yb).backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
        if vl:
            m.eval()
            vl_loss = sum(loss_fn(m(xb.to(device)), yb.to(device)).item() for xb, yb in vl) / len(vl)
            if vl_loss < best:
                best, best_w, wait = vl_loss, {k: v.clone() for k, v in m.state_dict().items()}, 0
            else:
                wait += 1
                if wait >= PATIENCE:
                    logger.info(f"early stop at epoch {epoch+1}")
                    break
    if best_w:
        m.load_state_dict(best_w)
    return m, sx, sy


def train_rf(subset):
    sx, sy = StandardScaler(), StandardScaler()
    X = sx.fit_transform(subset[FEATURES].values)
    y = sy.fit_transform(subset[["target"]].values).flatten()
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    return rf, sx, sy


def main():
    logger.info(f"loading {DATA_FILE}...")
    df = pd.read_csv(DATA_FILE)
    df["datetime"] = pd.to_datetime(df["datetime"])
    hourly = prepare(df)
    subset = hourly[(hourly["datetime"] >= TRAIN_START) & (hourly["datetime"] <= TRAIN_END)]
    logger.info(f"training on {TRAIN_START} to {TRAIN_END} ({len(subset)} hours), HORIZON={HORIZON}h")

    logger.info("training RF (fast)...")
    rf, sx, sy = train_rf(subset)
    save("rf_2018", sx, sy, rf=rf)

    logger.info("training LSTM (this takes a few minutes)...")
    lstm, sx, sy = train_lstm(subset)
    save("lstm_2018", sx, sy, model=lstm)

    logger.info("done. both base models saved to models/")


if __name__ == "__main__":
    main()
