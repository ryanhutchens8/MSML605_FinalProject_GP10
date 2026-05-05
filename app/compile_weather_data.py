import requests
import pandas as pd
import os

OUTPUT_FILE = "data/weather_combined.csv"

# Reagan National (DCA), close enough for Mid-Atlantic load
LAT = 38.8521
LON = -77.0377

START_DATE = "2018-01-01"
END_DATE = "2024-12-31"

URL = "https://archive-api.open-meteo.com/v1/archive"


def fetch_weather(start, end):
    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": start,
        "end_date": end,
        "hourly": "temperature_2m,relative_humidity_2m,apparent_temperature",
        "temperature_unit": "fahrenheit",
        "timezone": "America/New_York"
    }

    print(f"fetching {start} to {end}...")
    response = requests.get(URL, params=params, timeout=60)

    if response.status_code != 200:
        print(f"error {response.status_code}: {response.text[:200]}")
        return pd.DataFrame()

    data = response.json()

    df = pd.DataFrame({
        "datetime": data["hourly"]["time"],
        "temp_f": data["hourly"]["temperature_2m"],
        "feels_like_f": data["hourly"]["apparent_temperature"],
        "humidity_pct": data["hourly"]["relative_humidity_2m"],
    })

    df["datetime"] = pd.to_datetime(df["datetime"])
    return df


def main():
    os.makedirs("data", exist_ok=True)

    df = fetch_weather(START_DATE, END_DATE)

    if df.empty:
        print("nothing returned")
        return

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"saved {len(df)} rows to {OUTPUT_FILE}")
    print(df.head())


if __name__ == "__main__":
    main()