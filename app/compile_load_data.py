import pandas as pd
import os
import glob

DATA_DIR = "data"
OUTPUT_FILE = "data/pjm_load_combined.csv"

# Mid-Atlantic zones (DOM would be data center alley)
LOAD_AREAS = ["PEPCO", "ME", "PE", "PS", "JC", "DPLCO", "DOM"]


def main():
    csv_files = glob.glob(os.path.join(DATA_DIR, "hrl_load_metered*.csv"))

    if not csv_files:
        print("no files found in data/")
        return

    frames = []
    for f in sorted(csv_files):
        df = pd.read_csv(f)
        frames.append(df)
        print(f"loaded {len(df)} rows from {os.path.basename(f)}")

    combined = pd.concat(frames, ignore_index=True)
    combined.columns = [c.strip().lower().replace(" ", "_") for c in combined.columns]

    combined = combined[combined["load_area"].isin(LOAD_AREAS)].copy()
    print(f"{len(combined)} rows after zone filter")

    combined["datetime"] = pd.to_datetime(combined["datetime_beginning_ept"])

    # sum MW across all zones per hour
    combined = combined.groupby("datetime")["mw"].sum().reset_index()
    combined = combined.sort_values("datetime").drop_duplicates(subset="datetime")

    combined.to_csv(OUTPUT_FILE, index=False)
    print(f"saved {len(combined)} rows to {OUTPUT_FILE}")
    print(f"{combined['datetime'].min()} to {combined['datetime'].max()}")


if __name__ == "__main__":
    main()