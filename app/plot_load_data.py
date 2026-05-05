import pandas as pd
import matplotlib.pyplot as plt
import os

INPUT_FILE = "data/pjm_load_combined.csv"
OUTPUT_FILE = "results/midatlantic_load_by_season.png"

SEASON_MAP = {
    12: ("Winter", "blue"), 1: ("Winter", "blue"), 2: ("Winter", "blue"),
    3: ("Spring", "green"), 4: ("Spring", "green"), 5: ("Spring", "green"),
    6: ("Summer", "red"), 7: ("Summer", "red"), 8: ("Summer", "red"),
    9: ("Fall", "orange"), 10: ("Fall", "orange"), 11: ("Fall", "orange"),
}


def main():
    os.makedirs("results", exist_ok=True)

    df = pd.read_csv(INPUT_FILE)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["month"] = df["datetime"].dt.month

    fig, ax = plt.subplots(figsize=(16, 6))

    plotted = set()
    for month, (season, color) in SEASON_MAP.items():
        subset = df[df["month"] == month]
        label = season if season not in plotted else "_nolegend_"
        plotted.add(season)
        ax.scatter(subset["datetime"], subset["mw"],
                   s=0.3, alpha=0.5, color=color, label=label, rasterized=True)

    ax.set_title("Mid-Atlantic Hourly Load by Season (PEPCO, ME, PE, PS, JC, DPLCO, DOM)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Load (MW)")
    ax.grid(True, alpha=0.2)
    ax.legend(title="Season", loc="upper right", markerscale=10)

    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=150)
    print(f"saved to {OUTPUT_FILE}")
    plt.show()


if __name__ == "__main__":
    main()