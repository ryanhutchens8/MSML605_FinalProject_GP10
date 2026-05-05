from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_PATH = BASE_DIR / "results" / "model_comparison_results.csv"
PLOTS_DIR = BASE_DIR / "results" / "plots"


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RESULTS_PATH)
    df["datetime"] = pd.to_datetime(df["datetime"])

    df = df.dropna(subset=["static_rolling_mae", "dynamic_rolling_mae", "inline_rolling_mae"])

    plt.figure(figsize=(12, 6))
    plt.plot(df["datetime"], df["static_rolling_mae"], label="Static RF")
    plt.plot(df["datetime"], df["inline_rolling_mae"], label="Inline Retrain RF")
    plt.plot(df["datetime"], df["dynamic_rolling_mae"], label="Dynamic RF")
    plt.axhline(2000, linestyle="--", color="red", label="Drift Threshold")
    plt.xlabel("Datetime")
    plt.ylabel("Rolling MAE (MW)")
    plt.title("Rolling MAE Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "rolling_mae_comparison.png", dpi=300)
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(df["datetime"], df["static_error"], label="Static RF")
    plt.plot(df["datetime"], df["inline_error"], label="Inline Retrain RF")
    plt.plot(df["datetime"], df["dynamic_error"], label="Dynamic RF")
    plt.xlabel("Datetime")
    plt.ylabel("Absolute Error (MW)")
    plt.title("Prediction Error Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "prediction_error_comparison.png", dpi=300)
    plt.close()

    final_static_mae = df["static_error"].mean()
    final_dynamic_mae = df["dynamic_error"].mean()
    final_inline_mae = df["inline_error"].mean()

    print("Plots saved to:", PLOTS_DIR)
    print(f"Final Static MAE:  {final_static_mae:.2f} MW")
    print(f"Final Inline MAE:  {final_inline_mae:.2f} MW")
    print(f"Final Dynamic MAE: {final_dynamic_mae:.2f} MW")


if __name__ == "__main__":
    main()
