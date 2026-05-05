import pandas as pd

df = pd.read_csv("results/MLOps Metric Sample1.csv")

summary = df.groupby("metric")["value"].agg(["mean", "max"])

def convert(metric, val):
    if "memory_bytes" in metric:
        return round(val / 1e6, 2), "MB"
    if "latency" in metric:
        return round(val * 1000, 2), "ms"
    if "cpu_cores" in metric:
        return round(val, 4), "cores"
    return round(val, 4), ""

rows = []
for metric, row in summary.iterrows():
    avg_val, unit = convert(metric, row["mean"])
    max_val, _ = convert(metric, row["max"])
    rows.append({"Metric": metric, "Unit": unit, "Average": avg_val, "Max": max_val})

out = pd.DataFrame(rows).set_index("Metric")
print(out.to_string())
