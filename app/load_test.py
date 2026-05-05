import argparse
import csv
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

BASE_URL = "http://localhost:8000"
REQUESTS_PER_LEVEL = 200

# fixed weather/date so the service doesn't hit the external API
PARAMS = {
    "date": "2020-06-15 14:00",
    "temp_f": 85.0,
    "feels_like_f": 90.0,
    "humidity_pct": 65.0,
}


def one_request(url):
    t0 = time.perf_counter()
    try:
        r = requests.get(f"{url}/predict", params=PARAMS, timeout=10)
        return (time.perf_counter() - t0) * 1000, r.status_code == 200
    except Exception:
        return (time.perf_counter() - t0) * 1000, False


def run_level(url, concurrency, n):
    latencies = []
    errors = 0
    t_start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(one_request, url) for _ in range(n)]
        for f in as_completed(futures):
            lat, ok = f.result()
            latencies.append(lat)
            if not ok:
                errors += 1

    elapsed = time.perf_counter() - t_start
    latencies.sort()

    def pct(p):
        return latencies[min(int(len(latencies) * p), len(latencies) - 1)]

    return {
        "concurrency": concurrency,
        "requests": n,
        "rps": round(n / elapsed, 1),
        "errors": errors,
        "p50_ms": round(statistics.median(latencies), 1),
        "p95_ms": round(pct(0.95), 1),
        "p99_ms": round(pct(0.99), 1),
        "elapsed_s": round(elapsed, 2),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=BASE_URL)
    parser.add_argument("--requests", type=int, default=REQUESTS_PER_LEVEL)
    args = parser.parse_args()

    try:
        resp = requests.get(f"{args.url}/health", timeout=5)
        if resp.status_code != 200 or not resp.json().get("model_loaded"):
            print(f"service not ready at {args.url}")
            return
    except Exception:
        print(f"can't reach {args.url}")
        return

    print(f"load testing {args.url}/predict  ({args.requests} req per concurrency level)\n")
    header = f"{'concurrency':>12}  {'rps':>8}  {'p50 ms':>8}  {'p95 ms':>8}  {'p99 ms':>8}  {'errors':>7}"
    print(header)
    print("-" * len(header))

    results = []
    for c in [1, 5, 10, 20, 50]:
        r = run_level(args.url, c, args.requests)
        results.append(r)
        print(f"{c:>12}  {r['rps']:>8.1f}  {r['p50_ms']:>8.1f}  {r['p95_ms']:>8.1f}  {r['p99_ms']:>8.1f}  {r['errors']:>7}")

    os.makedirs("results", exist_ok=True)
    out = "results/load_test_results.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nsaved to {out}")


if __name__ == "__main__":
    main()
