import json
import time
import urllib.request
import statistics


URL = "http://127.0.0.1:8000/predict"


payload = {
    "Time": 406,
    "V1": -1.36,
    "V2": -0.07,
    "V3": 2.54,
    "V4": 1.38,
    "V5": -0.34,
    "V6": 0.46,
    "V7": -0.11,
    "V8": 0.10,
    "V9": 0.32,
    "V10": -0.17,
    "V11": 1.02,
    "V12": -0.43,
    "V13": 0.12,
    "V14": -0.29,
    "V15": 0.63,
    "V16": -0.12,
    "V17": 0.25,
    "V18": -0.08,
    "V19": 0.11,
    "V20": 0.04,
    "V21": -0.02,
    "V22": 0.07,
    "V23": -0.01,
    "V24": 0.02,
    "V25": 0.03,
    "V26": -0.04,
    "V27": 0.01,
    "V28": 0.02,
    "Amount": 149.62,
    "Hour_Sin": 0.0,
    "Hour_Cos": 1.0,
    "Day": 0,
    "Amount_Log": 5.014,
    "Amount_Rolling_Mean_10": 149.62,
    "Amount_Rolling_Std_10": 0.0,
    "Amount_Deviation": 0.0,
    "Time_Since_Previous": 0.0,
    "Previous_1H_Transaction_Count": 0
}


data = json.dumps(payload).encode("utf-8")


def send_request():

    request = urllib.request.Request(
        URL,
        data=data,
        headers={
            "Content-Type": "application/json"
        },
        method="POST"
    )

    start = time.perf_counter()

    with urllib.request.urlopen(request) as response:
        response.read()

    end = time.perf_counter()

    return (end - start) * 1000


print("=" * 70)
print("ShieldML - API Latency Benchmark")
print("=" * 70)

print("\nWarming up API...")

for _ in range(10):
    send_request()

print("Warm-up completed.")

print("\nRunning 100 prediction requests...")

latencies = []

for i in range(100):

    latency = send_request()

    latencies.append(latency)

    if (i + 1) % 20 == 0:
        print(
            f"Completed {i + 1}/100 requests"
        )


latencies_sorted = sorted(latencies)

average = statistics.mean(latencies)
minimum = min(latencies)
maximum = max(latencies)

p95_index = int(
    0.95 * len(latencies_sorted)
) - 1

p95 = latencies_sorted[p95_index]


print("\n" + "=" * 70)
print("LATENCY RESULTS")
print("=" * 70)

print(
    f"Average latency : {average:.2f} ms"
)

print(
    f"p95 latency     : {p95:.2f} ms"
)

print(
    f"Minimum latency : {minimum:.2f} ms"
)

print(
    f"Maximum latency : {maximum:.2f} ms"
)

print("\nTarget: < 50 ms")

if p95 < 50:
    print("STATUS: PASS - p95 latency is below 50 ms.")
else:
    print("STATUS: NEEDS OPTIMIZATION - p95 exceeds 50 ms.")


# ------------------------------------------------------------
# Save benchmark report
# ------------------------------------------------------------

with open(
    "reports/latency_benchmark.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "ShieldML API Latency Benchmark\n"
    )

    file.write(
        "=" * 50 + "\n\n"
    )

    file.write(
        "Requests: 100\n"
    )

    file.write(
        f"Average latency: {average:.2f} ms\n"
    )

    file.write(
        f"p95 latency: {p95:.2f} ms\n"
    )

    file.write(
        f"Minimum latency: {minimum:.2f} ms\n"
    )

    file.write(
        f"Maximum latency: {maximum:.2f} ms\n"
    )

    file.write(
        "Target: < 50 ms\n"
    )

    file.write(
        f"Status: "
        f"{'PASS' if p95 < 50 else 'NEEDS OPTIMIZATION'}\n"
    )

print(
    "\nSaved: reports/latency_benchmark.txt"
)

print("\nBenchmark completed.")