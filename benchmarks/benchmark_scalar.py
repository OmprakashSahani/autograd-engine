import time
from autograd.engine import Value


def run_workload(n):
    x = Value(1.0)

    for _ in range(n):
        x = x * 1.0001 + 0.0001

    start = time.perf_counter()
    x.backward()
    end = time.perf_counter()

    return end - start


if __name__ == "__main__":
    for n in [50, 100, 200, 400]:
        elapsed = run_workload(n)
        print(f"graph_size={n}, backward_time={elapsed:.6f}s")