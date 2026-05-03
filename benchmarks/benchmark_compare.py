import time
import torch

from autograd.engine import Value


def run_my_engine(n):
    x = Value(1.0)

    for _ in range(n):
        x = x * 1.0001 + 0.0001

    start = time.perf_counter()
    x.backward()
    end = time.perf_counter()

    return end - start


def run_pytorch(n):
    x = torch.tensor(1.0, requires_grad=True)

    for _ in range(n):
        x = x * 1.0001 + 0.0001

    start = time.perf_counter()
    x.backward()
    end = time.perf_counter()

    return end - start


if __name__ == "__main__":
    for n in [50, 100, 200, 400]:
        my_time = run_my_engine(n)
        torch_time = run_pytorch(n)

        print(
            f"graph_size={n}, "
            f"my_engine={my_time:.6f}s, "
            f"pytorch={torch_time:.6f}s, "
            f"ratio={my_time / torch_time:.2f}x"
        )
