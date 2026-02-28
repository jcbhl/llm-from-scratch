import matplotlib

matplotlib.use("qtagg")

import time

import matplotlib.pyplot as plt
import numpy as np
import torch


def matmul(size, device):
    start = time.time()
    A = torch.randn(size=(size, size), device=device)
    B = torch.randn(size=(size, size), device=device)

    A @ B
    end = time.time()
    return end - start


def run_comparison():
    print(f"Torch version: {torch.__version__}")
    print(f"GPU status: {torch.cuda.is_available()}")

    sizes = np.logspace(2, 3.8, dtype=np.int64)
    cpu_times = []
    gpu_times = []
    cpu = torch.device("cpu")
    gpu = torch.device("cuda")
    for size in sizes:
        print(size)
        cpu_time = matmul(size, cpu)
        gpu_time = matmul(size, gpu)
        cpu_times.append(cpu_time)
        gpu_times.append(gpu_time)
        print(f"CPU time: {cpu_time}\nGPU time: {gpu_time}")

    plt.plot(sizes, cpu_times, label="CPU")
    plt.plot(sizes, gpu_times, label="GPU")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    run_comparison()
