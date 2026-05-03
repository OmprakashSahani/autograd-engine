import matplotlib.pyplot as plt

graph_sizes = [100, 1000, 5000, 10000]
backward_times = [0.000323, 0.002406, 0.010410, 0.115016]

plt.figure(figsize=(7, 4))
plt.plot(graph_sizes, backward_times, marker="o")
plt.xlabel("Graph Size")
plt.ylabel("Backward Time (seconds)")
plt.title("Backward Pass Time vs Graph Size (Iterative Traversal)")
plt.grid(True)

plt.savefig("results/backward_benchmark.png", dpi=150, bbox_inches="tight")
print("Saved graph to results/backward_benchmark.png")
