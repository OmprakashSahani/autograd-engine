import matplotlib.pyplot as plt

graph_sizes = [50, 100, 200, 400]
backward_times = [0.000072, 0.000167, 0.000370, 0.001140]

plt.figure(figsize=(7, 4))
plt.plot(graph_sizes, backward_times, marker="o")
plt.xlabel("Graph Size")
plt.ylabel("Backward Time (seconds)")
plt.title("Backward Pass Time vs Graph Size")
plt.grid(True)

plt.savefig("results/backward_benchmark.png", dpi=150, bbox_inches="tight")
print("Saved graph to results/backward_benchmark.png")
