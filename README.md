# Autograd Engine
git add README.md
A reverse-mode automatic differentiation engine built from scratch, designed to explore the core mechanics of backpropagation, computation graphs, and neural network training.

---

## Overview

This project implements a minimal PyTorch-like autograd system with:

* Dynamic computation graph construction
* Reverse-mode automatic differentiation
* Custom neural network modules (MLP)
* Unit testing for correctness
* Performance benchmarking of backward pass

The goal is to understand how modern ML frameworks handle gradients at a systems level.

---

## Features

* Scalar reverse-mode autodiff
* Backpropagation from first principles
* Support for operations:

  * Addition, subtraction, multiplication, division
  * Power
  * `tanh`, `exp`
* Neural network components:

  * Neuron
  * Layer
  * Multi-Layer Perceptron (MLP)
* Unit tests (7 passing)
* Benchmarking of backward pass performance

---

## Example Usage

```python
from autograd.engine import Value

a = Value(2.0)
b = Value(3.0)

d = a * b + a
d.backward()

print("a.grad =", a.grad)  # 4.0
print("b.grad =", b.grad)  # 2.0
```

---

## Training a Neural Network

```bash
PYTHONPATH=. python examples/train_mlp.py
```

Example output:

```
epoch 0, loss = 0.90
epoch 5, loss = 0.83
epoch 10, loss = 0.81
epoch 19, loss = 0.76
```

---

## Gradient Checking

This project validates autograd gradients against numerical finite-difference gradients.

```bash
PYTHONPATH=. python scripts/grad_check.py
```

Output:

```
numerical_grad = 7.000000000090267
autograd_grad  = 7.0
error          = 9.026734915096313e-11
```

## Benchmark

Run:

```bash
PYTHONPATH=. python benchmarks/benchmark_scalar.py
```

Results:

```
graph_size=50,  backward_time=0.000072s
graph_size=100, backward_time=0.000167s
graph_size=200, backward_time=0.000370s
graph_size=400, backward_time=0.001140s
```

### Observations

* Backward pass time increases with computation graph size
* Recursive graph traversal introduces a recursion depth limitation
* Larger graphs require iterative traversal for scalability

---

## Project Structure

```
autograd/            # core autograd engine
  ├── engine.py
  ├── nn.py
tests/               # unit tests
examples/            # usage examples
benchmarks/          # performance benchmarks
results/             # benchmark outputs
```

---

## Testing

```bash
PYTHONPATH=. pytest -q
```

Output:

```
7 passed
```

---

## Engineering Notes

* Implements topological sorting for backpropagation
* Uses dynamic graph construction (define-by-run)
* Gradients are accumulated during backward pass
* Identified recursion limits in deep computation graphs

---

## Future Work

* Iterative (non-recursive) graph traversal
* Tensor support (beyond scalars)
* Vectorized operations
* GPU / parallel computation experiments
* Distributed gradient aggregation (all-reduce simulation)

---

## Tech Stack

* Python
* PyTest

---

## Author

Omprakash Sahani
Machine Learning Systems Engineer (Early Career)
