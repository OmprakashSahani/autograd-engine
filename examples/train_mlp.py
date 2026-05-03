from autograd.nn import MLP
from autograd.engine import Value

# simple dataset: XOR-like
xs = [
    [Value(0.0), Value(0.0)],
    [Value(0.0), Value(1.0)],
    [Value(1.0), Value(0.0)],
    [Value(1.0), Value(1.0)],
]

ys = [Value(0.0), Value(1.0), Value(1.0), Value(0.0)]

model = MLP(2, [4, 1])

for epoch in range(20):
    ypred = [model(x)[0] for x in xs]

    loss = sum((yout - ygt) ** 2 for yout, ygt in zip(ypred, ys))

    # zero gradients
    for p in model.parameters():
        p.grad = 0.0

    # backward
    loss.backward()

    # update
    for p in model.parameters():
        p.data += -0.05 * p.grad

    print(f"epoch {epoch}, loss = {loss.data}")
