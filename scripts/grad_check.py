from autograd.engine import Value


def f(x):
    return x * x + x * 3.0  # simple function


def numerical_grad(x, eps=1e-6):
    x1 = Value(x.data + eps)
    x2 = Value(x.data - eps)

    y1 = f(x1)
    y2 = f(x2)

    return (y1.data - y2.data) / (2 * eps)


def autograd_grad(x):
    y = f(x)
    y.backward()
    return x.grad


if __name__ == "__main__":
    x = Value(2.0)

    num_g = numerical_grad(x)
    auto_g = autograd_grad(x)

    print(f"numerical_grad = {num_g}")
    print(f"autograd_grad  = {auto_g}")
    print(f"error          = {abs(num_g - auto_g)}")
