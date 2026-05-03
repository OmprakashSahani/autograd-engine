from autograd.engine import Value
import math


def test_addition_backward():
    a = Value(2.0)
    b = Value(3.0)

    c = a + b
    c.backward()

    assert c.data == 5.0
    assert a.grad == 1.0
    assert b.grad == 1.0


def test_multiplication_backward():
    a = Value(2.0)
    b = Value(3.0)

    c = a * b
    c.backward()

    assert c.data == 6.0
    assert a.grad == 3.0
    assert b.grad == 2.0


def test_composed_expression_backward():
    a = Value(2.0)
    b = Value(3.0)

    d = a * b + a
    d.backward()

    assert d.data == 8.0
    assert a.grad == 4.0
    assert b.grad == 2.0


def test_subtraction_backward():
    a = Value(5.0)
    b = Value(3.0)

    c = a - b
    c.backward()

    assert c.data == 2.0
    assert a.grad == 1.0
    assert b.grad == -1.0


def test_power_backward():
    a = Value(3.0)

    b = a ** 2
    b.backward()

    assert b.data == 9.0
    assert a.grad == 6.0


def test_tanh_backward():
    a = Value(2.0)

    b = a.tanh()
    b.backward()

    expected_grad = 1 - math.tanh(2.0) ** 2

    assert abs(b.data - math.tanh(2.0)) < 1e-9
    assert abs(a.grad - expected_grad) < 1e-9


def test_exp_backward():
    a = Value(2.0)

    b = a.exp()
    b.backward()

    assert abs(b.data - math.exp(2.0)) < 1e-9
    assert abs(a.grad - math.exp(2.0)) < 1e-9