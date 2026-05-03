from autograd.engine import Value

a = Value(2.0)
b = Value(3.0)

c = a * b
d = c + a

d.backward()

print("d =", d)
print("a =", a)
print("b =", b)