from autograd.engine import Value


def trace(root):
    nodes, edges = set(), set()

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)

    build(root)
    return nodes, edges


def to_dot(root):
    nodes, edges = trace(root)
    lines = [
        "digraph G {",
        "rankdir=LR;",
        "node [fontsize=10];",
    ]

    for n in nodes:
        uid = f"node_{id(n)}"
        label = f"data={n.data:.4f}\\ngrad={n.grad:.4f}"
        lines.append(f'{uid} [label="{label}", shape=record];')

        if n._op:
            op_id = f"op_{id(n)}"
            lines.append(f'{op_id} [label="{n._op}", shape=circle];')
            lines.append(f"{op_id} -> {uid};")

    for n1, n2 in edges:
        child_id = f"node_{id(n1)}"
        parent_id = f"node_{id(n2)}"

        if n2._op:
            op_id = f"op_{id(n2)}"
            lines.append(f"{child_id} -> {op_id};")
        else:
            lines.append(f"{child_id} -> {parent_id};")

    lines.append("}")
    return "\n".join(lines)


if __name__ == "__main__":
    a = Value(2.0)
    b = Value(3.0)

    d = a * b + a
    d.backward()

    dot = to_dot(d)

    with open("results/graph.dot", "w") as f:
        f.write(dot)

    print("Saved graph to results/graph.dot")