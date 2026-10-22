import random

from .engine import Value


class Neuron:
    def __init__(self, nin, nonlin=True):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0.0)
        self.nonlin = nonlin

    def __call__(self, x):
        if len(x) != len(self.w):
            raise ValueError(f"期望 {len(self.w)} 个输入，实际收到 {len(x)} 个")

        res = sum([(self.w[i] * (xi if isinstance(xi, Value) else Value(xi)) ) for i, xi in enumerate(x)], self.b)
        return res.relu() if self.nonlin else res

    def parameters(self):
        return [*self.w, self.b]


class Layer:
    def __init__(self, nin, nout, nonlin=True):
        self.neurons = [Neuron(nin, nonlin=nonlin) for _ in range(nout)]

    def __call__(self, x):
        return [n(x) for n in self.neurons]

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


class MLP:
    def __init__(self, nin, data):
        inputs = [nin, *data]
        self.layers = [Layer(inputs[i], d, nonlin=(i < len(data) - 1)) for i, d in enumerate(data)]

    def __call__(self, x):
        result = x
        for l in self.layers:
            result = l(result)
        return result

    def parameters(self):
        return [p for l in self.layers for p in l.parameters()]
