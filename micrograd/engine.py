import math

if __package__:
    from .draw import draw_dot
else:
    from draw import draw_dot


class Value:
    def __init__(self, data, _children=(), _op='', label = ''):
        self.data = data
        self._prev = set(_children)
        self._op = _op
        self.label = label
        self.grad = 0.0
        self._backward = lambda: None

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), '+')
        
        def _backward():
            self.grad += out.grad * 1.0
            other.grad += out.grad * 1.0
        out._backward = _backward
        return out

    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data
        out._backward = _backward    
        return out 

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), 'tanh')
        
        def _backward():
            self.grad += (1 - t**2)*out.grad

        out._backward = _backward
        return out
        
    def __neg__(self):
        out = Value(-self.data, (self, ), '-')

        def _backward():
            self.grad += -out.grad

        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)

                for child in v._prev:
                    build_topo(child)

                topo.append(v)
                
        build_topo(self)

        self.grad = 1.0

        for node in reversed(topo):
            node._backward()
    
if __name__ == '__main__':
    a = Value(1.0, label='a')
    b = Value(2.0, label='b')
    c = Value(3.0, label='c')

    d = a + b; d.label = 'd'
    e = d * c; e.label = 'e'

    f = Value(0.1, label='f')
    L = e * f; L.label = 'L'

    O = L.tanh(); O.label = 'O'

    # 先反向传播，再绘图，图中才会显示计算后的梯度。
    O.backward()
    dot = draw_dot(O)
    # 相对于运行命令时的目录保存，cleanup=True 删除中间 DOT 文件。
    output_path = dot.render(filename='computation_graph', directory='outputs',
                             format='svg', cleanup=True)
    print(f'计算图已导出到：{output_path}')
