import numpy as np
class Value:
    def __init__(self, data, _children=(), _op="",label=''):
        self.data = np.array(data, dtype=float)
        
        self._prev = set(_children)
        self._op = _op
        self.label = label
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None

    @staticmethod 
    def unbroadcast(grad, target_shape):
        while len(grad.shape) > len(target_shape):
            grad = grad.sum(axis=0)
        for i, dim in enumerate(target_shape):
            if dim == 1:
                grad = grad.sum(axis=i, keepdims=True)
        return grad
    
    def __repr__(self):
        return f"Value(data={self.data},\n grad={self.grad})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += Value.unbroadcast(1.0 * out.grad, self.grad.shape)
            other.grad += Value.unbroadcast(1.0 * out.grad, other.grad.shape)

        out._backward = _backward
        return out
        
    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += Value.unbroadcast(other.data * out.grad, self.grad.shape)
            other.grad += Value.unbroadcast(self.data * out.grad, other.grad.shape)

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data / other.data, (self, other), '/', )

        def _backward():
            
            self.grad += Value.unbroadcast(out.grad / other.data, self.grad.shape)

            raw = out.grad * (- self.data / other.data**2)
            other.grad += Value.unbroadcast(raw, other.grad.shape)

        out._backward = _backward
        return out
    

    def __rtruediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return other / self
        
    def __sub__(self, other):
        return self + (-other)
        
    def __rsub__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return other - self 
        
    def __neg__(self):
        return self * -1
    
    def __pow__(self, other):
        assert isinstance(other, (float, int))
        out = Value(self.data ** other, (self, ), f"**{other}")

        def _backward():
            self.grad += other * self.data**(other - 1) * out.grad
        out._backward = _backward
        return out

    def __matmul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data @ other.data, (self, other), "@")

        def _backward():
            # self.grad += Value.unbroadcast(out.grad @ other.data.T, self.grad.shape)
            # other.grad += Value.unbroadcast(self.data.T @ out.grad, other.grad.shape)
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad

        out._backward = _backward
        return out

    @property
    def T(self):
        out = Value(self.data.T, (self, ), 'T')

        def _backward():
            self.grad += out.grad.T

        out._backward = _backward
        return out
        
    def sum(self, axis=None, keepdims=False):
        out = Value(self.data.sum(axis=axis, keepdims=keepdims), (self, ), 'sum')

        def _backward():
            grad = out.grad
            if axis is not None:
                axes = axis if isinstance(axis, tuple) else (axis,)

                if not keepdims:
                    for ax in sorted(axes):
                        grad = np.expand_dims(grad, ax)
            self.grad += np.ones_like(self.data) * grad
        out._backward = _backward
        return out

    def mean(self):
        n = self.data.size
        out = Value(np.mean(self.data), (self, ), 'mean')

        def _backward():
            self.grad += np.ones_like(self.data) * (out.grad/n)
        out._backward = _backward
        return out

    
    def tanh(self):
        ex = self.exp()
        enx = (-self).exp()
        return (ex - enx) / (ex + enx)

    def sigmoid(self):
        enx = (-self).exp()
        return 1 / (1 + enx)

    def relu(self):
        z = self.data
        relu_z = np.maximum(0, z)
        out = Value(relu_z, _children = (self, ), _op='relu')

        def _backward():
            self.grad += (self.data>0) * out.grad

        out._backward = _backward
        return out

    def softmax(self):
        max_val = self.data.max(axis=1, keepdims=True)
        shifted = self - max_val
        exps = shifted.exp()
        sum_exps = exps.sum(axis=1, keepdims=True)
        return exps / sum_exps
        
    def exp(self):
        out = Value(np.exp(self.data), _children = (self, ), _op = "exp")

        def _backward():
            self.grad += out.data * out.grad
            
        out._backward = _backward
        return out

    def log(self):
        out = Value(np.log(self.data), _children=(self,), _op='log')

        def _backward():
            self.grad += (1/self.data) * out.grad
        out._backward = _backward
        return out
    
    def backward(self):
        sorted_nodes = []
        visited_nodes = set()

        def topological_sort(v):
            if v not in visited_nodes:
                visited_nodes.add(v)
                for child in v._prev:
                    topological_sort(child)
                sorted_nodes.append(v)
        topological_sort(self)
        self.grad = np.ones_like(self.data)
        for node in reversed(sorted_nodes):
            node._backward()