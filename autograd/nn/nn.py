import numpy as np
from engine.engine import Value


class Layer:
    def __init__(self, nin, nout, activation='tanh'):
        self.w = Value(np.random.uniform(-1, 1, size=(nin, nout)))
        self.b = Value(np.random.uniform(-1, 1, size=(nout, )))
        self.activation = activation

    def parameters(self):
        return [self.w, self.b]

    def __call__(self, x):
        x = x if isinstance(x, Value) else Value(x)
        z = x @ self.w + self.b
        
        if self.activation == 'tanh':
            return z.tanh()
        elif self.activation == 'sigmoid':
            return z.sigmoid()
        elif self.activation == 'relu':
            return z.relu()
        elif self.activation == 'softmax':
            return z.softmax()
        elif self.activation == 'linear':
            return z
        else:
            raise ValueError("Activation function is not supported.")
    
class MLP:
    def __init__(self, nin, nouts,activations):
        """
        nin: number of input features
        nouts: list like [10,20,1]. length represents
        the number of layers in the network and value
        represents the number of output from each layer
        activations: list like ['tanh', 'relu', 'softmax']
        representing activation function to each layer
        """
        self.layers = []

        assert len(nouts) == len(activations)
        size = [nin] + nouts
        for i in range(len(nouts)):
            self.layers.append(
                Layer(size[i], size[i+1], activations[i])
        )

    def parameters(self):
        params = []
        for layer in  self.layers:
            params.extend(layer.parameters())
        return params

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x