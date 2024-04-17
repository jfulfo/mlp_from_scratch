import numpy as np
from loss import LossFunction
from layer import Layer
from typing import List, Tuple

class MultiLayerPerceptron:
    layers: List[Layer]
    loss: LossFunction

    def __init__(self, loss: LossFunction):
        self.layers = []
        self.loss = loss

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.predict(x)

    def add_layer(self, layer: Layer):
        self.layers.append(layer)

    def predict(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer(x)
        return x

    def forward(self, x: np.ndarray) -> np.ndarray:
        return [x := layer(x) for layer in self.layers]

    def backward(self, layer_outputs: List[np.ndarray], input: np.ndarray, target: np.ndarray) -> List[np.ndarray]:
        """
        Returns the gradients back to the optimizer
        """
        deltas = [self.loss.gradient(target, layer_outputs[-1])]
        gradients = []
        for i in range(len(self.layers) - 1, -1, -1):
            layer_input = layer_outputs[i - 1] if i > 0 else input
            next_weights = self.layers[i + 1].weights if i < len(self.layers) - 1 else None
            new_deltas, new_gradients = self.layers[i].backward(layer_input, layer_outputs[i], deltas[-1], next_weights)
            deltas.append(new_deltas)
            gradients.append(new_gradients)
        return deltas[::-1], gradients[::-1]

    def detect_nan(self):
        for layer in self.layers:
            if np.isnan(layer.weights).any() or np.isnan(layer.biases).any():
                raise ValueError("NaN detected in weights or biases")