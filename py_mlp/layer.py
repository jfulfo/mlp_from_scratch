import numpy as np
from typing import List, Tuple
from activation import ActivationFunction
from initializer import Initializer

class Layer:
    weights: np.ndarray
    biases: np.ndarray
    activation: ActivationFunction

    def __init__(self, shape: Tuple[int], activation: ActivationFunction, initializer: Initializer):
        self.weights = initializer(shape)
        self.biases = np.zeros(shape[1])
        self.activation = activation

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.activation(np.dot(x, self.weights) + self.biases)

    def backward(self, input: np.ndarray, output: np.ndarray, next_deltas: np.ndarray, next_weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if next_weights is None: # if output layer
            deltas = next_deltas * self.activation.gradient(output)
        else:
            deltas = np.dot(next_deltas, next_weights.T) * self.activation.gradient(output)
        gradients = np.outer(input, deltas)
        return deltas, gradients
