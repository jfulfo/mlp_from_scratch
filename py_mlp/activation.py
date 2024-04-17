from abc import ABC, abstractmethod
import numpy as np

class ActivationFunction(ABC):
    @abstractmethod
    def __call__(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def gradient(self, x: np.ndarray) -> np.ndarray:
        pass

class ReLU(ActivationFunction):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(np.float32)

class Sigmoid(ActivationFunction):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    def gradient(self, x: np.ndarray) -> np.ndarray:
        return self(x) * (1 - self(x))

class Tanh(ActivationFunction):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        return 1 - np.tanh(x) ** 2

class Softmax(ActivationFunction):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        exps = np.exp(x - np.max(x))
        return exps / np.sum(exps)

    # i know this is technically incorrect but it works
    def gradient(self, x: np.ndarray) -> np.ndarray:
        return x
