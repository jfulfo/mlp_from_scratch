import numpy as np
from abc import ABC, abstractmethod

class LossFunction(ABC):
    @abstractmethod
    def __call__(self, target: np.ndarray, output: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def gradient(self, target: np.ndarray, output: np.ndarray) -> np.ndarray:
        pass

class CrossEntropy(LossFunction):
    def __call__(self, target: np.ndarray, output: np.ndarray) -> np.ndarray:
        return -np.sum(target * np.log(output))

    def gradient(self, target: np.ndarray, output: np.ndarray) -> np.ndarray:
        return output - target

class MeanSquaredError(LossFunction):
    def __call__(self, target: np.ndarray, output: np.ndarray) -> np.ndarray:
        return np.sum((target - output) ** 2)

    def gradient(self, target: np.ndarray, output: np.ndarray) -> np.ndarray:
        return -2 * (target - output)

