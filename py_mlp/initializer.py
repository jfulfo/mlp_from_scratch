import numpy as np
from abc import ABC, abstractmethod

class Initializer(ABC):
    @abstractmethod
    def __call__(self, shape):
        pass

class RandomNormal(Initializer):
    def __call__(self, shape):
        return np.random.randn(*shape)

class HeNormal(Initializer):
    def __call__(self, shape):
        return np.random.randn(*shape) * np.sqrt(2 / shape[0])