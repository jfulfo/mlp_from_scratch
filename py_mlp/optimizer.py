import numpy as np
from abc import ABC, abstractmethod
from mlp import MultiLayerPerceptron

class Optimizer(ABC):
    @abstractmethod
    def train(self, inputs: np.ndarray, targets: np.ndarray, validatoin_split: float, epochs: int) -> float:
        pass

    def evaluate(self, input: np.ndarray, target: np.ndarray) -> float:
        return self.mlp.loss(target, self.mlp.predict(input))
    
    def test(self, inputs: np.ndarray, targets: np.ndarray) -> float:
        accuracy = sum([np.argmax(self.mlp.predict(x)) == np.argmax(y) for x, y in zip(inputs, targets)]) / len(inputs)
        print(f"\nAccuracy: {round(accuracy * 100, 5)}%")

    def print_progress(self, epoch: int, n: int, total: int):
        print(f"Epoch {epoch + 1} - {n}/{total}", end='\r')


class StochasticGradientDescent(Optimizer):
    mlp: MultiLayerPerceptron
    learning_rate: float

    def __init__(self, mlp: MultiLayerPerceptron, learning_rate: float):
        self.mlp = mlp
        self.learning_rate = learning_rate

    def shuffle(self, inputs: np.ndarray, targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pass

    def train(self, inputs: np.ndarray, targets: np.ndarray, validation_split: float, epochs: int, learning_rate_decay: float = 0.95) -> float:
        validation_inputs = inputs[:int(len(inputs) * validation_split)]
        validation_outputs = targets[:int(len(targets) * validation_split)]
        inputs = inputs[int(len(inputs) * validation_split):]
        targets = targets[int(len(targets) * validation_split):]
        for epoch in range(epochs):
            count = 0
            for x, y in zip(inputs, targets):
                predictions = self.mlp.forward(x)
                deltas, grads = self.mlp.backward(predictions, x, y)
                for i in range(len(self.mlp.layers)):
                    self.mlp.layers[i].weights -= self.learning_rate * grads[i]
                for i in range(len(self.mlp.layers)):
                    for j in range(len(self.mlp.layers[i].biases)):
                        self.mlp.layers[i].biases[j] -= self.learning_rate * deltas[i][j]
                count += 1
                self.print_progress(epoch, count, len(inputs))
            if validation_split != 0.0: self.test(validation_inputs, validation_outputs)
            self.learning_rate *= learning_rate_decay
        

class Adam(Optimizer):
    pass