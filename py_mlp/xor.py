import numpy as np
import pickle
from mlp import MultiLayerPerceptron
from optimizer import StochasticGradientDescent
from layer import Layer
from loss import MeanSquaredError
from activation import ReLU, Sigmoid
from initializer import HeNormal

inputs_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
targets_train = np.array([[0], [1], [1], [0]])

mlp = MultiLayerPerceptron(MeanSquaredError())
mlp.add_layer(Layer((2, 3), ReLU(), HeNormal()))
mlp.add_layer(Layer((3, 1), Sigmoid(), HeNormal()))

optimizer = StochasticGradientDescent(mlp, learning_rate=0.1)
optimizer.train(inputs_train, targets_train, validation_split=0, epochs=2000, learning_rate_decay=1.0)

print()
correct = 0
for x, y in zip(inputs_train, targets_train):
    print(f"Input: {x} - Prediction: {round(mlp.predict(x)[0], 5)} - Target: {y[0]}")
    if np.round(mlp.predict(x)[0]) == y[0]: correct += 1
if correct == len(inputs_train):
    with open("xor_model.pkl", "wb") as f:
        f.write(pickle.dumps(mlp))