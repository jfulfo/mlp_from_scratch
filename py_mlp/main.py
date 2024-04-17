import pandas as pd
import numpy as np
import pickle
from mlp import MultiLayerPerceptron
from optimizer import StochasticGradientDescent
from layer import Layer
from loss import CrossEntropy
from activation import ReLU, Softmax
from initializer import HeNormal

# Load the data, targets is the first column
mnist_train = pd.read_csv("../mnist_train.csv", header=None)
inputs_train = mnist_train.iloc[:, 1:].values
targets_raw = mnist_train.iloc[:, 0].values
targets_train = np.zeros((len(targets_raw), 10))
for i in range(len(targets_raw)): targets_train[i][targets_raw[i]] = 1
inputs_train = inputs_train / 255

mlp = MultiLayerPerceptron(CrossEntropy())
mlp.add_layer(Layer((784, 128), ReLU(), HeNormal()))
mlp.add_layer(Layer((128, 64), ReLU(), HeNormal()))
mlp.add_layer(Layer((64, 10), Softmax(), HeNormal()))

optimizer = StochasticGradientDescent(mlp, learning_rate=0.01)
optimizer.train(inputs_train, targets_train, validation_split=0.2, epochs=3)

mnist_test = pd.read_csv("../mnist_test.csv", header=None)
inputs_test = mnist_test.iloc[:, 1:].values
targets_raw = mnist_test.iloc[:, 0].values
targets_test = np.zeros((len(targets_raw), 10))
for i in range(len(targets_raw)): targets_test[i][targets_raw[i]] = 1
inputs_test = inputs_test / 255

optimizer.test(inputs_test, targets_test)

with open("model.pkl", "wb") as f:
    f.write(pickle.dumps(mlp))