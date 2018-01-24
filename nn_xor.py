from neuralnetwork import NeuralNetwork
import numpy as np

# construct the XOR dataset
X = np.array([[0, 0],[0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# define our 2-2-1 neural network and train it
nn = NeuralNetwork([2, 2, 1], alpha=0.5)
nn.fit(X, y, epochs=20000)

# [INFO] epoch=20000, loss=0.0002594