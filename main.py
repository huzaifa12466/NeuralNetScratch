import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from src.train import train_nn

# Dataset
x, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
X_train, X_test = X_train.T, X_test.T
y_train, y_test = y_train.reshape(1, -1).astype(float), y_test.reshape(1, -1).astype(float)

# Train
layer_dims = [2, 2, 1]
parameters, Loss = train_nn(X_train, y_train, layer_dims, epochs=10, eta=0.001)
