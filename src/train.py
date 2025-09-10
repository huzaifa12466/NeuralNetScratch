import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from .model import initialize_parameters, update_parameters

def train_nn(X_train, y_train, layer_dims, epochs=10, eta=0.001):
    # Ensure plots folder exists
    if not os.path.exists("plots"):
        os.makedirs("plots")

    parameters = initialize_parameters(layer_dims)
    Loss = []

    for i in range(epochs):
        # Forward propagation
        Z1 = np.dot(parameters['W1'], X_train) + parameters['b1']
        A1 = np.maximum(0, Z1)  # ReLU
        Z2 = np.dot(parameters['W2'], A1) + parameters['b2']
        y_hat = 1 / (1 + np.exp(-Z2))  # Sigmoid

        t = i + 1
        parameters = update_parameters(parameters, y_train, y_hat, Z1, A1, X_train, eta, t)

        # Loss calculation
        y_hat_clipped = np.clip(y_hat, 1e-15, 1 - 1e-15)
        loss = -np.mean(y_train * np.log(y_hat_clipped) + (1 - y_train) * np.log(1 - y_hat_clipped))
        Loss.append(loss)
        print(f"Epoch {i+1}/{epochs}: Loss = {loss}")

    # Plot training loss curve
    plt.figure(figsize=(8,5))
    plt.plot(range(1, epochs+1), Loss, marker='o')
    plt.title("Training Loss Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig("plots/loss_curve.png")
    plt.show()

    # Decision boundary plot
    x_min, x_max = X_train[0, :].min() - 0.5, X_train[0, :].max() + 0.5
    y_min, y_max = X_train[1, :].min() - 0.5, X_train[1, :].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))

    # Forward pass on meshgrid
    Z1_mesh = np.dot(parameters['W1'], np.c_[xx.ravel(), yy.ravel()].T) + parameters['b1']
    A1_mesh = np.maximum(0, Z1_mesh)
    Z2_mesh = np.dot(parameters['W2'], A1_mesh) + parameters['b2']
    y_pred_mesh = 1 / (1 + np.exp(-Z2_mesh))
    y_pred_mesh = (y_pred_mesh > 0.5).astype(int).reshape(xx.shape)

    plt.figure(figsize=(8,5))
    plt.contourf(xx, yy, y_pred_mesh, alpha=0.8, cmap=ListedColormap(('red','blue')))
    plt.scatter(X_train[0, :], X_train[1, :], c=y_train.ravel(), edgecolor='k', s=20)
    plt.title("Decision Boundary")
    plt.savefig("plots/decision_boundary.png")  # Save BEFORE show
    plt.show()

    return parameters, Loss
