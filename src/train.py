import numpy as np
import matplotlib.pyplot as plt
from .model import initialize_parameters, update_parameters

def train_nn(X_train, y_train, layer_dims, epochs=10, eta=0.001):
    parameters = initialize_parameters(layer_dims)
    Loss = []

    for i in range(epochs):
        # Forward prop
        Z1 = np.dot(parameters['W1'], X_train) + parameters['b1']
        A1 = np.maximum(0, Z1)
        Z2 = np.dot(parameters['W2'], A1) + parameters['b2']
        y_hat = 1 / (1 + np.exp(-Z2))

        t = i + 1
        parameters = update_parameters(parameters, y_train, y_hat, Z1, A1, X_train, eta, t)

        y_hat_clipped = np.clip(y_hat, 1e-15, 1 - 1e-15)
        loss = -np.mean(y_train * np.log(y_hat_clipped) + (1 - y_train) * np.log(1 - y_hat_clipped))
        Loss.append(loss)
        print(f"Epoch {i+1}/{epochs}: Loss = {loss}")

    # Plot
    plt.figure(figsize=(8,5))
    plt.plot(range(1, epochs+1), Loss, marker='o')
    plt.title("Training Loss Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig("plots/loss_curve.png")
    plt.show()

    return parameters, Loss
