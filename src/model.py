import numpy as np

def initialize_parameters(layer_dims):
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)

    for i in range(1, L):
        # He initialization for ReLU layers
        parameters['W' + str(i)] = np.random.randn(layer_dims[i], layer_dims[i-1]) * np.sqrt(2.0 / layer_dims[i-1])
        parameters['b' + str(i)] = np.zeros((layer_dims[i], 1))

    return parameters

def update_parameters(parameters, y_train, y_hat, Z1, A1, X, eta=0.001, t=1):
    L = len(parameters) // 2
    m = X.shape[1]

    grads = {}
    # Output layer Sigmoid
    dZ2 = y_hat - y_train
    dW2 = (1/m) * np.dot(dZ2, A1.T)
    db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)

    # Hidden layer RELU
    dA1 = np.dot(parameters['W2'].T, dZ2)
    dZ1 = dA1 * (Z1 > 0)
    dW1 = (1/m) * np.dot(dZ1, X.T)
    db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)

    grads['dW2'] = dW2
    grads['db2'] = db2
    grads['dW1'] = dW1
    grads['db1'] = db1

    beta1, beta2, epsilon = 0.9, 0.999, 1e-8

    if not hasattr(update_parameters, "m"):
        update_parameters.m = {}
        update_parameters.v = {}
        for i in range(1, L+1):
            update_parameters.m["dW"+str(i)] = np.zeros_like(parameters["W"+str(i)])
            update_parameters.m["db"+str(i)] = np.zeros_like(parameters["b"+str(i)])
            update_parameters.v["dW"+str(i)] = np.zeros_like(parameters["W"+str(i)])
            update_parameters.v["db"+str(i)] = np.zeros_like(parameters["b"+str(i)])

    for i in range(1, L+1):
        update_parameters.m["dW"+str(i)] = beta1 * update_parameters.m["dW"+str(i)] + (1-beta1)*grads["dW"+str(i)]
        update_parameters.m["db"+str(i)] = beta1 * update_parameters.m["db"+str(i)] + (1-beta1)*grads["db"+str(i)]
        update_parameters.v["dW"+str(i)] = beta2 * update_parameters.v["dW"+str(i)] + (1-beta2)*(grads["dW"+str(i)]**2)
        update_parameters.v["db"+str(i)] = beta2 * update_parameters.v["db"+str(i)] + (1-beta2)*(grads["db"+str(i)]**2)

        m_hat_dW = update_parameters.m["dW"+str(i)] / (1 - beta1**t)
        m_hat_db = update_parameters.m["db"+str(i)] / (1 - beta1**t)
        v_hat_dW = update_parameters.v["dW"+str(i)] / (1 - beta2**t)
        v_hat_db = update_parameters.v["db"+str(i)] / (1 - beta2**t)

        parameters["W"+str(i)] -= eta * m_hat_dW / (np.sqrt(v_hat_dW) + epsilon)
        parameters["b"+str(i)] -= eta * m_hat_db / (np.sqrt(v_hat_db) + epsilon)

    return parameters
