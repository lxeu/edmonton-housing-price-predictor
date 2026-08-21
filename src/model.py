import numpy as np

def zscore_normalize_features(x):
    """
    Normalize each feature to have mean 0 and standard deviation 1.
    Returns X_norm, mu, and sigma
    """
    mu = np.mean(x, axis=0)
    sigma = np.std(x, axis=0)
    x_norm = (x - mu) / sigma
    return x_norm, mu, sigma

def compute_cost(x, y, w, b):
    """
    Returns a single number, the cost J.
    """
    m = x.shape[0]
    f_wb = np.dot(x, w) + b
    error = f_wb - y
    cost = np.sum(error ** 2) / (2*m)
    return cost


def compute_gradient(x, y, w, b):
    """
    Calculates the gradient for gradient descent
    Returns dj_dw and dj_db
    """
    m = x.shape[0]
    n = x.shape[1]
    dj_dw = np.zeros(n)
    dj_db = 0.0

    for i in range(m):
        error = np.dot(x[i], w) + b - y[i]
        for j in range(n):
            dj_dw[j] += error * x[i, j]
        dj_db += error

    dj_dw = dj_dw / m
    dj_db = dj_db / m
    return dj_dw, dj_db


def gradient_descent(x, y, w_in, b_in, alpha, num_iters):
    """
    Runs gradient descent. Returns final w, b, and the cost at each step.
    """
    w = w_in.copy()
    b = b_in
    cost_history = []

    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient(x, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        cost_history.append(compute_cost(x, y, w, b))

        if i % 50 == 0:
            print(f"iter {i:5d}  cost {cost_history[-1]:.2e}")

    return w, b, cost_history