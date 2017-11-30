import numpy as np


def sigmoid_dash(x):
    return sigmoid(x) * (1 - sigmoid(x))


def sigmoid(x):
    """
    If x is a numpy array, applies element wise
    """
    return 1.0 / (1.0 + np.exp(-x))
