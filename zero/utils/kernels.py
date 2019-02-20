import numpy as np


def linear(**kwargs):
    def f(x1, x2):
        return np.dot(x1, x2)
    return f


def poly(degree, coef0, **kwargs):
    def f(x1, x2):
        return (np.dot(x1, x2) + coef0)**degree
    return f


def rbf(gamma, **kwargs):
    def f(x1, x2):
        distance = np.linalg.norm(x1 - x2) ** 2
        return np.exp(-gamma * distance)
    return f
