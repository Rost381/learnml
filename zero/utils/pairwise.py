import numpy as np


def euclidean_distances(X, Y):
    """dist(x, y) = sqrt(dot(x, x) - 2 * dot(x, y) + dot(y, y))"""
    distance = 0
    for i in range(len(X)):
        distance += pow((X[i] - Y[i]), 2)
    return np.sqrt(distance)
