import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ml_student.math_tools import mt


class Regression(object):
    """ Base Regression model
    """

    def __init__(self, n_iterations, learning_rate):
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate

    def _init_weights(self, n_features):
        limit = 1 / math.sqrt(n_features)
        self.w = np.random.uniform(-limit, limit, (n_features,))

    def fit(self, X, y):
        X = np.insert(X, 0, 1, axis=1)
        self.training_errors = []
        self._init_weights(n_features=X.shape[1])

        for i in range(self.n_iterations):
            print(i)

    def predict(self, X):
        # Insert constant ones for bias weights
        X = np.insert(X, 0, 1, axis=1)
        y_pred = X.dot(self.w)
        return y_pred


class LinearRegression(Regression):
    def __init__(self, n_iterations=100, learning_rate=0.001, gradient_descent=False):
        self.gradient_descent = gradient_descent

        super(LinearRegression, self).__init__(n_iterations=n_iterations,
                                               learning_rate=learning_rate)

    def fit(self, X, y):
        if not self.gradient_descent:
            """
            Insert constant value 1 in first column.
            [[1, 1] =>  [[3, 6]
             [1, 2]      [6, 14]]
             [1, 3]]
            """
            X = np.insert(X, 0, 1, axis=1)
            A = X.T.dot(X)

            """
            Singular Value Decomposition.
            U: Unitary array(s).
            Sigma: Vector(s) with the singular values, sorted in descending order. 
            V: Unitary array(s).
            """
            U, Sigma, V = np.linalg.svd(A)

            """
            Construct a diagonal array.
            [1, 4] => [[1, 0],
                       [0, 4]]

            Compute the (Moore-Penrose) pseudo-inverse of a matrix.  
            """
            Sigma = np.diag(Sigma)
            Sigma_pinv = np.linalg.pinv(Sigma)

            """
            Calculate weights by least squares
            X+ = (X_t * X)^-1 * X_t
            X = U * Sigma * Adjugate(V)
            X+ = V * pseudo-inverse(Sigma) * Adjugate(U)
            w = V * pseudo-inverse(Sigma) * Adjugate(U) * X * y

            https://math.stackexchange.com/questions/772039/how-does-the-svd-solve-the-least-squares-problem/2173715#2173715
            https://math.stackexchange.com/questions/1816364/the-svd-solution-to-linear-least-squares-linear-system-of-equations
            """
            w = V.dot(Sigma_pinv).dot(U).dot(X.T).dot(y)
            self.w = w

        else:
            super(LinearRegression, self).fit(X, y)
