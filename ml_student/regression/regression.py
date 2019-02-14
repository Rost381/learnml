import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ml_student.math_tools import mt


class l2_regularization():
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, w):
        return self.alpha * 0.5 * w.T.dot(w)

    def grad(self, w):
        return self.alpha * w


class Regression(object):
    """
    Base Regression model
    """

    def __init__(self, n_iterations, learning_rate):
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate

    def _init_weights(self, n_features):
        """
        Draw samples from a uniform distribution.
        n_features : Output shape. 
        """
        limit = 1 / math.sqrt(n_features)
        self.w = np.random.uniform(-limit, limit, (n_features,))

    def fit(self, X, y):
        X = np.insert(X, 0, 1, axis=1)
        self.training_errors = []
        self._init_weights(n_features=X.shape[1])

        for i in range(self.n_iterations):
            y_pred = X.dot(self.w)
            mse = np.mean(0.5 * (y - y_pred)**2 + self.regularization(self.w))
            self.training_errors.append(mse)
            grad_w = -(y - y_pred).dot(X) + self.regularization.grad(self.w)
            self.w -= self.learning_rate * grad_w

    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)
        y_pred = X.dot(self.w)
        return y_pred


class LinearRegression(Regression):
    def __init__(self, n_iterations=100, learning_rate=0.001, gradient_descent=False):
        self.gradient_descent = gradient_descent

        self.regularization = lambda x: 0
        self.regularization.grad = lambda x: 0

        super(LinearRegression, self).__init__(n_iterations=n_iterations,
                                               learning_rate=learning_rate)

    def fit(self, X, y):
        if not self.gradient_descent:
            """
            Insert constant value 1 in first column and caculate XX^T.

            Examples
            --------
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
            U, s, V_T = np.linalg.svd(A)

            """
            create m x n D matrix
            populate D with n x n diagonal matrix
            [c, b] => [[c, 0],
                       [0, d]]
            """
            d = 1.0 / s
            S = np.zeros(A.shape)
            S[:A.shape[1], :A.shape[1]] = np.diag(d)

            """
            Calculate weights by least squares
            https://math.stackexchange.com/questions/772039/how-does-the-svd-solve-the-least-squares-problem/2173715#2173715
            https://math.stackexchange.com/questions/1816364/the-svd-solution-to-linear-least-squares-linear-system-of-equations

            Actually, we can just use this simple formula: 
            w = np.linalg.pinv(A).dot(X.T).dot(y)

            np.linalg.pinv(A) = V_T.T.dot(S.T).dot(U.T)
            """
            w = V_T.T.dot(S.T).dot(U.T).dot(X.T).dot(y)
            self.w = w

        else:
            super(LinearRegression, self).fit(X, y)


class RidgeRegression(Regression):
    def __init__(self, alpha=0.01, n_iterations=1000, learning_rate=0.001):
        """
        alpha:
        The L2 norm term in ridge regression is weighted by the regularization parameter alpha
        alpha value is 0 = Ordinary Least Squares Regression model.
        the larger is the alpha, the higher is the smoothness constraint.
        So, the alpha parameter need not be small. But, for a larger alpha, the flexibility of the fit would be very strict.
        """
        self.regularization = l2_regularization(alpha=alpha)
        super(RidgeRegression, self).__init__(n_iterations, learning_rate)
