import math

import numpy as np
import pandas as pd

from alphalearn.utils.api import (PolynomialFeatures, l1_regularization,
                                  l2_regularization, normalize)


class Regression(object):
    """Base Regression model

    Parameters:
    -----------
    max_iter : int
        The maximum number of iterations.
    learning_rate : float
        The step length that will be used when updating the weights.
    """

    def __init__(self, max_iter, learning_rate):
        self.max_iter = max_iter
        self.learning_rate = learning_rate

    def _init_weights(self, n_features):
        """Draw samples from a uniform distribution.
        n_features : Output shape. 
        """
        limit = 1 / math.sqrt(n_features)
        self.w = np.random.uniform(-limit, limit, (n_features,))

    def fit(self, X, y):
        X = np.insert(X, 0, 1, axis=1)
        self.training_errors = []
        self._init_weights(n_features=X.shape[1])
        m = len(y)
        for _ in range(self.max_iter):
            y_pred = X.dot(self.w)
            mse = np.mean(0.5 * (y - y_pred)**2 + self.regularization(self.w))
            self.training_errors.append(mse)
            """Gradient Descent is a general function for minimizing a function,
            in this case the Mean Squared Error(MSE) cost function.
            """
            grad_w = -(y - y_pred).dot(X) + self.regularization.grad(self.w)
            self.w -= self.learning_rate * grad_w

    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)
        y_pred = X.dot(self.w)
        return y_pred


class LinearRegression(Regression):
    """ Linear Regression

    Parameters:
    -----------
    max_iter : int
        The maximum number of iterations.
    learning_rate : float
        The step length that will be used when updating the weights.
    gradient_descent : boolean
        True => gradient descent.
        False => optimization by least squares.
    """

    def __init__(self, max_iter=100, learning_rate=0.001, gradient_descent=False):
        self.gradient_descent = gradient_descent
        self.regularization = lambda x: 0
        self.regularization.grad = lambda x: 0

        super(LinearRegression, self).__init__(max_iter=max_iter,
                                               learning_rate=learning_rate)

    def fit(self, X, y):
        if not self.gradient_descent:
            """ Insert constant value 1 in first column and caculate XX^T.

            [[1, 1] =>  [[3, 6]
             [1, 2]      [6, 14]]
             [1, 3]]
            """
            X = np.insert(X, 0, 1, axis=1)
            A = X.T.dot(X)

            """ Singular Value Decomposition.
            U: Unitary array(s).
            Sigma: Vector(s) with the singular values, sorted in descending order. 
            V: Unitary array(s).
            """
            U, s, V_T = np.linalg.svd(A)

            """ Create m x n D matrix
            populate D with n x n diagonal matrix.
            
            [c, b] => [[c, 0],
                       [0, d]]
            """
            d = 1.0 / s
            S = np.zeros(A.shape)
            S[:A.shape[1], :A.shape[1]] = np.diag(d)

            """ Calculate weights by least squares
            *** Actually, we can just use this formula: ***
                w = np.linalg.pinv(A).dot(X.T).dot(y)
            
            because:
                np.linalg.pinv(A) = V_T.T.dot(S.T).dot(U.T)
            """
            w = V_T.T.dot(S.T).dot(U.T).dot(X.T).dot(y)
            # w = np.linalg.pinv(A).dot(X.T).dot(y)
            self.w = w

        else:
            super(LinearRegression, self).fit(X, y)


class RidgeRegression(Regression):
    """Ridge Regression

    Linear least squares with l2 regularization.
    Minimizes the objective function:
        ||y - Xw||^2_2 + alpha * ||w||^2_2

    Parameters:
    -----------
    alpha : float
        The L2 norm term in ridge regression is weighted by the regularization parameter alpha
        alpha = 0 is equivalent to an ordinary least square.
        the larger is the alpha, the higher is the smoothness constraint.
        So, the alpha parameter need not be small.
        But, for a larger alpha, the flexibility of the fit would be very strict.
    max_iter : int
        The maximum number of iterations
    learning_rate : float
        The step length that will be used when updating the weights.
    """

    def __init__(self, alpha=0.01, max_iter=1000, learning_rate=0.001):
        self.regularization = l2_regularization(alpha=alpha)
        super(RidgeRegression, self).__init__(max_iter, learning_rate)


class LassoRegression(Regression):
    """Lasso Regression

    The algorithm used to fit the model is coordinate descent.
    Coordinate descent is an algorithm that considers each column of data
    at a time hence it will automatically convert the X input
    as a Fortran-contiguous numpy array if necessary.

    Parameters:
    -----------
    alpha : float
        Constant that multiplies the L1 term.
        alpha = 0 is equivalent to an ordinary least square.
    max_iter : int
        The maximum number of iterations
    fit_intercept : boolean
        Whether to calculate the intercept for this model.
    """

    def __init__(self, alpha=0.01, max_iter=1000, fit_intercept=True):
        self.alpha = alpha
        self.max_iter = max_iter
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = None

    def _S(self, z, gamma):
        """Soft-thresholding operator
        coordinate descent algorithm for LASSO
        """
        if z > 0.0 and gamma < z:
            return z - gamma
        elif z < 0.0 and gamma < abs(z):
            return z + gamma
        else:
            return 0.0

    def fit(self, X, y):
        if self.fit_intercept:
            X = np.insert(X, 0, 1, axis=1)

        """Initialize beta, which means coef_
        fit_intercept = false:
            beta = [0, 0, ...]
        fit_intercept = true:
            beta = [x, 0, ...]
        """
        beta = np.zeros(X.shape[1])
        if self.fit_intercept:
            beta[0] = np.sum(y - np.dot(X[:, 1:], beta[1:])) / (X.shape[0])

        for i in range(self.max_iter):
            start = 1 if self.fit_intercept else 0
            for j in range(start, len(beta)):
                beta[j] = 0.0
                z = (y - X.dot(beta)).dot(X[:, j].T)
                gamma = self.alpha * X.shape[0]
                """Update beta[1:]"""
                beta[j] = self._S(z, gamma) / X[:, j].dot(X[:, j].T)

                if self.fit_intercept:
                    """Update beta[0], which means intercept_"""
                    beta[0] = np.sum(
                        y - np.dot(X[:, 1:], beta[1:])) / (X.shape[0])

        """Caculate intercept_ and coef_"""
        if self.fit_intercept:
            self.intercept_ = beta[0]
            self.coef_ = beta[1:]
            self.w = np.insert(self.coef_, 0, self.intercept_, axis=0)
        else:
            self.coef_ = beta
            self.w = self.coef_

        return self
        super(LassoRegression, self).fit(X, y)

    def predict(self, X):
        if self.fit_intercept:
            X = np.insert(X, 0, 1, axis=1)
            y_pred = X.dot(self.w)
        else:
            y_pred = X.dot(self.w)
        return y_pred
        super(LassoRegression, self).predict(X)


class PolynomialRidgeRegression(Regression):
    """PolynomialRidgeRegression

    Parameters:
    -----------
    degree : integer
        The degree of the polynomial features.
    reg_factor : float
        The factor that will determine the amount of regularization and feature shrinkage. 
    max_iter : int
        The maximum number of iterations
    learning_rate : float
        The step length that will be used when updating the weights.
    """

    def __init__(self, degree, reg_factor, max_iter=3000, learning_rate=0.001, gradient_descent=True):
        self.degree = degree
        self.regularization = l2_regularization(alpha=reg_factor)
        self.regularization.grad = lambda x: 0
        super(PolynomialRidgeRegression, self).__init__(
            max_iter=max_iter, learning_rate=learning_rate)

    def fit(self, X, y):
        X = normalize(PolynomialFeatures(X, degree=self.degree))
        super(PolynomialRidgeRegression, self).fit(X, y)

    def predict(self, X):
        X = normalize(PolynomialFeatures(X, degree=self.degree))
        return super(PolynomialRidgeRegression, self).predict(X)
