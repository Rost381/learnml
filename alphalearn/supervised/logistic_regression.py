import numpy as np
from alphalearn.utils.api import Sigmoid


class LogisticRegression():
    """Logistic Regression

    Parameters:
    -----------
    max_iter : int
        The maximum number of iterations
    learning_rate : float
        The step length that will be used when updating the weights.
    fit_intercept : boolean
        Whether to calculate the intercept for this model.
    """

    def __init__(self, max_iter=100, learning_rate=0.001, fit_intercept=True):
        self.w = None
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.fit_intercept = fit_intercept
        self.sigmoid = Sigmoid()

    def log_likelihood(self, features, target, weights):
        scores = np.dot(features, weights)
        ll = np.sum(target * scores - np.log(1 + np.exp(scores)))
        return ll

    def fit(self, X, y):
        if self.fit_intercept:
            X = np.insert(X, 0, 1, axis=1)

        """Step 1 initialize the weight"""
        self.w = np.zeros(X.shape[1])

        for _ in range(self.max_iter):
            """Step 2 caculate the y_pred"""
            scores = np.dot(X, self.w)
            y_pred = self.sigmoid(scores)

            """Step 3 update the weight
            new weight = old weight - X . (y - y_pred) * learning rate"""
            grad_w = (y_pred - y).dot(X)
            self.w -= self.learning_rate * grad_w

        return self.w

    def predict_proba(self, X):
        """Predict_proba
        Probability estimates.
        The returned estimates for all classes are ordered by the label of classes.
        """
        y_proba = list()
        if self.fit_intercept:
            X = np.insert(X, 0, 1, axis=1)
            y_pred = self.sigmoid(X.dot(self.w))
        else:
            y_pred = self.sigmoid(X.dot(self.w))

        y_proba = y_pred.T
        if y_proba.ndim == 1:
            y_proba = np.c_[1 - y_proba, y_proba]
        return y_proba

    def predict(self, X):
        if self.fit_intercept:
            X = np.insert(X, 0, 1, axis=1)
            y_pred = np.round(self.sigmoid(X.dot(self.w)))
        else:
            y_pred = np.round(self.sigmoid(X.dot(self.w)))
        return y_pred
