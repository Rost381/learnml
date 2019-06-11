import numpy as np

from learnml.utils.api import l2_loss, Sigmoid


class Perceptron():
    """Perceptron

    Parameters:
    -----------
    n_iter : int
        The maximum number of passes over the training data (aka epochs).
    penalty : 'l2_loss’
        The penalty (aka regularization term) to be used. 
    learning_rate : float
        The step length that will be used when updating the weights.
    fit_intercept : boolean
        Whether to calculate the intercept for this model.
    """
    def __init__(self, n_iter=1000, penalty=l2_loss, learning_rate=0.01, fit_intercept=True):

        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.penalty = loss()
        self.fit_intercept = fit_intercept
        self.w0 = None
        self.w1 = None
        self.sigmoid = Sigmoid()

    def fit(self, X, y):
        n_sample, n_features = np.shape(X)
        _, n_outputs = np.shape(y)
        limit = 1 / np.sqrt(n_features)
        if self.fit_intercept:
            self.w0 = np.zeros((1, n_outputs))
        self.w1 = np.random.uniform(-limit, limit, (n_features, n_outputs))

        for i in range(self.n_iter):
            """Calculate the actual output"""
            if self.fit_intercept:
                linear_output = np.dot(X, self.w1) + self.w0
            else:
                linear_output = np.dot(X, self.w1)
            y_pred = self.sigmoid(linear_output)
            grad = self.penalty.gradient(
                y, y_pred) * self.sigmoid(linear_output) * (1 - self.sigmoid(linear_output))

            """Update the weights"""
            if self.fit_intercept:
                self.w0 -= self.learning_rate * \
                    np.sum(grad, axis=0, keepdims=True)
            self.w1 -= self.learning_rate * np.dot(X.T, grad)

    def predict(self, X):
        if self.fit_intercept:
            y_pred = self.sigmoid(np.dot(X, self.w1) + self.w0)
        else:
            y_pred = self.sigmoid(np.dot(X, self.w1))
        return y_pred
