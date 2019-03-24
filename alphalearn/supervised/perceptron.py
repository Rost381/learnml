import numpy as np

from alphalearn.utils.api import l2_loss, sigmoid


class Perceptron():
    def __init__(self, n_iterations=1000, loss=l2_loss, learning_rate=0.01):
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.loss = loss()

    def fit(self, X, y):
        self.w = []
        n_sample, n_features = np.shape(X)
        _, n_outputs = np.shape(y)
        limit = 1 / np.sqrt(n_features)
        self.w0 = np.zeros((1, n_outputs))
        self.w1 = np.random.uniform(-limit, limit, (n_features, n_outputs))

        for i in range(self.n_iterations):
            """Calculate the actual output"""
            linear_output = np.dot(X, self.w1) + self.w0
            y_pred = sigmoid(linear_output)
            grad = self.loss.gradient(
                y, y_pred) * sigmoid(linear_output) * (1 - sigmoid(linear_output))

            """Update the weights"""
            self.w0 -= self.learning_rate * \
                np.sum(grad, axis=0, keepdims=True)
            self.w1 -= self.learning_rate * np.dot(X.T, grad)

    def predict(self, X):
        y_pred = sigmoid(np.dot(X, self.w1) + self.w0)
        return y_pred
