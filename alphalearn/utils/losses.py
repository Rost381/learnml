import numpy as np
from .metrics import calculate_accuracy_score
from .activations import sigmoid


class l1_regularization():
    """L1 regularization
    For Lasso regression, also known as least absolute deviations, least absolute errors.
    *** but here use Coordinate Descent to solve LassoRegression ***
    """

    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, w):
        """np.linalg.norm"""
        return self.alpha * np.linalg.norm(w)

    def grad(self, w):
        return self.alpha * np.sign(w)


class l2_regularization():
    """L2 regularization
    For Ridge regression, gives an estimate which minimise the sum of square error.
    """

    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, w):
        return self.alpha * 0.5 * w.T.dot(w)

    def grad(self, w):
        return self.alpha * w


class l1_loss():
    """L1 Loss function 
    minimizes the absolute differences between 
    the estimated values and the existing target values.
    """

    def __init__(self): pass

    def loss(self, y, y_pred):
        return abs(y - y_pred)


class l2_loss():
    """L2 loss function(Least squared error)
    minimizes the squared differences between
    the estimated and existing target values.

    Used in GradientBoostingClassifier
    """

    def __init__(self): pass

    def loss(self, y, y_pred):
        return 0.5 * np.power((y - y_pred), 2)

    def gradient(self, y, y_pred):
        return -(y - y_pred)


class cross_entropy_loss():
    """Define a Cross Entropy loss
    Used in GradientBoostingRegressor

    In binary classification, cross-entropy can be calculated as:
    -{(y\log(p) + (1 - y)\log(1 - p))}
    """

    def __init__(self): pass

    def loss(self, y, p):
        p = np.clip(p, 1e-7, 1 - 1e-7)
        return - (y * np.log(p) + (1 - y) * np.log(1 - p))

    def acc(self, y, p):
        return calculate_accuracy_score(np.argmax(y, axis=1), np.argmax(p, axis=1))

    def gradient(self, y, p):
        p = np.clip(p, 1e-7, 1 - 1e-7)
        return - (y / p) + (1 - y) / (1 - p)


class logistic_loss():
    def __init__(self): pass

    def loss(self, y, y_pred):
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        p = sigmoid(y_pred)
        return y * np.log(p) + (1 - y) * np.log(1 - p)

    def gradient(self, y, y_pred):
        p = sigmoid(y_pred)
        return -(y - p)

    def hess(self, y, y_pred):
        p = sigmoid(y_pred)
        return p * (1 - p)
