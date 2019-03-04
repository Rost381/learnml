import numpy as np
from .metrics import calculate_accuracy_score


class l1_regularization():
    """ L1 regularization
    For Lasso regression, also known as least absolute deviations, least absolute errors.
    *** but 'Zero' use Coordinate Descent to solve LassoRegression ***
    """

    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, w):
        """ np.linalg.norm """
        return self.alpha * np.linalg.norm(w)

    def grad(self, w):
        return self.alpha * np.sign(w)


class l2_regularization():
    """  L2 regularization
    For Ridge regression, gives an estimate which minimise the sum of square error.
    """

    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, w):
        return self.alpha * 0.5 * w.T.dot(w)

    def grad(self, w):
        return self.alpha * w


class loss(object):
    def loss(self, y_true, y_pred):
        return NotImplementedError()

    def gradient(self, y, y_pred):
        raise NotImplementedError()

    def acc(self, y, y_pred):
        return 0


class l1_loss(loss):
    """ L1 Loss """

    def __init__(self): pass

    def loss(self, y, y_pred):
        return abs(y - y_pred)


class l2_loss(loss):
    """ L2 Loss
    Used in GradientBoostingClassifier
    """

    def __init__(self): pass

    def loss(self, y, y_pred):
        return 0.5 * np.power((y - y_pred), 2)

    def gradient(self, y, y_pred):
        return -(y - y_pred)


class cross_entropy_loss(loss):
    """ Define a Cross Entropy loss
    In binary classification, cross-entropy can be calculated as:
    -{(y\log(p) + (1 - y)\log(1 - p))}
    """

    def __init__(self): pass

    def loss(self, y, p):
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return - (y * np.log(p) + (1 - y) * np.log(1 - p))

    def acc(self, y, p):
        return calculate_accuracy_score(np.argmax(y, axis=1), np.argmax(p, axis=1))

    def gradient(self, y, p):
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return - (y / p) + (1 - y) / (1 - p)
