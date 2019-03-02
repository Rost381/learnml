import math

import numpy as np


class DecisionStump():
    def __init__(self):
        self.polarity = 1
        self.feature_index = None
        self.threshold = None
        self.alpha = None


class AdaBoostClassifier():
    """ Discrete Adaboost
    An AdaBoost classifier.

    Parameters:
    -----------
    n_estimators : int
        The maximum number of estimators at which boosting is terminated.
    """

    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.w = None

    def H(self, X, y_init, stump):
        """
        The goal of the weak
        learner is to obtain a classifier
        """
        negative_idx = (
            stump.polarity * X[:, stump.feature_index] < stump.polarity * stump.threshold)
        y_init[negative_idx] = -1
        return y_init

    def fit(self, X, y):
        n_samples, n_features = np.shape(X)
        self.stumps = []

        """ Initialize the weights """
        self.w = np.full(n_samples, (1 / n_samples))

        for _ in range(self.n_estimators):
            stump = DecisionStump()
            error_min = float('inf')

            """ Calculate minimum error """
            for feature_i in range(n_features):
                feature_values = np.expand_dims(X[:, feature_i], axis=1)
                unique_values = np.unique(feature_values)

                for threshold in unique_values:
                    p = 1
                    prediction = np.ones(np.shape(y))
                    prediction[X[:, feature_i] < threshold] = -1

                    error = sum(self.w[y != prediction])

                    if error > 0.5:
                        error = 1 - error
                        p = -1

                    if error < error_min:
                        stump.polarity = p
                        stump.threshold = threshold
                        stump.feature_index = feature_i
                        error_min = error

            """ Caculate alpha """
            stump.alpha = 0.5 * math.log((1 - error_min) / (error_min + 1e-6))

            """ Update w """
            y_init = np.ones(np.shape(y))

            self.w *= np.exp(-stump.alpha * y * self.H(X, y_init, stump))
            self.w /= np.sum(self.w)

            """ Update stumps """
            self.stumps.append(stump)

    def predict(self, X):
        n_samples = np.shape(X)[0]
        y_pred = np.zeros((n_samples, 1))

        for stump in self.stumps:
            y_init = np.ones(np.shape(y_pred))
            y_pred += stump.alpha * self.H(X, y_init, stump)

        y_pred = np.sign(y_pred).flatten()

        return y_pred
