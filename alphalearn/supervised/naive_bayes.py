import numpy as np


class GaussianNB():
    """Gaussian Naive Bayes

    http://i.stanford.edu/pub/cstr/reports/cs/tr/79/773/CS-TR-79-773.pdf
    """

    def __init__(self):
        pass

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.classes = np.unique(y)
        self.parameters = []
        """Calculate the mean and variance of each feature for each class"""
        for i, c in enumerate(self.classes):
            X_c = X[np.where(y == c)]
            self.parameters.append([])

            for feature in X_c.T:
                """
                feature is 'column'
                [[-1 -1] => feature:[-1 -2 -3]
                 [-2 -1]    feature:[-1 -1 -2]
                 [-3 -2]]
                """
                parameters = {"mean": feature.mean(), "var": feature.var()}
                self.parameters[i].append(parameters)

    def _likelihood(self, mean, var, x):
        """Gaussian likelihood of the data x given mean and var """
        coef = 1.0 / np.sqrt(2.0 * np.pi * var + 1e-7)
        exponent = np.exp(-(np.power(x - mean, 2) / (2 * var + 1e-7)))
        return coef * exponent

    def _prior_probability(self, c):
        """Calculate the prior probability of class c
        (samples where class == c / total number of samples)"""
        p = np.mean(self.y == c)
        return p

    def _classify(self, sample):
        """P(Y|X) = P(X|Y)*P(Y)/P(X)
        -------------------------------------------
        P(X|Y) | likelihood(Gaussian distribution)|
        P(Y)   | prior probability                |
        P(X)   | ignored here                     |
        -------------------------------------------
        """
        posterior_probability = []
        for i, c in enumerate(self.classes):
            p = self._prior_probability(c)
            """independence
            P(x1,x2,x3|Y) = P(x1|Y)*P(x2|Y)*P(x3|Y)
            """
            for feature_value, params in zip(sample, self.parameters[i]):
                likelihood = self._likelihood(
                    params["mean"], params["var"], feature_value)
                p *= likelihood
            posterior_probability.append(p)
        return self.classes[np.argmax(posterior_probability)]

    def predict(self, X):
        """Predict the class labels of the samples in X"""
        y_pred = [self._classify(sample) for sample in X]
        return y_pred
