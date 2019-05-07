import numpy as np

from learnml.supervised.api import XGBoostRegressionTree
from learnml.utils.api import logistic_loss, to_categorical


class XGBoost():
    """XGBoost

    Parameters:
    -----------
    n_estimators : int
        The number of boosting stages to perform.
    learning_rate : float
        learning rate shrinks the contribution of each tree by learning_rate.
    min_samples_split : int
        The minimum number of samples needed to make a split when building a tree.
    min_impurity_split : float
        The minimum impurity required to split tree.
    max_depth : int
        The maximum depth of a tree.
    """

    def __init__(self,
                 n_estimators=100,
                 learning_rate=0.1,
                 min_samples_split=2,
                 min_impurity_split=1e-7,
                 max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.min_samples_split = min_samples_split
        self.min_impurity_split = min_impurity_split
        self.max_depth = max_depth
        self.loss = logistic_loss()
        self.trees = []

        for _ in range(self.n_estimators):
            tree = XGBoostRegressionTree(
                min_samples_split=self.min_samples_split,
                min_impurity_split=self.min_impurity_split,
                max_depth=self.max_depth,
                loss=self.loss)

            self.trees.append(tree)

    def fit(self, X, y):
        y = to_categorical(y)

        y_pred = np.zeros(np.shape(y))
        for i in range(self.n_estimators):
            tree = self.trees[i]
            y_y_pred = np.concatenate((y, y_pred), axis=1)
            tree.fit(X, y_y_pred)
            update_pred = tree.predict(X)
            y_pred -= np.multiply(self.learning_rate, update_pred)

    def predict(self, X):
        y_pred = None
        for tree in self.trees:
            update_pred = tree.predict(X)
            if y_pred is None:
                y_pred = np.zeros_like(update_pred)
            y_pred -= np.multiply(self.learning_rate, update_pred)

        y_pred = np.exp(y_pred) / np.sum(np.exp(y_pred), axis=1, keepdims=True)
        y_pred = np.argmax(y_pred, axis=1)
        return y_pred
