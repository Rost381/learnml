import numpy as np

from alphalearn.api import ClassificationTree
from alphalearn.utils.api import random_subsets


class RandomForestClassifier():
    """RandomForestClassifier

    A random forest classifier.

    Parameters:
    -----------
    n_estimators : integer
        The number of trees in the forest.
    max_features : int
        The number of features to consider when looking for the best split:
    min_samples_split : int
        The minimum number of samples required to split an internal node:
    min_impurity_split : float
        Threshold for early stopping in tree growth.
    max_depth : integer
        The maximum depth of the tree.
    """

    def __init__(self,
                 n_estimators=10,
                 max_features=2,
                 min_samples_split=2,
                 min_impurity_split=1e-7,
                 max_depth=2):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.min_impurity_split = min_impurity_split
        self.max_depth = max_depth
        self.trees = []
        for _ in range(self.n_estimators):
            self.trees.append(ClassificationTree(min_samples_split=self.min_samples_split,
                                                 min_impurity_split=min_impurity_split,
                                                 max_depth=self.max_depth
                                                 ))

    def fit(self, X, y):
        n_features = np.shape(X)[1]
        if not self.max_features:
            self.max_features = int(np.sqrt(n_features))
        """Step 1
        create a bootstrapped dataset
        """
        subsets = random_subsets(X, y, self.n_estimators)

        for i in range(self.n_estimators):
            X_subset, y_subset = subsets[i]

            """Step 2
            only use a random subset of variables(or columns) at each step"""
            idx = np.random.choice(
                range(n_features), size=self.max_features, replace=True)
            self.trees[i].feature_indices = idx
            X_subset = X_subset[:, idx]
            self.trees[i].fit(X_subset, y_subset)

    def predict(self, X):
        """Predict
        The predicted class of an input sample is a vote by the trees in the forest.
        """
        y_pred_all = np.empty((X.shape[0], len(self.trees)))
        for i, tree in enumerate(self.trees):
            idx = tree.feature_indices
            prediction = tree.predict(X[:, idx])
            y_pred_all[:, i] = prediction

        y_pred = []
        for sample_predictions in y_pred_all:
            y_pred.append(np.bincount(
                sample_predictions.astype('int')).argmax())
        return y_pred
