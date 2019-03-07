import numpy as np
import pandas as pd

from zero.utils.stats import covariance_matrix


class LinearDiscriminantAnalysis():
    """Linear Discriminant Analysis

    Parameters:
    -----------
    n_components : int
        Number of components (< n_classes - 1) for dimensionality reduction.
    """

    def __init__(self, n_components):
        self.n_components = n_components

    def _labels(self, y):
        return np.unique(y)

    def _n_features(self, X):
        return X.shape[1]

    def _S_W(self, X, y):
        """Caculate SW"""
        labels = self._labels(y)
        n_features = self._n_features(X)
        S_W = np.empty((n_features, n_features))
        for label in labels:
            _X = X[y == label]

            """S_W = \sum\limits_{i=1}^{c} (N_{i}-1) \Sigma_i"""
            S_W += (len(_X) - 1) * covariance_matrix(_X)
        return S_W

    def _S_B(self, X, y):
        """Caculate SB"""
        labels = self._labels(y)
        n_features = self._n_features(X)
        overall_mean = np.mean(X, axis=0)
        S_B = np.empty((n_features, n_features))
        for label in labels:
            _X = X[y == label]
            _mean = np.mean(_X, axis=0)
            S_B += len(_X) * \
                (_mean - overall_mean).dot((_mean - overall_mean).T)
        return S_B

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.s_w = self._S_W(self.X, self.y)
        self.s_b = self._S_B(self.X, self.y)
        return self

    def transform(self, X):
        """Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
        """

        """SW^-1 * SB"""
        A = np.linalg.inv(self.s_w).dot(self.s_b)

        """Caculate eigenvalues and eigenvectors of SW^-1 * SB

        Examples:
        eigenvalues: [3, 2, 1]
        eigenvectors:
        [[1 ,2, 3]
        [1, 2, 3]
        [1, 2, 3]]
        """
        eigenvalues, eigenvectors = np.linalg.eig(A)

        """Sort eigenvectors from largest to smallest
        idx: [3, 2, 1]
        """
        idx = eigenvalues.argsort()[::-1]

        """Select the first n_components of eigenvalues and eigenvectors
        
        Examples:
        eigenvalues: [3, 2]
        eigenvectors:
        [[1, 2]
        [1, 2]
        [1, 2]]
        """
        eigenvalues = eigenvalues[idx][:self.n_components]
        eigenvectors = eigenvectors[:, idx][:, :self.n_components]

        """Project the data onto eigenvectors"""
        X_transformed = X.dot(eigenvectors)

        return X_transformed
