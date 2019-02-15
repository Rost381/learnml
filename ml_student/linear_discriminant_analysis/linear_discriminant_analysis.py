import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ml_student.math_tools import mt


class LDA():
    def __init__(self):
        None

    def _labels(self, y):
        return np.unique(y)

    def _n_features(self, X):
        return X.shape[1]

    def _S_W(self, X, y):
        """
        Caculate s_w
        """
        labels = self._labels(y)
        n_features = self._n_features(X)
        S_W = np.empty((n_features, n_features))
        for label in labels:
            _X = X[y == label]

            """
            S_W = \sum\limits_{i=1}^{c} (N_{i}-1) \Sigma_i
            """
            S_W += (len(_X) - 1) * mt.covariance_matrix(_X)
        return S_W

    def _S_B(self, X, y):
        """
        Caculate s_b
        """
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

    def _transform(self, X, y, n_components):
        S_W, S_B = self._S_W(X, y), self._S_B(X, y)

        """SW^-1 * SB"""
        A = np.linalg.inv(S_W).dot(S_B)

        """
        Caculate eigenvalues and eigenvectors of SW^-1 * SB

        Examples:
        eigenvalues: [3, 2, 1]
        eigenvectors:
        [[1 ,2, 3]
        [1, 2, 3]
        [1, 2, 3]]
        """
        eigenvalues, eigenvectors = mt.calculate_eig(A)

        """
        Sort eigenvectors from largest to smallest
        idx: [3, 2, 1]
        """
        idx = eigenvalues.argsort()[::-1]

        """
        Select the first n_components of eigenvalues and eigenvectors
        
        Examples:
        eigenvalues: [3, 2]
        eigenvectors:
        [[1, 2]
        [1, 2]
        [1, 2]]
        """
        eigenvalues = eigenvalues[idx][:n_components]
        eigenvectors = eigenvectors[:, idx][:, :n_components]

        """
        Project the data onto eigenvectors
        """
        X_transformed = X.dot(eigenvectors)

        return X_transformed

    def plot_lda(self, X, y, filename):
        X_transformed = self._transform(X, y, n_components=2)
        x1 = X_transformed[:, 0]
        x2 = X_transformed[:, 1]
        plt.scatter(x1, x2, c=y)
        # plt.show()
        plt.savefig(filename)
