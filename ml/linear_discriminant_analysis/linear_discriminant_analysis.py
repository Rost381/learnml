import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ml.math_tools import mt


class LDA():
    def __init__(self):
        None

    def _labels(self, y):
        return np.unique(y)

    def _n_features(self, X):
        return X.shape[1]

    def _S_W(self, X, y):
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
        Get eigenvalues and eigenvectors of SW^-1 * SB
        eigenvalues: [ -35.12243062  205.94594317  420.207018   3228.70484396]
        eigenvectors:
        [[-0.03606865  0.10724823 -0.76021496 -0.6397422]
        [-0.70732579 -0.23566355  0.42929382 -0.50976439]
        [ 0.70509418 -0.27697698  0.37633114 -0.53338619]
        [ 0.03509243  0.92533467  0.31008854 -0.21533546]]
        """
        eigenvalues, eigenvectors = mt.eig(A)

        """
        Sort eigenvectors from largest to smallest
        idx: [3 2 1 0]
        """
        idx = eigenvalues.argsort()[::-1]

        """
        select the first n_components of eigenvalues
        eigenvalues: [3228.70484396  420.207018]
        eigenvectors:
        [[-0.6397422   -0.76021496]
        [-0.50976439  0.42929382]
        [-0.53338619  -0.37633114]
        [-0.21533546  0.31008854]]
        """
        eigenvalues = eigenvalues[idx][:n_components]
        eigenvectors = eigenvectors[:, idx][:, :n_components]

        """Project the data onto eigenvectors"""
        X_transformed = X.dot(eigenvectors)

        return X_transformed

    def plot_lda(self, X, y, filename):
        """ Plot the dataset X and the corresponding labels y in transformation."""
        X_transformed = self._transform(X, y, n_components=2)
        x1 = X_transformed[:, 0]
        x2 = X_transformed[:, 1]
        plt.scatter(x1, x2, c=y)
        # plt.show()
        plt.savefig(filename)
