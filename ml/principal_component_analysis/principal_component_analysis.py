import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ml.math_tools import mt


class PCA():
    def _transform(self, X, n_components):
        covariance_matrix = mt.covariance_matrix(X)

        """
        Get eigenvalues and eigenvectors of SW^-1 * SB
        """
        eigenvalues, eigenvectors = mt.eig(covariance_matrix)

        """
        Sort eigenvectors from largest to smallest
        """
        idx = eigenvalues.argsort()[::-1]

        """
        select the first n_components of eigenvalues
        """
        eigenvalues = eigenvalues[idx][:n_components]
        eigenvectors = eigenvectors[:, idx][:, :n_components]

        X_transformed = X.dot(eigenvectors)

        return X_transformed

    def plot_pca(self, X, y, filename):
        X_transformed = self._transform(X, n_components=2)
        x1 = X_transformed[:, 0]
        x2 = X_transformed[:, 1]
        plt.scatter(x1, x2, c=y)

        cmap = plt.get_cmap('viridis')
        colors = [cmap(i) for i in np.linspace(0, 1, len(np.unique(y)))]

        class_distr = []
        for i, l in enumerate(np.unique(y)):
            _x1 = x1[y == l]
            _x2 = x2[y == l]
            _y = y[y == l]
            class_distr.append(plt.scatter(_x1, _x2, color=colors[i]))

        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        # plt.show()
        plt.savefig(filename)
