import numpy as np
import pandas as pd

from alphalearn.utils.stats import covariance_matrix


class PCA():
    """Principal component analysis (PCA)

    Parameters:
    -----------
    n_components : int
        Number of components to keep. 
    """

    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X):
        self.X = X
        return self

    def transform(self, X):
        """Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
        """
        cov_matrix = covariance_matrix(X)

        """Caculate eigenvalues and eigenvectors of covariance_matrix(X) """
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        """Sort eigenvectors from largest to smallest """
        idx = eigenvalues.argsort()[::-1]

        """Select the first n_components of eigenvalues """
        eigenvalues = eigenvalues[idx][:self.n_components]
        eigenvectors = eigenvectors[:, idx][:, :self.n_components]

        X_transformed = X.dot(eigenvectors)

        return X_transformed
