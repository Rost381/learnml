import numpy as np
from learnml.utils.api import euclidean_distances


class KNeighborsClassifier():
    """k-nearest neighbors

    Classifier implementing the k-nearest neighbors vote.

    Parameters:
    -----------
    n_neighbors : int, optional (default = 5)
        Number of neighbors to use by default for kneighbors queries.
    """

    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X_test):
        y_pred = np.zeros(X_test.shape[0])
        for i, x_test in enumerate(X_test):
            """Distances between sample and train data"""
            distances = [euclidean_distances(x_test, x) for x in self.X]

            """Returns the indices of distances"""
            distances = np.argsort(distances)

            """get the first n_neighbors"""
            idx = distances[:self.n_neighbors]

            """k_nearest_neighbors"""
            k_nearest_neighbors = np.array([self.y[i] for i in idx])

            """Count number of occurrences of each value in k_nearest_neighbors"""
            counts = np.bincount(k_nearest_neighbors.astype('int'))

            """Returns the indices of the maximum values of counts"""
            y_pred[i] = counts.argmax()

        return y_pred
