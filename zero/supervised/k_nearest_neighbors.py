import numpy as np
from zero.utils.api import euclidean_distances


class KNeighborsClassifier():

    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors

    def predict(self, X_test, X_train, y_train):
        y_pred = np.empty(X_test.shape[0])
        for i, test_sample in enumerate(X_test):
            idx = np.argsort([euclidean_distances(test_sample, x)
                              for x in X_train])[:self.n_neighbors]
            k_nearest_neighbors = np.array([y_train[i] for i in idx])

            counts = np.bincount(k_nearest_neighbors.astype('int'))
            y_pred[i] = counts.argmax()
        return y_pred
