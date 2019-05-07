import numpy as np

from learnml.utils.api import euclidean_distances


class KMeans():
    """K-Means clustering

    https://en.wikipedia.org/wiki/K-means_clustering#Standard_algorithm

    Parameters:
    -----------
    n_clusters : int
        The number of clusters to form as well as the number of centroids to generate.
    max_iter : int
        Maximum number of iterations of the k-means algorithm for a single run.
    random_state : int, RandomState instance or None (default)
        Determines random number generation for centroid initialization
    """

    def __init__(self, n_clusters=8, max_iter=300, tol=1e-4, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.inertia_ = float('inf')
        self.cluster_centers_ = None
        self.labels_ = None

    def _nearest_centroid(self, sample, centroids):
        """Return the index of the closest centroid to the sample"""
        closest_i = 0
        closest_dist = float('inf')
        for i, centroid in enumerate(centroids):
            distance = euclidean_distances(sample, centroid)
            if distance < closest_dist:
                closest_i = i
                closest_dist = distance
        return closest_i

    def _nearest(self, clusters, x):
        return np.argmin([euclidean_distances(x, c) for c in clusters])

    def fit(self, X):
        if self.random_state:
            np.random.seed(self.random_state)

        n_samples, n_features = np.shape(X)

        for _ in range(self.max_iter):
            """Step 1
            Initial means.
            randomly chooses k observations from the dataset.
            """
            self.cluster_centers_temp_ = X[np.random.permutation(n_samples)[
                :self.n_clusters]]

            """Step 2
            k clusters are created by associating every observation with the nearest mean. 
            """
            self.labels_temp_ = [self._nearest(
                self.cluster_centers_temp_, x) for x in X]

            indices = [[i for i, l in enumerate(self.labels_temp_) if l == j]
                       for j in range(self.n_clusters)]
            X_cluster = [X[i] for i in indices]

            """Step 3
            The centroid of each of the k clusters becomes the new mean.
            """
            self.cluster_centers_temp_ = [x_c.sum(axis=0) / len(x_c)
                                          for x_c in X_cluster]

            error = 0
            for x, l_t in zip(X, self.labels_temp_):
                error += np.sum(
                    np.square((self.cluster_centers_temp_[l_t] - x)))

            if error < self.tol:
                break

            if self.inertia_ > error:
                self.inertia_ = error
                self.cluster_centers_ = self.cluster_centers_temp_
                self.labels_ = self.labels_temp_

        return self

    def predict(self, X):
        y_pred = []
        for x in X:
            y_pred.append(self._nearest_centroid(x, self.cluster_centers_))
        return y_pred
