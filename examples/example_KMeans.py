import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from alphalearn.api import KMeans
from alphalearn.datasets.api import load_iris


def main():
    # Example 1
    X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(X)

    print(kmeans.cluster_centers_)
    print(kmeans.labels_)
    print(kmeans.inertia_)
    print(kmeans.predict([[0, 0], [12, 3]]))

    # Example 2
    iris = load_iris()
    X = iris.data
    y = iris.target

    kmeans = KMeans(n_clusters=3)
    kmeans.fit(X)
    labels = kmeans.labels_

    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(1, 2, 1, projection='3d', elev=48, azim=134)
    for name, label in [('Setosa', 0),
                        ('Versicolour', 1),
                        ('Virginica', 2)]:
        ax.text3D(X[y == label, 3].mean(),
                  X[y == label, 0].mean(),
                  X[y == label, 2].mean() + 2, name,
                  horizontalalignment='center',
                  bbox=dict(alpha=.2, edgecolor='w', facecolor='w'))

    ax.scatter(X[:, 3], X[:, 0], X[:, 2],
               c=labels, edgecolor='k')

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_title('3 clusters')
    ax.dist = 12

    ax = fig.add_subplot(1, 2, 2, projection='3d', elev=48, azim=134)
    for name, label in [('Setosa', 0),
                        ('Versicolour', 1),
                        ('Virginica', 2)]:
        ax.text3D(X[y == label, 3].mean(),
                  X[y == label, 0].mean(),
                  X[y == label, 2].mean() + 2, name,
                  horizontalalignment='center',
                  bbox=dict(alpha=.2, edgecolor='w', facecolor='w'))

    y = np.choose(y, [1, 2, 0]).astype(np.float)
    ax.scatter(X[:, 3], X[:, 0], X[:, 2],
               c=y, edgecolor='k')

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_xlabel('Petal width')
    ax.set_ylabel('Sepal length')
    ax.set_zlabel('Petal length')
    ax.set_title('Ground Truth')
    ax.dist = 12
    plt.savefig("./examples/example_KMeans.png")


if __name__ == "__main__":
    main()
