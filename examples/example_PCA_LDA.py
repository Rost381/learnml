import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from alphalearn.api import PCA, LinearDiscriminantAnalysis
from alphalearn.datasets.api import load_iris


def main():
    iris = load_iris()

    X = iris.data
    y = iris.target
    target_names = iris.target_names

    lda = LinearDiscriminantAnalysis(n_components=2)
    pca = PCA(n_components=2)

    X_r = pca.fit(X).transform(X)
    X_r2 = lda.fit(X, y).transform(X)

    # plot
    plt.figure(figsize=(15, 6))

    # pca, left
    plt.subplot(121)
    colors = ['navy', 'turquoise', 'darkorange']
    lw = 2

    for color, i, target_name in zip(colors, [0, 1, 2], target_names):
        plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, lw=lw,
                    label=target_name)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('PCA of IRIS dataset')

    # lda, right
    plt.subplot(122)
    for color, i, target_name in zip(colors, [0, 1, 2], target_names):
        plt.scatter(X_r2[y == i, 0], X_r2[y == i, 1], alpha=.8, color=color,
                    label=target_name)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('LDA of IRIS dataset')
    plt.savefig('./examples/example_PCA_LDA.png')


if __name__ == "__main__":
    main()
