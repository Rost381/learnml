import matplotlib.pyplot as plt
import numpy as np

from learnml.supervised.api import svm
from learnml.utils.api import make_meshgrid, plot_contours


def main():
    np.random.seed(0)
    X = np.random.randn(150, 2)
    y = np.logical_xor(X[:, 0] > 0,
                       X[:, 1] > 0)
    y = np.where(y, 1, -1)

    fig, sub = plt.subplots(2, 2)

    models = (svm(kernel='linear', C=1),
              svm(kernel='poly', degree=3, C=1),
              svm(kernel='rbf', gamma=1, C=1),
              svm(kernel='rbf', gamma=0.01, C=100))

    titles = ('linear kernel',
              'polynomial (degree 3) kernel',
              'RBF (gamma 1) kernel',
              'RBF (gamma 0.01) kernel')

    for clf, title, ax in zip(models, titles, sub.flatten()):
        clf.fit(X, y)
        xx1, xx2 = make_meshgrid(X, y, h=0.02)
        plot_contours(ax, clf, xx1, xx2, alpha=0.8, cmap=plt.cm.coolwarm)
        ax.scatter(X[:, 0], X[:, 1], c=y,
                   cmap=plt.cm.coolwarm, s=20, edgecolors='k')

        ax.set_xlim(xx1.min(), xx1.max())
        ax.set_ylim(xx2.min(), xx2.max())
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(title)

    plt.savefig('./examples/example_SVM_02.png')


if __name__ == "__main__":
    main()
