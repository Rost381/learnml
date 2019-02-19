import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.realpath(__file__)))

from zero.api import svmCVXOPT


def main():
    df = pd.read_csv('data/svm.csv', header=None)
    y = df.iloc[:, -1].values
    X = df.iloc[:, :-1].values

    model = svmCVXOPT()
    model.fit(X, y)

    print(model.coef_, model.intercept_)
    print(model.support_vectors_)

    w = model.coef_
    a = -w[0] / w[1]
    xx = np.linspace(2, 8)
    yy = a * xx - (model.intercept_[0]) / w[1]

    # plot the parallels to the separating hyperplane that pass through the
    # support vectors (margin away from hyperplane in direction
    # perpendicular to hyperplane). This is sqrt(1+a^2) away vertically in
    # 2-d.
    margin = 1 / np.sqrt(np.sum(model.coef_ ** 2))
    yy_down = yy - np.sqrt(1 + a ** 2) * margin
    yy_up = yy + np.sqrt(1 + a ** 2) * margin

    # plot the line, the points, and the nearest vectors to the plane
    plt.plot(xx, yy, 'k-')
    plt.plot(xx, yy_down, 'k--')
    plt.plot(xx, yy_up, 'k--')

    plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=180,
                facecolors='none', zorder=10, edgecolors='k')
    plt.scatter(X[:, 0], X[:, 1], c=y, zorder=10, cmap=plt.cm.Paired,
                edgecolors='k')

    plt.savefig('example_svmCVXOPT.png')


if __name__ == "__main__":
    main()
