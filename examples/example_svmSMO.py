import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.realpath(__file__)))

from zero.api import svmSMO


def main():
    df = pd.read_csv('data/svm.csv', header=None)
    y = df.iloc[:, -1].values
    X = df.iloc[:, :-1].values

    model = svmSMO(max_iter=1000, kernel='linear')
    model.fit(X, y)

    print(model.w)
    print(model.b)

    plt.scatter(X[:, 0], X[:, 1], c=y, zorder=10, cmap=plt.cm.Paired,
                edgecolors='k')
    x = np.linspace(2, 8)
    y = model.w * x + model.b
    plt.plot(x, y, 'k-')
    plt.savefig('example_svmSMO.png')


if __name__ == "__main__":
    main()
