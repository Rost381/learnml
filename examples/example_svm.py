import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.realpath(__file__)))

from mlalgo.api import SVM


def main():
    df = pd.read_csv('data/svm.csv', header=None)
    y = df.iloc[:, -1].values
    X = df.iloc[:, :-1].values

    model = SVM(max_iter=1000, kernel='linear')
    model.fit(X, y)

    print(model.w)
    print(model.b)

    plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)
    x = np.linspace(0, 10)
    y = model.w * x + model.b
    plt.plot(x, y, 'k-')
    plt.savefig('example_SVM.png')


if __name__ == "__main__":
    main()
