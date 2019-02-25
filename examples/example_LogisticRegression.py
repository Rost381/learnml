import math

import matplotlib.pyplot as plt
import numpy as np

from zero.api import LogisticRegression
from zero.datasets.api import load_iris
from zero.utils.api import calculate_accuracy_score


def main():
    iris = load_iris()
    X = iris["data"][:, 3:]  # petal width
    y = (iris["target"] == 2).astype(np.int)  # 1 if Iris-Virginica, else 0

    model = LogisticRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    print(y_pred, y)
    accuracy = calculate_accuracy_score(y, y_pred)
    print("Accuracy Score: {:.2%}".format(accuracy))

    X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
    y_proba = model.predict_proba(X_new)
    plt.plot(X_new, y_proba[:, 1], "g-", linewidth=2, label="Iris-Virginica")
    plt.plot(X_new, y_proba[:, 0], "b--",
             linewidth=2, label="Not Iris-Virginica")
    plt.plot(X[y == 0], y[y == 0], "bs")
    plt.plot(X[y == 1], y[y == 1], "g^")
    plt.legend()
    # plt.show()
    plt.savefig("./examples/example_LogisticRegression.png")


if __name__ == "__main__":
    main()
