import matplotlib.pyplot as plt
import numpy as np

from zero.api import LogisticRegression
from zero.datasets.api import load_iris
from zero.utils.api import calculate_accuracy_score


def main():
    iris = load_iris()
    X = iris["data"][:, 3:]  # petal width
    y = (iris["target"] == 2).astype(np.int)  # 1 if Iris-Virginica, else 0

    model = LogisticRegression(
        max_iter=5000, learning_rate=0.001, fit_intercept=True)
    model.fit(X, y)
    y_pred = model.predict(X)

    accuracy = calculate_accuracy_score(y, y_pred)
    print("Accuracy Score: {:.2%}".format(accuracy))

    X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
    y_proba = model.predict_proba(X_new)

    plt.plot(X_new, y_proba[:, 0], "r--", label="Not Iris-Virginica")
    plt.plot(X_new, y_proba[:, 1], "g-", label="Iris-Virginica")

    plt.plot(X[y == 0], y[y == 0], "ro")
    plt.plot(X[y == 1], y[y == 1], "go")
    plt.legend()
    plt.show()
    plt.savefig("./examples/example_LogisticRegression.png")


if __name__ == "__main__":
    main()
