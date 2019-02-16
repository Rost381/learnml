import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.realpath(__file__)))

from ml_student.decision_tree import decision_tree
from ml_student.math_tools import mt


def main():
    """ example 1
    """
    df = pd.read_csv("data/tempature.csv")

    X = np.atleast_2d(df["time"].as_matrix()).T
    y = np.atleast_2d(df["temp"].as_matrix()).T
    X = mt.standardize(X)
    y = y[:, 0]

    X_train, X_test, y_train, y_test = mt.data_train_test_split(
        X, y, test_size=0.4)
    rt = decision_tree.RegressionTree()
    rt.fit(X_train, y_train)

    y_pred = rt.predict(X_test)
    mse = mt.calculate_mean_squared_error(y_test, y_pred)
    print("MSE: {0}".format(mse))

    """ example 2
    The decision trees is used to fit a sine curve with addition noisy observation.
    As a result, it learns local linear regressions approximating the sine curve.
    """
    rng = np.random.RandomState(1)
    X = np.sort(5 * rng.rand(80, 1), axis=0)
    y = np.sin(X).ravel()
    y[::5] += 3 * (0.5 - rng.rand(16))

    # Fit regression model
    model_1 = decision_tree.RegressionTree(max_depth=2)
    model_2 = decision_tree.RegressionTree(max_depth=5)
    model_1.fit(X, y)
    model_2.fit(X, y)

    # Predict
    X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
    y_1 = model_1.predict(X_test)
    y_2 = model_2.predict(X_test)

    # Plot the results
    plt.figure()
    plt.scatter(X, y, s=20, edgecolor="black", c="darkorange", label="data")
    plt.plot(X_test, y_1, color="cornflowerblue",
             label="max_depth=2", linewidth=2)
    plt.plot(X_test, y_2, color="yellowgreen",
             label="max_depth=5", linewidth=2)
    plt.xlabel("data")
    plt.ylabel("target")
    plt.title("Decision Tree Regression")
    plt.legend()
    # plt.show()
    plt.savefig('example_RegressionTree.png')


if __name__ == "__main__":
    main()
