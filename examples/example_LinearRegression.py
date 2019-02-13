import os
import sys

import numpy as np
import pandas as pd
from sklearn.datasets import make_regression

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.realpath(__file__)))

from ml_student.regression import regression
from ml_student.math_tools import mt
import statsmodels.api as sm


def main():
    X, y = make_regression(n_samples=10, n_features=1, noise=10)
    X_train, X_test, y_train, y_test = mt.data_train_test_split(
        X, y, test_size=0.4)

    n_samples, n_features = np.shape(X)

    lr = regression.LinearRegression()

    lr.fit(X_train, y_train)
    print(lr.w)
    y_pred = lr.predict(X_test)
    mse = mt.calculate_mean_squared_error(y_test, y_pred)

    "============================"
    X_train = sm.add_constant(X_train)
    model = sm.OLS(y_train, X_train)
    results = model.fit()
    print(results.params)

    lr = regression.LinearRegression(
        n_iterations=10, learning_rate=0.001, gradient_descent=True)
    X = np.array([[1, 1, 1], [1, 1, 2], [1, 2, 1]])
    y = np.array([6, 9, 8])
    lr.fit(X, y)


if __name__ == "__main__":
    main()
