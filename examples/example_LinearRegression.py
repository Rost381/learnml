import os
import sys

import numpy as np
import pandas as pd
from sklearn.datasets import make_regression

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.realpath(__file__)))

from ml_student.regression import regression
from ml_student.math_tools import mt


def main():
    X, y = make_regression(n_samples=10, n_features=1, noise=10)
    X_train, X_test, y_train, y_test = mt.data_train_test_split(
        X, y, test_size=0.4)

    n_samples, n_features = np.shape(X)

    model = regression.LinearRegression()

    model.fit(X_train, y_train)
    print(model.w)
    y_pred = model.predict(X_test)
    mse = mt.calculate_mean_squared_error(y_test, y_pred)
    print("MSE: {0}".format(mse))


if __name__ == "__main__":
    main()
