import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from zero.api import PolynomialRidgeRegression
from zero.datasets.api import load_temperature
from zero.utils.api import calculate_mean_squared_error


def main():
    temp = load_temperature()

    X = np.atleast_2d((temp.data)).T
    y = temp.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

    poly_degree = 16
    lowest_error = float("inf")
    best_reg_factor = None

    for reg_factor in np.arange(0, 0.1, 0.01):
        mse = 0
        model = PolynomialRidgeRegression(degree=poly_degree,
                                          reg_factor=reg_factor,
                                          learning_rate=0.001,
                                          max_iter=10000)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        _mse = calculate_mean_squared_error(y_test, y_pred)
        mse += _mse
        print("Mean squared error: %s (given by reg. factor: %s)" %
              (lowest_error, best_reg_factor))
        if mse < lowest_error:
            best_reg_factor = reg_factor
            lowest_error = mse

    print(best_reg_factor, lowest_error)

    model = PolynomialRidgeRegression(degree=poly_degree,
                                      reg_factor=best_reg_factor,
                                      learning_rate=0.001,
                                      max_iter=10000)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = calculate_mean_squared_error(y_test, y_pred)
    print("Mean squared error: %s (given by reg. factor: %s)" %
          (lowest_error, best_reg_factor))

    y_pred_line = model.predict(X)

    cmap = plt.get_cmap('viridis')

    m1 = plt.scatter(365 * X_train, y_train, color=cmap(0.9),s=10)
    m2 = plt.scatter(365 * X_test, y_test, color=cmap(0.5),s=10)
    plt.plot(365 * X, y_pred_line, color='black',
             linewidth=2, label="Prediction")

    plt.legend((m1, m2), ("Training data", "Test data"), loc='lower right')
    plt.savefig('./examples/example_PolynomialRidgeRegression.png')


if __name__ == "__main__":
    main()
