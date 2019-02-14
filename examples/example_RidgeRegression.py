import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.realpath(__file__)))

from ml_student.math_tools import mt
from ml_student.regression import regression


def main():
    X, y, w = make_regression(n_samples=10, n_features=10, coef=True,
                              random_state=1, bias=3.5)

    coefs = []
    errors = []

    alphas = np.logspace(-4, 4, 200)

    for alpha in alphas:
        model = regression.RidgeRegression(
            alpha=alpha, n_iterations=1000, learning_rate=1e-4)
        model.fit(X, y)

        coef = model.w
        y_pred = model.predict(X)
        error = mt.calculate_mean_squared_error(y, y_pred)

        coefs.append(coef)
        errors.append(error)

    plt.figure(figsize=(20, 6))

    plt.subplot(121)
    ax = plt.gca()
    ax.plot(alphas, coefs)
    ax.set_xscale('log')
    plt.xlabel('alpha')
    plt.ylabel('weights')
    plt.title('Ridge coefficients as a function of the regularization')
    plt.axis('tight')

    plt.subplot(122)
    ax = plt.gca()
    ax.plot(alphas, errors)
    ax.set_xscale('log')
    plt.xlabel('alpha')
    plt.ylabel('error')
    plt.title('Coefficient error as a function of the l2 regularization')
    # plt.show()
    plt.savefig('example_RidgeRegression.png')


if __name__ == "__main__":
    main()
