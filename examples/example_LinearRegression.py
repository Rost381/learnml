import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

from learnml.supervised.api import LinearRegression
from learnml.utils.api import calculate_mean_squared_error


def main():
    # Example 1 One-dimensional
    X, y = make_regression(n_samples=10, n_features=1, noise=10)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4)

    n_samples, n_features = np.shape(X)

    model = LinearRegression()

    model.fit(X_train, y_train)
    print(model.w)
    y_pred = model.predict(X_test)
    mse = calculate_mean_squared_error(y_test, y_pred)
    print("MSE: {0}".format(mse))

    # Example 2 Two-dimensional
    """
    y = 1 * x_0 + 2 * x_1 + 3
    1 * 3 + 2 * 5 + 3 = 16
    """
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([1, 2])) + 3

    X_test = np.array([[3, 5]])

    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X_test)
    print(y_pred)


if __name__ == "__main__":
    main()
