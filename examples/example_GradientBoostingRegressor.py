import matplotlib.pyplot as plt
import numpy as np

from learnml.datasets.api import load_temperature
from learnml.supervised.api import GradientBoostingRegressor
from learnml.utils.api import calculate_mean_squared_error


def main():
    np.random.seed(1)

    def f(x):
        """The function to predict."""
        return x * np.sin(x)

    X = np.atleast_2d(np.random.uniform(0, 10.0, size=100)).T
    X = X.astype(np.float32)

    # Observations
    y = f(X).ravel()

    dy = 1.5 + 1.0 * np.random.random(y.shape)
    noise = np.random.normal(0, dy)
    y += noise
    y = y.astype(np.float32)

    model = GradientBoostingRegressor(n_estimators=200,
                                      learning_rate=0.5,
                                      min_samples_split=2,
                                      min_impurity_split=1e-7,
                                      max_depth=4)
    model.fit(X, y)

    y_pred = model.predict(X)

    xx = np.atleast_2d(np.linspace(0, 10, 1000)).T
    xx = xx.astype(np.float32)

    fig = plt.figure()
    plt.plot(xx, f(xx), 'g:', label=r'$f(x) = x\,\sin(x)$')
    plt.plot(X, y_pred, 'r.', label=u'Prediction')
    plt.plot(X, y, 'b.', label=u'Observations')
    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')
    plt.ylim(-10, 20)
    plt.legend(loc='upper left')
    plt.savefig("./examples/example_GradientBoostingRegressor.png")


if __name__ == "__main__":
    main()
