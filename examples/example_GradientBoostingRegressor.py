import matplotlib.pyplot as plt
import numpy as np

from zero.api import GradientBoostingRegressor
from zero.utils.api import calculate_mean_squared_error
from zero.datasets.api import load_temperature


def main():
    temp = load_temperature()

    X = np.atleast_2d((temp.data)).T
    y = np.atleast_2d((temp.target)).T

    X = X.reshape((-1, 1))
    X = np.insert(X, 0, values=1, axis=1) 
    y = y[:, 0]

    model = GradientBoostingRegressor(n_estimators=100,
                                      learning_rate=0.5,
                                      min_samples_split=2,
                                      min_impurity_split=1e-7,
                                      max_depth=4)
    model.fit(X, y)

    y_pred = model.predict(X)

    mse = calculate_mean_squared_error(y, y_pred)
    print("MSE: {0}".format(mse))
    
    fig = plt.figure()
    plt.plot(365 * X, y, 'b.', label='Observations')
    plt.plot(365 * X, y_pred, 'r-', label='Prediction')
    plt.legend(loc='upper left')
    plt.savefig("./examples/example_GradientBoostingRegressor.png")


if __name__ == "__main__":
    main()
