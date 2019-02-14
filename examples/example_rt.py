import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.realpath(__file__)))

from ml_student.decision_tree import decision_tree
from ml_student.math_tools import mt


def main():
    """
    Regression Tree
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


if __name__ == "__main__":
    main()
