import os
import sys

import numpy as np
import pandas as pd
from sklearn import datasets

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.realpath(__file__)))

from ml.decision_tree import decision_tree
from ml.math_tools import mt


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
    clf = decision_tree.RT()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    mt.mean_squared_error(y_test, y_pred)


if __name__ == "__main__":
    main()
