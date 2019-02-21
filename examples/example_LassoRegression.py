import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from zero.api import LassoRegression
from zero.datasets.api import load_boston
from zero.utils.api import standardize


def main():
    boston = load_boston()
    y = boston.target
    X = standardize(boston.data)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4)

    model = LassoRegression(
        alpha=0.1, max_iter=1000, fit_intercept=True)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    Score = r2_score(y_test, y_pred)

    print(model.intercept_)
    print(model.coef_)
    print("Score : %f" % Score)


if __name__ == "__main__":
    main()
