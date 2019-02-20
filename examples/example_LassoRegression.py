import os
import sys

import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.realpath(__file__)))

from zero.api import LassoRegression


def main():

    df = pd.read_csv("data/Boston.csv", index_col=0)
    y = df.iloc[:,  13].values
    df = (df - df.mean()) / df.std()
    X = df.iloc[:, :13].values
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
