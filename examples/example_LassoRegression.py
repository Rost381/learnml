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

from sklearn.metrics import r2_score


def main():

    df = pd.read_csv("data/Boston.csv", index_col=0)
    y = df.iloc[:,  13].values
    df = (df - df.mean()) / df.std()  # 基準化
    X = df.iloc[:, :13].values
    X_train, X_test, y_train, y_test = mt.data_train_test_split(
        X, y, test_size=0.4)

    model = regression.LassoRegression(
        alpha=0.1, max_iter=1000, fit_intercept=True)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    Score = r2_score(y_test, y_pred)

    print(model.intercept_)
    print(model.coef_)
    print("r^2 on test data : %f" % Score)


if __name__ == "__main__":
    main()
