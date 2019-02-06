import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.realpath(__file__)))

from ml.linear_regression import linear_regression
from ml.math_tools import mt


def main():

    df = pd.read_csv('data/insurance.csv', header=None)

    train, test = train_test_split(df, test_size=0.4)
    train, test = train.values.tolist(), test.values.tolist()

    lr = linear_regression.LR()
    b0, b1 = lr.coefficients(train)

    y_pred = lr.predict(train, test)
    y_test = [row[-1] for row in test]

    mse = lr.mse(y_test, y_pred)
    print('b0={0}, b1={1}, MSE={2}'.format(b0, b1, mse))


if __name__ == "__main__":
    main()
