import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.realpath(__file__)))

from ml_student.simple_linear_regression import simple_linear_regression
from ml_student.math_tools import mt


def main():

    df = pd.read_csv('data/insurance.csv', header=None)

    train, test = train_test_split(df, test_size=0.4)
    train, test = train.values.tolist(), test.values.tolist()

    lr = simple_linear_regression.LinearRegression()
    b0, b1 = lr.fit(train)

    y_pred = lr.predict(train, test)
    y_test = np.array([row[-1] for row in test])

    print('b0={0}, b1={1}'.format(b0, b1))
    mt.calculate_mean_squared_error(y_test, y_pred)


if __name__ == "__main__":
    main()
