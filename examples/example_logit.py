import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.realpath(__file__)))

from ml.logistic_regression import logistic_regression
from ml.math_tools import mt


def main():

    dataset = [[2.7810836, 2.550537003, 0],
               [1.465489372, 2.362125076, 0],
               [3.396561688, 4.400293529, 0],
               [1.38807019, 1.850220317, 0],
               [3.06407232, 3.005305973, 0],
               [7.627531214, 2.759262235, 1],
               [5.332441248, 2.088626775, 1],
               [6.922596716, 1.77106367, 1],
               [8.675418651, -0.242068655, 1],
               [7.673756466, 3.508563011, 1]]

    l_rate = 0.3
    n_epoch = 100
    train, test = train_test_split(dataset, test_size=0.4)

    logit = logistic_regression.LOGIT()
    coef = logit.fit(train, l_rate, n_epoch)
    print(coef)

    correct_count = 0
    test_total = len(test)
    for row in test:
        y_pred = round(logit.predict(row, coef))
        y_test = row[-1]
        if y_pred == y_test:
            correct_count += 1
        print('Actual=%d, Predict=%d' % (y_test, y_pred))

    print('Accuracy Score: {:.2%}'.format(float(correct_count / test_total)))


if __name__ == "__main__":
    main()
