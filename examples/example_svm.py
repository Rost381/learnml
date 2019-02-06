import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.realpath(__file__)))

from ml.support_vector_machine import support_vector_machine
from ml.math_tools import mt


def main():

    df = pd.read_csv('data/svm.csv', header=None)
    y = df.iloc[:, -1].values
    X = df.iloc[:, :-1].values
    X_train, X_test, y_train, y_test = mt.data_train_test_split(
        X, y, test_size=0.4)

    svm = support_vector_machine.SVM(max_iter=1000, kernel='poly')
    svm.fit(X_train, y_train)

    print(svm.score(X_test, y_test))


if __name__ == "__main__":
    main()
