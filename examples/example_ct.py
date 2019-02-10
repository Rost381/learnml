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
    Classification Tree
    """
    df = pd.read_csv("data/banknote.csv", header=None)

    y = df.iloc[:, -1].values
    X = df.iloc[:, :-1].values
    X_train, X_test, y_train, y_test = mt.data_train_test_split(
        X, y, test_size=0.4)

    ct = decision_tree.CT()
    ct.fit(X_train, y_train)
    ct.print_tree()

    y_pred = ct.predict(X_test)
    mt.accuracy_score(y_test, y_pred)


if __name__ == "__main__":
    main()
