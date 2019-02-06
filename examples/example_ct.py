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
    Classification Tree
    """
    df = pd.read_csv("data/banknote.csv", header=None)

    y = df.iloc[:, -1].values
    X = df.iloc[:, :-1].values
    X_train, X_test, y_train, y_test = mt.data_train_test_split(
        X, y, test_size=0.4)

    clf = decision_tree.CT()
    clf.fit(X_train, y_train)
    clf.print_tree()

    y_pred = clf.predict(X_test)
    mt.accuracy_score(y_test, y_pred)


if __name__ == "__main__":
    main()
