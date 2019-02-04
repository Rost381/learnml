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

    df = pd.read_csv("data/banknote.csv", header=None)
    train, test, test_total = mt.data_train_test(df)

    dt = decision_tree.DT()
    tree = dt.build_tree(train, 5, 1)

    correct_count = 0
    for row in test:
        prediction = dt.predict(tree, row)
        if row[-1] == prediction:
            correct_count += 1
        print('Actual=%d, Predict=%d' % (row[-1], prediction))

    print('{:.2%}'.format(float(correct_count / test_total)))


if __name__ == "__main__":
    main()
