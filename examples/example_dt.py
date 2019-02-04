import os
import sys

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.realpath(__file__)))

from ml.decision_tree import decision_tree
from ml.math_tools import mt


def main():

    df = pd.read_csv("data/banknote.csv", header=None)
    train, test = train_test_split(df, test_size=0.4)
    total = test.shape[0]
    train, test = train.values.tolist(), test.values.tolist()

    dt = decision_tree.DT()
    tree = dt.build_tree(train, 5, 1)

    correct_count = 0
    for row in test:
        prediction = dt.predict(tree, row)
        if row[-1] == prediction:
            correct_count += 1
        print('Actual=%d, Predict=%d' % (row[-1], prediction))

    print('{:.2%}'.format(float(correct_count / total)))


if __name__ == "__main__":
    main()
